"""taste-5 API — the contract surface over UDS/Granian/FastAPI.

Every endpoint maps 1:1 to CONTRACT.md §3 (plus POST /distill, an allowed
extension — the contract is a minimum surface). create_app() is the
conformance entry point (TASTE_APP=taste.api:create_app + TASTE_APP_FACTORY=1).
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .core import StandardsEngine
from .distill import distill
from .embed import embed
from .feedback import record_feedback
from .hone import hone_segment
from .ingest import ingest_docs
from .inject import compile_packet
from .interview import run_interview
from .mirror import build_mirror
from .models import (
    DistillRequest,
    DistillResponse,
    FeedbackRequest,
    FeedbackResponse,
    HoneRequest,
    HoneResponse,
    ImportRequest,
    IngestRequest,
    IngestResponse,
    InterviewRequest,
    InterviewResponse,
    LayerState,
    MetaResponse,
    MirrorRequest,
    MirrorResponse,
    Packet,
    PacketRequest,
    Provenance,
    ScoreRequest,
    ScoreResult,
    Standard,
)
from .score import score_candidates
from .store import HoneBudgetExhausted, Store

PERMUTATION = {
    "schema": "A1-atomic", "retrieval": "B1-cosine-exemplar-biased",
    "embed": "C2-Qwen3-0.6B", "scorer": "D4-hybrid-judge-tiebreak",
    "interview": "E1-adaptive", "hone": "F4-escalating-BoN",
    "storage": "G1-sqlite-wal", "memory": "H3-episode-to-pattern",
    "injection": "I3-contrastive-pairs", "pipeline": "J1-fixed",
}


def _error_body(status: int, message: str) -> dict:
    return {"error": {"code": f"e{status}", "message": message}}


def create_app(db_path: str | Path | None = None) -> FastAPI:
    store = Store(db_path) if db_path else Store()
    engine = StandardsEngine(store)

    app = FastAPI(title="taste-5", version="0.1.0")
    app.state.store = store
    app.state.engine = engine

    # contract §1 error shape: {"error": {"code", "message"}}
    @app.exception_handler(HTTPException)
    async def _http_error(_request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content=_error_body(exc.status_code,
                                                str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request,
                                exc: RequestValidationError):
        first = exc.errors()[:1]
        message = first[0].get("msg", "invalid request") if first else "invalid request"
        prefix = "Value error, "
        message = message.removeprefix(prefix)
        return JSONResponse(status_code=422,
                            content=_error_body(422, message))

    # ------------------------------------------------------------ meta & check
    @app.get("/meta", response_model=MetaResponse)
    def meta():
        return MetaResponse(name="taste-5", version="0.1.0",
                            permutation=PERMUTATION)

    @app.get("/check")
    def check(profile_id: str = "owner"):
        active = engine.active(profile_id)
        all_stds = store.list_standards(profile_id)
        cats: dict[str, int] = {}
        pols: dict[str, int] = {}
        for s in active:
            c = s.category.value if hasattr(s.category, "value") else str(s.category)
            p = s.polarity.value if hasattr(s.polarity, "value") else str(s.polarity)
            cats[c] = cats.get(c, 0) + 1
            pols[p] = pols.get(p, 0) + 1
        return {
            "status": "healthy",
            "name": "taste-5",
            "version": "0.1.0",
            "active_standards": len(active),
            "total_standards": len(all_stds),
            "categories": cats,
            "polarities": pols,
        }


    # ------------------------------------------------------------ task flow
    @app.post("/interview", response_model=InterviewResponse)
    def interview(req: InterviewRequest):
        return run_interview(engine, req.profile_id, req.task_ctx)

    @app.post("/mirror", response_model=MirrorResponse)
    def mirror(req: MirrorRequest):
        return build_mirror(engine, store, req)

    @app.post("/packet", response_model=Packet)
    def packet(req: PacketRequest):
        return compile_packet(engine, req)

    @app.post("/score", response_model=ScoreResult)
    def score(req: ScoreRequest):
        return ScoreResult(scores=score_candidates(
            engine, req.profile_id, req.task_ctx, req.candidates))

    @app.post("/hone", response_model=HoneResponse)
    def hone(req: HoneRequest):
        try:
            return hone_segment(engine, req)
        except HoneBudgetExhausted as exc:
            raise HTTPException(429, str(exc)) from exc

    # ------------------------------------------------------------ learning
    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest):
        return record_feedback(engine, store, req)

    @app.post("/distill", response_model=DistillResponse)
    def distill_endpoint(req: DistillRequest):
        return distill(engine, store, req.profile_id,
                       since=req.since, min_cluster=req.min_cluster)

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest):
        resp = ingest_docs(req.profile_id, req.docs)
        for cand in resp.candidates:
            engine.propose(cand)   # lands provisional — never auto-active
        return resp

    # ------------------------------------------------------------ corpus
    @app.get("/standards")
    def standards(profile_id: str = "owner",
                  state: str | None = None):
        try:
            st = LayerState(state) if state else None
        except ValueError as exc:
            raise HTTPException(
                422, f"unknown state '{state}' — expected one of "
                f"{[s.value for s in LayerState]}") from exc
        rows = store.list_standards(profile_id, state=st)
        return {"standards": [s.model_dump(mode="json") for s in rows],
                "count": len(rows)}

    @app.post("/standards/import")
    def standards_import(req: ImportRequest):
        existing = {(s.profile_id, re.sub(r"\s+", " ",
                    s.directive.lower())) for s in
                    store.list_standards(req.profile_id)}
        imported = 0
        for s in req.standards:
            key = (s.profile_id, re.sub(r"\s+", " ", s.directive.lower()))
            if key in existing:
                continue
            # imported layers never silently activate if unconfirmed at
            # source, or if they lack a rationale (NEVER rule 1)
            if (s.provenance in (Provenance.AMENDMENT, Provenance.INFERENCE)
                    or not s.rationale.strip()):
                s.state = LayerState.PROVISIONAL
            # Declared state round-trips. It does not approve the row here.
            meta = dict(s.variant_meta)
            meta["local_approved"] = False
            s.variant_meta = meta
            vec_text = s.directive + " " + s.rationale
            store.put_standard(s, embed([vec_text])[0])
            existing.add(key)
            imported += 1
        return {"imported": imported, "skipped": len(req.standards) - imported}

    # ------------------------------------------------------ owner authority
    @app.post("/standards/{sid}/confirm")
    def confirm(sid: str, body: dict | None = None):
        rationale = (body or {}).get("rationale")
        try:
            ok = engine.confirm(sid, rationale=rationale)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not ok:
            raise HTTPException(404, f"standard {sid} not found")
        return {"activated": sid}

    @app.post("/standards/{sid}/supersede")
    def supersede(sid: str, new: Standard):
        if not store.get_standard(sid):
            raise HTTPException(404, f"standard {sid} not found")
        return engine.supersede(sid, new)

    @app.get("/review-queue")
    def review_queue(profile_id: str = "owner"):
        return {"pending": store.pending_inferred(profile_id)}

    @app.post("/review-queue/{iid}/resolve")
    def resolve(iid: str):
        store.resolve_inferred(iid)
        return {"resolved": iid}

    return app


app = create_app()
