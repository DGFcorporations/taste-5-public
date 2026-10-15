"""hone — F4 permutation: escalating best-of-N.

N=1 first. If the single shot clears the threshold, done — most segments
finish in one call and the cost saving is the point of the axis. Otherwise
generate 2 more, pick argmax, and flag iterate if still under threshold.

Every call writes exactly one hone_events row — the episode store H3
distills from. n_generated makes the escalation observable in SQL.

The driver is a factory: driver(draft, rules) -> gen(n) emitting n fresh
candidates per call. The offline default emits a deterministic pool lazily,
so a strong draft provably produces exactly one generated candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

from .core import StandardsEngine
from .inject import compile_packet
from .models import (
    Candidate,
    HoneEvent,
    HoneRequest,
    HoneResponse,
    PacketRequest,
)
from .score import score_candidates

HONE_LIMIT = 15


def task_key(ctx: dict) -> str:
    """Segments of one task share a hone budget. Attendance does not reset it."""
    for field in ("task_ref", "ref"):
        value = ctx.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    relevant = {key: value for key, value in ctx.items() if key not in {"segment", "attended"}}
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return "task_" + hashlib.sha256(canonical.encode()).hexdigest()


SCORE_THRESHOLD = 0.55
ESCALATE_TO = 3          # total candidates generated when first shot fails

# driver(draft, rules) -> gen(n) -> list[str] of n fresh candidates
DriverFactory = Callable[[str | None, list[str]], Callable[[int], list[str]]]

HEDGE_WORDS = ["just", "really", "very", "basically", "actually",
               "in today's world", "it's important to note"]


class _OfflineDriver:
    """Deterministic lazy candidate pool — conformance-safe, zero spend.

    Pool order: tightened draft → restructured → directive-framed. Emitted
    lazily so escalation spend is real: gen(1) yields only the first.
    """

    def __init__(self, draft: str | None, rules: list[str]):
        base = (draft or "").strip()
        tight = base
        for w in HEDGE_WORDS:
            tight = re.sub(rf"\b{re.escape(w)}\b[:,]?\s*", "", tight,
                           flags=re.IGNORECASE)
        tight = re.sub(r"\s+", " ", tight).strip()

        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", base)
                 if s.strip()]
        reordered = " ".join(sorted(sents, key=len, reverse=True)) or base

        framed = base
        if rules:
            framed = f"{rules[0].rstrip('.')} — {base}"

        self._pool = [tight or base, reordered or base, framed or base]
        self._i = 0

    def __call__(self, n: int) -> list[str]:
        out = self._pool[self._i:self._i + n]
        self._i += len(out)
        return out


def _log_event(engine: StandardsEngine, req: HoneRequest,
               candidates: list[Candidate], pick: str, score: float,
               violations: list[str], n_generated: int):
    engine.store.put_hone_event(HoneEvent(
        profile_id=req.profile_id,
        task_type=str(req.segment_ctx.get("type", "")),
        segment=str(req.segment_ctx.get("segment", "")),
        draft=req.draft or "", candidates=candidates, pick=pick,
        score=score, violations=violations, n_generated=n_generated))


def hone_segment(engine: StandardsEngine, req: HoneRequest,
                 driver: DriverFactory | None = None) -> HoneResponse:
    reservation = engine.store.reserve_hone(
        req.profile_id, task_key(req.segment_ctx), HONE_LIMIT)
    room = reservation < HONE_LIMIT
    ctx = {**req.segment_ctx, "attended": False}
    packet = compile_packet(engine, PacketRequest(
        profile_id=req.profile_id, task_ctx=ctx))
    do_rules = [d.text for d in packet.directives
                if d.polarity.value == "do"]

    gen = (driver or _OfflineDriver)(req.draft, do_rules)
    labels = iter("ABCDEFGHIJ")
    generated: list[Candidate] = []

    def _emit(n: int) -> list[Candidate]:
        new = [Candidate(label=next(labels), content=t) for t in gen(n)]
        generated.extend(new)
        return new

    def _score(cands: list[Candidate]):
        scored = score_candidates(engine, req.profile_id,
                                  req.segment_ctx, cands)
        by_ref = {s.ref: s.score for s in scored}
        for c in cands:
            c.score = by_ref.get(c.label, c.score)  # episodes carry scores
        return scored

    # F4: one shot first — a clear draft never pays for a second call
    first = _emit(1)
    if not first:
        _log_event(engine, req, generated, "", 0.0, [], n_generated=0)
        return HoneResponse(candidates=[], pick="", score=0.0, iterate=room)
    s1 = _score(first)[0]
    if s1.score >= SCORE_THRESHOLD:
        _log_event(engine, req, generated, s1.ref, s1.score,
                   s1.violations, n_generated=1)
        return HoneResponse(candidates=generated, pick=s1.ref,
                            score=s1.score, iterate=False)

    # escalate: fill out the BoN-3 pool, pick argmax
    _emit(ESCALATE_TO - 1)
    scored = _score(generated)
    best = max(scored, key=lambda s: s.score)
    _log_event(engine, req, generated, best.ref, best.score,
               best.violations, n_generated=len(generated))
    return HoneResponse(candidates=generated, pick=best.ref,
                        score=best.score,
                        iterate=best.score < SCORE_THRESHOLD and room)
