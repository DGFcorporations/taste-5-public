"""feedback — the corpus builder, shared by api.py and mcp_server.py.

Contract: a decision record never exists without rationale (NEVER rule 1 —
enforced upstream by pydantic min_length).

taste-5 writes TWO things per decision:
  1. a provisional why-layer (the rationale as reusable doctrine — kept
     from control; propose-never-apply)
  2. contrast pairs — when rejected[] and the chosen side resolve to real
     text, {rejected, chosen, why} lands in the pair store, immediately
     injectable (owner-decision fact, not a generalization)
"""

from __future__ import annotations

from .core import StandardsEngine
from .embed import embed
from .models import (
    Candidate,
    Category,
    ContrastPair,
    DecisionRecord,
    FeedbackRequest,
    FeedbackResponse,
    Polarity,
    Provenance,
    Standard,
)
from .store import Store

MIN_PAIR_TEXT = 3   # bare labels/empties never become pair text


def _resolve(token: str, candidates: list[Candidate]) -> str:
    """'A' → candidate A's content when candidates are supplied."""
    for c in candidates:
        if c.label == token:
            return (c.content or c.content_ref or "").strip()
    return ""


def record_feedback(engine: StandardsEngine, store: Store,
                    req: FeedbackRequest) -> FeedbackResponse:
    rec = DecisionRecord(
        profile_id=req.profile_id, task_ctx=req.task_ctx,
        candidates=req.candidates, pick=req.pick, rationale=req.rationale,
        agent_role=str(req.task_ctx.get("role", "")),
        provenance="owner_pick")
    store.put_decision(rec)

    task_type = str(req.task_ctx.get("type", ""))

    # provisional why-layer — the rationale as reusable doctrine
    engine.propose(Standard(
        profile_id=req.profile_id, category=Category.DOCTRINE,
        polarity=Polarity.WHY,
        directive=f"When choosing for {task_type or 'this kind of task'},"
                  f" prefer what won as '{req.pick}'",
        rationale=req.rationale, provenance=Provenance.DECISION,
        variant_meta={"decision_id": rec.id}))

    # contrast pairs — resolve both sides to real text or write nothing
    chosen = (req.chosen or "").strip() or _resolve(req.pick, req.candidates)
    pairs_written = 0
    for token in req.rejected:
        rejected = _resolve(token, req.candidates) or token.strip()
        if len(chosen) < MIN_PAIR_TEXT or len(rejected) < MIN_PAIR_TEXT:
            continue
        pair = ContrastPair(
            profile_id=req.profile_id, task_type=task_type,
            rejected=rejected, chosen=chosen, why=req.rationale,
            source="feedback", source_id=rec.id)
        store.put_pair(pair, embed([pair.chosen + " " + pair.rejected])[0])
        pairs_written += 1

    return FeedbackResponse(
        accepted=True, record_id=rec.id,
        extracted={"candidates": 1, "pairs": pairs_written})
