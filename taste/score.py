"""score — D4 permutation: rules gate → judge on borderline.

Deterministic rules score every candidate (control mechanics). When the top
two land within BORDERLINE_DELTA of each other, an LLM judge reranks only
that borderline set — shown the packet's own contrast pairs as evidence, so
the tiebreak is decided by the same teaching signal the model saw.

Judge is opt-in (TASTE_JUDGE_MODEL); unset or failed → rules stand, logged.
"""

from __future__ import annotations

import logging
import re

from . import llm
from .core import StandardsEngine
from .embed import cosine, embed
from .inject import compile_packet
from .models import (
    Candidate,
    CandidateScore,
    PacketRequest,
    Polarity,
)

log = logging.getLogger("taste.score")

BORDERLINE_DELTA = 0.15   # |top − second| below this → judge tiebreak

QUOTED = re.compile(
    r'"([^"]{4,})"|“([^”]{4,})”'      # double / smart quotes
    r"|(?<![\w'])'([^']{4,})'(?!\w)"  # single quotes, apostrophe-safe
)
LEADIN = re.compile(
    r"^(never|avoid|do not|don't|no|stop|refrain from)\s+", re.IGNORECASE)
SUB_ACTION = re.compile(
    r"^(use|write|include|say|post|generate|employ)\s+", re.IGNORECASE)


def _quoted_strings(text: str) -> list[str]:
    return [next(g for g in m if g) for m in QUOTED.findall(text)]


def _forbidden_strings(directive: str) -> list[str]:
    """Extract the concrete forbidden material from a never-directive.

    The full stripped phrase is always a candidate — short rules like
    "No emoji" must still fire; the ≥4-words gate used to drop them.
    """
    quoted = _quoted_strings(directive)
    stripped = LEADIN.sub("", directive).strip().rstrip(".")
    phrases = list(quoted)
    if stripped:
        phrases.append(stripped)
        sub = SUB_ACTION.sub("", stripped).strip()
        if sub and sub != stripped and len(sub.split()) >= 1:
            phrases.append(sub)
    # deduplicate while preserving order
    seen: set[str] = set()
    res: list[str] = []
    for p in phrases:
        low = p.lower()
        if low and low not in seen:
            seen.add(low)
            res.append(low)
    return res


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower())


def _rules_score(engine: StandardsEngine, profile_id: str,
                 task_ctx: dict, candidates: list[Candidate]
                 ) -> list[CandidateScore]:
    """D1 mechanics — the gate every candidate passes first."""
    query = " ".join(str(v) for v in task_ctx.values() if isinstance(v, str))
    hits = engine.retrieve(query or "general", profile_id)

    nevers = [s for s in engine.active(profile_id)
              if s.polarity == Polarity.NEVER]
    positives = [s for s, sim in hits
                 if s.polarity in (Polarity.DO, Polarity.WHY) and sim > 0.3]

    centroid = None
    if positives:
        vecs = embed([s.directive + " " + s.rationale for s in positives])
        dim = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

    out: list[CandidateScore] = []
    for cand in candidates:
        text = cand.content or cand.content_ref
        norm = _norm(text)

        violations: list[str] = []
        for s in nevers:
            for phrase in _forbidden_strings(s.directive):
                if phrase and phrase in norm:
                    violations.append(f"violates {s.id}: '{phrase}'")

        positive = 0.0
        if centroid is not None and text.strip():
            positive = max(0.0, cosine(embed([text])[0], centroid))

        score = max(-1.0, min(1.0, positive - 0.5 * len(violations)))
        out.append(CandidateScore(
            ref=cand.label, score=round(score, 4),
            breakdown={
                "positive": round(positive, 4),
                "negative": float(len(violations)),
                "bloat_veto": float(len(violations)),
                "leverage_fit": round(positive, 4),
                "cost_tokens": float(len(text) // 4),
                "judge_conf": 0.0,
                "borderline": 0.0,
            },
            violations=violations))
    return out


_JUDGE_SYSTEM = (
    "You are an objective judge breaking ties between near-equal candidate "
    "texts. You are given the owner's taste directives AND contrast pairs "
    "showing what lost vs won before, with reasons. Judge each candidate "
    "against that evidence. Output JSON: "
    '{"scores": [{"ref": "A", "score": 0.0-1.0, "why": "one line"}], '
    '"judge_conf": 0.0-1.0}')


def _judge_rerank(engine: StandardsEngine, profile_id: str, task_ctx: dict,
                  tied: list[Candidate], model: str) -> dict[str, float]:
    """Ask the judge to score only the borderline candidates.

    Returns {ref: judged_score}. Raises on failure — caller degrades.
    """
    pkt = compile_packet(engine, PacketRequest(
        profile_id=profile_id, task_ctx=task_ctx))
    directives_text = "\n".join(
        f"- [{d.polarity.value}] {d.text}" for d in pkt.directives)
    pairs_text = "\n".join(
        f"- LOST: {e.rejected}\n  WON:  {e.text}\n  WHY:  {e.why}"
        if e.rejected else f"- WON: {e.text}\n  WHY: {e.why}"
        for e in pkt.exemplars)

    cand_block = "\n\n".join(
        f"CANDIDATE {c.label}:\n'''{c.content or c.content_ref}'''"
        for c in tied)
    prompt = (
        f"TIE-BREAK THESE CANDIDATES:\n\n{cand_block}\n\n"
        f"DIRECTIVES:\n{directives_text or 'None'}\n\n"
        f"WHAT LOST VS WON BEFORE:\n{pairs_text or 'None'}\n\n"
        "Return JSON with a score for every ref shown.")

    res = llm.chat_json(model, _JUDGE_SYSTEM, prompt)
    out: dict[str, float] = {}
    for item in res.get("scores", []):
        ref = str(item.get("ref", "")).strip()
        if ref:
            out[ref] = max(0.0, min(1.0, float(item.get("score", 0.5))))
    return out


def score_candidates(engine: StandardsEngine, profile_id: str,
                     task_ctx: dict, candidates: list[Candidate]
                     ) -> list[CandidateScore]:
    scores = _rules_score(engine, profile_id, task_ctx, candidates)

    model = llm.judge_model()
    if len(scores) < 2 or not model:
        return scores

    ranked = sorted(scores, key=lambda s: s.score, reverse=True)
    if ranked[0].score - ranked[1].score >= BORDERLINE_DELTA:
        return scores  # decisive — rules stand alone

    leader = ranked[0].score
    tied_refs = {s.ref for s in ranked if leader - s.score < BORDERLINE_DELTA}
    tied = [c for c in candidates if c.label in tied_refs]
    for s in scores:
        if s.ref in tied_refs:
            s.breakdown["borderline"] = 1.0

    try:
        judged = _judge_rerank(engine, profile_id, task_ctx, tied, model)
    except Exception as exc:  # noqa: BLE001 — a judge outage must keep the rules score
        log.warning("LLM judge failed (%s: %s); rules scores stand",
                    type(exc).__name__, exc)
        return scores

    for s in scores:
        if s.ref in judged:
            s.score = round(judged[s.ref], 4)
            s.breakdown["judge_conf"] = 1.0
    return scores
