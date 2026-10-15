"""inject — I3 permutation: packet compiler → contrastive exemplars.

The control injects "never do X" blocks. This variant injects what lost vs
what won, with the reason — the model learns the boundary, not just the
rule. The packet IS the product; deterministic: same corpus + same input →
same packet (contract §4).

Exemplar composition (bounded, deterministic order):
  1. contrast pairs — cosine-retrieved decision evidence {rejected, chosen,
     why}; the teaching signal. Raw owner decisions — no provisional gate.
  2. exemplar-kind standards — owner-confirmed winning shapes
     (polarity=why, kind=exemplar), already retrieval-biased in core.
  3. ordinary why-layers — control behavior, the remaining budget.
"""

from __future__ import annotations

from .core import StandardsEngine
from .embed import cosine, embed
from .models import (
    Constraint,
    Directive,
    Exemplar,
    Packet,
    PacketRequest,
    Polarity,
)

import re

SIM_THRESHOLD = 0.40          # a do/why layer must be relevant to apply
PAIR_THRESHOLD = 0.30         # a pair must relate to the task to teach
MAX_DIRECTIVES = 8            # packet size is bounded — every token is cost
MAX_NEVERS = 4                # guardrails always apply, bounded
MAX_PAIRS = 3                 # contrastive pairs — the I3 headline
MAX_STD_EXEMPLARS = 2         # confirmed exemplar standards
MAX_WHY_EXEMPLARS = 1         # ordinary why-layers fill the tail

# Domain synonym keywords for boosting relevance
DOMAIN_KEYWORDS = {
    "media": ["video", "audio", "mp4", "render", "youtube", "tiktok", "reels", "voice", "sync", "lip", "thumbnail", "kallaway", "hormozi", "elevenlabs", "qwen", "deepgram", "camera", "frame", "cut"],
    "content": ["copy", "text", "headline", "hook", "story", "framing", "script", "editorial", "email", "angle", "jargon", "fluff"],
    "engineering": ["code", "bug", "fix", "feature", "git", "commit", "test", "tdd", "branch", "refactor", "security", "fail-closed", "clean", "architecture"],
    "systems": ["container", "docker", "service", "systemd", "server", "port", "daemon", "proxy", "cloudflare", "postgres", "redis", "bridge", "rclone"],
    "governance": ["gate", "audit", "compliance", "phi", "legal", "secret", "env", "credential", "permission", "sterile-field"],
}


def score_standard_relevance(s, query: str, tokens: set[str], hit_score: float = 0.0) -> float:
    text = (s.directive + " " + (s.rationale or "")).lower()
    score = hit_score * 10.0
    for t in tokens:
        if len(t) > 3 and t in text:
            score += 3.0
        elif t in text:
            score += 1.0
    q_low = query.lower()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(k in q_low for k in kws):
            if any(k in text for k in kws):
                score += 5.0
    conf = float(s.confidence if s.confidence is not None else 0.5)
    return score * (0.5 + conf)



def _task_query(req: PacketRequest) -> str:
    query = " ".join(
        str(v) for v in req.task_ctx.values() if isinstance(v, str))
    for ans in req.answers.values():
        query += " " + ans
    return query.strip() or "general task"


def compile_packet(engine: StandardsEngine,
                   req: PacketRequest) -> Packet:
    query = _task_query(req)
    hits = engine.retrieve(query, req.profile_id)

    directives: list[Directive] = []
    exemplars: list[Exemplar] = []
    std_exemplars = 0
    why_exemplars = 0

    hit_dict = {s.id: sim for s, sim in hits}
    tokens = set(re.findall(r"\w+", query.lower()))

    # Rank NEVER rules by task relevance + confidence, break ties by s.id for determinism
    active_nevers = [
        s for s in engine.active(req.profile_id)
        if s.polarity == Polarity.NEVER
    ]
    ranked_nevers = sorted(
        active_nevers,
        key=lambda s: (score_standard_relevance(s, query, tokens, hit_dict.get(s.id, 0.0)), s.id),
        reverse=True,
    )
    selected_nevers = sorted(ranked_nevers[:MAX_NEVERS], key=lambda s: s.id) if ranked_nevers else []
    directives.extend(
        Directive(polarity=s.polarity, text=s.directive, standard_id=s.id)
        for s in selected_nevers)

    ranked_hits = sorted(
        hits,
        key=lambda item: (score_standard_relevance(item[0], query, tokens, item[1]), item[1]),
        reverse=True,
    )
    for s, sim in ranked_hits:
        if s.polarity == Polarity.NEVER:
            continue  # already emitted from the active set above
        if sim < SIM_THRESHOLD:
            continue
        if s.polarity == Polarity.WHY:
            if s.is_exemplar and std_exemplars < MAX_STD_EXEMPLARS:
                exemplars.append(Exemplar(
                    text=s.directive, why=s.rationale,
                    rejected=s.variant_meta.get("rejected") or None,
                    kind="exemplar"))
                std_exemplars += 1
            elif not s.is_exemplar and why_exemplars < MAX_WHY_EXEMPLARS:
                exemplars.append(Exemplar(text=s.directive, why=s.rationale))
                why_exemplars += 1
            continue
        if len(directives) >= MAX_DIRECTIVES:
            break
        directives.append(Directive(
            polarity=s.polarity, text=s.directive, standard_id=s.id))


    # contrast pairs — decision evidence retrieved by task similarity.
    # Pairs are facts about what the owner chose, so they inject
    # immediately; the provisional gate governs generalizations, not facts.
    pair_rows = engine.store.list_pairs(req.profile_id, with_vectors=True)
    if pair_rows:
        qv = embed([query])[0]
        scored = [(p, cosine(qv, v)) for p, v in pair_rows if v is not None]
        scored.sort(key=lambda t: (t[1], t[0].id), reverse=True)
        pair_count = 0
        for p, sim in scored:
            if pair_count >= MAX_PAIRS:
                break
            if sim < PAIR_THRESHOLD:
                continue
            exemplars.append(Exemplar(
                text=p.chosen, why=p.why, rejected=p.rejected,
                kind="contrast"))
            pair_count += 1

    constraints = [Constraint(hard=True, text=g) for g in req.guidelines]

    # NEVER rule 6 — one bounded free degree minimum, always
    free_degrees = [
        (
            "voice and rhythm within the directives — vary phrasing, "
            "keep the pattern alive"
        )
    ]

    return Packet(
        profile_id=req.profile_id, task_ctx=req.task_ctx,
        directives=directives, exemplars=exemplars,
        constraints=constraints, free_degrees=free_degrees,
        variant_meta={"engine": "taste-5", "hits": len(hits),
                      "pairs": sum(1 for e in exemplars
                                   if e.kind == "contrast")})
