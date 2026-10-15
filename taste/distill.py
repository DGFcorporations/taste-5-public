"""distill — H3 permutation: episode→pattern memory.

A pass over hone_events turns "what keeps winning" and "what keeps failing"
into new provisional standards — the system writes its own exemplars from
its own history.

  wins  (pick score >= 0.7), clustered by task_type
        → provisional EXEMPLAR standard (directive = representative winning
          text) + a contrast_pairs row per event (winner vs worst loser)
  fails (best score < 0.4), clustered by dominant violation
        → provisional ANTI standard (never-directive synthesized from the
          recurring violation)

Everything lands provisional — owner confirms via the comprehension flow.
Deterministic synthesis, no LLM: every rationale carries its evidence
counts, so confirmation is auditable. Idempotent: directives already in the
corpus (any state) are skipped, not re-proposed.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .core import StandardsEngine
from .embed import embed
from .models import (
    Category,
    ContrastPair,
    DistillResponse,
    HoneEvent,
    Polarity,
    Provenance,
    Standard,
)
from .store import Store

WIN_SCORE = 0.7
FAIL_SCORE = 0.4
MAX_DIRECTIVE_LEN = 240


def _pick_content(e: HoneEvent) -> str:
    for c in e.candidates:
        if c.label == e.pick:
            return (c.content or c.content_ref or "").strip()
    return ""


def _worst_loser(e: HoneEvent) -> str:
    losers = [c for c in e.candidates if c.label != e.pick]
    if not losers:
        return ""
    worst = min(losers, key=lambda c: c.score)
    return (worst.content or worst.content_ref or "").strip()


def _dominant_violation(events: list[HoneEvent]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        for v in e.violations:
            m = re.search(r"'([^']{3,})'", v)
            if m:
                counts[m.group(1)] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _existing_directives(store: Store, profile_id: str) -> set[str]:
    return {re.sub(r"\s+", " ", s.directive.lower()).strip()
            for s in store.list_standards(profile_id)}


def distill(engine: StandardsEngine, store: Store, profile_id: str,
            since: str | None = None, min_cluster: int = 3
            ) -> DistillResponse:
    events = store.list_hone_events(profile_id, since=since)
    wins: dict[str, list[HoneEvent]] = defaultdict(list)
    fails: dict[str, list[HoneEvent]] = defaultdict(list)
    for e in events:
        if e.score >= WIN_SCORE:
            wins[e.task_type or "general"].append(e)
        elif e.score < FAIL_SCORE:
            fails[e.task_type or "general"].append(e)

    existing = _existing_directives(store, profile_id)
    proposed = 0
    pairs = 0

    for task_type, cluster in wins.items():
        for e in cluster:
            # every win with a loser is a ready-made contrast pair —
            # evidence, so it lands directly in the pair store
            loser = _worst_loser(e)
            chosen = _pick_content(e)
            if chosen and loser and not store.pair_exists_for(e.id):
                pair = ContrastPair(
                    profile_id=profile_id, task_type=e.task_type,
                    segment=e.segment, rejected=loser, chosen=chosen,
                    why=(f"hone pick for {e.task_type}/{e.segment}: "
                         f"scored {e.score:.2f} against the pool"),
                    source="distill", source_id=e.id)
                store.put_pair(pair, embed([chosen + " " + loser])[0])
                pairs += 1

        if len(cluster) < min_cluster:
            continue
        rep = max(cluster, key=lambda e: e.score)
        rep_text = _pick_content(rep)
        if not rep_text:
            continue
        directive = rep_text[:MAX_DIRECTIVE_LEN]
        key = re.sub(r"\s+", " ", directive.lower()).strip()
        if key in existing:
            continue
        scores = sorted(e.score for e in cluster)
        engine.propose(Standard(
            profile_id=profile_id, category=Category.DOCTRINE,
            polarity=Polarity.WHY, directive=directive,
            rationale=(f"distilled: {len(cluster)} winning hone segments "
                       f"for '{task_type}' (scores {scores[0]:.2f}–"
                       f"{scores[-1]:.2f}) — this shape keeps winning"),
            provenance=Provenance.INFERENCE,
            variant_meta={
                "kind": "exemplar", "task_type": task_type,
                "chosen": rep_text, "cluster_size": len(cluster),
                "rejected": _worst_loser(rep) or None,
                "source": "distill", "source_id": rep.id}))
        existing.add(key)
        proposed += 1

    for task_type, cluster in fails.items():
        if len(cluster) < min_cluster:
            continue
        violation = _dominant_violation(cluster)
        # quoted violation → the rules scorer can enforce it verbatim
        directive = (f"Never include '{violation}'" if violation else
                     f"Never ship under-taste '{task_type}' segments")
        key = re.sub(r"\s+", " ", directive.lower()).strip()
        if key in existing:
            continue
        engine.propose(Standard(
            profile_id=profile_id, category=Category.ANTI,
            polarity=Polarity.NEVER, directive=directive,
            rationale=(f"distilled: {len(cluster)} hone segments scored "
                       f"<{FAIL_SCORE} for '{task_type}'"
                       + (f"; recurring violation '{violation}'"
                          if violation else "")),
            provenance=Provenance.INFERENCE,
            variant_meta={"task_type": task_type,
                          "cluster_size": len(cluster),
                          "violation": violation or None,
                          "source": "distill"}))
        existing.add(key)
        proposed += 1

    return DistillResponse(
        events_seen=len(events), proposed=proposed, pairs_written=pairs,
        clusters={**{f"win:{k}": len(v) for k, v in wins.items()},
                  **{f"fail:{k}": len(v) for k, v in fails.items()}})
