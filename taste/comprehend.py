"""comprehend — restatement protocol.

A new layer is never trusted until restated as concrete implications and
confirmed by the owner. This module generates the restatement the owner
confirms or corrects — corrections are themselves corpus.

taste-5 extension: exemplar-kind layers restate as "show this pattern,"
not "follow this rule" — the implication names the winning text it carries.
"""

from __future__ import annotations

from .models import Polarity, Restatement, Standard


def restate(standard: Standard) -> Restatement:
    """Turn a candidate layer into 'what this means next time' implications."""
    d = standard.directive.rstrip(".")
    why = standard.rationale.strip() or "declared as part of the source material"

    if standard.is_exemplar:
        implications = [
            f"When drafting, this is a shape that won: \"{d[:120]}\".",
            (
                f"Prefer output in that shape over the rejected alternative — "
                f"because {why}."
            ),
        ]
    elif standard.polarity == Polarity.NEVER:
        implications = [
            f"Do not produce output that {d.lower()}.",
            f"If a draft does this, it fails the audit — because {why}.",
        ]
    elif standard.polarity == Polarity.WHY:
        implications = [
            f"Apply this reasoning when choosing between options: {d}.",
            f"Decisions that ignore it will read as generic — because {why}.",
        ]
    else:
        implications = [
            f"Output should {d[0].lower() + d[1:] if d else d}.",
            f"Missing this marks the output as not-yours — because {why}.",
        ]
    implications.append(f"Category: {standard.category.value} — applies to "
                        f"{_scope(standard)} tasks.")
    return Restatement(layer=standard, implications=implications)


def _scope(s: Standard) -> str:
    return {
        "structure": "structure/organization-sensitive",
        "aesthetic": "tone/voice-sensitive",
        "procedural": "process-step",
        "doctrine": "principle-level",
        "anti": "all output (violation check)",
    }.get(s.category.value, "all")
