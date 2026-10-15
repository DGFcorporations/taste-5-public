"""segments — J1 permutation: fixed declarative segmentation by task type.

Rigid mechanics: every task decomposes the same way per type. Each segment
gets its own packet + hone budget — ~10–15 iterations total across a task.
"""

from __future__ import annotations

SEGMENT_MAP: dict[str, list[str]] = {
    "write_thread": ["hook", "body", "close"],
    "write_doc": ["lead", "structure", "detail", "close"],
    "security_review": ["triage", "findings", "verdict"],
    "daily_digest": ["numbers", "flags", "recommendation"],
    "decision_brief": ["verdict", "tradeoffs", "recommendation"],
    "default": ["draft", "refine", "finalize"],
}


def segments_for(task_type: str) -> list[str]:
    return SEGMENT_MAP.get(task_type, SEGMENT_MAP["default"])
