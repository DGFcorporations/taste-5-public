"""interview — E1 permutation: adaptive 3-question interview.

The questions target what taste CANNOT resolve — never re-ask what the
corpus knows. Max 3, fewer when coverage is already good.
"""

from __future__ import annotations

from uuid import uuid4

from .core import StandardsEngine
from .models import InterviewQuestion, InterviewResponse

GAP_QUESTIONS = {
    "structure": (
        "How should this be organized — what leads, what follows?",
        ["verdict first", "problem→solution", "chronological", "hierarchical"]),
    "aesthetic": (
        "What should this feel like — tone, density, register?",
        ["direct/operational", "warm/conversational", "formal/precise", "bold/punchy"]),
    "procedural": (
        "Any process rules for this — must-do steps or must-avoid moves?",
        []),
    "doctrine": (
        "Is there a principle at stake here — a line this output shouldn't cross?",
        []),
    "anti": (
        "What would make this output feel like slop to you — the thing to never do?",
        []),
}

MAX_QUESTIONS = 3


def run_interview(engine: StandardsEngine, profile_id: str,
                  task_ctx: dict) -> InterviewResponse:
    query = " ".join(str(v) for v in task_ctx.values()) or "general task"
    cov = engine.coverage(profile_id, query)

    questions: list[InterviewQuestion] = []
    for gap in cov["gaps"][:MAX_QUESTIONS]:
        text, options = GAP_QUESTIONS.get(gap, (f"Preference for {gap}?", []))
        questions.append(InterviewQuestion(
            id=f"q_{uuid4().hex[:8]}", text=text, options=options,
            resolves=gap))

    return InterviewResponse(questions=questions, coverage=cov)
