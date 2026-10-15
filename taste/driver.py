"""driver — the model that fills connective tissue (P4 wiring).

The model is just the driver — taste supplies the output. Config below is
the owner's spec from the TASTE heuristic research: high-conviction
inference params tuned against generic creative drift.

EXCLUDED BY OWNER DIRECTIVE: local generation models (Ollama, vLLM,
llama.cpp, any on-box LLM). Drivers are flash-tier API models only.
(Local *embedding* — Qwen3 — is the C2 axis and unrelated to generation.)

F4 shape: a driver is a callable factory — `driver(draft, rules)` returns a
generator callable `gen(n)` that emits n fresh candidates per call, so
escalating hone pays only for what it needs.
"""

from __future__ import annotations

from collections.abc import Callable

# Inference parameters — owner's spec (Project Taste §4)
TEMPERATURE = 0.35        # against generic creative drift
TOP_P = 0.85              # high-conviction vocabulary only
PRESENCE_PENALTY = 0.10   # varied angles across candidates
REPETITION_PENALTY = 1.05 # no circular justification
MAX_TOKENS = 1024         # punchy outputs, no conversational bloat

# Flash-tier driver candidates — cloud APIs only, cheapest sufficient
DRIVER_CANDIDATES = [
    "gemini-2.5-flash",
    "grok-4-fast",
    "qwen-flash",          # DashScope API — cloud, not local
    "deepseek-chat",
]

# The driver protocol — P4 wires real providers behind this.
# DriverFactory(draft, rules) -> gen(n) -> list[str] of n fresh candidates
DriverGen = Callable[[int], list[str]]
DriverFactory = Callable[[str | None, list[str]], DriverGen]


def call_driver(prompt: str, n: int = 3,
                provider: str | None = None) -> list[str]:
    """Generate n candidates via the configured flash driver.

    P4: implement provider dispatch (LiteLLM or direct SDK). Until then the
    offline deterministic generator in hone.py keeps the loop honest.
    """
    raise NotImplementedError(
        "flash driver wiring lands in P4 — pass the offline driver for now")
