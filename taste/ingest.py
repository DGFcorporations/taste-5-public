"""ingest — cold-start pipeline: docs → candidate layers + restatements.

Deterministic extraction (the control's bias — LLM-assisted extraction is a
later axis to test). Rule-bearing lines become provisional standards marked
provenance=persona; nothing activates here, ever (contract NEVER rules).
"""

from __future__ import annotations

import json
import re

from .comprehend import restate
from .models import (
    Category,
    IngestResponse,
    Polarity,
    Provenance,
    Restatement,
    Standard,
)

NEVER_RE = re.compile(
    r"\b(never|avoid|do not|don't|no\b.*\bever|ban|cut|ditch|scrap)\b", re.IGNORECASE)
WHY_RE = re.compile(r"\b(because|why|reason|since)\b", re.IGNORECASE)
RULE_RE = re.compile(
    r"\b(never|always|avoid|prefer|lead|keep|use|must|should|"
    r"cut|ban|require|start|end|write|organize|verify)\b", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

CAT_HINTS = [
    (Category.ANTI, re.compile(r"slop|generic|bloat|never|forbidden", re.IGNORECASE)),
    (Category.STRUCTURE, re.compile(
        r"structure|organiz|lead|order|first|section|hook|close", re.IGNORECASE)),
    (Category.AESTHETIC, re.compile(
        r"tone|voice|style|rhythm|sentence|word|punchy|warm", re.IGNORECASE)),
    (Category.PROCEDURAL, re.compile(
        r"step|process|protocol|verify|check|test|commit|handoff", re.IGNORECASE)),
    (Category.DOCTRINE, re.compile(
        r"principle|doctrine|belief|value|rule\b", re.IGNORECASE)),
]


def _classify(line: str, section: str) -> tuple[Category, Polarity]:
    text = line.lower()
    for cat, rx in CAT_HINTS:
        if rx.search(section) or rx.search(text):
            category = cat
            break
    else:
        category = Category.DOCTRINE
    if NEVER_RE.search(text):
        return (Category.ANTI if category == Category.DOCTRINE else category,
                Polarity.NEVER)
    if WHY_RE.search(text):
        return category, Polarity.WHY
    return category, Polarity.DO


def _extract_lines(text: str) -> list[tuple[str, str]]:
    """(rule_text, nearest_section_header) pairs."""
    pairs, section = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        if BULLET_RE.match(line):
            line = BULLET_RE.sub("", line)
        if RULE_RE.search(line) and 4 <= len(line.split()) <= 80:
            pairs.append((line.rstrip("."), section))
    return pairs


# ------------------------------------------------------------------ profile

def _mk(profile_id: str, category: Category, polarity: Polarity,
        directive: str, rationale: str, source: str, **meta) -> Standard:
    return Standard(
        profile_id=profile_id, category=category, polarity=polarity,
        directive=directive.rstrip("."), rationale=rationale,
        provenance=Provenance.PERSONA,
        variant_meta={"source_doc": source, **meta})


def _ingest_profile(profile_id: str, doc_id: str, data: dict
                    ) -> list[Standard]:
    """taste_profile.json → candidate standards."""
    out: list[Standard] = []
    why = "persona-declared preference"

    for trig in data.get("veto_triggers", []):
        out.append(_mk(profile_id, Category.ANTI, Polarity.NEVER,
                       trig, why, doc_id, trigger="veto"))

    for k, v in (data.get("axioms") or {}).items():
        if v is True:
            d = k.replace("_", " ")
        elif v is False or v is None:
            continue
        elif isinstance(v, (int, float)):
            d = f"{k.replace('_', ' ')}: {v}"
        else:
            d = str(v)
        out.append(_mk(profile_id, Category.DOCTRINE, Polarity.DO,
                       d, why, doc_id, axiom=k))

    for k, v in (data.get("author_profile") or {}).items():
        out.append(_mk(profile_id, Category.DOCTRINE, Polarity.DO,
                       f"{k.replace('_', ' ')}: {v}", why, doc_id,
                       profile_key=k))

    for k, v in (data.get("aesthetic_and_tone") or {}).items():
        out.append(_mk(profile_id, Category.AESTHETIC, Polarity.DO,
                       f"{k.replace('_', ' ')}: {v}", why, doc_id,
                       aesthetic_key=k))

    for lane, caps in (data.get("agent_lanes") or {}).items():
        out.append(_mk(profile_id, Category.PROCEDURAL, Polarity.DO,
                       f"{lane} lane operates: {', '.join(caps)}",
                       why, doc_id, lane=lane, capabilities=caps))

    return out


def ingest_docs(profile_id: str, docs: list[dict[str, str]]
                ) -> IngestResponse:
    candidates: list[Standard] = []
    restatements: list[Restatement] = []

    for doc in docs:
        text = doc.get("text", "")
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and (
                "axioms" in parsed or "veto_triggers" in parsed):
            new = _ingest_profile(profile_id, doc.get("id", ""), parsed)
            candidates.extend(new)
            restatements.extend(restate(s) for s in new)
            continue

        for line, section in _extract_lines(text):
            category, polarity = _classify(line, section)
            rationale = ""
            m = re.search(r"because\s+(.+)$", line, re.IGNORECASE)
            if m:
                line, rationale = line[: m.start()].strip(" .—-"), m.group(1)
            if not rationale:
                rationale = "persona-declared preference"
            std = Standard(
                profile_id=profile_id, category=category, polarity=polarity,
                directive=line, rationale=rationale,
                provenance=Provenance.PERSONA,
                variant_meta={"source_doc": doc.get("id", ""),
                              "source_section": section})
            candidates.append(std)
            restatements.append(restate(std))

    return IngestResponse(candidates=candidates, restatements=restatements)
