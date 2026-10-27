# taste-5

This repository contains the core mechanics for TASTE variant 5, an exemplar-driven engine designed to manage and apply personalized development standards through contrastive learning and feedback distillation. It enables automated extraction, scoring, and application of an owner's codebase preferences without exposing private, proprietary data.

The public mechanics of TASTE variant 5 — engine code and docs only.

**What's never here:** the owner's corpus (standards, decision records,
rationale, persona-derived layers). Those live in the private repo and are
the system's learned knowledge — publishing them would publish the person.

Each trial cycle: corpus pruned → keepers stay private → mechanics worth
sharing land here via the publish decision.

## Language and Stack

- **Python 3.11+**
- **FastAPI** & **Pydantic**
- **FastEmbed**
- **MCP** (Model Context Protocol)

## Key Features

- **Fleet Interface via MCP:** A fully integrated Model Context Protocol server exposing taste mechanics to language models.
- **Contrastive Pair Capture:** Learns what is preferred versus what is rejected to map nuanced preferences.
- **Feedback Distillation:** Aggregates hone-episodes into refined, codified layers of preference over time.

## How to Run

To set up the project locally:

```bash
pip install -e .
```

You can then run the CLI tools provided via `taste`:

- `taste ingest <file.md>` — extract candidate layers
- `taste standards` — list corpus
- `taste confirm <std_id>` — activate a provisional layer
- `taste packet` — compile the packet for a task
- `taste feedback` — log an owner decision
- `taste pairs` — contrast-pair evidence store
- `taste distill` — distill hone episodes into layers
- `taste queue` — pending inferred items
- `taste serve` — run an HTTP server
