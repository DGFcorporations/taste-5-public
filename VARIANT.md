# VARIANT: taste-5 — exemplar-driven

**Thesis under test:** do examples beat rules? Where taste-1 injects
"never do X," taste-5 injects rejected-vs-chosen pairs WITH the why —
contrastive teaching signal. Memory distills hone-decision logs into
exemplars/anti-patterns over time (do-memory shaped).

## Permutation

| Axis | Choice | Divergence from control |
|---|---|---|
| A schema | A1 atomic micro-layers | held w/ control |
| B retrieval | B1 cosine top-N | held w/ control |
| C embedding | C2 Qwen3-Embedding-0.6B | better exemplar recall |
| D scorer | D4 hybrid (rules gate → judge on borderline) | judge as tiebreaker |
| E interview | E1 adaptive | held w/ control |
| F hone | F4 escalating BoN (N=1 → multi only if score fails) | cost-optimal |
| G storage | G1 SQLite WAL | held w/ control |
| H memory | **H3 episode→pattern distillation** (hone logs → exemplars) | the axis under test |
| I injection | **I3 contrastive pairs** | the axis under test |
| J pipeline | J1 fixed segments | held w/ control |

## The question it answers

Owner said "show, don't tell" might beat "rule, don't violate." If
contrastive exemplars outperform directive blocks, injection is the lever —
and H3 means the system writes its own best examples from hone history.
