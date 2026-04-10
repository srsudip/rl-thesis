# src/ — Core Application Logic

All non-UI source code: data generation pipeline and reinforcement learning module.

## Structure

```
src/
├── data/       # IRT-based synthetic CBC student data generator + real-data loader
└── rl/         # Deep Q-Network agent, environment, HITL workflow, evaluation
```

## Sub-module Docs

- [`src/data/README.md`](data/README.md) — IRT model, vectorized data generation, curriculum CSVs, derived subjects
- [`src/rl/README.md`](rl/README.md) — DQN architecture, 78-dim state, 9 actions, HITL, baselines, evaluation

## Design Principles

- **No database** — all data is generated as CSV and loaded into memory via Pandas DataFrames
- **Reproducible** — every component accepts a `seed` parameter; same seed → identical outputs
- **Vectorized** — NumPy broadcasting throughout; O(n) data generation scales to 10k+ students without per-student Python loops
- **Decoupled** — `src/data` and `src/rl` are independent packages; `pages/dashboard.py` is the only consumer
