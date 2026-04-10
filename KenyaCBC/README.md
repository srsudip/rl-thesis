# Kenya CBC Pathway Recommendation System

A deep reinforcement learning prototype for recommending senior secondary school pathways under Kenya's Competency-Based Curriculum (CBC). Aligned with KNEC's 2025 KJSEA grading and pathway placement standards.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8050** → Click **Generate Data** → Select a student on the **Analysis** page.

## Features

- **KJSEA Result Slip** — Official KNEC format with 12 subjects (901–912), 8-level grading
- **Multi-signal Fusion** — Final recommendation blends 60% cluster weight + 25% cosine similarity + 15% PSI
- **Two-Gate Pathway Eligibility** — KNEC cluster weight threshold (STEM 20%, SS/Arts 25%) + core subject validation (AE1)
- **Three Pathways** — STEM, Social Sciences, Arts and Sports Science, each with tracks
- **Pathway Suitability History** — Per-grade PSI chart showing recommended pathway trajectory and competing alternatives
- **Pathway Strength Display** — Student's strength within the recommended pathway computed from top-3 core subjects
- **Human-in-the-Loop** — Students request pathway changes, teachers approve/reject with full audit log
- **DQN Agent** — Supplementary AI recommendations with XAI explanations (78-dim state, 9 coaching actions)
- **Vectorized Data Generation** — IRT pipeline using NumPy broadcasting; 500 students in < 0.2 s
- **Benchmarks** — DKW confidence bands, convergence analysis, per-pathway accuracy

## Pathway Eligibility

A student is recommended for a pathway only if:

1. **Suitability ≥ KNEC minimum** — STEM ≥ 20%, SS/Arts ≥ 25% weighted cluster score
2. **Core subjects ≥ AE1 (31%)** — STEM requires Mathematics + at least one science subject

If the top pathway fails validation, the next eligible pathway is recommended. Students with all subjects below expectations receive an academic support warning.

## Project Structure

```
KenyaCBC/
├── app.py                  # Dash entry point
├── requirements.txt        # Python dependencies
├── USER_GUIDE.md           # Supervisor guide
├── config/                 # Pathways, grading, competencies, RL config
├── src/
│   ├── data/               # IRT data generator + curriculum CSVs
│   └── rl/                 # DQN agent, HITL, evaluation, baselines
├── pages/
│   ├── dashboard.py        # DataManager (data access layer)
│   └── dash_pages/         # Home, Analysis, Teacher, Advanced, About
├── benchmarks/             # Statistical evaluation framework
├── data/                   # Generated CSVs + HITL state (runtime)
├── models/                 # Trained DQN weights (runtime)
└── tests/                  # Integration test suite
```

## Technology

Python 3.9+, Dash, Plotly, NumPy, Pandas. No database required — all data generated in-memory and persisted as CSV.

## Documentation

| File | Content |
|------|---------|
| `USER_GUIDE.md` | End-user guide for running and using the dashboard |
| `config/README.md` | Grading system, pathway weights, multi-signal fusion, core subject requirements |
| `src/README.md` | Source module overview and design principles |
| `src/data/README.md` | IRT model, vectorized generation, derived subjects, curriculum CSVs |
| `src/rl/README.md` | DQN architecture (78-dim state, 9 actions), HITL workflow, evaluation API |
| `pages/README.md` | Dashboard architecture, DataManager, multi-signal recommendation |
| `pages/dash_pages/README.md` | Per-page breakdown (Analysis, Teacher, Advanced) |
| `data/README.md` | Runtime data files, regeneration instructions, data flow |
| `models/README.md` | Model checkpoint format, training and loading API |
| `assets/README.md` | CSS conventions, grade-level colour classes |
| `tests/README.md` | Test suite overview, how to run, coverage focus |
| `benchmarks/README.md` | Statistical benchmarking methodology |
| `docs/RL_CORE.md` | Theoretical justification for DQN; state/action/reward specification |
| `docs/HITL.md` | Formal HITL workflow specification |

## Thesis

This prototype is part of a Bachelor's thesis at Universität Leipzig / ScaDS.AI on applying Deep Reinforcement Learning to educational pathway recommendations in the Kenyan CBC context.
