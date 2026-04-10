# Kenya CBC Pathway Recommendation System — Documentation

## Architecture Overview

```
KenyaCBC/
├── app.py                          # Dash entry point
├── config/                         # Pathways, grading, competencies, RL params
│   ├── pathways.py                 # Multi-signal fusion, two-gate eligibility
│   ├── competencies.py             # 7 CBC competencies + ideal pathway profiles
│   ├── rl_config.py                # DQN hyperparameters
│   └── settings.py                 # Paths, grade ranges, thresholds
├── src/
│   ├── data/                       # IRT-based synthetic data generator
│   │   ├── cbc_data_generator.py   # NumPy-vectorized IRT pipeline
│   │   └── real_data_loader.py     # Per-student data access layer
│   └── rl/                         # Reinforcement Learning core
│       ├── agent.py                # Dueling Double DQN with PER (78-dim state)
│       ├── environment.py          # Grade-to-grade episodic MDP
│       ├── trainer.py              # Training loop, k-fold CV
│       ├── dqn_coaching.py         # 9-action space, reward shaping, XAI
│       ├── hitl.py                 # Human-in-the-Loop workflow
│       ├── baselines.py            # Random, majority, rule-based, LR, bandit
│       ├── evaluation.py           # Multi-seed eval, hyperparameter search
│       └── evaluation_extended.py  # Bootstrap CIs, Cohen's d, Wilcoxon
├── benchmarks/                     # Jordan et al. (2020) statistical benchmarks
│   └── benchmark_runner.py         # DKW bands, PBP-t convergence, per-pathway CIs
├── tests/                          # Integration test suite (no mocks)
│   ├── test_all.py                 # 6 categories, full pipeline
│   ├── test_verification.py        # Phase 6: cosine, PSI, history, coaching
│   └── test_audit.py               # Exhaustive data-path audit
├── pages/                          # Plotly Dash pages
│   ├── dashboard.py                # DataManager singleton + app factory
│   └── dash_pages/                 # Home, Analysis, Teacher, Advanced, About
├── data/                           # Generated CSVs + HITL state (runtime)
├── models/                         # Trained DQN weights + metadata (runtime)
└── docs/                           # This documentation
```

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:8050
```

## Key Workflows

### 1. Generate Data + Train
Home page → Generate Data → Train Model

### 2. Get Recommendation
Analysis page → Select student → View pathway recommendation with XAI explanation

### 3. Teacher Override (HITL)
Teacher page → Review pending requests → Approve / Reject → Audit log updated

### 4. Run Benchmarks
```bash
python -m benchmarks.benchmark_runner --trials 5 --episodes 500
```
Or use the **Advanced** page in the dashboard.

### 5. Run Tests
```bash
python -m pytest tests/ -v
```

## Module Documentation

| Module | Doc |
|--------|-----|
| Configuration | [CONFIG.md](CONFIG.md) |
| RL Core | [RL_CORE.md](RL_CORE.md) |
| Data Generator | [DATA.md](DATA.md) |
| Benchmarks | [BENCHMARKS.md](BENCHMARKS.md) |
| HITL Workflow | [HITL.md](HITL.md) |
| Tests | [TESTS.md](TESTS.md) |
