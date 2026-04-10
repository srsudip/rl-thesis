# Pages Module

This directory contains the Plotly Dash dashboard pages.

## Files Overview

| File | Description |
|------|-------------|
| `dashboard.py` | **Main dashboard** - DataManager, app factory, session state |
| `dash_pages/home.py` | Home page - Generate data, train model, system overview |
| `dash_pages/analysis.py` | Analysis page - KJSEA result slip, pathway suitability, history |
| `dash_pages/teacher_review.py` | Teacher page - HITL requests, student data editing, audit log |
| `dash_pages/advanced.py` | Advanced page - Benchmarks, tests, multi-seed eval |
| `dash_pages/about.py` | About page - CBC structure, grading system, architecture |

## Dashboard Architecture

```
app.py
└── pages/dashboard.py (create_app)
    ├── DataManager (singleton)
    │   ├── CBC data storage (IRT-generated)
    │   ├── RL agent instance (DQN)
    │   ├── KNEC pathway suitability (z-score normalized)
    │   ├── HITL workflow manager
    │   └── Population stats cache
    └── Multi-page routing
        ├── / → home.py
        ├── /analysis → analysis.py
        ├── /teacher → teacher_review.py
        ├── /advanced → advanced.py
        └── /about → about.py
```

## Key Components

### DataManager Class
```python
class DataManager:
    def generate_data(n, seed)              # Generate IRT-based student data
    def get_recommendation(id)              # Multi-signal fusion → pathway recommendation
    def get_knec_result_slip(id)            # Official KJSEA result slip format
    def get_performance(id)                 # Subject scores (grade 9)
    def get_competencies(id)                # 7 CBC competencies
    def get_transitions(id)                 # Grade 4-9 progression
    def get_sub_pathway_recommendation(id)  # Track within pathway
    def get_psi(id)                         # Pathway Suitability Index per pathway
    def train_model(episodes)               # Train DQN agent
```

### Multi-signal Pathway Recommendation

`get_recommendation()` and `get_knec_result_slip()` both use the full three-signal fusion:

```
composite = 0.60 × cluster_weight + 0.25 × cosine_similarity + 0.15 × PSI
```

All three signals are computed inside the `DataManager` call and passed to `config/pathways.py:recommend_pathway()`. The function degrades gracefully when cosine or PSI signals are unavailable (weights redistributed to cluster weight).

### Pathway Suitability (KNEC methodology)
- Raw subject scores → z-score normalized using population stats
- Weighted average per pathway (0-100%)
- Two-gate eligibility: STEM ≥ 20%, Social Sciences/Arts ≥ 25% + core subject AE1 check
- Primary recommendation; RL model provides supplementary AI suggestion when it disagrees

### Session State
- Selected student persists across pages via `dcc.Store`
- Training status shown in navbar badges
- Data regeneration clears all caches

## Running the Dashboard

```bash
python app.py
# Opens at http://localhost:8050
```
