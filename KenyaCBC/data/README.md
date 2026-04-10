# data/ — Runtime Data Store

This directory holds all generated and persisted data files. Contents are created at runtime — nothing here is hand-authored or version-controlled (except the curriculum CSVs, which live in `src/data/curriculum/`).

## Files

| File / Folder | Created by | Purpose |
|---------------|-----------|---------|
| `generated_theta.csv` | `cbc_data_generator.py` | Latent ability (θ) per student per competency, sampled from N(0, 0.7) |
| `generated_ind.csv` | `cbc_data_generator.py` | Item discrimination (a) and difficulty (b) parameters per indicator |
| `generated_resp.csv` | `cbc_data_generator.py` | Raw IRT response scores (Grade 4–9, all subjects) |
| `gradess1.csv` | `cbc_data_generator.py` | Final aggregated student dataset: subject scores, competencies, pathway labels |
| `subject_4_9.csv` | curriculum (static) | Subject master list (Grade 4–9) |
| `strand_4_9.csv` | curriculum (static) | Strand definitions per subject |
| `substrand_4_9.csv` | curriculum (static) | Substrand definitions per strand |
| `indicator_4_9.csv` | curriculum (static) | Assessment indicators per substrand |
| `substrandindicator_4_9.csv` | curriculum (static) | Substrand–indicator mapping |
| `competencysubstrand_4_9.csv` | curriculum (static) | Competency–substrand mapping |
| `hitl_state.json` | `hitl.py` | Human-in-the-Loop state: pending requests, teacher decisions, audit log |
| `session_state.json` | `dashboard.py` | Dash session state cache (last selected student, generation params) |
| `exports/` | dashboard export actions | User-triggered CSV/PDF exports |

## Regenerating Data

From the dashboard **Home** page, click **Generate Data** and choose the student count. This overwrites all `generated_*.csv` and `gradess1.csv` files. HITL state is **not** cleared on regeneration.

Programmatically:

```python
from src.data.cbc_data_generator import generate_dashboard_data
data = generate_dashboard_data(n_students=500, seed=42, save_csv=True)
```

## Data Flow

```
cbc_data_generator.py
  1. simulate_latent_competencies()   → generated_theta.csv
  2. simulate_indicator_responses()   → generated_ind.csv + generated_resp.csv
  3. aggregate_to_subjects()          → subject scores per grade
  4. derive_competencies_from_subjects() → 7 CBC competencies
  5. determine_pathways()             → STEM / SOCIAL_SCIENCES / ARTS_SPORTS labels
  → gradess1.csv (master table used by all dashboard pages)
```

## Note on Scale

Generating 500 students takes < 0.2 s (NumPy-vectorized). The earlier O(n) per-student loop was replaced with 3D tensor broadcasts and matrix multiplications in the vectorization pass.
