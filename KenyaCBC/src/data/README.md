# src/data/ — Data Generation Pipeline

Generates synthetic CBC student data aligned with KNEC assessment standards, plus the data access layer used by the dashboard.

## Pipeline

```
CBC Curriculum CSVs → IRT Model → Raw Scores → Aggregation → Derived Subjects → Pathway Suitability
```

## Files

| File | Purpose |
|------|---------|
| `cbc_data_generator.py` | Main generator: IRT model, derived subjects, CSV persistence |
| `real_data_loader.py` | Data access layer: per-student queries, pathway history, competencies |
| `curriculum/` | 6 CSV files defining subjects, strands, substrands, indicators |

## IRT Model Parameters

- Student ability: θ ~ N(0, 0.7)
- Item difficulty: b ~ N(0, 1)
- Item discrimination: a ~ U(0.8, 2.0)
- Score formula: `score = 100 × sigmoid(a × (θ - b)) + noise`, clipped to [1, 100]

All scores have a floor of 1 (matching CBC's minimum BE2 grade of 1–10%).

## Derived Subjects

Certain subjects exist only in specific grade ranges and are derived from curriculum indicators:

**Grade 4–6 only:**

| Subject | Source |
|---------|--------|
| HOME_SCI | Home Science indicators |
| PHE | Physical & Health Education indicators |

**Grade 7–9 only (derived by blending related areas):**

| Subject | Source Blend |
|---------|-------------|
| INT_SCI | 60% Science & Technology + 40% Agriculture |
| HEALTH_ED | 50% PHE + 30% Science & Technology + 20% Home Science |
| PRE_TECH | 50% Science & Technology + 30% Mathematics + 20% Creative Arts |
| BUS_STUD | 40% Mathematics + 30% Social Studies + 30% Agriculture |
| LIFE_SKILLS | 40% Social Studies + 30% Religious Education + 30% English |
| SPORTS_PE | 60% PHE + 40% Health Education |

## Curriculum CSV Inventory

| File | Records | Content |
|------|---------|---------|
| `subject_4_9.csv` | 10 | Subject definitions (Grade 4–9) |
| `strand_4_9.csv` | ~40 | Strand definitions per subject |
| `substrand_4_9.csv` | ~120 | Substrand definitions per strand |
| `indicator_4_9.csv` | ~400 | Assessment indicators per substrand |
| `substrandindicator_4_9.csv` | ~400 | Substrand–indicator mappings |
| `competencysubstrand_4_9.csv` | ~200 | Competency–substrand mappings |

## API

```python
from src.data.cbc_data_generator import generate_dashboard_data, load_dashboard_data

# Generate fresh data
data = generate_dashboard_data(n_students=500, seed=42, save_csv=True)

# Load from CSV
data = load_dashboard_data()
```

## real_data_loader.py — Data Access Layer

Provides per-student data views consumed by `pages/dashboard.py`:

| Function | Returns |
|----------|---------|
| `get_student_scores(id)` | Grade-9 subject scores dict |
| `get_student_competencies(id)` | 7 CBC competency scores |
| `get_grade_transitions(id)` | Grade 4–9 progression DataFrame |
| `get_suggested_pathway_per_grade(id)` | Per-grade pathway label using multi-signal fusion |
| `simulate_earlier_grades(id)` | Reconstructed earlier-grade score trajectories |
| `derive_competencies_from_subjects(df)` | Vectorized competency derivation via weight matrix multiply |

### Per-grade Pathway History

`get_suggested_pathway_per_grade()` passes cosine similarities to `recommend_pathway()` at each grade point, producing the suitability trajectory rendered in the Analysis page history chart.

## Vectorization

All inner loops were replaced with NumPy operations. Key improvements:

| Function | Before | After |
|----------|--------|-------|
| `simulate_latent_competencies()` | Triple Python loop (n_students × n_grades × n_comp) | 3D tensor broadcast |
| `simulate_indicator_responses()` | Per-grade inner loop with dot-product | `Theta_g @ A.T − b` matrix multiply |
| `determine_pathways()` | Per-student `.loc` filter + manual argmax | Vectorized z-score + weight matrix multiply + `idxmax` |
| `derive_competencies_from_subjects()` | Double `iterrows` loop | `scores_mat @ W_norm.T` |
| `simulate_earlier_grades()` | O(n) per-student `.loc` scan | `groupby(['student_id', 'subject'])` |

Benchmark: 500 students in ~0.14 s (vs ~30 s before, ~430× speedup).

## Reproducibility

Same seed produces identical data. Default seed: 42.
