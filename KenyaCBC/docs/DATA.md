# Data Generator — `src/data/cbc_data_generator.py`

## IRT-Based Latent Competency Model

The generator creates synthetic student data grounded in Item Response Theory (IRT).

### Pipeline

```
Curriculum CSVs (data/*.csv)
        │
        ▼
  define_indicators_by_grade()       ← joins subjects, strands, substrands, indicators
        │
        ▼
  simulate_latent_competencies()     ← θ = α + β × (grade - min_grade)
        │
        ▼
  simulate_indicator_parameters()    ← IRT discrimination (a) and difficulty (b)
        │
        ▼
  simulate_indicator_responses()     ← score = 100 × logistic(a·θ - b) + noise
        │
        ▼
  aggregate_to_subjects_wide()       ← indicator → subject means
        │
        ▼
  theta_to_competencies_wide()       ← latent θ → competency scores (0-100)
        │
        ▼
  determine_pathways()               ← weighted competency + subject → pathway label
```

### IRT Model Details

Each student has:
- **α** (baseline ability): 7-dimensional, sampled from multivariate normal
- **β** (growth rate): 7-dimensional, correlated with α (negative correlation → regression to mean)

Latent competency at grade g:
```
θ(g) = α + β × (g - 4)
```

Indicator score generation:
```
η = Σ aₖ × θₖ - b
p = logistic(η)
score = clip(N(100·p, σ), 0, 100)
```

### Curriculum Data Files

| File | Content |
|------|---------|
| `subject_4_9.csv` | 25 subjects across grades 4-9 |
| `strand_4_9.csv` | Strands per subject |
| `substrand_4_9.csv` | Sub-strands per strand |
| `indicator_4_9.csv` | Assessment indicators |
| `competency_4_9.csv` | 7 CBC competencies |
| `competencysubstrand_4_9.csv` | Competency ↔ substrand mapping |
| `substrandindicator_4_9.csv` | Substrand ↔ indicator mapping with weights |

### Pathway Determination

Uses weighted combination:
- 60% competency alignment (cosine similarity to ideal profile)
- 40% subject-based scoring (KNEC-aligned weights)
