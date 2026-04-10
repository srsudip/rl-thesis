# Configuration Module — `config/`

## Files

| File | Purpose |
|------|---------|
| `settings.py` | Directory paths, grade levels, placement weights, performance thresholds |
| `pathways.py` | CBC grading system, 3 pathways + sub-pathways, subject weights, grade subjects |
| `competencies.py` | 7 CBC competencies, competency-subject weight matrix, ideal pathway profiles |
| `rl_config.py` | DQN hyperparameters, reward structure, consistency checker config |

## Grade Structure

| Level | Grades | Subjects |
|-------|--------|----------|
| Lower Primary | 1-3 | 8 subjects (MATH_LP, ENG_ACT, etc.) |
| Upper Primary | 4-6 | 15 subjects (MATH, ENG, SCI_TECH, etc.) |
| Junior Secondary | 7-9 | 16 subjects (adds INT_SCI, PRE_TECH) |

## Pathway Subject Weights

Each subject has a weight [0, 1] per pathway indicating relevance:

- **STEM**: MATH (1.0), INT_SCI (0.95), SCI_TECH (0.95), PRE_TECH (0.85)
- **Social Sciences**: SOC_STUD (1.0), ENG (0.95), KIS (0.90), Religious Ed (0.65)
- **Arts & Sports**: CRE_ARTS (1.0), ENG (0.70), KIS (0.60)

## Assessment Weights

Per KNEC placement policy:
- 20% Grade 6 KPSEA
- 20% School-based G7-G8
- 60% Grade 9 KJSEA

## CBC 8-Level Grading System

| Level | Score Range | Points |
|-------|-------------|--------|
| EE1 | 90-100 | 4.0 |
| EE2 | 75-89 | 3.5 |
| ME1 | 58-74 | 3.0 |
| ME2 | 41-57 | 2.5 |
| AE1 | 31-40 | 2.0 |
| AE2 | 21-30 | 1.5 |
| BE1 | 11-20 | 1.0 |
| BE2 | 1-10 | 0.5 |
