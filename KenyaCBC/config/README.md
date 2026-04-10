# config/ — System Configuration

Central configuration for the Kenya CBC Pathway Recommendation System.

## Files

| File | Purpose |
|------|---------|
| `pathways.py` | Pathways, subjects, grading, cluster weights, core validation |
| `competencies.py` | 7 CBC core competencies, ideal pathway profiles |
| `rl_config.py` | DQN hyperparameters, reward shaping, training settings |
| `settings.py` | Directory paths, grade ranges, performance levels |
| `__init__.py` | Re-exports all public constants and functions |

## KNEC 8-Level Grading System

| Level | Range | Points | Category |
|-------|-------|--------|----------|
| EE1 | 90–100% | 8 | Exceeding Expectations |
| EE2 | 75–89% | 7 | Exceeding Expectations |
| ME1 | 58–74% | 6 | Meeting Expectations |
| ME2 | 41–57% | 5 | Meeting Expectations |
| AE1 | 31–40% | 4 | Approaching Expectations |
| AE2 | 21–30% | 3 | Approaching Expectations |
| BE1 | 11–20% | 2 | Below Expectations |
| BE2 | 1–10% | 1 | Below Expectations |

## Subjects

**Upper Primary (Grade 4–6):** English, Kiswahili, Mathematics, Science & Technology, Social Studies, Religious Education (CRE/IRE/HRE), Creative Arts, Agriculture, Home Science, Physical & Health Education.

**Junior Secondary (Grade 7–9):** English, Kiswahili, Mathematics, Integrated Science, Health Education, Pre-Technical & Pre-Career Education, Social Studies, Religious Education, Business Studies, Agriculture, Life Skills, Sports & Physical Education.

## Pathway Suitability

### Step 1 — Cluster Weights (`compute_cluster_weights()`)

1. Each pathway has weighted subjects (e.g., STEM weights Mathematics at 1.0, Integrated Science at 1.0).
2. When population statistics are available, z-score normalization is applied: `normalized = 50 + 10 × (raw − mean) / std`.
3. The weighted average produces a suitability percentage (0–100%).

### Step 2 — Multi-signal Fusion (`recommend_pathway()`)

The final recommendation uses three signals:

```
composite = 0.60 × cluster_weight + 0.25 × cosine_similarity + 0.15 × PSI
```

| Signal | Source | Weight |
|--------|--------|--------|
| Cluster weight | KNEC subject-weighted average | 60% |
| Cosine similarity | Student profile vs ideal pathway vector | 25% |
| PSI (Pathway Suitability Index) | Competency-profile closeness (0–1) | 15% |

When cosine or PSI signals are absent, weights redistribute: CW takes 70–80% share. This graceful degradation is implemented in `_composite_alignment_score()`.

## Pathway Eligibility (Two-Gate System)

A student is eligible for a pathway only if BOTH conditions are met:

### Gate 1: Minimum Suitability Threshold (KNEC)

STEM requires ≥ 20% cluster weight; Social Sciences and Arts & Sports require ≥ 25% (per KNEC 2025 placement criteria).

### Gate 2: Core Subject Validation

| Pathway | Required (ALL ≥ AE1) | Language Gate (≥1 ≥ AE1) | Content Gate (≥1 ≥ AE1) |
|---------|----------------------|--------------------------|-------------------------|
| STEM | Mathematics | — | Integrated Science, Sci & Tech, Pre-Technical |
| Social Sciences | — | English, Kiswahili | Social Studies, Business Studies, Life Skills |
| Arts & Sports | — | — | Creative Arts, Sports/PE, Health Education |

AE1 threshold = 31%. If the top pathway fails core validation, the system picks the pathway requiring the least remediation (smallest gap to meeting requirements). All fallback recommendations carry a `below_expectations_warning`.

## Competencies

7 CBC Core Competencies with ideal pathway profiles:

| Competency | STEM | Social Sciences | Arts & Sports |
|-----------|------|----------------|---------------|
| Communication & Collaboration | 0.6 | 0.9 | 0.7 |
| Critical Thinking & Problem Solving | 0.9 | 0.7 | 0.5 |
| Creativity & Imagination | 0.5 | 0.5 | 0.9 |
| Digital Literacy | 0.8 | 0.5 | 0.4 |
| Learning to Learn | 0.7 | 0.7 | 0.6 |
| Citizenship | 0.5 | 0.8 | 0.5 |
| Self-efficacy | 0.6 | 0.6 | 0.8 |

## RL Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 0.001 | DQN optimizer step size |
| gamma | 0.99 | Discount factor |
| epsilon_start | 1.0 | Initial exploration rate |
| epsilon_end | 0.01 | Minimum exploration rate |
| epsilon_decay | 0.995 | Per-episode decay factor |
| batch_size | 32 | Training batch size |
| buffer_size | 10000 | Experience replay capacity |
| target_update | 10 | Target network sync interval |

## Placement Formula

`60% KJSEA (Grade 9) + 20% SBA (Grade 7–8) + 20% KPSEA (Grade 6)`
