# Test Suite — `tests/test_all.py`

## Running Tests

```bash
# Standard runner
python tests/test_all.py

# With pytest (more features)
python -m pytest tests/test_all.py -v
```

## Test Categories

### 1. Data Generator (5 tests)

| Test | Validates |
|------|-----------|
| `test_student_count` | Correct number of students generated |
| `test_competency_shape` | 7 competency score columns present |
| `test_assessment_scores_range` | All scores in [0, 100] |
| `test_pathway_distribution` | Multiple pathways represented |
| `test_grades_present` | Grades 4-9 all present |

### 2. Environment (4 tests)

| Test | Validates |
|------|-----------|
| `test_state_dimension` | State vector is 7-dimensional |
| `test_action_space` | 3 valid actions |
| `test_episode_single_step` | Done after one step |
| `test_correct_match_higher_reward` | Correct pathway gives higher reward |

### 3. Agent (3 tests)

| Test | Validates |
|------|-----------|
| `test_action_selection` | Actions in {0, 1, 2} |
| `test_recommend_output` | Recommendation structure correct |
| `test_save_load` | Model persistence works |

### 4. Trainer (2 tests)

| Test | Validates |
|------|-----------|
| `test_training_produces_history` | History has expected keys and length |
| `test_evaluation` | Per-pathway evaluation produces valid accuracy |

### 5. HITL Workflow (3 tests)

| Test | Validates |
|------|-----------|
| `test_submit_and_approve` | Submit → approve creates active override |
| `test_submit_and_reject` | Submit → reject leaves no override |
| `test_duplicate_pending_blocked` | Cannot double-submit for same student |

### 6. Benchmarks (3 tests)

| Test | Validates |
|------|-----------|
| `test_dkw_bound` | DKW ε decreases with more samples |
| `test_t_confidence_interval` | CI brackets the mean |
| `test_mini_benchmark_pipeline` | End-to-end benchmark completes |

## Total: 20 tests across 6 categories
