# tests/ — Integration Test Suite

Three test files covering every major data path, RL component, and dashboard computation.

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_verification.py -v

# Specific class
python -m pytest tests/test_verification.py::TestCosineSimilarity -v
```

## Test Files

### `test_all.py` — Comprehensive System Tests

Seven test categories exercising the full pipeline end-to-end:

| Class | What it tests |
|-------|--------------|
| `TestDataGenerator` | IRT generation, reproducibility, score bounds, derived subjects |
| `TestEnvironment` | 78-dim state vector, reset/step semantics, reward signal |
| `TestAgent` | DQN forward pass, action selection, PER buffer, double-Q update |
| `TestTrainer` | Training loop convergence, early stopping, model persistence |
| `TestHITL` | Request submission, teacher approve/reject, audit log integrity |
| `TestBenchmarks` | DKW bands, convergence curves, per-pathway accuracy |

### `test_verification.py` — Phase 6 Verification Tests

Fine-grained checks added during the multi-signal fusion and pathway history implementation:

| Class | What it tests |
|-------|--------------|
| `TestDataConsistency` | `gradess1.csv` column integrity, grade ranges, pathway label coverage |
| `TestCSVUpload` | Curriculum CSV loading and relationship integrity |
| `TestDQNCoaching` | 9-action space, `deepen_current_pathway` subject targets, reward shaping |
| `TestCosineSimilarity` | Cosine similarity computation vs ideal pathway vectors |
| `TestPathwayHistory` | Per-grade PSI history, suitability annotations on recommended pathway |
| `TestPreferencesHITL` | Student preference signals in reward function |
| `TestEdgeCases` | Single-subject students, all-zeros profiles, boundary grade scores |
| `TestTrajectoryStability` | Monotonicity checks on improving/declining student trajectories |
| `TestStudentFeedback` | Feedback signal (`satisfied`/`wants_different`) state encoding |
| `TestDQNFeedbackState` | Feedback dimension in 78-dim state vector wired correctly |

### `test_audit.py` — Critical System Audit

Exhaustive consistency tests tracing every data path:
- Every function that computes scores, cluster weights, and recommendations is called and its output validated
- Cross-checks between `real_data_loader.py` and `cbc_data_generator.py` outputs
- KJSEA result slip subject codes (901–912) and grade mappings

## Coverage Focus

Tests do **not** mock the data layer — they use real generated data (seed=42, 50 students by default) to catch integration failures that unit mocks would miss. This design choice was made after a prior incident where mocked tests passed but the real pipeline failed on edge-case grade distributions.
