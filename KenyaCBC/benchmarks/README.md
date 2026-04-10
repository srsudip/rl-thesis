# benchmarks/ — Statistical Evaluation

Benchmarking framework based on Jordan et al. (2020) ICML methodology.

## Benchmarks

### 1. Performance Distribution (DKW)

Empirical CDF of agent accuracy with Dvoretzky–Kiefer–Wolfowitz confidence bands. Measures distributional stability across independent runs.

### 2. Convergence Analysis (PBP-t)

Learning curves plotting accuracy vs. training episode with Student's t confidence intervals. Measures learning speed and stability.

### 3. Per-Pathway Accuracy

Accuracy broken down by STEM, Social Sciences, and Arts & Sports with 95% CIs. Detects pathway-specific biases.

## Key Metrics

| Metric | Description |
|--------|-------------|
| DKW ε | Confidence band width (lower = more stable) |
| 95% CI | Accuracy confidence interval |
| PBP-t | Probable Best Policy per Student's t |
| First 50% episode | Episode at which 50% accuracy first reached |

## Usage

Run benchmarks from the Advanced page in the dashboard, or programmatically:

```python
from benchmarks.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run_all(n_seeds=5, episodes=200)
```

Output saved to `data/benchmark_results.json`.

## Files

| File | Purpose |
|------|---------|
| `benchmark_runner.py` | Orchestrates all three benchmarks |

## Statistical Tests (Advanced page)

The Advanced dashboard page exposes additional evaluation beyond the three core benchmarks:

- **Multi-seed evaluation** — mean ± std across N independent training seeds
- **Paired t-test / Wilcoxon signed-rank** — DQN vs each baseline
- **Cohen's d** — effect size for practical significance
- **Bootstrap CIs** — non-parametric confidence intervals on per-pathway accuracy

These are implemented in `src/rl/evaluation_extended.py` and called from `pages/dash_pages/advanced.py`.
