# Benchmarking Suite — `benchmarks/`

## Reference

Jordan, Laber, Davidian, Tsiatis, Zeng (2020).
*Evaluating the Performance of Reinforcement Learning Algorithms.*
ICML 2020.

## Three Benchmarks

### Benchmark 1: Performance Distribution

Collects average returns from the last 50 episodes of each of T independent trials.
Reports the empirical CDF with **DKW (Dvoretzky–Kiefer–Wolfowitz) confidence bands**:

```
ε = sqrt( ln(2/δ) / (2n) )       (Massart 1990 tight constant)
```

For n = 5 trials and δ = 0.05: ε ≈ 0.61
For n = 30 trials: ε ≈ 0.25

**Outputs**: sorted returns, ECDF, DKW bands, quantile function at 19 points, mean/median/std with 95% CI.

### Benchmark 2: Convergence Analysis

Learning curves with **PBP-t (Student's t)** confidence intervals at each episode:

```
CI(ep) = mean(acc[ep]) ± t_{α/2, n-1} × std(acc[ep]) / sqrt(n)
```

Smoothed with rolling window of 20 episodes. Reports episode at which 50% accuracy is first reached.

**Outputs**: per-episode mean accuracy with CI bands, smoothed reward curves with std bands.

### Benchmark 3: Per-Pathway Accuracy

Breakdown by STEM, Social Sciences, Arts & Sports with 95% CI on each:

```
For each pathway p:
    acc_p[trial] = correct_p / total_p
    CI_p = t_confidence_interval(acc_p)
```

**Outputs**: mean accuracy + CI per pathway, overall aggregated accuracy.

## Usage

### Command Line

```bash
python -m benchmarks.benchmark --trials 5 --episodes 500 --students 200
```

### Python API

```python
from benchmarks import RLBenchmark

bm = RLBenchmark(n_students=200, n_episodes=500, n_trials=5)
results = bm.run_all(save_path='models/benchmark_results.json')
```

### Recommended Settings

| Context | Trials | Episodes | Students |
|---------|--------|----------|----------|
| Quick check | 3 | 100 | 50 |
| Validation | 5 | 500 | 200 |
| Publication | 30+ | 800+ | 500+ |

## Statistical Tools

| Tool | Purpose | Reference |
|------|---------|-----------|
| DKW bound | Non-parametric CI on empirical CDF | Massart (1990) |
| PBP-t | Parametric CI on mean | Student's t-distribution |
| Quantile function | Distribution shape | Jordan et al. (2020) §3 |
