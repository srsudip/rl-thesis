"""
RL Benchmarking Suite — Jordan et al. (2020) Methodology
=========================================================

Implements three benchmarks from:
    "Evaluating the Performance of Reinforcement Learning Algorithms"
    Jordan, Laber, Davidian, Tsiatis, Zeng (ICML 2020)

Benchmark 1 — Performance Distribution:
    Empirical CDF of final returns across T independent trials with
    DKW (Dvoretzky–Kiefer–Wolfowitz) confidence bands.

Benchmark 2 — Convergence Analysis:
    Learning curves with PBP-t (Student's t) confidence intervals
    on mean accuracy at each episode.

Benchmark 3 — Per-Pathway Accuracy:
    Breakdown by pathway (STEM, Social Sciences, Arts & Sports)
    with 95 % confidence intervals on each.

Usage:
    from benchmarks import RLBenchmark
    bm = RLBenchmark(n_students=200, n_episodes=500, n_trials=5)
    results = bm.run_all()
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats as sp_stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PATHWAYS, COMPETENCIES, RL_CONFIG, MODELS_DIR
from src.rl.agent import PathwayRecommendationAgent
from src.rl.environment import PathwayEnvironment
from src.data.cbc_data_generator import generate_dashboard_data


# ─────────────────────────────────────────────────────────────
#  Statistical helpers
# ─────────────────────────────────────────────────────────────

def dkw_epsilon(n: int, delta: float = 0.05) -> float:
    """
    DKW inequality confidence band half-width.

    For n samples and confidence level 1 - δ:
        ε = sqrt( ln(2/δ) / (2n) )

    Reference: Massart (1990) tight constant.
    """
    return np.sqrt(np.log(2.0 / delta) / (2.0 * n))


def t_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute mean and symmetric CI using Student's t-distribution (PBP-t).

    Returns (mean, ci_lower, ci_upper).
    """
    n = len(values)
    if n < 2:
        m = float(values[0]) if n == 1 else 0.0
        return m, m, m
    m = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    t_val = sp_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(m), float(m - t_val * se), float(m + t_val * se)


# ─────────────────────────────────────────────────────────────
#  Benchmark runner
# ─────────────────────────────────────────────────────────────

class RLBenchmark:
    """
    Run reproducible RL benchmarks following Jordan et al. (2020).

    Parameters
    ----------
    n_students : int
        Synthetic students per trial.
    n_episodes : int
        Training episodes per trial.
    n_trials : int
        Independent training runs (different seeds).
        5-10 for quick validation; 30+ for publication-grade results.
    base_seed : int
        Starting seed; trial i uses base_seed + i.
    delta : float
        DKW confidence parameter (default 0.05 → 95 % bands).
    """

    PATHWAY_NAMES = list(PATHWAYS.keys())

    def __init__(
        self,
        n_students: int = 200,
        n_episodes: int = 500,
        n_trials: int = 5,
        base_seed: int = 42,
        delta: float = 0.05,
        verbose: bool = True,
    ):
        self.n_students = n_students
        self.n_episodes = n_episodes
        self.n_trials = n_trials
        self.base_seed = base_seed
        self.delta = delta
        self.verbose = verbose

        # Collected across trials
        self._trial_returns: List[float] = []
        self._trial_accuracies: List[float] = []
        self._trial_curves: List[List[float]] = []       # per-episode accuracy
        self._trial_reward_curves: List[List[float]] = [] # per-episode reward
        self._trial_pathway_acc: List[Dict[str, float]] = []
        self._trial_histories: List[Dict] = []

    # ─────── core training loop ──────────────────────────────

    def _run_single_trial(self, seed: int) -> Dict:
        """Train one agent from scratch and collect metrics."""
        np.random.seed(seed)

        # Generate fresh data
        data = generate_dashboard_data(
            n_students=self.n_students,
            grades=(4, 5, 6, 7, 8, 9),
            seed=seed,
            save_csv=False,
        )

        env = PathwayEnvironment(data['assessments'], data['competencies'])
        agent = PathwayRecommendationAgent()

        # Train
        history = agent.train(
            env,
            episodes=self.n_episodes,
            verbose=False,
            batch_per_episode=min(128, len(env.students)),
            early_stopping=False,
        )

        # Final greedy evaluation
        correct = 0
        pathway_correct = {p: 0 for p in self.PATHWAY_NAMES}
        pathway_total = {p: 0 for p in self.PATHWAY_NAMES}

        for sid in env.students:
            state = env.reset(sid)
            rec = agent.recommend(state)
            gt = env.get_ground_truth_pathway(sid)
            pathway_total[gt] += 1
            if rec['recommended_pathway'] == gt:
                correct += 1
                pathway_correct[gt] += 1

        total = len(env.students)
        accuracy = correct / total if total > 0 else 0.0

        # Compute per-pathway accuracy
        pw_acc = {}
        for p in self.PATHWAY_NAMES:
            if pathway_total[p] > 0:
                pw_acc[p] = pathway_correct[p] / pathway_total[p]
            else:
                pw_acc[p] = 0.0

        # Average return from last 50 episodes
        rewards = history.get('rewards', [])
        tail = rewards[-50:] if len(rewards) >= 50 else rewards
        avg_return = float(np.mean(tail)) if tail else 0.0

        return {
            'accuracy': accuracy,
            'avg_return': avg_return,
            'pathway_accuracy': pw_acc,
            'reward_curve': rewards,
            'accuracy_curve': history.get('accuracy', []),
            'history': history,
        }

    # ─────── public API ──────────────────────────────────────

    def run_all(self, save_path: Optional[Path] = None) -> Dict:
        """
        Execute all three benchmarks across n_trials.

        Returns a dict with keys:
            performance_distribution, convergence, per_pathway, summary
        """
        t0 = time.time()
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  JORDAN et al. (2020) RL BENCHMARKS")
            print(f"  Trials={self.n_trials}  Episodes={self.n_episodes}  "
                  f"Students={self.n_students}")
            print(f"{'='*60}\n")

        # Run trials
        for i in range(self.n_trials):
            seed = self.base_seed + i
            if self.verbose:
                print(f"  Trial {i+1}/{self.n_trials} (seed={seed}) ...", end=" ", flush=True)
            t1 = time.time()
            result = self._run_single_trial(seed)
            elapsed = time.time() - t1

            self._trial_returns.append(result['avg_return'])
            self._trial_accuracies.append(result['accuracy'])
            self._trial_curves.append(result['accuracy_curve'])
            self._trial_reward_curves.append(result['reward_curve'])
            self._trial_pathway_acc.append(result['pathway_accuracy'])
            self._trial_histories.append(result['history'])

            if self.verbose:
                print(f"acc={result['accuracy']:.1%}  "
                      f"return={result['avg_return']:.2f}  ({elapsed:.1f}s)")

        total_time = time.time() - t0

        # Compute benchmarks
        bm1 = self._benchmark_performance_distribution()
        bm2 = self._benchmark_convergence()
        bm3 = self._benchmark_per_pathway()

        summary = {
            'n_trials': self.n_trials,
            'n_episodes': self.n_episodes,
            'n_students': self.n_students,
            'total_time_s': round(total_time, 1),
            'mean_accuracy': float(np.mean(self._trial_accuracies)),
            'std_accuracy': float(np.std(self._trial_accuracies)),
        }
        summary['accuracy_ci_95'] = t_confidence_interval(
            np.array(self._trial_accuracies)
        )

        results = {
            'performance_distribution': bm1,
            'convergence': bm2,
            'per_pathway': bm3,
            'summary': summary,
        }

        if self.verbose:
            self._print_summary(results)

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert numpy types for JSON
            self._save_json(results, save_path)

        return results

    # ─────── Benchmark 1: Performance Distribution ───────────

    def _benchmark_performance_distribution(self) -> Dict:
        """
        Empirical CDF of average returns with DKW confidence bands.
        """
        returns = np.array(self._trial_returns)
        n = len(returns)
        eps = dkw_epsilon(n, self.delta)

        sorted_returns = np.sort(returns)
        ecdf = np.arange(1, n + 1) / n

        # Quantile function at selected probabilities
        probs = np.linspace(0.05, 0.95, 19)
        quantiles = {f"{p:.2f}": float(np.percentile(returns, p * 100)) for p in probs}

        mean_ret, ci_lo, ci_hi = t_confidence_interval(returns)

        return {
            'returns': sorted_returns.tolist(),
            'ecdf': ecdf.tolist(),
            'dkw_epsilon': float(eps),
            'dkw_lower': np.clip(ecdf - eps, 0, 1).tolist(),
            'dkw_upper': np.clip(ecdf + eps, 0, 1).tolist(),
            'mean_return': mean_ret,
            'median_return': float(np.median(returns)),
            'std_return': float(np.std(returns)),
            'ci_95_return': (ci_lo, ci_hi),
            'quantiles': quantiles,
        }

    # ─────── Benchmark 2: Convergence Analysis ───────────────

    def _benchmark_convergence(self) -> Dict:
        """
        Learning curves with PBP-t confidence intervals on accuracy.
        """
        # Pad curves to same length
        max_len = max(len(c) for c in self._trial_curves) if self._trial_curves else 0
        if max_len == 0:
            return {'episodes': [], 'mean_accuracy': [], 'ci_lower': [], 'ci_upper': []}

        padded = []
        for curve in self._trial_curves:
            if len(curve) < max_len:
                pad_val = curve[-1] if curve else 0.0
                padded.append(curve + [pad_val] * (max_len - len(curve)))
            else:
                padded.append(curve[:max_len])

        mat = np.array(padded)  # (n_trials, n_episodes)

        # Smooth with rolling window
        window = 20
        smoothed = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                start = max(0, j - window + 1)
                smoothed[i, j] = np.mean(mat[i, start:j+1])

        # Per-episode CI
        episodes = list(range(1, max_len + 1))
        mean_acc = []
        ci_lower = []
        ci_upper = []

        for ep in range(max_len):
            m, lo, hi = t_confidence_interval(smoothed[:, ep])
            mean_acc.append(m)
            ci_lower.append(lo)
            ci_upper.append(hi)

        # Find episode where 50% accuracy first reached
        first_50 = None
        for ep, m in enumerate(mean_acc):
            if m >= 0.50:
                first_50 = ep + 1
                break

        # Reward curves
        max_rew_len = max(len(c) for c in self._trial_reward_curves) if self._trial_reward_curves else 0
        rew_padded = []
        for curve in self._trial_reward_curves:
            if len(curve) < max_rew_len:
                pad_val = curve[-1] if curve else 0.0
                rew_padded.append(curve + [pad_val] * (max_rew_len - len(curve)))
            else:
                rew_padded.append(curve[:max_rew_len])

        rew_mat = np.array(rew_padded) if rew_padded else np.array([[]])

        # Smooth reward curves
        rew_smoothed = np.zeros_like(rew_mat)
        for i in range(rew_mat.shape[0]):
            for j in range(rew_mat.shape[1]):
                start = max(0, j - window + 1)
                rew_smoothed[i, j] = np.mean(rew_mat[i, start:j+1])

        mean_rew = rew_smoothed.mean(axis=0).tolist() if rew_smoothed.size > 0 else []
        std_rew = rew_smoothed.std(axis=0).tolist() if rew_smoothed.size > 0 else []

        return {
            'episodes': episodes,
            'mean_accuracy': mean_acc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'smoothing_window': window,
            'first_50pct_episode': first_50,
            'mean_reward': mean_rew,
            'std_reward': std_rew,
        }

    # ─────── Benchmark 3: Per-Pathway Accuracy ───────────────

    def _benchmark_per_pathway(self) -> Dict:
        """
        Per-pathway accuracy breakdown with 95 % CIs.
        """
        result = {}
        for pathway in self.PATHWAY_NAMES:
            accs = [trial[pathway] for trial in self._trial_pathway_acc if pathway in trial]
            if accs:
                m, lo, hi = t_confidence_interval(np.array(accs))
                result[pathway] = {
                    'mean_accuracy': m,
                    'ci_lower': lo,
                    'ci_upper': hi,
                    'std': float(np.std(accs)),
                    'n_trials': len(accs),
                    'all_values': accs,
                }
            else:
                result[pathway] = {
                    'mean_accuracy': 0.0,
                    'ci_lower': 0.0,
                    'ci_upper': 0.0,
                    'std': 0.0,
                    'n_trials': 0,
                }

        # Aggregate overall
        all_accs = np.array(self._trial_accuracies)
        m, lo, hi = t_confidence_interval(all_accs)
        result['OVERALL'] = {
            'mean_accuracy': m,
            'ci_lower': lo,
            'ci_upper': hi,
            'std': float(np.std(all_accs)),
            'n_trials': len(all_accs),
        }

        return result

    # ─────── output helpers ──────────────────────────────────

    def _print_summary(self, results: Dict):
        s = results['summary']
        pd_ = results['performance_distribution']
        pp = results['per_pathway']

        print(f"\n{'='*60}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"  Trials: {s['n_trials']}  |  Episodes: {s['n_episodes']}  |  Time: {s['total_time_s']}s")
        print(f"\n  [1] Performance Distribution")
        print(f"      Mean return:   {pd_['mean_return']:.3f}")
        print(f"      Median return: {pd_['median_return']:.3f}")
        print(f"      95% CI:        [{pd_['ci_95_return'][0]:.3f}, {pd_['ci_95_return'][1]:.3f}]")
        print(f"      DKW ε:         {pd_['dkw_epsilon']:.4f}")

        conv = results['convergence']
        print(f"\n  [2] Convergence")
        if conv['first_50pct_episode']:
            print(f"      First 50% acc: episode {conv['first_50pct_episode']}")
        final_acc = conv['mean_accuracy'][-1] if conv['mean_accuracy'] else 0
        print(f"      Final mean acc: {final_acc:.1%}")

        print(f"\n  [3] Per-Pathway Accuracy")
        for pathway in self.PATHWAY_NAMES:
            p = pp[pathway]
            print(f"      {pathway:20s}: {p['mean_accuracy']:.1%}  "
                  f"[{p['ci_lower']:.1%}, {p['ci_upper']:.1%}]")
        o = pp['OVERALL']
        print(f"      {'OVERALL':20s}: {o['mean_accuracy']:.1%}  "
              f"[{o['ci_lower']:.1%}, {o['ci_upper']:.1%}]")
        print(f"{'='*60}\n")

    def _save_json(self, results: Dict, path: Path):
        """Save results converting numpy types."""
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        with open(path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        if self.verbose:
            print(f"  Results saved to {path}")


# ─────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run RL benchmarks")
    parser.add_argument('--students', type=int, default=200)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    bm = RLBenchmark(
        n_students=args.students,
        n_episodes=args.episodes,
        n_trials=args.trials,
        base_seed=args.seed,
    )

    save_path = Path(args.output) if args.output else MODELS_DIR / 'benchmark_results.json'
    bm.run_all(save_path=save_path)
