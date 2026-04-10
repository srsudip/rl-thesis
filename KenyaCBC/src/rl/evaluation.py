"""
Comprehensive Evaluation Module for RL Pathway Recommendation.

Merged module providing all evaluation capabilities:

A) Cross-Validation & Bootstrap
   - K-Fold Cross-Validation with confidence intervals
   - Bootstrap confidence intervals
   - Learning curve analysis

B) Multi-Seed Robustness
   - Multi-seed evaluation (run N seeds, report mean ± std)
   - Quick multi-seed evaluation helper

C) Statistical Testing
   - Paired t-test and Wilcoxon signed-rank tests
   - Effect size calculation (Cohen's d)
   - Method comparison framework (CV-based and seed-based)

D) Baseline Comparisons
   - DQN vs all baselines with statistical significance
   - Comprehensive comparison report

E) Hyperparameter Tuning
   - Grid search with configurable parameter space

F) Reporting
   - Markdown report generation
   - JSON export for results

This provides rigorous evaluation required for thesis-level research.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PATHWAYS, get_pathway_from_index, get_pathway_index


# =============================================================================
# A) CROSS-VALIDATION EVALUATION
# =============================================================================

def cross_validate_agent(data: Dict,
                         agent_factory: Callable,
                         train_fn: Callable,
                         n_folds: int = 5,
                         episodes: int = 300,
                         random_state: int = 42,
                         verbose: bool = True) -> Dict:
    """
    Perform K-fold cross-validation on the RL agent.
    
    Args:
        data: Generated CBC data dictionary
        agent_factory: Function that creates a new agent instance
        train_fn: Function that trains the agent (agent, train_data) -> agent
        n_folds: Number of cross-validation folds
        episodes: Training episodes per fold
        random_state: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        Dictionary with cross-validation results and confidence intervals
    """
    from src.rl.environment import PathwayEnvironment
    
    profiles = data['students']
    student_ids = profiles['student_id'].values
    pathways = data['pathways']['recommended_pathway'].values
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Fall back to regular KFold if only 1 class
    n_classes = len(set(pathways))
    if n_classes < 2:
        from sklearn.model_selection import KFold
        skf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {n_folds}-FOLD CROSS-VALIDATION")
        print(f"{'='*60}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(student_ids, pathways) if n_classes >= 2 else skf.split(student_ids)):
        if verbose:
            print(f"\n  Fold {fold + 1}/{n_folds}:")
        
        train_students = set(student_ids[train_idx])
        test_students = set(student_ids[test_idx])
        
        train_data = _subset_data(data, train_students)
        test_data = _subset_data(data, test_students)
        
        if verbose:
            print(f"    Train: {len(train_students)} students")
            print(f"    Test:  {len(test_students)} students")
        
        agent = agent_factory()
        agent = train_fn(agent, train_data, episodes=episodes, verbose=False)
        
        env = PathwayEnvironment(test_data['assessments'], test_data['competencies'])
        
        y_true = []
        y_pred = []
        y_prob = []

        # Only evaluate students that have transitions in the test environment.
        # Students with <2 grade records are excluded by extract_transitions, so
        # iterating raw test_students would silently return wrong states for them.
        for student_id in env.students:
            state = env.reset(student_id)
            gt = env.get_ground_truth_pathway(student_id)
            rec = agent.recommend(state)
            
            y_true.append(gt)
            y_pred.append(rec['recommended_pathway'])

            # confidence_scores is keyed by action name; aggregate to pathway level
            # by summing confidence for actions whose target_pathway matches each pathway
            from src.rl.dqn_coaching import ACTIONS as _ACTIONS
            pw_conf = {p: 0.0 for p in PATHWAYS.keys()}
            for action_name, conf in rec.get('confidence_scores', {}).items():
                action_obj = next((a for a in _ACTIONS if a.name == action_name), None)
                if action_obj and action_obj.target_pathway in pw_conf:
                    pw_conf[action_obj.target_pathway] += conf
            probs = [pw_conf[p] for p in PATHWAYS.keys()]
            y_prob.append(probs)
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'n_train': len(train_students),
            'n_test': len(test_students)
        })
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        
        if verbose:
            print(f"    Accuracy: {accuracy:.1%}, F1: {f1_macro:.1%}")
    
    # Aggregate results
    accuracies = [r['accuracy'] for r in fold_results]
    f1_macros = [r['f1_macro'] for r in fold_results]
    
    def ci_95(values):
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / np.sqrt(len(values))
        ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=se)
        return mean, std, ci[0], ci[1]
    
    acc_mean, acc_std, acc_ci_low, acc_ci_high = ci_95(accuracies)
    f1_mean, f1_std, f1_ci_low, f1_ci_high = ci_95(f1_macros)
    
    labels = list(PATHWAYS.keys())
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=labels, zero_division=0
    )
    
    per_pathway = {}
    for i, pathway in enumerate(labels):
        per_pathway[pathway] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    results = {
        'n_folds': n_folds,
        'fold_results': fold_results,
        'accuracy': {
            'mean': float(acc_mean),
            'std': float(acc_std),
            'ci_95_lower': float(acc_ci_low),
            'ci_95_upper': float(acc_ci_high)
        },
        'f1_macro': {
            'mean': float(f1_mean),
            'std': float(f1_std),
            'ci_95_lower': float(f1_ci_low),
            'ci_95_upper': float(f1_ci_high)
        },
        'confusion_matrix': cm.tolist(),
        'per_pathway': per_pathway,
        'classification_report': classification_report(
            all_y_true, all_y_pred,
            labels=labels,
            target_names=[PATHWAYS[p]['name'] for p in labels],
            output_dict=True
        )
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"\n  Accuracy: {acc_mean:.1%} ± {acc_std:.1%}")
        print(f"  95% CI:   [{acc_ci_low:.1%}, {acc_ci_high:.1%}]")
        print(f"\n  F1 Macro: {f1_mean:.1%} ± {f1_std:.1%}")
        print(f"  95% CI:   [{f1_ci_low:.1%}, {f1_ci_high:.1%}]")
        print(f"\n  Per-Pathway Performance:")
        print(f"  {'-'*50}")
        print(f"  {'Pathway':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*50}")
        for pathway in labels:
            m = per_pathway[pathway]
            print(f"  {PATHWAYS[pathway]['name']:<20} {m['precision']:>10.1%} "
                  f"{m['recall']:>10.1%} {m['f1']:>10.1%}")
    
    return results


def _subset_data(data: Dict, student_ids: set) -> Dict:
    """Create a subset of data for specific students."""
    subset = {}
    for key, df in data.items():
        if isinstance(df, pd.DataFrame) and 'student_id' in df.columns:
            subset[key] = df[df['student_id'].isin(student_ids)].copy()
        else:
            subset[key] = df
    return subset


# =============================================================================
# A.2) BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_evaluation(y_true: List,
                         y_pred: List,
                         n_bootstrap: int = 1000,
                         confidence_level: float = 0.95,
                         random_state: int = 42) -> Dict:
    """
    Compute bootstrap confidence intervals for evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 95%)
        random_state: Random seed
        
    Returns:
        Dictionary with bootstrap statistics
    """
    np.random.seed(random_state)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    
    boot_accuracies = []
    boot_f1_macros = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        acc = accuracy_score(y_true[idx], y_pred[idx])
        f1 = f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0)
        boot_accuracies.append(acc)
        boot_f1_macros.append(f1)
    
    alpha = 1 - confidence_level
    
    def compute_ci(values):
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'ci_lower': float(np.percentile(values, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(values, 100 * (1 - alpha / 2))),
            'median': float(np.median(values))
        }
    
    return {
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
        'accuracy': compute_ci(boot_accuracies),
        'f1_macro': compute_ci(boot_f1_macros)
    }


# =============================================================================
# A.3) LEARNING CURVE ANALYSIS
# =============================================================================

def learning_curve_analysis(data: Dict,
                            agent_factory: Callable,
                            train_fn: Callable,
                            train_sizes: List[float] = None,
                            episodes: int = 300,
                            n_iterations: int = 3,
                            random_state: int = 42,
                            verbose: bool = True) -> Dict:
    """
    Analyze how model performance changes with training data size.
    
    Args:
        data: Generated CBC data
        agent_factory: Function to create agent
        train_fn: Training function
        train_sizes: Fractions of training data to use
        episodes: Training episodes
        n_iterations: Number of iterations per size
        random_state: Random seed
        verbose: Print progress
        
    Returns:
        Learning curve results
    """
    from src.rl.environment import PathwayEnvironment
    
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    profiles = data['students']
    student_ids = profiles['student_id'].values
    pathways = data['pathways']['recommended_pathway'].values
    
    results = {'train_sizes': train_sizes, 'curves': []}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  LEARNING CURVE ANALYSIS")
        print(f"{'='*60}")
    
    stratify_arg = pathways if len(set(pathways)) >= 2 else None
    try:
        train_ids, test_ids = train_test_split(
            student_ids, test_size=0.2, stratify=stratify_arg, random_state=random_state
        )
    except ValueError:
        train_ids, test_ids = train_test_split(
            student_ids, test_size=0.2, random_state=random_state
        )
    
    test_data = _subset_data(data, set(test_ids))
    test_env = PathwayEnvironment(test_data['assessments'], test_data['competencies'])
    
    for size in train_sizes:
        size_results = {'train_size': size, 'accuracies': [], 'f1_scores': []}
        
        for iteration in range(n_iterations):
            n_train = int(len(train_ids) * size)
            np.random.seed(random_state + iteration)
            sample_ids = np.random.choice(train_ids, n_train, replace=False)
            
            train_data = _subset_data(data, set(sample_ids))
            
            agent = agent_factory()
            agent = train_fn(agent, train_data, episodes=episodes, verbose=False)
            
            y_true, y_pred = [], []
            for student_id in test_env.students:
                state = test_env.reset(student_id)
                gt = test_env.get_ground_truth_pathway(student_id)
                rec = agent.recommend(state)
                y_true.append(gt)
                y_pred.append(rec['recommended_pathway'])
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            size_results['accuracies'].append(acc)
            size_results['f1_scores'].append(f1)
        
        size_results['accuracy_mean'] = float(np.mean(size_results['accuracies']))
        size_results['accuracy_std'] = float(np.std(size_results['accuracies']))
        size_results['f1_mean'] = float(np.mean(size_results['f1_scores']))
        size_results['f1_std'] = float(np.std(size_results['f1_scores']))
        
        results['curves'].append(size_results)
        
        if verbose:
            print(f"  Train size {size*100:.0f}%: Acc={size_results['accuracy_mean']:.1%} "
                  f"± {size_results['accuracy_std']:.1%}")
    
    return results


# =============================================================================
# B) MULTI-SEED EVALUATION
# =============================================================================

def multi_seed_evaluation(
    data: Dict,
    seeds: List[int] = None,
    n_folds: int = 5,
    episodes: int = 300,
    verbose: bool = True
) -> Dict:
    """
    Run evaluation with multiple random seeds for robust results.
    
    Args:
        data: Generated CBC data dictionary
        seeds: List of random seeds (default: [42, 123, 456, 789, 1011])
        n_folds: Number of CV folds per seed
        episodes: Training episodes
        verbose: Print progress
        
    Returns:
        Dictionary with mean, std, CI for all metrics
    """
    from src.rl.environment import PathwayEnvironment
    from src.rl.agent import PathwayRecommendationAgent
    
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]
    
    all_results = {
        'accuracies': [],
        'f1_scores': [],
        'pathway_accuracies': {'STEM': [], 'SOCIAL_SCIENCES': [], 'ARTS_SPORTS': []},
        'training_times': [],
        'seed_details': []
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  MULTI-SEED EVALUATION ({len(seeds)} seeds)")
        print(f"{'='*60}")
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n  Seed {i+1}/{len(seeds)}: {seed}")
        
        np.random.seed(seed)
        start_time = time.time()
        
        env = PathwayEnvironment(data['assessments'], data['competencies'])
        
        agent = PathwayRecommendationAgent()
        agent.train(env, episodes=episodes, verbose=False,
                    batch_per_episode=min(128, len(env.students)))
        
        correct = 0
        pathway_correct = {'STEM': 0, 'SOCIAL_SCIENCES': 0, 'ARTS_SPORTS': 0}
        pathway_total = {'STEM': 0, 'SOCIAL_SCIENCES': 0, 'ARTS_SPORTS': 0}
        y_true, y_pred = [], []
        
        for student_id in env.students:
            state = env.reset(student_id)
            rec = agent.recommend(state)
            ground_truth = env.get_ground_truth_pathway(student_id)
            
            y_true.append(ground_truth)
            y_pred.append(rec['recommended_pathway'])
            
            pathway_total[ground_truth] += 1
            if rec['recommended_pathway'] == ground_truth:
                correct += 1
                pathway_correct[ground_truth] += 1
        
        accuracy = correct / len(env.students)
        training_time = time.time() - start_time
        
        f1 = f1_score(y_true, y_pred, average='weighted', labels=list(PATHWAYS.keys()))
        
        all_results['accuracies'].append(accuracy)
        all_results['f1_scores'].append(f1)
        all_results['training_times'].append(training_time)
        
        for pathway in pathway_correct:
            if pathway_total[pathway] > 0:
                all_results['pathway_accuracies'][pathway].append(
                    pathway_correct[pathway] / pathway_total[pathway]
                )
        
        all_results['seed_details'].append({
            'seed': seed,
            'accuracy': accuracy,
            'f1': f1,
            'time': training_time
        })
        
        if verbose:
            print(f"    Accuracy: {accuracy:.2%}, F1: {f1:.3f}, Time: {training_time:.1f}s")
    
    # Summary statistics
    accuracies = np.array(all_results['accuracies'])
    f1_scores = np.array(all_results['f1_scores'])
    
    n = len(seeds)
    ci_multiplier = stats.t.ppf(0.975, n-1) if n > 1 else 1.96
    
    std_acc = float(np.std(accuracies, ddof=1)) if n > 1 else 0.0
    std_f1 = float(np.std(f1_scores, ddof=1)) if n > 1 else 0.0
    
    summary = {
        'accuracy': {
            'mean': float(np.mean(accuracies)),
            'std': std_acc,
            'ci_95_lower': float(np.mean(accuracies) - ci_multiplier * std_acc / np.sqrt(n)),
            'ci_95_upper': float(np.mean(accuracies) + ci_multiplier * std_acc / np.sqrt(n)),
            'min': float(np.min(accuracies)),
            'max': float(np.max(accuracies))
        },
        'f1': {
            'mean': float(np.mean(f1_scores)),
            'std': std_f1,
            'ci_95_lower': float(np.mean(f1_scores) - ci_multiplier * std_f1 / np.sqrt(n)),
            'ci_95_upper': float(np.mean(f1_scores) + ci_multiplier * std_f1 / np.sqrt(n))
        },
        'pathway_accuracies': {},
        'n_seeds': n,
        'seeds': seeds,
        'raw_results': all_results
    }
    
    for pathway in all_results['pathway_accuracies']:
        accs = np.array(all_results['pathway_accuracies'][pathway])
        if len(accs) > 0:
            summary['pathway_accuracies'][pathway] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  SUMMARY ({n} seeds)")
        print(f"{'='*60}")
        print(f"  Accuracy: {summary['accuracy']['mean']:.2%} ± {summary['accuracy']['std']:.2%}")
        print(f"  95% CI: [{summary['accuracy']['ci_95_lower']:.2%}, "
              f"{summary['accuracy']['ci_95_upper']:.2%}]")
        print(f"  F1 Score: {summary['f1']['mean']:.3f} ± {summary['f1']['std']:.3f}")
        print(f"\n  Per-Pathway Accuracy:")
        for pathway, stats_dict in summary['pathway_accuracies'].items():
            print(f"    {pathway}: {stats_dict['mean']:.2%} ± {stats_dict['std']:.2%}")
    
    return summary


def audit_action_distribution(env) -> Dict:
    """
    Report the ground-truth action frequency distribution across all students
    in the environment to detect label imbalance in the inferred training set.

    A dominant action exceeding 40 % of the dataset makes the headline
    action-accuracy figure misleading (a majority-class predictor achieves
    that floor trivially).  Pathway-level aggregation shows whether the
    imbalance also propagates to pathway labels.

    Args:
        env: A fitted PathwayEnvironment instance.

    Returns:
        Dict with per-action counts/fractions, per-pathway aggregation,
        dominant-action fraction, and an imbalance warning flag.
    """
    from collections import Counter
    from src.rl.dqn_coaching import ACTIONS

    counts = Counter(env.get_ground_truth_action(sid) for sid in env.students)
    total = max(sum(counts.values()), 1)

    per_action: Dict = {}
    pathway_agg: Dict[str, int] = {}

    for action_idx in range(len(ACTIONS)):
        action = ACTIONS[action_idx]
        count = counts.get(action_idx, 0)
        pw = action.target_pathway or 'ambiguous'
        per_action[action.name] = {
            'count': count,
            'fraction': round(count / total, 4),
            'target_pathway': pw,
        }
        pathway_agg[pw] = pathway_agg.get(pw, 0) + count

    per_pathway = {
        pw: {'count': c, 'fraction': round(c / total, 4)}
        for pw, c in pathway_agg.items()
    }

    dominant_fraction = max(counts.values(), default=0) / total
    dominant_action   = ACTIONS[counts.most_common(1)[0][0]].name if counts else 'none'

    return {
        'per_action':              per_action,
        'per_pathway':             per_pathway,
        'total_students':          total,
        'dominant_action':         dominant_action,
        'dominant_action_fraction': round(dominant_fraction, 4),
        'imbalance_warning':       dominant_fraction > 0.40,
        'note': (
            'imbalance_warning=True means one action accounts for >40 % of labels. '
            'Action accuracy above that threshold is achievable by a trivial majority predictor.'
        ),
    }


def quick_multi_seed_eval(data: Dict, seeds: List[int] = None,
                          episodes: int = 200) -> str:
    """Quick evaluation returning formatted string for display."""
    if seeds is None:
        seeds = [42, 123, 456]
    results = multi_seed_evaluation(data, seeds=seeds, episodes=episodes, verbose=False)
    
    return (f"Accuracy: {results['accuracy']['mean']:.2%} ± {results['accuracy']['std']:.2%}\n"
            f"95% CI: [{results['accuracy']['ci_95_lower']:.2%}, "
            f"{results['accuracy']['ci_95_upper']:.2%}]\n"
            f"F1 Score: {results['f1']['mean']:.3f} ± {results['f1']['std']:.3f}")


# =============================================================================
# C) STATISTICAL SIGNIFICANCE TESTS
# =============================================================================

def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def statistical_comparison(
    dqn_results: List[float],
    baseline_results: List[float],
    baseline_name: str = "Baseline",
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict:
    """
    Perform statistical significance tests comparing DQN to a baseline.
    
    Uses paired t-test (parametric), Wilcoxon (non-parametric), and Cohen's d.
    
    Args:
        dqn_results: List of DQN accuracies (one per seed)
        baseline_results: List of baseline accuracies (same seeds)
        baseline_name: Name of baseline for reporting
        alpha: Significance level
        verbose: Print results
        
    Returns:
        Dictionary with test results
    """
    dqn = np.array(dqn_results)
    baseline = np.array(baseline_results)
    
    # Paired t-test (parametric)
    t_stat, t_pvalue = stats.ttest_rel(dqn, baseline)
    
    # Wilcoxon signed-rank test (non-parametric)
    differences = dqn - baseline
    if np.all(differences == 0):
        w_stat, w_pvalue = np.nan, 1.0
    else:
        try:
            w_stat, w_pvalue = stats.wilcoxon(dqn, baseline, alternative='two-sided')
        except ValueError:
            w_stat, w_pvalue = np.nan, 1.0
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(dqn, ddof=1) + np.var(baseline, ddof=1)) / 2)
    cohens_d = (np.mean(dqn) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0
    
    results = {
        'baseline_name': baseline_name,
        'dqn_mean': float(np.mean(dqn)),
        'dqn_std': float(np.std(dqn, ddof=1)),
        'baseline_mean': float(np.mean(baseline)),
        'baseline_std': float(np.std(baseline, ddof=1)),
        'difference': float(np.mean(dqn) - np.mean(baseline)),
        'paired_ttest': {
            't_statistic': float(t_stat),
            'p_value': float(t_pvalue),
            'significant': bool(t_pvalue < alpha)
        },
        'wilcoxon': {
            'w_statistic': float(w_stat) if not np.isnan(w_stat) else None,
            'p_value': float(w_pvalue) if not np.isnan(w_pvalue) else None,
            'significant': bool(w_pvalue < alpha) if not np.isnan(w_pvalue) else False
        },
        'effect_size': {
            'cohens_d': float(cohens_d),
            'interpretation': interpret_cohens_d(cohens_d)
        },
        'alpha': alpha
    }
    
    if verbose:
        print(f"\n  Statistical Comparison: DQN vs {baseline_name}")
        print(f"  {'-'*50}")
        print(f"  DQN:      {results['dqn_mean']:.2%} ± {results['dqn_std']:.2%}")
        print(f"  {baseline_name}: {results['baseline_mean']:.2%} ± {results['baseline_std']:.2%}")
        print(f"  Difference: {results['difference']:+.2%}")
        print(f"  Paired t-test: t={t_stat:.3f}, p={t_pvalue:.4f} "
              f"{'*' if t_pvalue < alpha else ''}")
        if not np.isnan(w_stat):
            print(f"  Wilcoxon test: W={w_stat:.1f}, p={w_pvalue:.4f} "
                  f"{'*' if w_pvalue < alpha else ''}")
        print(f"  Effect size (Cohen's d): {cohens_d:.3f} "
              f"({results['effect_size']['interpretation']})")
        if t_pvalue < alpha:
            print(f"  → DQN is SIGNIFICANTLY "
                  f"{'better' if results['difference'] > 0 else 'worse'} (p < {alpha})")
        else:
            print(f"  → No significant difference at α = {alpha}")
    
    return results


def compare_methods_statistical(results_a: Dict,
                                results_b: Dict,
                                method_name_a: str = "Method A",
                                method_name_b: str = "Method B") -> Dict:
    """
    Perform statistical comparison between two methods using CV fold results.
    
    Uses paired t-test, Wilcoxon signed-rank, and Cohen's d on fold accuracies.
    
    Args:
        results_a: CV results from method A (must contain 'fold_results')
        results_b: CV results from method B
        method_name_a: Name of method A
        method_name_b: Name of method B
        
    Returns:
        Statistical comparison results
    """
    acc_a = [r['accuracy'] for r in results_a['fold_results']]
    acc_b = [r['accuracy'] for r in results_b['fold_results']]
    
    f1_a = [r['f1_macro'] for r in results_a['fold_results']]
    f1_b = [r['f1_macro'] for r in results_b['fold_results']]
    
    t_acc, p_acc = stats.ttest_rel(acc_a, acc_b)
    t_f1, p_f1 = stats.ttest_rel(f1_a, f1_b)
    
    def cohens_d_paired(x, y):
        diff = np.array(x) - np.array(y)
        return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    
    d_acc = cohens_d_paired(acc_a, acc_b)
    d_f1 = cohens_d_paired(f1_a, f1_b)
    
    try:
        w_acc, wp_acc = stats.wilcoxon(acc_a, acc_b)
        w_f1, wp_f1 = stats.wilcoxon(f1_a, f1_b)
    except ValueError:
        w_acc, wp_acc = np.nan, 1.0
        w_f1, wp_f1 = np.nan, 1.0
    
    return {
        'methods': [method_name_a, method_name_b],
        'accuracy': {
            'mean_a': float(np.mean(acc_a)),
            'mean_b': float(np.mean(acc_b)),
            'difference': float(np.mean(acc_a) - np.mean(acc_b)),
            't_statistic': float(t_acc),
            'p_value': float(p_acc),
            'significant_at_05': p_acc < 0.05,
            'significant_at_01': p_acc < 0.01,
            'cohens_d': float(d_acc),
            'wilcoxon_p': float(wp_acc) if not np.isnan(wp_acc) else None
        },
        'f1_macro': {
            'mean_a': float(np.mean(f1_a)),
            'mean_b': float(np.mean(f1_b)),
            'difference': float(np.mean(f1_a) - np.mean(f1_b)),
            't_statistic': float(t_f1),
            'p_value': float(p_f1),
            'significant_at_05': p_f1 < 0.05,
            'significant_at_01': p_f1 < 0.01,
            'cohens_d': float(d_f1),
            'wilcoxon_p': float(wp_f1) if not np.isnan(wp_f1) else None
        }
    }


# =============================================================================
# D) BASELINE COMPARISONS
# =============================================================================

def run_all_baseline_comparisons(
    data: Dict,
    seeds: List[int] = None,
    episodes: int = 300,
    verbose: bool = True
) -> Dict:
    """
    Run multi-seed evaluation and compare DQN to all baselines with statistical tests.
    
    Args:
        data: Generated CBC data
        seeds: Random seeds (default: [42, 123, 456, 789, 1011])
        episodes: Training episodes per seed
        verbose: Print progress
        
    Returns:
        Summary with all results and statistical comparisons
    """
    from src.rl.baselines import (
        RandomBaseline, MajorityBaseline, RuleBasedBaseline,
        LogisticBaseline, RandomForestBaseline
    )
    from src.rl.environment import PathwayEnvironment
    from src.rl.agent import PathwayRecommendationAgent
    
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  COMPREHENSIVE BASELINE COMPARISON")
        print(f"  {len(seeds)} seeds, {episodes} episodes")
        print(f"{'='*60}")
    
    # Prepare data
    X, y = [], []
    env = PathwayEnvironment(data['assessments'], data['competencies'])
    
    # Use actual pathway labels if available (real CSV data), else computed
    actual_labels = data.get('pathway_labels', {})
    
    for student_id in env.students:
        state = env.reset(student_id)
        if student_id in actual_labels and actual_labels[student_id]:
            ground_truth = actual_labels[student_id]
        else:
            ground_truth = env.get_ground_truth_pathway(student_id)
        X.append(state)
        y.append(ground_truth)
    
    X = np.array(X)
    y = np.array(y)
    
    unique_classes = set(y)
    n_classes = len(unique_classes)
    if verbose:
        from collections import Counter
        print(f"\n  Class distribution: {dict(Counter(y))}")
        if n_classes < 2:
            print(f"  ⚠ Only {n_classes} class found — sklearn classifiers will be skipped")
    
    all_results = {
        'DQN': [],
        'Random': [],
        'Majority': [],
        'RuleBased': [],
    }
    
    baselines = {
        'Random': RandomBaseline(),
        'Majority': MajorityBaseline(),
        'RuleBased': RuleBasedBaseline(),
    }
    
    # Only add sklearn classifiers if we have 2+ classes
    if n_classes >= 2:
        all_results['Logistic'] = []
        all_results['RandomForest'] = []
        baselines['Logistic'] = LogisticBaseline()
        baselines['RandomForest'] = RandomForestBaseline()
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n  Seed {i+1}/{len(seeds)}: {seed}...")
        
        np.random.seed(seed)
        
        # Train and evaluate DQN
        agent = PathwayRecommendationAgent()
        agent.train(env, episodes=episodes, verbose=False,
                    batch_per_episode=min(128, len(env.students)))
        
        dqn_correct = 0
        for j, student_id in enumerate(env.students):
            state = env.reset(student_id)
            rec = agent.recommend(state)
            if rec['recommended_pathway'] == y[j]:
                dqn_correct += 1
        all_results['DQN'].append(dqn_correct / len(env.students))
        
        # Evaluate baselines
        for name, baseline in baselines.items():
            try:
                baseline.fit(X, y)
                preds = baseline.predict(X)
                acc = np.mean(preds == y)
                all_results[name].append(acc)
            except Exception as e:
                if verbose:
                    print(f"    {name} failed: {e}")
                all_results[name].append(0.0)
        
        if verbose:
            rf_str = f", RF: {all_results['RandomForest'][-1]:.2%}" if 'RandomForest' in all_results else ""
            print(f"    DQN: {all_results['DQN'][-1]:.2%}{rf_str}")
    
    # Statistical comparisons
    comparisons = {}
    for baseline_name in baselines.keys():
        comparisons[baseline_name] = statistical_comparison(
            all_results['DQN'],
            all_results[baseline_name],
            baseline_name,
            verbose=verbose
        )
    
    summary = {
        'dqn_results': {
            'mean': float(np.mean(all_results['DQN'])),
            'std': float(np.std(all_results['DQN'], ddof=1)),
            'raw': all_results['DQN']
        },
        'baseline_results': {
            name: {
                'mean': float(np.mean(results)),
                'std': float(np.std(results, ddof=1)),
                'raw': results
            }
            for name, results in all_results.items() if name != 'DQN'
        },
        'statistical_comparisons': comparisons,
        'seeds': seeds,
        'episodes': episodes
    }
    
    return summary


# =============================================================================
# E) HYPERPARAMETER TUNING
# =============================================================================

def hyperparameter_grid_search(
    data: Dict,
    param_grid: Dict = None,
    n_trials_per_config: int = 3,
    episodes: int = 200,
    verbose: bool = True
) -> Dict:
    """
    Grid search for hyperparameter tuning.
    
    Args:
        data: Generated CBC data
        param_grid: Dict of param_name -> list of values to try
        n_trials_per_config: Runs per configuration for stability
        episodes: Training episodes per trial
        verbose: Print progress
        
    Returns:
        Best config and all results sorted by accuracy
    """
    from src.rl.environment import PathwayEnvironment
    from src.rl.agent import PathwayRecommendationAgent
    from itertools import product
    
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.0005, 0.001, 0.005],
            'hidden_dim': [64, 128, 256],
            'epsilon_decay': [0.99, 0.995, 0.999],
        }
    
    env = PathwayEnvironment(data['assessments'], data['competencies'])
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    configurations = list(product(*param_values))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  HYPERPARAMETER GRID SEARCH")
        print(f"  {len(configurations)} configurations × {n_trials_per_config} trials")
        print(f"{'='*60}")
    
    results = []
    best_accuracy = 0
    best_config = None
    
    for i, config_values in enumerate(configurations):
        config = dict(zip(param_names, config_values))
        
        if verbose:
            print(f"\n  Config {i+1}/{len(configurations)}: "
                  f"lr={config.get('learning_rate')}, "
                  f"hidden={config.get('hidden_dim')}, "
                  f"decay={config.get('epsilon_decay')}")
        
        trial_accuracies = []
        
        for trial in range(n_trials_per_config):
            np.random.seed(42 + trial)
            
            agent = PathwayRecommendationAgent(
                learning_rate=config.get('learning_rate', 0.001),
                hidden_dim=config.get('hidden_dim', 128),
                epsilon_decay=config.get('epsilon_decay', 0.995),
                gamma=config.get('gamma', 0.95)
            )
            
            agent.train(env, episodes=episodes, verbose=False,
                        batch_per_episode=min(128, len(env.students)))
            
            correct = 0
            for student_id in env.students:
                state = env.reset(student_id)
                rec = agent.recommend(state)
                if rec['recommended_pathway'] == env.get_ground_truth_pathway(student_id):
                    correct += 1
            
            accuracy = correct / len(env.students)
            trial_accuracies.append(accuracy)
        
        mean_acc = np.mean(trial_accuracies)
        std_acc = np.std(trial_accuracies, ddof=1) if len(trial_accuracies) > 1 else 0
        
        results.append({
            'config': config,
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'trial_accuracies': trial_accuracies
        })
        
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_config = config
        
        if verbose:
            print(f"    Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    
    results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
    
    summary = {
        'best_config': best_config,
        'best_accuracy': float(best_accuracy),
        'all_results': results,
        'param_grid': param_grid,
        'n_trials': n_trials_per_config
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  BEST CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Accuracy: {best_accuracy:.2%}")
        for param, value in best_config.items():
            print(f"  {param}: {value}")
        print(f"\n  Top 5 Configurations:")
        for i, res in enumerate(results[:5]):
            print(f"    {i+1}. {res['mean_accuracy']:.2%} ± {res['std_accuracy']:.2%} "
                  f"- {res['config']}")
    
    return summary


# =============================================================================
# F) REPORTING
# =============================================================================

def generate_evaluation_report(cv_results: Dict,
                               bootstrap_results: Optional[Dict] = None,
                               learning_curve: Optional[Dict] = None,
                               comparison_results: Optional[List[Dict]] = None) -> str:
    """
    Generate a comprehensive evaluation report in markdown format.
    
    Args:
        cv_results: Cross-validation results
        bootstrap_results: Bootstrap confidence intervals
        learning_curve: Learning curve analysis
        comparison_results: Method comparison results
        
    Returns:
        Markdown-formatted report string
    """
    report = []
    report.append("# Comprehensive Evaluation Report\n")
    report.append("## DQN Pathway Recommendation Model\n")
    
    # Cross-validation
    report.append("### Cross-Validation Results\n")
    report.append(f"- **Number of Folds:** {cv_results['n_folds']}\n")
    report.append(f"- **Accuracy:** {cv_results['accuracy']['mean']:.1%} "
                  f"± {cv_results['accuracy']['std']:.1%}\n")
    report.append(f"- **95% CI:** [{cv_results['accuracy']['ci_95_lower']:.1%}, "
                  f"{cv_results['accuracy']['ci_95_upper']:.1%}]\n")
    report.append(f"- **F1 Macro:** {cv_results['f1_macro']['mean']:.1%} "
                  f"± {cv_results['f1_macro']['std']:.1%}\n")
    report.append(f"- **95% CI:** [{cv_results['f1_macro']['ci_95_lower']:.1%}, "
                  f"{cv_results['f1_macro']['ci_95_upper']:.1%}]\n\n")
    
    # Per-pathway table
    report.append("### Per-Pathway Performance\n\n")
    report.append("| Pathway | Precision | Recall | F1 Score | Support |\n")
    report.append("|---------|-----------|--------|----------|--------|\n")
    for pathway, metrics in cv_results['per_pathway'].items():
        report.append(f"| {PATHWAYS[pathway]['name']} | {metrics['precision']:.1%} | "
                      f"{metrics['recall']:.1%} | {metrics['f1']:.1%} | {metrics['support']} |\n")
    report.append("\n")
    
    # Bootstrap
    if bootstrap_results:
        report.append("### Bootstrap Confidence Intervals\n")
        report.append(f"- **Bootstrap Samples:** {bootstrap_results['n_bootstrap']}\n")
        report.append(f"- **Confidence Level:** "
                      f"{bootstrap_results['confidence_level']*100:.0f}%\n")
        report.append(f"- **Accuracy CI:** [{bootstrap_results['accuracy']['ci_lower']:.1%}, "
                      f"{bootstrap_results['accuracy']['ci_upper']:.1%}]\n")
        report.append(f"- **F1 Macro CI:** [{bootstrap_results['f1_macro']['ci_lower']:.1%}, "
                      f"{bootstrap_results['f1_macro']['ci_upper']:.1%}]\n\n")
    
    # Learning curve
    if learning_curve:
        report.append("### Learning Curve Analysis\n\n")
        report.append("| Train Size | Accuracy | Std |\n")
        report.append("|------------|----------|-----|\n")
        for curve in learning_curve['curves']:
            report.append(f"| {curve['train_size']*100:.0f}% | "
                          f"{curve['accuracy_mean']:.1%} | "
                          f"±{curve['accuracy_std']:.1%} |\n")
        report.append("\n")
    
    # Method comparisons
    if comparison_results:
        report.append("### Statistical Comparisons\n\n")
        for comp in comparison_results:
            report.append(f"#### {comp['methods'][0]} vs {comp['methods'][1]}\n")
            report.append(f"- **Accuracy Difference:** "
                          f"{comp['accuracy']['difference']:.1%}\n")
            report.append(f"- **p-value:** {comp['accuracy']['p_value']:.4f}")
            if comp['accuracy']['significant_at_05']:
                report.append(" **(significant at α=0.05)**")
            report.append("\n")
            report.append(f"- **Cohen's d:** {comp['accuracy']['cohens_d']:.3f}\n\n")
    
    return "".join(report)


# =============================================================================
# F.2) FULL EVALUATION PIPELINE
# =============================================================================

def run_full_evaluation(data: Dict,
                        n_folds: int = 5,
                        episodes: int = 300,
                        n_bootstrap: int = 1000,
                        run_learning_curve: bool = True,
                        verbose: bool = True) -> Dict:
    """
    Run complete evaluation pipeline: CV + bootstrap + learning curve + report.
    
    Args:
        data: Generated CBC data
        n_folds: Number of CV folds
        episodes: Training episodes
        n_bootstrap: Bootstrap samples
        run_learning_curve: Whether to run learning curve analysis
        verbose: Print progress
        
    Returns:
        Complete evaluation results with markdown report
    """
    from src.rl.agent import PathwayRecommendationAgent
    from src.rl.trainer import train_pathway_model
    from src.rl.environment import PathwayEnvironment
    
    def agent_factory():
        return PathwayRecommendationAgent()
    
    def train_fn(agent, train_data, episodes=300, verbose=False):
        env = PathwayEnvironment(train_data['assessments'], train_data['competencies'])
        agent.train(env, episodes=episodes, verbose=verbose)
        return agent
    
    results = {}
    
    # 1. Cross-validation
    results['cross_validation'] = cross_validate_agent(
        data, agent_factory, train_fn,
        n_folds=n_folds, episodes=episodes, verbose=verbose
    )
    
    # 2. Bootstrap CI
    if verbose:
        print(f"\n  Computing bootstrap confidence intervals...")
    
    agent = agent_factory()
    agent = train_fn(agent, data, episodes=episodes, verbose=False)
    env = PathwayEnvironment(data['assessments'], data['competencies'])
    
    y_true, y_pred = [], []
    for student_id in env.students:
        state = env.reset(student_id)
        gt = env.get_ground_truth_pathway(student_id)
        rec = agent.recommend(state)
        y_true.append(gt)
        y_pred.append(rec['recommended_pathway'])
    
    results['bootstrap'] = bootstrap_evaluation(y_true, y_pred, n_bootstrap=n_bootstrap)
    
    # 3. Learning curve
    if run_learning_curve:
        results['learning_curve'] = learning_curve_analysis(
            data, agent_factory, train_fn,
            episodes=episodes, verbose=verbose
        )
    
    # 4. Generate report
    results['report'] = generate_evaluation_report(
        results['cross_validation'],
        results['bootstrap'],
        results.get('learning_curve')
    )
    
    return results


# =============================================================================
# UTILITY: Save/Export
# =============================================================================

def save_evaluation_results(results: Dict, filepath: str):
    """Save evaluation results to JSON with numpy type conversion."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {filepath}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Comprehensive Evaluation Module")
    print("=" * 60)
    
    from src.data.cbc_data_generator import generate_dashboard_data
    
    print("\n1. Generating data...")
    data = generate_dashboard_data(n_students=500, seed=42)
    
    print("\n2. Running full evaluation...")
    results = run_full_evaluation(
        data,
        n_folds=5,
        episodes=200,
        n_bootstrap=500,
        run_learning_curve=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n  Final Accuracy: {results['cross_validation']['accuracy']['mean']:.1%}")
    print(f"  95% CI: [{results['cross_validation']['accuracy']['ci_95_lower']:.1%}, "
          f"{results['cross_validation']['accuracy']['ci_95_upper']:.1%}]")
    
    import tempfile
    report_path = Path(tempfile.gettempdir()) / 'evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write(results['report'])
    print(f"\n  Report saved to {report_path}")
