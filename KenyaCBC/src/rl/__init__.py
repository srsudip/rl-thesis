"""
Reinforcement Learning module for pathway recommendations.

Includes:
- DQN agent with Double DQN, Dueling, PER, Reward Shaping
- DQN coaching agent (numpy-only, Dr. Mayeku design)
- Training environment
- Baseline methods for comparison
- Comprehensive evaluation (CV, bootstrap, multi-seed, grid search)
- HITL (Human-in-the-Loop) workflow
"""

from .agent import PathwayRecommendationAgent, TORCH_AVAILABLE, ReplayBuffer
from .environment import PathwayEnvironment
from .hitl import HITLManager, HITLRequest, RequestStatus
from .baselines import (
    RandomBaseline, MajorityBaseline, RuleBasedBaseline,
    LogisticBaseline, RandomForestBaseline, ContextualBanditBaseline,
    compare_all_methods, print_comparison_table
)
from .evaluation import (
    cross_validate_agent,
    bootstrap_evaluation,
    learning_curve_analysis,
    run_full_evaluation,
    generate_evaluation_report,
    multi_seed_evaluation,
    quick_multi_seed_eval,
    statistical_comparison,
    compare_methods_statistical,
    interpret_cohens_d,
    run_all_baseline_comparisons,
    hyperparameter_grid_search,
    save_evaluation_results,
)

__all__ = [
    'PathwayRecommendationAgent',
    'PathwayEnvironment',
    'ReplayBuffer',
    'HITLManager',
    'HITLRequest',
    'RequestStatus',
    'TORCH_AVAILABLE',
    # Baselines
    'RandomBaseline',
    'MajorityBaseline',
    'RuleBasedBaseline',
    'LogisticBaseline',
    'RandomForestBaseline',
    'ContextualBanditBaseline',
    'compare_all_methods',
    'print_comparison_table',
    # Evaluation
    'cross_validate_agent',
    'bootstrap_evaluation',
    'learning_curve_analysis',
    'run_full_evaluation',
    'generate_evaluation_report',
    'multi_seed_evaluation',
    'quick_multi_seed_eval',
    'statistical_comparison',
    'compare_methods_statistical',
    'interpret_cohens_d',
    'run_all_baseline_comparisons',
    'hyperparameter_grid_search',
    'save_evaluation_results',
]
