"""
Training utilities for the pathway recommendation model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RL_CONFIG
from .agent import PathwayRecommendationAgent
from .environment import PathwayEnvironment
from .dqn_coaching import ACTIONS as _COACHING_ACTIONS


def train_pathway_model(data: Dict[str, pd.DataFrame],
                        episodes: int = None,
                        save_model: bool = True,
                        verbose: bool = True) -> PathwayRecommendationAgent:
    """
    Train the pathway recommendation model.
    
    Uses DQN with all improvements:
    - Double DQN (reduces Q-value overestimation)
    - Dueling Architecture (separate value/advantage)
    - Prioritized Experience Replay
    - Reward Shaping
    
    Args:
        data: Dictionary with 'assessments' and 'competencies' DataFrames
        episodes: Number of training episodes (default: auto based on data size)
        save_model: Whether to save the trained model
        verbose: Print progress
        
    Returns:
        Trained recommendation agent
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  PATHWAY RECOMMENDATION MODEL TRAINING")
        print(f"{'='*60}\n")
        print("  Initializing environment...")

    # Build environment first — extract_transitions may drop students with < 2 grade records,
    # so base episode count on actual env.students, not raw competencies count.
    env = PathwayEnvironment(data['assessments'], data['competencies'], verbose=verbose)
    n_students = len(env.students)

    # Auto-adjust episodes based on actual trainable students.
    # Cap raised to 1000 to match larger network (hidden_dim=256) and the
    # slower epsilon decay (0.997) which needs ~750+ episodes to reach epsilon_end.
    if episodes is None:
        episodes = min(RL_CONFIG.get('episodes', 1000), max(200, n_students // 3))

    if verbose:
        print(f"  Students (with multi-grade data): {n_students}")
        print(f"  Episodes: {episodes}")
        print(f"{'='*60}\n")
    
    if verbose:
        print("  Initializing DQN agent...")
    
    agent = PathwayRecommendationAgent(verbose=verbose)
    
    if verbose:
        print(f"  Starting training...\n")
    
    history = agent.train(
        env,
        episodes=episodes,
        verbose=verbose,
        batch_per_episode=min(128, n_students),
        early_stopping=False,  # Train full episodes
        patience=50,
        min_accuracy=0.95
    )
    
    if save_model:
        agent.save()
    
    # Final evaluation on ALL students (pathway-level match)
    if verbose:
        print("\n  === Final Evaluation (All Students) ===")

    pathway_correct = 0
    action_correct = 0
    total = len(env.students)

    for student_id in env.students:
        # Use latest (most recent grade) state for consistent evaluation.
        # env.reset() picks a random transition, which could be an earlier grade
        # whose suitability may differ from the ground-truth (latest) grade.
        state = env.get_latest_state(student_id)
        rec = agent.recommend(state)
        gt_pathway = env.get_ground_truth_pathway(student_id)
        gt_action = env.get_ground_truth_action(student_id)

        if rec['recommended_pathway'] == gt_pathway:
            pathway_correct += 1
        if rec.get('recommended_action') == _COACHING_ACTIONS[gt_action].name:
            action_correct += 1

    if verbose:
        print(f"  Pathway Accuracy: {pathway_correct/total:.1%} ({pathway_correct}/{total})")
        print(f"  Action Accuracy:  {action_correct/total:.1%} ({action_correct}/{total})")
        print(f"{'='*60}\n")

    return agent


def evaluate_model(agent: PathwayRecommendationAgent,
                   data: Dict[str, pd.DataFrame],
                   verbose: bool = True) -> Dict:
    """
    Evaluate model performance.
    
    Returns:
        Dictionary with evaluation metrics
    """
    env = PathwayEnvironment(data['assessments'], data['competencies'], verbose=verbose)

    correct = 0
    action_correct = 0
    total = len(env.students)
    pathway_stats = {p: {'correct': 0, 'total': 0} for p in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']}

    predictions = []

    for student_id in env.students:
        state = env.get_latest_state(student_id)   # most recent grade; avoids random-transition ambiguity
        rec = agent.recommend(state)
        ground_truth = env.get_ground_truth_pathway(student_id)
        gt_action = env.get_ground_truth_action(student_id)

        pathway_stats[ground_truth]['total'] += 1

        pathway_match = rec['recommended_pathway'] == ground_truth
        action_match = rec.get('recommended_action') == _COACHING_ACTIONS[gt_action].name

        if pathway_match:
            correct += 1
            pathway_stats[ground_truth]['correct'] += 1
        if action_match:
            action_correct += 1

        predictions.append({
            'student_id': student_id,
            'predicted_pathway': rec['recommended_pathway'],
            'predicted_action': rec.get('recommended_action', ''),
            'ground_truth_pathway': ground_truth,
            'pathway_correct': pathway_match,
            'action_correct': action_match,
            'confidence': rec['confidence'],
        })

    results = {
        'overall_accuracy': correct / total,
        'action_accuracy': action_correct / total,
        'pathway_accuracy': {},
        'predictions': predictions,
    }

    for pathway, stats in pathway_stats.items():
        if stats['total'] > 0:
            results['pathway_accuracy'][pathway] = stats['correct'] / stats['total']
    
    if verbose:
        print(f"\nPathway Accuracy: {results['overall_accuracy']:.2%}")
        print(f"Action Accuracy:  {results['action_accuracy']:.2%}")
        print("\nPer-Pathway Accuracy:")
        for pathway, acc in results['pathway_accuracy'].items():
            print(f"  {pathway}: {acc:.2%}")
    
    return results


def cross_validate(data: Dict[str, pd.DataFrame],
                   n_folds: int = 5,
                   episodes: int = 300,
                   verbose: bool = True) -> Dict:
    """
    Perform k-fold cross-validation.
    
    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import KFold
    
    student_ids = data['competencies']['student_id'].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(student_ids)):
        if verbose:
            print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        
        train_students = student_ids[train_idx]
        test_students = student_ids[test_idx]
        
        # Filter data
        train_data = {
            'assessments': data['assessments'][data['assessments']['student_id'].isin(train_students)],
            'competencies': data['competencies'][data['competencies']['student_id'].isin(train_students)]
        }
        
        test_data = {
            'assessments': data['assessments'][data['assessments']['student_id'].isin(test_students)],
            'competencies': data['competencies'][data['competencies']['student_id'].isin(test_students)]
        }
        
        # Train
        agent = train_pathway_model(train_data, episodes=episodes, save_model=False, verbose=False)
        
        # Evaluate
        results = evaluate_model(agent, test_data, verbose=False)
        fold_results.append(results['overall_accuracy'])
        
        if verbose:
            print(f"  Fold {fold + 1} Accuracy: {results['overall_accuracy']:.2%}")
    
    mean_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)
    
    if verbose:
        print(f"\n=== Cross-Validation Summary ===")
        print(f"Mean Accuracy: {mean_accuracy:.2%} (+/- {std_accuracy:.2%})")
    
    return {
        'fold_accuracies': fold_results,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    }
