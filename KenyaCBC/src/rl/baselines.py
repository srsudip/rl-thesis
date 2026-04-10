"""
Baseline Methods for Pathway Recommendation.

These baselines are essential for thesis evaluation to demonstrate
that the DQN approach provides meaningful improvement.

Baselines included:
1. RandomBaseline - Lower bound (33% expected accuracy)
2. MajorityBaseline - Always predict most common class
3. RuleBasedBaseline - Hand-crafted rules using competency thresholds
4. LogisticBaseline - Sklearn LogisticRegression
5. ContextualBanditBaseline - Epsilon-greedy without temporal modeling

Usage:
    from src.rl.baselines import compare_all_methods
    results = compare_all_methods(data, episodes=300)
"""

import numpy as np
from typing import Dict, List, Any
from collections import Counter

# For sklearn baselines
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PATHWAYS, COMPETENCIES, get_pathway_from_index, get_pathway_index
from config.competencies import IDEAL_PATHWAY_PROFILES


# =============================================================================
# BASELINE CLASSES
# =============================================================================

class RandomBaseline:
    """
    Random pathway selection.
    
    Expected accuracy: ~33.3% (1/3 for 3 classes)
    This is the lower bound - any useful method should beat this.
    """
    
    def __init__(self, seed: int = None):
        self.name = "Random"
        self.rng = np.random.RandomState(seed)
        self.pathways = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
    
    def fit(self, X, y):
        """No fitting needed for random baseline."""
        pass
    
    def recommend(self, state: np.ndarray) -> Dict:
        pathway = self.rng.choice(self.pathways)
        return {
            'recommended_pathway': pathway,
            'confidence': 1.0 / len(self.pathways),
            'method': 'Random'
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for array of states."""
        return np.array([self.recommend(x)['recommended_pathway'] for x in X])


class MajorityBaseline:
    """
    Always predict the most common pathway in training data.
    
    This baseline tests whether the model learns anything beyond
    the class distribution.
    """
    
    def __init__(self):
        self.name = "Majority Class"
        self.majority_class = None
        self.class_distribution = None
    
    def fit(self, X, y):
        """Learn the majority class from training data."""
        counter = Counter(y)
        self.majority_class = counter.most_common(1)[0][0]
        total = sum(counter.values())
        self.class_distribution = {k: v/total for k, v in counter.items()}
        print(f"  Majority class: {self.majority_class} ({self.class_distribution[self.majority_class]:.1%})")
    
    def recommend(self, state: np.ndarray) -> Dict:
        return {
            'recommended_pathway': self.majority_class,
            'confidence': self.class_distribution.get(self.majority_class, 0.33),
            'method': 'Majority Class'
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.majority_class] * len(X))


class RuleBasedBaseline:
    """
    Hand-crafted rules based on competency profiles.
    
    Rules:
    - If Mathematical & Scientific are top 2 → STEM
    - If Communication & Citizenship are top 2 → Social Sciences
    - If Cultural & Self-Efficacy are top 2 → Arts & Sports
    - Otherwise, use alignment scores
    
    This baseline tests whether simple domain knowledge is sufficient.
    """
    
    def __init__(self):
        self.name = "Rule-Based"
        self.competency_names = list(COMPETENCIES.keys())
        self.ideal_profiles = IDEAL_PATHWAY_PROFILES
    
    def fit(self, X, y):
        """No fitting needed for rule-based."""
        pass
    
    def recommend(self, state: np.ndarray) -> Dict:
        # Use cosine similarities encoded in state (indices 66-68: STEM, SS, Arts)
        # and cluster weights (indices 63-65) for rule-based decision.
        # Falls back to pure cosine if state is shorter (e.g. unit tests).
        if len(state) >= 69:
            cos_sims = state[66:69]   # [STEM, SS, Arts]
            cw = state[63:66]         # [STEM CW, SS CW, Arts CW]
            # Weighted combination: 60% cosine + 40% CW
            scores = 0.6 * cos_sims + 0.4 * (cw / (cw.sum() + 1e-8))
        else:
            scores = state[:3]

        pw_names = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
        best_idx = int(np.argmax(scores))
        pathway = pw_names[best_idx]
        confidence = float(scores[best_idx] / (scores.sum() + 1e-8))

        return {
            'recommended_pathway': pathway,
            'confidence': confidence,
            'method': 'Rule-Based',
        }

    def _best_alignment(self, state: np.ndarray) -> str:
        """Find pathway with best cosine similarity (read from state)."""
        if len(state) >= 69:
            pw_names = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
            return pw_names[int(np.argmax(state[66:69]))]
        return 'STEM'

    def _compute_confidence(self, state: np.ndarray, pathway: str) -> float:
        if len(state) >= 69:
            pw_idx = {'STEM': 0, 'SOCIAL_SCIENCES': 1, 'ARTS_SPORTS': 2}
            return float(state[66 + pw_idx.get(pathway, 0)])
        return 0.33
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.recommend(x)['recommended_pathway'] for x in X])


class LogisticBaseline:
    """
    Logistic Regression classifier.
    
    This is a strong baseline that learns linear decision boundaries.
    If DQN can't significantly beat this, the problem may not need deep RL.
    """
    
    def __init__(self):
        self.name = "Logistic Regression"
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for LogisticBaseline")
        # Note: multi_class parameter removed in scikit-learn 1.5+
        # multinomial is now the default for lbfgs solver
        self.model = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.fitted = False
    
    def fit(self, X, y):
        """Fit logistic regression on training data."""
        self.model.fit(X, y)
        self.fitted = True
        print(f"  Logistic Regression fitted on {len(X)} samples")
    
    def recommend(self, state: np.ndarray) -> Dict:
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        pathway = self.model.predict(state)[0]
        proba = self.model.predict_proba(state)[0]
        confidence = float(max(proba))
        
        return {
            'recommended_pathway': pathway,
            'confidence': confidence,
            'method': 'Logistic Regression',
            'probabilities': {
                c: float(p) for c, p in zip(self.model.classes_, proba)
            }
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestBaseline:
    """
    Random Forest classifier.
    
    Non-linear baseline that can capture feature interactions.
    """
    
    def __init__(self, n_estimators: int = 100):
        self.name = "Random Forest"
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for RandomForestBaseline")
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.fitted = False
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.fitted = True
        print(f"  Random Forest fitted on {len(X)} samples")
    
    def recommend(self, state: np.ndarray) -> Dict:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        pathway = self.model.predict(state)[0]
        proba = self.model.predict_proba(state)[0]
        
        return {
            'recommended_pathway': pathway,
            'confidence': float(max(proba)),
            'method': 'Random Forest'
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores using 78-dim feature names."""
        from src.rl.dqn_coaching import _get_feature_names
        feature_names = _get_feature_names()
        imps = self.model.feature_importances_
        # Pad or truncate names to match actual feature count
        names = (feature_names + [f'feat_{i}' for i in range(len(imps))])[:len(imps)]
        return {names[i]: float(imp) for i, imp in enumerate(imps)}


class ContextualBanditBaseline:
    """
    Epsilon-greedy contextual bandit.
    
    Learns action values for each context without temporal modeling.
    This tests whether the sequential RL aspect adds value.
    """
    
    def __init__(self, epsilon: float = 0.1, learning_rate: float = 0.01):
        self.name = "Contextual Bandit"
        self.epsilon = epsilon
        self.lr = learning_rate
        self.pathways = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
        
        # Simple linear model for Q-values: Q(s,a) = w_a · s
        # Weights are initialised lazily in fit() once state_dim is known
        self.weights = {p: None for p in self.pathways}
        self.counts = {p: 0 for p in self.pathways}
        self._state_dim: int = None
    
    def fit(self, X, y, n_iterations: int = 100):
        """Train bandit on labeled data."""
        # Lazily initialise weights to match actual state_dim (supports 78-dim state)
        self._state_dim = X.shape[1]
        self.weights = {p: np.zeros(self._state_dim) for p in self.pathways}

        # Normalize input
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        for iteration in range(n_iterations):
            indices = np.random.permutation(len(X_norm))
            for i in indices:
                state = X_norm[i]
                true_pathway = y[i]
                
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    action = np.random.choice(self.pathways)
                else:
                    q_values = {p: np.dot(self.weights[p], state) for p in self.pathways}
                    action = max(q_values, key=q_values.get)
                
                # Reward: +1 if correct, 0 otherwise
                reward = 1.0 if action == true_pathway else 0.0
                
                # Update weights for chosen action with gradient clipping
                q_pred = np.dot(self.weights[action], state)
                gradient = (reward - q_pred) * state
                gradient = np.clip(gradient, -1.0, 1.0)  # Clip gradients
                self.weights[action] += self.lr * gradient
                self.counts[action] += 1
        
        print(f"  Contextual Bandit trained for {n_iterations} iterations")
    
    def recommend(self, state: np.ndarray) -> Dict:
        # Normalize state
        state_norm = state / (np.linalg.norm(state) + 1e-8)
        
        q_values = {p: np.dot(self.weights[p], state_norm) for p in self.pathways}
        pathway = max(q_values, key=q_values.get)
        
        # Softmax for confidence
        q_array = np.array(list(q_values.values()))
        q_array = np.clip(q_array, -10, 10)  # Prevent overflow
        exp_q = np.exp(q_array - np.max(q_array))
        probs = exp_q / exp_q.sum()
        
        return {
            'recommended_pathway': pathway,
            'confidence': float(max(probs)),
            'method': 'Contextual Bandit',
            'q_values': q_values
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.recommend(x)['recommended_pathway'] for x in X])


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def prepare_data_for_baselines(data: Dict, normalize: bool = True) -> tuple:
    """
    Convert CBC data format to X, y arrays for sklearn-style baselines
    (Random, Majority, Rule-Based, Logistic, RF, Bandit).

    NOTE: Returns 7-dim grade-9 competency vectors — suitable for sklearn
    baselines only.  The DQN agent requires the full 78-dim trajectory state
    produced by PathwayEnvironment; use the environment directly for DQN eval.

    Args:
        data: CBC data dictionary
        normalize: If True, normalize competencies to 0-1 range

    Returns:
        X: np.ndarray of shape (n_students, 7) - competency scores
        y: np.ndarray of shape (n_students,) - pathway labels
    """
    competencies_df = data['competencies']
    pathways_df = data['pathways']
    
    # Get Grade 9 competencies (final state)
    grade9 = competencies_df[competencies_df['grade'] == 9].copy()
    
    # Get competency columns (with _score suffix)
    comp_cols = [f"{c}_score" for c in COMPETENCIES.keys()]
    
    X_list = []
    y_list = []
    
    for student_id in grade9['student_id'].unique():
        student_comps = grade9[grade9['student_id'] == student_id][comp_cols].values
        if len(student_comps) > 0:
            X_list.append(student_comps[0])
            pathway = pathways_df[pathways_df['student_id'] == student_id]['recommended_pathway'].values[0]
            y_list.append(pathway)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Normalize to 0-1 range (same as DQN environment)
    if normalize:
        X = X / 100.0
    
    return X, y


def evaluate_baseline(baseline, X_test, y_test) -> Dict:
    """Evaluate a baseline model."""
    y_pred = baseline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    labels = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
    
    results = {
        'name': baseline.name,
        'accuracy': accuracy,
        'per_pathway': {},
        'predictions': y_pred
    }
    
    if SKLEARN_AVAILABLE:
        f1_per_class = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
        for i, pathway in enumerate(labels):
            mask = y_test == pathway
            if mask.sum() > 0:
                pathway_acc = (y_pred[mask] == y_test[mask]).mean()
                results['per_pathway'][pathway] = {
                    'accuracy': float(pathway_acc),
                    'f1': float(f1_per_class[i]),
                    'count': int(mask.sum())
                }
        
        results['macro_f1'] = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        results['weighted_f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    
    return results


def compare_all_methods(data: Dict, 
                        dqn_agent=None,
                        test_size: float = 0.2,
                        dqn_episodes: int = 300,
                        verbose: bool = True) -> Dict:
    """
    Compare all baseline methods against DQN.
    
    Args:
        data: CBC data dictionary
        dqn_agent: Trained DQN agent (optional, will train if None)
        test_size: Fraction for test set
        dqn_episodes: Episodes to train DQN if not provided
        verbose: Print progress
    
    Returns:
        Dictionary with comparison results
    """
    from sklearn.model_selection import train_test_split
    
    if verbose:
        print("\n" + "="*60)
        print("  BASELINE COMPARISON")
        print("="*60)
    
    # Prepare data
    X, y = prepare_data_for_baselines(data)
    
    n_classes = len(set(y))
    
    # Use stratify only if we have 2+ classes with enough samples
    stratify_arg = y if n_classes >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_arg
        )
    except ValueError:
        # Stratify can fail if a class has too few samples
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    n_train_classes = len(set(y_train))
    
    if verbose:
        print(f"\n  Data: {len(X)} total, {len(X_train)} train, {len(X_test)} test")
        print(f"  Class distribution: {Counter(y)}")
        if n_train_classes < 2:
            print(f"  ⚠ Only {n_train_classes} class in train set — sklearn classifiers will be skipped")
    
    results = {}
    
    # 1. Random Baseline
    if verbose:
        print("\n  Training Random Baseline...")
    random_bl = RandomBaseline(seed=42)
    random_bl.fit(X_train, y_train)
    results['Random'] = evaluate_baseline(random_bl, X_test, y_test)
    
    # 2. Majority Baseline
    if verbose:
        print("\n  Training Majority Baseline...")
    majority_bl = MajorityBaseline()
    majority_bl.fit(X_train, y_train)
    results['Majority'] = evaluate_baseline(majority_bl, X_test, y_test)
    
    # 3. Rule-Based Baseline
    if verbose:
        print("\n  Training Rule-Based Baseline...")
    rule_bl = RuleBasedBaseline()
    rule_bl.fit(X_train, y_train)
    results['Rule-Based'] = evaluate_baseline(rule_bl, X_test, y_test)
    
    # 4. Logistic Regression (needs 2+ classes)
    if SKLEARN_AVAILABLE and n_train_classes >= 2:
        if verbose:
            print("\n  Training Logistic Regression...")
        try:
            logistic_bl = LogisticBaseline()
            logistic_bl.fit(X_train, y_train)
            results['Logistic'] = evaluate_baseline(logistic_bl, X_test, y_test)
        except Exception as e:
            if verbose:
                print(f"    Logistic failed: {e}")
    
    # 5. Random Forest (needs 2+ classes)
    if SKLEARN_AVAILABLE and n_train_classes >= 2:
        if verbose:
            print("\n  Training Random Forest...")
        try:
            rf_bl = RandomForestBaseline()
            rf_bl.fit(X_train, y_train)
            results['Random Forest'] = evaluate_baseline(rf_bl, X_test, y_test)
            results['Random Forest']['feature_importance'] = rf_bl.feature_importance()
        except Exception as e:
            if verbose:
                print(f"    Random Forest failed: {e}")
    
    # 6. Contextual Bandit
    if verbose:
        print("\n  Training Contextual Bandit...")
    bandit_bl = ContextualBanditBaseline(epsilon=0.1, learning_rate=0.01)  # Lower LR to prevent overflow
    bandit_bl.fit(X_train, y_train, n_iterations=200)
    results['Bandit'] = evaluate_baseline(bandit_bl, X_test, y_test)
    
    # 7. DQN Agent - Train on same train split for fair comparison
    if verbose:
        print("\n  Training DQN Agent on train split...")
    
    # Create filtered data for training
    from src.rl.environment import PathwayEnvironment
    from src.rl.agent import PathwayRecommendationAgent
    
    # Build train student IDs
    competencies_df = data['competencies']
    pathways_df = data['pathways']
    grade9 = competencies_df[competencies_df['grade'] == 9]
    
    all_student_ids = list(grade9['student_id'].unique())
    train_indices, test_indices = train_test_split(
        range(len(all_student_ids)), test_size=test_size, random_state=42
    )
    train_student_ids = [all_student_ids[i] for i in train_indices]
    test_student_ids = [all_student_ids[i] for i in test_indices]
    
    # Filter data for training
    train_data = {
        'assessments': data['assessments'][data['assessments']['student_id'].isin(train_student_ids)],
        'competencies': data['competencies'][data['competencies']['student_id'].isin(train_student_ids)],
        'pathways': data['pathways'][data['pathways']['student_id'].isin(train_student_ids)]
    }
    
    # Train DQN on train data only
    env = PathwayEnvironment(train_data['assessments'], train_data['competencies'])
    dqn_agent = PathwayRecommendationAgent()
    
    history = dqn_agent.train(
        env,
        episodes=dqn_episodes,
        verbose=verbose,
        batch_per_episode=min(64, len(env.students)),
        early_stopping=False,  # Train full episodes
        patience=50,
        min_accuracy=0.95
    )
    
    # Evaluate DQN on test set using proper 78-dim states from the test environment.
    # X_test rows are 7-dim (sklearn baselines only); DQN requires the full
    # trajectory state produced by PathwayEnvironment / extract_transitions.
    if verbose:
        print("\n  Evaluating DQN on test set...")

    test_env = PathwayEnvironment(test_data['assessments'], test_data['competencies'])
    y_pred_dqn = []
    y_true_dqn = []
    for student_id in test_env.students:
        state = test_env.reset(student_id)
        rec = dqn_agent.recommend(state)
        y_pred_dqn.append(rec['recommended_pathway'])
        y_true_dqn.append(test_env.get_ground_truth_pathway(student_id))
    y_pred_dqn = np.array(y_pred_dqn)
    y_true_dqn = np.array(y_true_dqn)

    dqn_results = {
        'name': 'DQN',
        'accuracy': accuracy_score(y_true_dqn, y_pred_dqn),
        'per_pathway': {},
        'predictions': y_pred_dqn
    }

    if SKLEARN_AVAILABLE:
        labels = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
        f1_per_class = f1_score(y_true_dqn, y_pred_dqn, labels=labels, average=None, zero_division=0)
        for i, pathway in enumerate(labels):
            mask = y_true_dqn == pathway
            if mask.sum() > 0:
                pathway_acc = (y_pred_dqn[mask] == y_true_dqn[mask]).mean()
                dqn_results['per_pathway'][pathway] = {
                    'accuracy': float(pathway_acc),
                    'f1': float(f1_per_class[i]),
                    'count': int(mask.sum())
                }
        dqn_results['macro_f1'] = float(f1_score(y_true_dqn, y_pred_dqn, average='macro', zero_division=0))
        dqn_results['weighted_f1'] = float(f1_score(y_true_dqn, y_pred_dqn, average='weighted', zero_division=0))
        dqn_results['confusion_matrix'] = confusion_matrix(y_true_dqn, y_pred_dqn, labels=labels).tolist()
    
    results['DQN'] = dqn_results
    
    # Print comparison table
    if verbose:
        print("\n" + "="*60)
        print("  RESULTS SUMMARY")
        print("="*60)
        print(f"\n  {'Method':<20} {'Accuracy':>10} {'Macro F1':>10}")
        print("  " + "-"*42)
        
        for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy']):
            acc = res['accuracy']
            f1 = res.get('macro_f1', 'N/A')
            f1_str = f"{f1:.1%}" if isinstance(f1, float) else f1
            print(f"  {name:<20} {acc:>10.1%} {f1_str:>10}")
        
        print("="*60)
    
    return results


def print_comparison_table(results: Dict, format: str = 'markdown'):
    """Print results as a formatted table."""
    
    if format == 'markdown':
        print("\n| Method | Accuracy | Macro F1 | STEM F1 | Social F1 | Arts F1 |")
        print("|--------|----------|----------|---------|-----------|---------|")
        
        for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            acc = f"{res['accuracy']:.1%}"
            f1 = f"{res.get('macro_f1', 0):.1%}" if 'macro_f1' in res else "N/A"
            
            stem_f1 = res.get('per_pathway', {}).get('STEM', {}).get('f1', 0)
            social_f1 = res.get('per_pathway', {}).get('SOCIAL_SCIENCES', {}).get('f1', 0)
            arts_f1 = res.get('per_pathway', {}).get('ARTS_SPORTS', {}).get('f1', 0)
            
            print(f"| {name:<14} | {acc:>8} | {f1:>8} | {stem_f1:>7.1%} | {social_f1:>9.1%} | {arts_f1:>7.1%} |")
    
    elif format == 'latex':
        print("\n\\begin{tabular}{lcccc}")
        print("\\hline")
        print("Method & Accuracy & Macro F1 & STEM & Social Sciences \\\\")
        print("\\hline")
        
        for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            acc = f"{res['accuracy']*100:.1f}\\%"
            f1 = f"{res.get('macro_f1', 0)*100:.1f}\\%" if 'macro_f1' in res else "N/A"
            stem = f"{res.get('per_pathway', {}).get('STEM', {}).get('f1', 0)*100:.1f}\\%"
            social = f"{res.get('per_pathway', {}).get('SOCIAL_SCIENCES', {}).get('f1', 0)*100:.1f}\\%"
            
            print(f"{name} & {acc} & {f1} & {stem} & {social} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Test the baselines
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data.cbc_data_generator import generate_dashboard_data
    from src.rl.trainer import train_pathway_model
    
    print("Generating test data...")
    data = generate_dashboard_data(n_students=500, seed=42)
    
    print("Training DQN agent...")
    dqn_agent = train_pathway_model(data, episodes=300, save_model=False, verbose=True)
    
    print("\nComparing all methods...")
    results = compare_all_methods(data, dqn_agent=dqn_agent)
    
    print("\n\nMarkdown Table:")
    print_comparison_table(results, format='markdown')
