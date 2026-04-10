"""
Reinforcement Learning configuration and hyperparameters.

Key hyperparameters explained:
- learning_rate: How fast the network learns. Higher = faster but unstable, Lower = slower but stable
- gamma: Discount factor for future rewards. 0.99 = long-term focus, 0.9 = short-term focus
- epsilon: Exploration rate. Starts high (explore) → decays low (exploit)
- batch_size: Experiences per gradient update. Larger = stable, Smaller = fast
- tau: Soft update rate for target network
"""

# =====================================================
# RL Model Configuration
# =====================================================
RL_CONFIG = {
    # State/Action dimensions
    'state_dim': 78,          # Full trajectory state (grades G4-G9, growth, CW, cosine, gaps, pref, feedback)
    'action_dim': 9,          # 9 improvement recommendations (from dqn_coaching.ACTIONS)

    # Network architecture
    # Increased from 128: 78-dim input with 9-action dueling streams needs more capacity
    # to learn the fine-grained action distinctions without bottlenecking.
    'hidden_dim': 256,

    # Learning hyperparameters
    'learning_rate': 5e-4,    # Adam optimizer LR
    'gamma': 0.95,            # Discount factor (appropriate for G4-G9 short horizons)
    'tau': 0.005,             # Soft update rate for target network

    # Exploration schedule
    'epsilon_start': 1.0,     # Initial exploration rate
    'epsilon_end': 0.01,      # Reduced from 0.05 — tighter floor for more decisive exploitation
    'epsilon_decay': 0.997,   # Slightly slower decay: more exploration early, sharper late convergence

    # Experience replay
    'batch_size': 128,        # Increased from 64 — larger batches stabilise gradient estimates
    'memory_size': 20000,     # Replay buffer size

    # Target network
    'target_update': 10,      # Soft update frequency (episodes)

    # Prioritized Experience Replay
    'per_alpha': 0.6,         # Priority exponent
    'per_beta_start': 0.4,    # Initial IS correction
    'per_beta_frames': 5000,  # Increased from 2000 — longer annealing matches more episodes

    # Default training episodes
    # Increased cap: with 256-dim network and 9 actions the model needs more steps to converge
    'episodes': 1000,

    # Gradient updates per episode (was hardcoded to 4)
    # 8 updates × batch 128 = ~4× learning signal vs previous 4 × 64
    'updates_per_episode': 8,

    # Conservative Q-Learning (CQL-lite) regularization
    # DISABLED (0.0) at this data scale: with only hundreds of students and 2-5 transitions
    # each, cql_alpha=0.1 directly fights the Bellman update and suppresses Q-value spread,
    # collapsing the argmax and sharply reducing accuracy.  Re-enable (0.01–0.1) only after
    # validating that the dataset size is sufficient (>5000 transitions) or when deploying
    # to production where OOD overestimation is a safety risk worth the accuracy trade-off.
    'cql_alpha': 0.0,
}

# =====================================================
# Reward Configuration for RL
# =====================================================
REWARD_CONFIG = {
    'pathway_match_excellent': 10.0,   # Perfect match with student strengths
    'pathway_match_good': 5.0,         # Good match
    'pathway_match_moderate': 2.0,     # Moderate match
    'pathway_match_poor': -5.0,        # Poor match
    'consistency_bonus': 3.0,          # Bonus for consistent recommendations
    'improvement_bonus': 2.0,          # Bonus for recommending pathway that encourages growth
    'penalty_mismatch': -10.0          # Penalty for recommending unsuitable pathway
}

# =====================================================
# Consistency Checker Configuration
# =====================================================
CONSISTENCY_CONFIG = {
    'n_checks': 10,           # Number of inference passes to check
    'agreement_threshold': 0.7,  # Minimum agreement ratio
    'temperature': 0.1        # Softmax temperature for consistency
}

# =====================================================
# Training Configuration
# =====================================================
TRAINING_CONFIG = {
    'verbose_interval': 50,   # Print progress every N episodes
    'save_interval': 100,     # Save model every N episodes
    'early_stopping_patience': 50,
    'min_accuracy_threshold': 0.6
}
