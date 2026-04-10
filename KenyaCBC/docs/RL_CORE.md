# RL Core — `src/rl/`

## Why Deep Q-Learning for Pathway Recommendation?

Pathway placement is a **sequential, feedback-driven decision problem** — not a one-shot classification task. Three properties make it a natural fit for Deep RL, and specifically DQN:

### 1. Delayed rewards justify RL over classifiers
A recommendation made in Grade 6 is only validated by the student's performance in Grade 7 or 8. Logistic regression and random forests optimise a static label; they cannot model the fact that a good action now produces payoff later. The Bellman equation in DQN explicitly accounts for this: `Q(s,a) = r + γ · max Q(s',a')`, discounting future improvement into the present decision.

### 2. Sequential state justifies the trajectory design
The state vector encodes **three consecutive grades** of subject scores, growth rates, and suitability deltas — not a snapshot. Each episode in the environment is a grade-to-grade transition, so the agent learns *how a student moves through competency space*, not just where they are today. A standard classifier would treat each grade independently.

### 3. Student feedback closes the loop
The reward function includes a feedback signal (`satisfied` / `wants_different`) that re-shapes the agent's policy based on the student's own preferences. This human-in-the-loop interaction is a core requirement and cannot be represented in a supervised model without re-labelling the entire dataset every time a student responds.

### Why DQN over policy gradient (PPO/A3C)?
- **Discrete, small action space** (9 improvement actions) — DQN is data-efficient here; policy gradients are better suited to continuous or very high-dimensional action spaces.
- **Off-policy replay** — Prioritised Experience Replay allows the agent to revisit rare but important transitions (e.g. students who later switch pathways), which is critical given the skewed CBC dataset distribution.
- **Stable convergence** — Double DQN + dueling architecture reduce overestimation variance, making training reproducible across seeds.

### Why not a simpler rule-based or logistic system?
Baselines (random, majority, rule-based, logistic regression, contextual bandit) are included for comparison. The rule-based system achieves acceptable accuracy on well-represented pathways but cannot adapt to student feedback, does not model grade trajectories, and has no mechanism to improve from new data. DQN subsumes the rule-based approach as a special case of the learned policy while adding temporal and interactive dimensions.

---

## Agent (`agent.py`)

Dueling Double DQN with Prioritised Experience Replay.

### Architecture

```
State (78 dims) → FC₁ (128) → BN → ReLU → FC₂ (128) → BN → ReLU
                                                            ├→ Value stream  → V(s)
                                                            └→ Advantage stream → A(s,a)
                 Q(s,a) = V(s) + A(s,a) - mean(A)
```

### Key Features

- **Double DQN**: Online network selects action, target network evaluates → reduces Q-value overestimation
- **Dueling Architecture**: Separate V(s) and A(s,a) streams → better value estimation
- **Prioritised Experience Replay (PER)**: Sum-tree sampling, importance sampling correction
- **Reward Shaping**: Dense signal via pathway alignment + partial credit
- **Dual Backend**: PyTorch (preferred) with NumPy fallback
- **XAI**: Perturbation-based feature importance, counterfactual analysis, sensitivity

### Hyperparameters (from `config/rl_config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| hidden_dim | 128 | Hidden layer size |
| learning_rate | 0.001 | Adam optimizer |
| gamma | 0.99 | Discount factor |
| tau | 0.01 | Soft target update rate |
| epsilon_start | 1.0 | Initial exploration |
| epsilon_decay | 0.995 | Per-episode decay |
| batch_size | 64 | PER batch size |
| memory_size | 10000 | Replay buffer capacity |

## Environment (`environment.py`)

Grade-to-grade episodic MDP — one episode = one student transition between grades.

- **State**: 78-dimensional trajectory vector (normalised 0-1), encoding:
  - Current / previous / oldest grade subject scores (9 subjects × 3 grades = 27)
  - Grade-to-grade growth rates (9 × 2 = 18)
  - Pathway suitability at oldest / middle / current grade + delta (3 × 4 = 12)
  - Per-subject gap to recommended pathway ideal (9)
  - Student preference one-hot (3)
  - Strength within recommended pathway — top-3 core subjects (3)
  - Recommendation confidence — softmax of suitability (3)
  - Student feedback signal: satisfied=+1, wants_different=−1 (1)
  - Padding (2)
- **Actions**: 9 improvement recommendations (see `dqn_coaching.ACTIONS`)
  - 0–4: Strengthen specific subjects toward STEM / SS / Arts
  - 5: Maintain and deepen current pathway strengths
  - 6–7: Explore alternative pathways
  - 8: General improvement across core subjects
- **Reward**: Δsuitability + Δstrength + preference_alignment + feedback_agreement
- **Done**: True after each grade transition (episodic)

## Trainer (`trainer.py`)

- `train_pathway_model(data, episodes)` — full training pipeline
- `evaluate_model(agent, data)` — greedy evaluation with per-pathway breakdown
- `cross_validate(data, n_folds)` — k-fold cross-validation

## Baselines (`baselines.py`)

For thesis comparison:
1. Random (33% expected)
2. Majority class
3. Rule-based (competency thresholds)
4. Logistic Regression
5. Contextual Bandit

## Evaluation (`evaluation.py`, `evaluation_extended.py`)

- K-fold cross-validation with bootstrap CIs
- Multi-seed evaluation (mean ± std across N seeds)
- Hyperparameter grid search
- Statistical significance tests (paired t-test, Wilcoxon)
- Effect size (Cohen's d)

## HITL (`hitl.py`)

See [HITL.md](HITL.md) for the formal workflow.

## Known Open Items

### Teacher feedback in reward function (deferred)
The supervisor specification includes teacher feedback as a reward signal component:
> *"(c) reward space — next state improvement in next class or term, teacher feedback — to be done later"*

The HITL system (`hitl.py`) captures teacher overrides and approval/rejection decisions. Wiring these into the reward function is a planned extension:
- Teacher approves a recommendation → positive reward signal on the transition that led to it
- Teacher overrides a recommendation → negative reward signal, triggering re-training on that episode

This is **not yet implemented** and is explicitly scoped out of the current thesis version per supervisor instruction. The reward currently uses: Δsuitability + Δstrength + preference_alignment + student_feedback_agreement.
