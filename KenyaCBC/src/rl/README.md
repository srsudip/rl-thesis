# src/rl/ — Reinforcement Learning Module

Deep Q-Network agent for supplementary pathway recommendations. See [`docs/RL_CORE.md`](../../docs/RL_CORE.md) for the full theoretical justification.

## Architecture

```
State (78 dims) → FC₁(128) → BN → ReLU → FC₂(128) → BN → ReLU
                                                        ├→ Value stream    → V(s)
                                                        └→ Advantage stream → A(s,a)
                          Q(s,a) = V(s) + A(s,a) − mean(A)
```

Key features: Double DQN, Dueling Architecture, Prioritized Experience Replay (PER), XAI (perturbation-based + counterfactuals).

### Why DQN?

Pathway placement is a **sequential, feedback-driven decision problem**. Three properties make DQN the right fit:

1. **Delayed rewards** — A Grade 6 recommendation is only validated by Grade 7–8 performance. The Bellman equation `Q(s,a) = r + γ·max Q(s',a')` captures this; a static classifier cannot.
2. **Sequential state** — The 78-dim state encodes three consecutive grades of scores + growth rates. The agent learns *how a student moves through competency space*, not just where they are today.
3. **Student feedback loop** — The reward signal includes `satisfied`/`wants_different` feedback, closing a human-in-the-loop that cannot be represented in a supervised model without re-labelling the full dataset on every response.

## Recommendation Priority

```
1. HITL Override (teacher-approved)  →  highest priority
2. KNEC Cluster Weights + Core Validation  →  primary recommendation
3. RL Agent Suggestion  →  supplementary (shown as "AI: ...")
```

The RL agent never overrides KNEC cluster weights. It provides a supplementary suggestion that appears below the main recommendation when it disagrees.

## HITL Workflow

```
Student/Counselor submits request → Pending
    → Teacher approves → Override active (pathway + key subjects update)
    → Teacher rejects  → No change
```

Persistent state stored in `data/hitl_state.json`.

## State Vector (78 dimensions)

| Dimensions | Content |
|-----------|---------|
| 0–26 | Subject scores — current / middle / oldest grade (9 subjects × 3 grades) |
| 27–44 | Grade-to-grade growth rates (9 subjects × 2 transitions) |
| 45–56 | Pathway suitability at oldest/middle/current grade + delta (3 pathways × 4 values) |
| 57–65 | Per-subject gap to recommended pathway ideal (9 subjects) |
| 66–68 | Student preference one-hot (STEM / SS / Arts) |
| 69–71 | Pathway strength — top-3 core subjects in recommended pathway |
| 72–74 | Recommendation confidence — softmax of suitability scores |
| 75 | Student feedback signal: satisfied=+1, wants_different=−1 |
| 76–77 | Padding |

## Action Space (9 actions)

| Action | Name | Description |
|--------|------|-------------|
| 0 | strengthen_math_science | Boost MATH + INT_SCI toward STEM |
| 1 | strengthen_social_studies | Boost SOC_STUD + BUS_STUD toward SS |
| 2 | strengthen_creative_arts | Boost CRE_ARTS + SPORTS_PE toward Arts |
| 3 | improve_language_skills | Boost ENG + KIS (language gate) |
| 4 | improve_core_subjects | Boost all pathway core subjects |
| 5 | deepen_current_pathway | Deepen MATH, ENG, INT_SCI, CRE_ARTS, SOC_STUD in current pathway |
| 6 | explore_stem_alternative | Explore STEM if currently non-STEM |
| 7 | explore_arts_alternative | Explore Arts if currently non-Arts |
| 8 | general_improvement | Raise all 9 subjects toward 60% baseline |

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Dueling Double DQN with PER; PyTorch backend with NumPy fallback |
| `environment.py` | Grade-to-grade episodic MDP (one episode = one student transition) |
| `trainer.py` | Training loop, early stopping, k-fold cross-validation |
| `hitl.py` | Human-in-the-Loop manager with JSON persistence |
| `dqn_coaching.py` | 9-action space, reward shaping, XAI coaching explanations |
| `hybrid_recommender.py` | Combines RL, rule-based, and cluster weight recommendations |
| `evaluation.py` | Multi-seed evaluation, hyperparameter grid search, statistical tests |
| `evaluation_extended.py` | Bootstrap CIs, Cohen's d effect size, Wilcoxon signed-rank |
| `baselines.py` | Random, majority-class, rule-based, logistic regression, contextual bandit |
| `consistency.py` | Temporal consistency checks across grades |
| `sequential_agent.py` | Sequential decision-making variant |

## Reward Function

```
reward = Δsuitability + Δstrength + preference_alignment + feedback_agreement
```

- **Δsuitability** — change in composite pathway suitability score
- **Δstrength** — change in strength within recommended pathway (top-3 core subjects)
- **preference_alignment** — bonus when recommended pathway matches stated student preference
- **feedback_agreement** — +1 if student is `satisfied`, −1 if `wants_different`

Teacher feedback as an additional reward signal is explicitly deferred per supervisor instruction (see `docs/RL_CORE.md` § Known Open Items).

## Evaluation API

```python
from src.rl.evaluation import run_all_baseline_comparisons, multi_seed_evaluation

# Compare against baselines (random, majority, rule-based, logistic, bandit)
results = run_all_baseline_comparisons(agent, env)

# Multi-seed reliability (mean ± std)
results = multi_seed_evaluation(n_seeds=5, episodes=200)
```
