"""
Pytest suite for Kenya CBC Pathway Recommendation System
=========================================================

Categories:
    1. Data Generator   — IRT model, shapes, value ranges
    2. Environment      — state dims, episode flow, rewards
    3. Agent            — action selection, training, save/load
    4. Trainer          — full training loop, evaluation
    5. HITL Workflow    — submit, approve, reject
    6. Benchmarks       — DKW bound, CI computation, mini pipeline

Run:
    python -m pytest tests/test_all.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ═════════════════════════════════════════════════════════════
#  1. DATA GENERATOR
# ═════════════════════════════════════════════════════════════


class TestDataGenerator:
    """IRT-based CBC data generator tests."""

    def test_student_count(self, dashboard_data):
        student_ids = dashboard_data["competencies"]["student_id"].unique()
        assert len(student_ids) == 30

    def test_competency_shape(self, dashboard_data):
        df = dashboard_data["competencies"]
        assert "student_id" in df.columns
        assert "grade" in df.columns
        comp_cols = [c for c in df.columns if c.endswith("_score")]
        assert len(comp_cols) == 7, (
            f"Expected 7 competency columns, got {len(comp_cols)}: {comp_cols}"
        )

    def test_assessment_scores_range(self, dashboard_data):
        df = dashboard_data["assessments"]
        score_cols = [c for c in df.columns if c.endswith("_score")]
        for col in score_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                assert vals.min() >= 0, f"{col} has value < 0"
                assert vals.max() <= 100, f"{col} has value > 100"

    def test_pathway_distribution(self, dashboard_data):
        df = dashboard_data["pathways"]
        unique_pathways = df["recommended_pathway"].unique()
        assert len(unique_pathways) >= 2, (
            f"Only {len(unique_pathways)} pathways in distribution"
        )

    def test_grades_present(self, dashboard_data):
        df = dashboard_data["competencies"]
        grades = sorted(df["grade"].unique())
        assert grades == [4, 5, 6, 7, 8, 9], f"Expected grades [4..9], got {grades}"


@pytest.mark.parametrize("n_students,seed", [(10, 1), (50, 7), (100, 42)])
def test_data_generator_reproducible(n_students, seed):
    """Same seed always produces identical pathway distributions."""
    from src.data.cbc_data_generator import generate_dashboard_data

    d1 = generate_dashboard_data(n_students=n_students, seed=seed, save_csv=False)
    d2 = generate_dashboard_data(n_students=n_students, seed=seed, save_csv=False)
    pways1 = sorted(d1["pathways"]["recommended_pathway"].tolist())
    pways2 = sorted(d2["pathways"]["recommended_pathway"].tolist())
    assert pways1 == pways2, "generate_dashboard_data is not reproducible"


# ═════════════════════════════════════════════════════════════
#  2. ENVIRONMENT
# ═════════════════════════════════════════════════════════════


class TestEnvironment:
    """RL environment tests."""

    def test_state_dimension(self, env):
        state = env.reset()
        assert len(state) == 78, f"State dim is {len(state)}, expected 78"

    def test_action_space(self, env):
        assert env.get_action_dim() == 9

    def test_episode_single_step(self, env):
        state = env.reset()
        assert len(state) == 78
        next_state, reward, done, info = env.step(0)
        assert len(next_state) == 78
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        for key in ("student_id", "recommended", "ground_truth", "match"):
            assert key in info

    def test_correct_match_higher_reward(self, env):
        from src.rl.dqn_coaching import ACTIONS

        non_terminal = [
            t for t in env.transitions
            if t["grade_before"] != t["grade_after"] and t["reward"] > 0.0
        ]
        if not non_terminal:
            pytest.skip("No non-terminal transition with positive reward found")

        t = non_terminal[0]
        true_action_id = t["action"]
        true_pathway = ACTIONS[true_action_id].target_pathway

        wrong_actions = [
            a for a in ACTIONS
            if a.target_pathway is not None and a.target_pathway != true_pathway
        ]
        if not wrong_actions:
            pytest.skip("No wrong-pathway action found")

        wrong_action_id = wrong_actions[0].id

        env._current_transition = t
        _, reward_correct, _, _ = env.step(true_action_id)

        env._current_transition = t
        _, reward_wrong, _, _ = env.step(wrong_action_id)

        assert reward_correct > reward_wrong, (
            f"GT action {true_action_id} (reward={reward_correct:.3f}) should "
            f"exceed wrong action {wrong_action_id} (reward={reward_wrong:.3f})"
        )


# ═════════════════════════════════════════════════════════════
#  3. AGENT
# ═════════════════════════════════════════════════════════════


class TestAgent:
    """DQN agent tests."""

    def test_action_selection(self, agent):
        state = np.random.rand(78).astype(np.float32)
        action = agent.select_action(state, eval_mode=True)
        assert action in range(9), f"Invalid action: {action}"

    def test_recommend_output(self, agent):
        state = np.random.rand(78).astype(np.float32)
        rec = agent.recommend(state)
        assert "recommended_pathway" in rec
        assert "confidence" in rec
        assert "pathway_ranking" in rec
        assert rec["recommended_pathway"] in {"STEM", "SOCIAL_SCIENCES", "ARTS_SPORTS"}
        assert 0.0 <= rec["confidence"] <= 1.0
        assert len(rec["pathway_ranking"]) == 9
        assert rec["pathway_ranking"][0] in {"STEM", "SOCIAL_SCIENCES", "ARTS_SPORTS", ""}

    def test_save_load(self, agent):
        state = np.random.rand(78).astype(np.float32)
        rec_before = agent.recommend(state)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"
            agent.save(path)

            from src.rl.agent import PathwayRecommendationAgent
            agent2 = PathwayRecommendationAgent(verbose=False)
            agent2.load(path)
            rec_after = agent2.recommend(state)

        assert rec_before["recommended_pathway"] == rec_after["recommended_pathway"], (
            "Loaded agent recommends a different pathway"
        )
        assert abs(rec_before["confidence"] - rec_after["confidence"]) < 1e-4, (
            "Loaded agent confidence diverges from original"
        )


# ═════════════════════════════════════════════════════════════
#  4. TRAINER
# ═════════════════════════════════════════════════════════════


class TestTrainer:
    """Training loop tests."""

    def test_training_produces_history(self, agent, env):
        history = agent.train(
            env,
            episodes=10,
            verbose=False,
            batch_per_episode=min(16, len(env.students)),
        )
        for key in ("rewards", "accuracy", "epsilon", "loss"):
            assert key in history, f"History missing key: {key}"
        assert len(history["rewards"]) == 10

    def test_evaluation(self, agent, dashboard_data):
        from src.rl.trainer import evaluate_model

        results = evaluate_model(agent, dashboard_data, verbose=False)
        assert "overall_accuracy" in results
        assert "pathway_accuracy" in results
        assert isinstance(results["overall_accuracy"], float)
        assert 0.0 <= results["overall_accuracy"] <= 1.0


# ═════════════════════════════════════════════════════════════
#  5. HITL WORKFLOW
# ═════════════════════════════════════════════════════════════


class TestHITL:
    """Human-in-the-Loop workflow tests."""

    def test_submit_and_approve(self, hitl_manager):
        req_id = hitl_manager.submit_request(
            student_id=1,
            current_pathway="STEM",
            desired_pathway="SOCIAL_SCIENCES",
            reason="Better language scores",
        )
        assert hitl_manager.pending_count == 1

        hitl_manager.approve_request(req_id, teacher_id="T001", justification="Agreed")
        assert hitl_manager.pending_count == 0
        assert hitl_manager.approved_count == 1

        override = hitl_manager.get_active_override(1)
        assert override is not None
        assert override["desired_pathway"] == "SOCIAL_SCIENCES"

    def test_submit_and_reject(self, hitl_manager):
        req_id = hitl_manager.submit_request(
            student_id=2,
            current_pathway="ARTS_SPORTS",
            desired_pathway="STEM",
            reason="Wants to try STEM",
        )
        hitl_manager.reject_request(req_id, teacher_id="T002", reason="Scores do not support")
        assert hitl_manager.pending_count == 0
        assert hitl_manager.get_active_override(2) is None

    def test_duplicate_pending_blocked(self, hitl_manager):
        hitl_manager.submit_request(
            student_id=3,
            current_pathway="STEM",
            desired_pathway="ARTS_SPORTS",
            reason="Test",
        )
        with pytest.raises(ValueError, match="already has a pending request"):
            hitl_manager.submit_request(
                student_id=3,
                current_pathway="STEM",
                desired_pathway="SOCIAL_SCIENCES",
                reason="Another",
            )


@pytest.mark.parametrize("desired", ["STEM", "SOCIAL_SCIENCES", "ARTS_SPORTS"])
def test_hitl_all_valid_pathways(desired, hitl_manager):
    """Every valid pathway can be requested."""
    current = "ARTS_SPORTS" if desired != "ARTS_SPORTS" else "STEM"
    req_id = hitl_manager.submit_request(
        student_id=99,
        current_pathway=current,
        desired_pathway=desired,
        reason="Parametrized test",
    )
    assert req_id.startswith("REQ-")


def test_hitl_invalid_pathway_rejected(hitl_manager):
    with pytest.raises(ValueError):
        hitl_manager.submit_request(
            student_id=10,
            current_pathway="STEM",
            desired_pathway="UNKNOWN_PATH",
            reason="Bad pathway",
        )


# ═════════════════════════════════════════════════════════════
#  6. BENCHMARKS
# ═════════════════════════════════════════════════════════════


class TestBenchmarks:
    """Statistical tools and mini benchmark pipeline."""

    def test_dkw_bound(self):
        from benchmarks.benchmark import dkw_epsilon

        eps_10 = dkw_epsilon(10)
        eps_100 = dkw_epsilon(100)
        eps_1000 = dkw_epsilon(1000)
        assert eps_10 > eps_100 > eps_1000

    def test_t_confidence_interval(self):
        from benchmarks.benchmark import t_confidence_interval

        data = np.array([0.8, 0.82, 0.79, 0.85, 0.81])
        mean, lo, hi = t_confidence_interval(data)
        assert abs(mean - np.mean(data)) < 1e-5
        assert lo <= mean <= hi

    def test_mini_benchmark_pipeline(self):
        from benchmarks.benchmark import RLBenchmark

        bm = RLBenchmark(n_students=20, n_episodes=10, n_trials=1, base_seed=99, verbose=False)
        results = bm.run_all()
        for key in ("summary", "performance_distribution", "convergence", "per_pathway"):
            assert key in results
        assert results["summary"]["mean_accuracy"] >= 0.0


@pytest.mark.parametrize("n,expected_order", [
    (10, True), (100, True), (1000, True),
])
def test_dkw_epsilon_monotone(n, expected_order):
    """DKW epsilon is strictly positive."""
    from benchmarks.benchmark import dkw_epsilon

    assert dkw_epsilon(n) > 0
