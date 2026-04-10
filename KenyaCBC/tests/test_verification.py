"""
Verification Tests — Phase 6
Covers: data consistency, CSV loading, DQN coaching, cosine similarity, PSI,
        pathway history, comparison table, gradient colors, KJSEA slip correctness.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def dm():
    """Create a DataManager loaded with real CSV data."""
    from pages.dashboard import DataManager
    from src.rl.hitl import HITLManager
    
    manager = DataManager.__new__(DataManager)
    manager.data = None
    manager.agent = None
    manager.env = None
    manager.overrides = {}
    manager.edits = {}
    manager._pop_stats_cache = None
    manager.coaching_model = None
    manager.coaching_stats = None
    manager.training_history = None
    manager.model_accuracy = None
    manager._student_preferences = {}
    manager._student_feedback = {}
    manager.hitl = HITLManager()
    manager._hitl_state_path = Path('/tmp/test_hitl.json')
    manager._session_state_path = Path('/tmp/test_session.json')
    
    csv_path = str(Path(__file__).parent.parent / 'data' / 'gradess1.csv')
    if not Path(csv_path).exists():
        csv_path = '/mnt/user-data/uploads/gradess1.csv'
    
    success = manager.load_csv_data(csv_path, simulate_g4_g6=True)
    assert success, "Failed to load CSV data"
    return manager


# ============================================================================
# 6.2: DATA CONSISTENCY — values match across all views
# ============================================================================

class TestDataConsistency:
    """Verify that scores match across slip, performance, charts, and history."""
    
    SPOT_CHECK_IDS = [1, 50, 100, 500, 1000]  # Item 1.12: 5 random students
    
    def test_slip_matches_performance(self, dm):
        """KJSEA slip values must match get_performance values. (1.11)"""
        for sid in self.SPOT_CHECK_IDS:
            if sid > dm.get_student_count():
                continue
            slip = dm.get_knec_result_slip(sid)
            perf = dm.get_performance(sid)
            
            slip_scores = {s['subject_key']: s['raw_score'] for s in slip['subjects']}
            perf_scores = {k: v['score'] for k, v in perf.get('subject_scores', {}).items()}
            
            # Every slip subject must match performance
            for subj, slip_score in slip_scores.items():
                if subj in perf_scores:
                    assert abs(slip_score - perf_scores[subj]) < 0.5, \
                        f"Student {sid}: {subj} slip={slip_score} != perf={perf_scores[subj]}"
    
    def test_grade_9_scores_consistent(self, dm):
        """Grade 9 subject scores from assessments match performance output. (1.15)"""
        df = dm.data['assessments']
        for sid in self.SPOT_CHECK_IDS:
            if sid > dm.get_student_count():
                continue
            g9 = df[(df['student_id'] == sid) & (df['grade'] == 9)]
            if len(g9) == 0:
                continue
            g9 = g9.iloc[0]
            
            perf = dm.get_performance(sid)
            for subj, data in perf.get('subject_scores', {}).items():
                col = f"{subj}_score"
                if col in g9.index and pd.notna(g9[col]):
                    assert abs(g9[col] - data['score']) < 0.5, \
                        f"Student {sid}: {subj} assessment={g9[col]} != perf={data['score']}"
    
    def test_cluster_weights_consistent(self, dm):
        """Cluster weights from slip match recommendation output."""
        for sid in self.SPOT_CHECK_IDS:
            if sid > dm.get_student_count():
                continue
            slip = dm.get_knec_result_slip(sid)
            rec = dm.get_recommendation(sid)
            
            if not rec.get('is_override'):
                slip_cw = slip['cluster_weights']
                rec_cw = rec.get('cluster_weights', {})
                if rec_cw:
                    for pw in slip_cw:
                        assert abs(slip_cw[pw] - rec_cw.get(pw, 0)) < 0.01, \
                            f"Student {sid}: {pw} slip_cw={slip_cw[pw]} != rec_cw={rec_cw.get(pw, 0)}"


# ============================================================================
# 6.3: CSV UPLOAD
# ============================================================================

class TestCSVUpload:
    
    def test_student_count(self, dm):
        assert dm.get_student_count() == 1039
    
    def test_grades_available(self, dm):
        df = dm.data['assessments']
        grades = sorted(df['grade'].unique())
        assert 4 in grades, "Grade 4 (simulated) missing"
        assert 9 in grades, "Grade 9 (real) missing"
        assert len(grades) == 6, f"Expected 6 grades, got {len(grades)}"
    
    def test_student_names_loaded(self, dm):
        opts = dm.get_student_options()
        assert len(opts) > 0
        # First student should have a name
        assert 'Jacob' in opts[0]['label'] or '#1' in opts[0]['label']
    
    def test_pathway_labels_exist(self, dm):
        labels = dm.data.get('pathway_labels', {})
        assert len(labels) > 0, "No pathway labels loaded"
        assert labels.get(1) in ['SOCIAL_SCIENCES', 'STEM', 'ARTS_SPORTS']
    
    def test_no_nan_scores_grade9(self, dm):
        df = dm.data['assessments']
        g9 = df[df['grade'] == 9]
        score_cols = [c for c in g9.columns if c.endswith('_score')]
        # Core subjects should not be NaN
        for col in ['MATH_score', 'ENG_score', 'KIS_KSL_score']:
            if col in g9.columns:
                nan_count = g9[col].isna().sum()
                assert nan_count == 0, f"{col} has {nan_count} NaN values in grade 9"


# ============================================================================
# 6.4: DQN COACHING
# ============================================================================

class TestDQNCoaching:
    
    def test_state_encoding(self):
        from src.rl.dqn_coaching import encode_state, get_state_dim
        scores = {9: {'MATH': 50, 'ENG': 60}}
        state = encode_state(scores)
        assert state.shape == (78,), f"Expected 78-dim state (75 + 3 feedback), got {state.shape}"
        assert state.dtype == np.float32
    
    def test_action_space(self):
        from src.rl.dqn_coaching import ACTIONS, NUM_ACTIONS
        assert NUM_ACTIONS == 9
        for a in ACTIONS:
            assert a.id >= 0
            assert len(a.name) > 0
            assert len(a.description) > 0
    
    def test_transition_extraction(self, dm):
        from src.rl.dqn_coaching import extract_transitions
        trans = extract_transitions(dm.data['assessments'], dm.data.get('pathways'),
                                    dm.data.get('pop_stats', {}))
        assert len(trans) > 1000, f"Expected 1000+ transitions, got {len(trans)}"
        # Each transition has required keys
        t = trans[0]
        for key in ['state', 'action', 'reward', 'next_state', 'done']:
            assert key in t, f"Missing key: {key}"
    
    def test_model_training(self, dm):
        from src.rl.dqn_coaching import extract_transitions, train_coaching_agent
        trans = extract_transitions(dm.data['assessments'], dm.data.get('pathways'),
                                    dm.data.get('pop_stats', {}))
        model, stats = train_coaching_agent(trans[:500], n_epochs=10, batch_size=32)
        assert stats['final_loss'] > 0
        assert stats['total_transitions'] == 500
        
        # Model can make recommendations
        rec = model.get_recommendation(trans[0]['state'])
        assert 'action_id' in rec
        assert 0 <= rec['action_id'] < 9
    
    def test_baselines(self, dm):
        from src.rl.dqn_coaching import (extract_transitions, train_coaching_agent,
                                          run_baseline_comparisons)
        trans = extract_transitions(dm.data['assessments'], dm.data.get('pathways'),
                                    dm.data.get('pop_stats', {}))
        model, _ = train_coaching_agent(trans[:300], n_epochs=10)
        baselines = run_baseline_comparisons(trans[:100], model)
        assert 'DQN' in baselines
        assert 'Random' in baselines


# ============================================================================
# COSINE SIMILARITY & PSI
# ============================================================================

class TestCosineSimilarity:
    
    def test_cosine_range(self, dm):
        cos = dm.get_cosine_similarities(1)
        for pw, val in cos.items():
            assert 0 <= val <= 1, f"{pw} cosine={val} out of range"
    
    def test_cosine_all_pathways(self, dm):
        cos = dm.get_cosine_similarities(1)
        assert 'STEM' in cos
        assert 'SOCIAL_SCIENCES' in cos
        assert 'ARTS_SPORTS' in cos
    
    def test_psi_range(self, dm):
        psi = dm.get_psi(1)
        for pw, val in psi.items():
            assert 0 <= val <= 1, f"{pw} PSI={val} out of range"
    
    def test_psi_all_pathways(self, dm):
        psi = dm.get_psi(1)
        assert len(psi) == 3


# ============================================================================
# PATHWAY HISTORY & COMPARISON
# ============================================================================

class TestPathwayHistory:
    
    def test_history_has_all_grades(self, dm):
        hist = dm.get_pathway_history(1)
        grades = sorted(hist['grade'].unique())
        assert len(grades) >= 3, f"Expected 3+ grades in history, got {len(grades)}"
    
    def test_history_has_all_pathways(self, dm):
        hist = dm.get_pathway_history(1)
        pathways = set(hist['pathway'].unique())
        assert 'STEM' in pathways
        assert 'SOCIAL_SCIENCES' in pathways
        assert 'ARTS_SPORTS' in pathways
    
    def test_per_grade_suggestion(self, dm):
        pg = dm.get_suggested_pathway_per_grade(1)
        assert len(pg) >= 3
        for g, pw in pg.items():
            assert pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
    
    def test_pathway_comparison(self, dm):
        df_comp, meta = dm.get_pathway_comparison(1, 'STEM')
        assert len(df_comp) > 0
        assert 'cosine_current' in meta
        assert 'cosine_desired' in meta
        assert 'Current Score' in df_comp.columns


# ============================================================================
# STUDENT PREFERENCES & HITL
# ============================================================================

class TestPreferencesHITL:
    
    def test_store_preference(self, dm):
        dm.set_student_preference(1, 'STEM')
        assert dm.get_student_preference(1) == 'STEM'
    
    def test_teacher_review_data(self, dm):
        data = dm.get_teacher_review_data(1)
        assert 'recommendation' in data
        assert 'psi' in data
        assert 'cosine_similarities' in data
        assert 'coaching' in data
        assert 'cluster_weights' in data


# ============================================================================
# EDGE CASES (6.6)
# ============================================================================

class TestEdgeCases:
    
    def test_nonexistent_student(self, dm):
        """Should handle gracefully without crash."""
        # Nonexistent student should return something without crashing
        cos = dm.get_cosine_similarities(99999)
        # It may return zeros or fallback values — key is no exception
        assert isinstance(cos, dict)
    
    def test_comparison_same_pathway(self, dm):
        """Comparing with same pathway should still work."""
        rec = dm.get_recommendation(1)
        df_comp, meta = dm.get_pathway_comparison(1, rec['recommended_pathway'])
        # Should return data (even if gaps are 0)
        assert isinstance(df_comp, pd.DataFrame)


# ============================================================================
# TRAJECTORY STABILITY (1.2-1.3)
# ============================================================================

class TestTrajectoryStability:
    
    def test_stability_affects_cw(self):
        from config.pathways import compute_cluster_weights
        scores = {'MATH': 70, 'INT_SCI': 65, 'PRE_TECH': 60, 'ENG': 50,
                  'KIS_KSL': 50, 'SOC_STUD': 50, 'REL_CRE': 50, 'CRE_ARTS': 50, 'AGRI': 50}
        
        # Without trajectory
        cw_no_traj = compute_cluster_weights(scores)
        
        # With consistent trajectory (should give slight boost for consistency)
        prev_cw = {'STEM': cw_no_traj['STEM'], 'SOCIAL_SCIENCES': cw_no_traj['SOCIAL_SCIENCES'],
                   'ARTS_SPORTS': cw_no_traj['ARTS_SPORTS']}
        cw_with_traj = compute_cluster_weights(scores, prev_cluster_weights=prev_cw)
        
        # With trajectory should be close to without (since prev == current)
        for pw in cw_no_traj:
            assert abs(cw_no_traj[pw] - cw_with_traj[pw]) < 2.0, \
                f"Stable trajectory should give similar CW: {cw_no_traj[pw]} vs {cw_with_traj[pw]}"


# ============================================================================
# STUDENT FEEDBACK & COACHING PLAN (Dr. Mayeku items a/b/c)
# ============================================================================

class TestStudentFeedback:
    
    def test_set_and_get_feedback(self, dm):
        dm.set_student_feedback(1, 'satisfied')
        fb = dm.get_student_feedback(1)
        assert fb['feedback'] == 'satisfied'
        assert fb['desired_pathway'] is None
    
    def test_set_feedback_wants_different(self, dm):
        dm.set_student_feedback(1, 'wants_different', 'STEM')
        fb = dm.get_student_feedback(1)
        assert fb['feedback'] == 'wants_different'
        assert fb['desired_pathway'] == 'STEM'
        # Should also store preference
        assert dm.get_student_preference(1) == 'STEM'
    
    def test_no_feedback_default(self, dm):
        fb = dm.get_student_feedback(99998)
        assert fb['feedback'] is None
    
    def test_coaching_plan_basic(self, dm):
        plan = dm.get_coaching_plan(1)
        assert 'status' in plan
        assert 'message' in plan
        assert 'focus_subjects' in plan
        assert plan['status'] in ('satisfied', 'coaching_to_desired', 'strengthen_current')
    
    def test_coaching_plan_with_feedback(self, dm):
        dm.set_student_feedback(1, 'wants_different', 'ARTS_SPORTS')
        plan = dm.get_coaching_plan(1)
        assert plan['status'] == 'coaching_to_desired'
        assert plan['target_pathway'] == 'ARTS_SPORTS'
        assert any(f['subject'] == 'CRE_ARTS' for f in plan['focus_subjects'])
    
    def test_coaching_plan_satisfied(self, dm):
        rec = dm.get_recommendation(1)
        dm.set_student_feedback(1, 'satisfied')
        plan = dm.get_coaching_plan(1)
        assert plan['status'] == 'satisfied'
        assert plan['target_pathway'] == rec['recommended_pathway']
    
    def test_coaching_focus_sorted_by_priority(self, dm):
        plan = dm.get_coaching_plan(1)
        priorities = [f['priority'] for f in plan['focus_subjects']]
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'maintain': 3}
        ranks = [priority_order.get(p, 9) for p in priorities]
        assert ranks == sorted(ranks), "Focus subjects should be sorted by priority"


class TestDQNFeedbackState:
    
    def test_state_with_feedback(self):
        from src.rl.dqn_coaching import encode_state, get_state_dim
        scores = {9: {'MATH': 70, 'ENG': 60, 'KIS_KSL': 55, 'INT_SCI': 65, 'AGRI': 50,
                      'SOC_STUD': 50, 'REL_CRE': 50, 'CRE_ARTS': 50, 'PRE_TECH': 55}}
        
        s_none = encode_state(scores, feedback=None)
        s_sat = encode_state(scores, feedback='satisfied')
        s_diff = encode_state(scores, feedback='wants_different')
        
        # All same dimension
        assert s_none.shape == s_sat.shape == s_diff.shape
        
        # Feedback is encoded as a single scalar at state[75]:
        #   0.0 = no feedback, 1.0 = satisfied, -1.0 = wants_different
        assert s_none[75] == 0.0, f"No feedback: expected state[75]=0.0, got {s_none[75]}"
        assert s_sat[75] == 1.0, f"Satisfied: expected state[75]=1.0, got {s_sat[75]}"
        assert s_diff[75] == -1.0, f"Wants_different: expected state[75]=-1.0, got {s_diff[75]}"
    
    def test_reward_with_feedback(self):
        from src.rl.dqn_coaching import compute_reward, ACTIONS
        
        s_before = {'MATH': 50, 'INT_SCI': 50}
        s_after = {'MATH': 60, 'INT_SCI': 55}
        action = ACTIONS[0]  # strengthen STEM math
        
        r_none = compute_reward(s_before, s_after, action, preference='STEM', feedback=None)
        r_sat = compute_reward(s_before, s_after, action, preference='STEM', feedback='satisfied')
        r_diff = compute_reward(s_before, s_after, action, preference='STEM', feedback='wants_different')
        
        # All should be valid floats
        assert isinstance(r_none, float)
        assert isinstance(r_sat, float)
        assert isinstance(r_diff, float)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
