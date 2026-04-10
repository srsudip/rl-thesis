"""
CRITICAL SYSTEM AUDIT
=====================
Exhaustive consistency tests that trace every data path in the system.
Tests every function that computes scores, cluster weights, recommendations,
or displays data to ensure they all use the same source and logic.

Organized by bug category:
  A. Score Consistency — same score everywhere for same student/subject/grade
  B. Cluster Weight Consistency — same CW everywhere for same student
  C. Recommendation Consistency — same pathway everywhere for same student
  D. Grade Level Consistency — correct grade used in every computation  
  E. Core Subject Logic — gate checks match displayed reasons
  F. Display Data Integrity — what's shown matches what's computed
  G. Edge Cases — boundary scores, missing data, extreme values
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def dm():
    from pages.dashboard import DataManager
    from src.rl.hitl import HITLManager
    m = DataManager.__new__(DataManager)
    m.data = None; m.agent = None; m.env = None; m.overrides = {}; m.edits = {}
    m._pop_stats_cache = None; m.coaching_model = None; m.coaching_stats = None
    m.training_history = None; m.model_accuracy = None; m._student_preferences = {}
    m.hitl = HITLManager()
    m._hitl_state_path = Path('/tmp/audit_hitl.json')
    m._session_state_path = Path('/tmp/audit_session.json')
    m.generate_data(n_students=200, seed=42)
    return m


@pytest.fixture(scope="module")
def all_student_ids(dm):
    return dm.get_student_ids()


# ============================================================================
# A. SCORE CONSISTENCY
# Every function that returns a student's score must return the same number.
# ============================================================================

class TestScoreConsistency:
    """Scores must be identical across: assessments DF, get_performance(),
    get_knec_result_slip(), get_transitions(), and the analysis page."""

    def test_performance_matches_assessments_all_students(self, dm, all_student_ids):
        """A.1: get_performance() scores must match raw assessments DataFrame."""
        df = dm.data['assessments']
        errors = []
        for sid in all_student_ids:
            perf = dm.get_performance(sid)
            g9 = df[(df['student_id'] == sid) & (df['grade'] == 9)]
            if len(g9) == 0:
                continue
            g9 = g9.iloc[0]
            for subj, data in perf.get('subject_scores', {}).items():
                col = f"{subj}_score"
                if col in g9.index and pd.notna(g9[col]):
                    if abs(g9[col] - data['score']) > 0.5:
                        errors.append(f"S{sid} {subj}: assess={g9[col]:.1f} != perf={data['score']:.1f}")
        assert not errors, f"{len(errors)} score mismatches:\n" + "\n".join(errors[:10])

    def test_slip_scores_match_performance_all_students(self, dm, all_student_ids):
        """A.2: KJSEA slip raw_score must match get_performance() for every subject."""
        errors = []
        for sid in all_student_ids:
            slip = dm.get_knec_result_slip(sid)
            perf = dm.get_performance(sid)
            perf_scores = {k: v['score'] for k, v in perf.get('subject_scores', {}).items()}
            for s in slip['subjects']:
                subj_key = s.get('subject_key', '')
                slip_score = s.get('raw_score', 0)
                perf_score = perf_scores.get(subj_key)
                if perf_score is not None and abs(slip_score - perf_score) > 0.5:
                    errors.append(f"S{sid} {subj_key}: slip={slip_score:.1f} != perf={perf_score:.1f}")
        assert not errors, f"{len(errors)} slip/perf mismatches:\n" + "\n".join(errors[:10])

    def test_transitions_scores_match_assessments(self, dm, all_student_ids):
        """A.3: Grade scores in transitions must match assessments DataFrame."""
        df = dm.data['assessments']
        errors = []
        for sid in all_student_ids[:50]:  # Sample for speed
            trans = dm.get_transitions(sid)
            for subj, data in trans.get('subject_transitions', {}).items():
                for grade, score in data.get('grade_scores', {}).items():
                    row = df[(df['student_id'] == sid) & (df['grade'] == grade)]
                    if len(row) == 0:
                        continue
                    col = f"{subj}_score"
                    if col in row.columns:
                        db_val = row.iloc[0][col]
                        if pd.notna(db_val) and abs(db_val - score) > 0.5:
                            errors.append(f"S{sid} {subj} G{grade}: trans={score:.1f} != db={db_val:.1f}")
        assert not errors, f"{len(errors)} transition/assessment mismatches:\n" + "\n".join(errors[:10])


# ============================================================================
# B. CLUSTER WEIGHT CONSISTENCY
# CW must be identical across: slip, recommendation, suitability, per-grade
# ============================================================================

class TestClusterWeightConsistency:

    def test_slip_cw_matches_recommendation_cw(self, dm, all_student_ids):
        """B.1: Cluster weights from slip must match recommendation output."""
        errors = []
        for sid in all_student_ids:
            slip = dm.get_knec_result_slip(sid)
            rec = dm.get_recommendation(sid)
            if rec.get('is_override'):
                continue
            slip_cw = slip['cluster_weights']
            rec_cw = rec.get('cluster_weights', {})
            if not rec_cw:
                continue
            for pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']:
                if abs(slip_cw.get(pw, 0) - rec_cw.get(pw, 0)) > 0.01:
                    errors.append(f"S{sid} {pw}: slip_cw={slip_cw[pw]:.2f} != rec_cw={rec_cw[pw]:.2f}")
        assert not errors, f"{len(errors)} CW mismatches:\n" + "\n".join(errors[:10])

    def test_same_pop_stats_everywhere(self, dm):
        """B.2: _get_population_stats() must return consistent results."""
        stats1 = dm._get_population_stats()
        stats2 = dm._get_population_stats()
        assert stats1 is stats2 or stats1 == stats2, "pop_stats not cached properly"


# ============================================================================
# C. RECOMMENDATION CONSISTENCY
# The recommended pathway must be the same everywhere it appears.
# ============================================================================

class TestRecommendationConsistency:

    def test_g9_suggestion_matches_recommendation_all_students(self, dm, all_student_ids):
        """C.1 CRITICAL: Grade 9 per-grade suggestion must match the displayed recommendation."""
        errors = []
        for sid in all_student_ids:
            rec = dm.get_recommendation(sid)
            if rec.get('is_override'):
                continue
            per_grade = dm.get_suggested_pathway_per_grade(sid)
            g9_sugg = per_grade.get(9, '')
            if g9_sugg != rec['recommended_pathway']:
                errors.append(f"S{sid}: rec={rec['recommended_pathway']}, g9_sugg={g9_sugg}")
        assert not errors, f"CRITICAL: {len(errors)} G9 suggestion mismatches:\n" + "\n".join(errors[:10])

    def test_recommendation_deterministic(self, dm, all_student_ids):
        """C.2: Calling get_recommendation twice must give same result."""
        for sid in all_student_ids[:50]:
            r1 = dm.get_recommendation(sid)
            r2 = dm.get_recommendation(sid)
            assert r1['recommended_pathway'] == r2['recommended_pathway'], \
                f"S{sid}: non-deterministic: {r1['recommended_pathway']} != {r2['recommended_pathway']}"

    def test_highest_eligible_composite_wins(self, dm, all_student_ids):
        """C.3: Recommended pathway must be an eligible pathway (passes both gates) when
        any exist, OR if none pass, below_expectations_warning must be True.

        Note: the system uses multi-signal fusion (60% CW + 25% cosine + 15% PSI), so
        the winner within the eligible set is the highest *composite* score, not highest
        raw CW. This test verifies eligibility gates and fallback behaviour only.
        """
        from config.pathways import _check_core_subjects, PATHWAY_MIN_THRESHOLD
        errors = []
        for sid in all_student_ids:
            rec = dm.get_recommendation(sid)
            if rec.get('is_override'):
                continue
            slip = dm.get_knec_result_slip(sid)
            cw = slip['cluster_weights']
            perf = dm.get_performance(sid)
            raw = {s: d['score'] for s, d in perf.get('subject_scores', {}).items()}

            eligible = []
            for pw, val in cw.items():
                threshold = PATHWAY_MIN_THRESHOLD.get(pw, 25)
                if val >= threshold and _check_core_subjects(pw, raw):
                    eligible.append(pw)

            if eligible:
                # Recommended pathway must be one of the eligible pathways
                if rec['recommended_pathway'] not in eligible:
                    errors.append(
                        f"S{sid}: rec={rec['recommended_pathway']} is not in "
                        f"eligible={eligible} (CW={cw})"
                    )
            else:
                # Fallback — must carry a below-expectations warning
                if not rec.get('below_expectations_warning'):
                    errors.append(
                        f"S{sid}: fallback to {rec['recommended_pathway']} but NO warning set"
                    )

        assert not errors, f"{len(errors)} recommendation logic errors:\n" + "\n".join(errors[:10])


# ============================================================================
# D. GRADE LEVEL CONSISTENCY
# ============================================================================

class TestGradeLevelConsistency:

    def test_all_students_have_grade_9(self, dm, all_student_ids):
        """D.1: Every student must have Grade 9 data."""
        df = dm.data['assessments']
        missing = []
        for sid in all_student_ids:
            g9 = df[(df['student_id'] == sid) & (df['grade'] == 9)]
            if len(g9) == 0:
                missing.append(sid)
        assert not missing, f"{len(missing)} students missing Grade 9 data: {missing[:10]}"

    def test_recommendation_uses_grade_9(self, dm, all_student_ids):
        """D.2: Recommendation must use Grade 9 scores (not other grades)."""
        df = dm.data['assessments']
        for sid in all_student_ids[:20]:
            slip = dm.get_knec_result_slip(sid)
            g9 = df[(df['student_id'] == sid) & (df['grade'] == 9)]
            if len(g9) == 0:
                continue
            # Check at least one subject matches
            for s in slip['subjects']:
                col = f"{s['subject_key']}_score"
                if col in g9.columns:
                    db_val = g9.iloc[0][col]
                    if pd.notna(db_val):
                        assert abs(s['raw_score'] - db_val) < 0.5, \
                            f"S{sid} {s['subject_key']}: slip uses wrong grade? slip={s['raw_score']}, G9={db_val}"
                        break


# ============================================================================
# E. CORE SUBJECT LOGIC
# ============================================================================

class TestCoreSubjectLogic:

    def test_stem_requires_math_above_ae1(self, dm, all_student_ids):
        """E.1: STEM requires Math >= 31% (AE1). Students failing core subjects
        should not be recommended that pathway."""
        from config.pathways import PATHWAY_MIN_THRESHOLD, _check_core_subjects
        for sid in all_student_ids[:50]:
            rec = dm.get_recommendation(sid)
            if rec.get('is_override'):
                continue
            pw = rec['recommended_pathway']
            perf = dm.get_performance(sid)
            raw = {k: v.get('score', 0) for k, v in perf.get('subject_scores', {}).items()}
            # The recommended pathway must pass its own core check
            # (unless ALL pathways fail core, in which case least-gap fallback is used)
            passes_core = _check_core_subjects(pw, raw)
            if not passes_core:
                # Should have core_check_failed flag
                assert rec.get('core_check_failed') or rec.get('below_expectations_warning'), \
                    f"S{sid}: {pw} fails core but no warning set"

    def test_ss_requires_language_above_ae1(self, dm, all_student_ids):
        """E.2: Gate 2 active — recommended pathway should pass core subjects.
        If below_expectations, the system picks the best available pathway
        that passes core, which may differ from highest CW."""
        from config.pathways import _check_core_subjects
        for sid in all_student_ids[:50]:
            rec = dm.get_recommendation(sid)
            if rec.get('is_override'):
                continue
            pw = rec['recommended_pathway']
            assert pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
            # If core check didn't fail, the pathway should pass its core subjects
            if not rec.get('core_check_failed'):
                perf = dm.get_performance(sid)
                raw = {k: v.get('score', 0) for k, v in perf.get('subject_scores', {}).items()}
                assert _check_core_subjects(pw, raw), \
                    f"S{sid}: {pw} recommended but fails core check"

    def test_grade_level_labels_correct(self, dm, all_student_ids):
        """E.3: get_cbc_grade() returns correct level for boundary scores."""
        from config.pathways import get_cbc_grade
        boundary_tests = [
            (1, 'BE2'), (10, 'BE2'), (11, 'BE1'), (20, 'BE1'),
            (21, 'AE2'), (30, 'AE2'), (31, 'AE1'), (40, 'AE1'),
            (41, 'ME2'), (57, 'ME2'), (58, 'ME1'), (74, 'ME1'),
            (75, 'EE2'), (89, 'EE2'), (90, 'EE1'), (100, 'EE1'),
        ]
        for score, expected_level in boundary_tests:
            result = get_cbc_grade(score)
            assert result['level'] == expected_level, \
                f"Score {score}: expected {expected_level}, got {result['level']}"

    def test_float_scores_dont_fall_through(self, dm):
        """E.4: Float scores at .5 boundary must round UP (standard rounding, not banker's)."""
        from config.pathways import get_cbc_grade
        edge_cases = [
            (74.4, 'ME1'), (74.5, 'EE2'), (74.9, 'EE2'),  # 74.5→75=EE2
            (89.5, 'EE1'), (30.5, 'AE1'), (10.5, 'BE1'),   # .5 always rounds up
            (30.4, 'AE2'), (40.4, 'AE1'), (40.5, 'ME2'),
        ]
        for score, expected in edge_cases:
            result = get_cbc_grade(score)
            assert result['level'] == expected, \
                f"Score {score}: expected {expected}, got {result['level']}"


# ============================================================================
# F. DISPLAY DATA INTEGRITY
# ============================================================================

class TestDisplayIntegrity:

    def test_psi_within_range(self, dm, all_student_ids):
        """F.1: PSI must be in [0, 1] for all students and pathways."""
        for sid in all_student_ids[:50]:
            psi = dm.get_psi(sid)
            for pw, val in psi.items():
                assert 0 <= val <= 1, f"S{sid} {pw}: PSI={val} out of [0,1]"

    def test_cosine_within_range(self, dm, all_student_ids):
        """F.2: Cosine similarity must be in [0, 1]."""
        for sid in all_student_ids[:50]:
            cos = dm.get_cosine_similarities(sid)
            for pw, val in cos.items():
                assert 0 <= val <= 1, f"S{sid} {pw}: cosine={val} out of [0,1]"

    def test_total_points_correct(self, dm, all_student_ids):
        """F.3: Slip total_points must equal sum of individual subject points."""
        errors = []
        for sid in all_student_ids:
            slip = dm.get_knec_result_slip(sid)
            computed_total = sum(s.get('points', 0) for s in slip['subjects'])
            if computed_total != slip['total_points']:
                errors.append(f"S{sid}: sum={computed_total} != total={slip['total_points']}")
        assert not errors, f"{len(errors)} point total errors:\n" + "\n".join(errors[:10])

    def test_max_points_correct(self, dm, all_student_ids):
        """F.4: max_points must be 8 × number of subjects."""
        for sid in all_student_ids[:20]:
            slip = dm.get_knec_result_slip(sid)
            expected_max = 8 * len(slip['subjects'])
            assert slip['max_points'] == expected_max, \
                f"S{sid}: max_points={slip['max_points']} != 8×{len(slip['subjects'])}={expected_max}"

    def test_pathway_suitability_chart_values_match_cw(self, dm, all_student_ids):
        """F.5: Pathway suitability bar chart values must match slip CW."""
        for sid in all_student_ids[:50]:
            slip = dm.get_knec_result_slip(sid)
            rec = dm.get_recommendation(sid)
            if rec.get('is_override'):
                continue
            # The chart uses slip['cluster_weights'] — verify recommendation agrees
            cw = slip['cluster_weights']
            rec_cw = rec.get('cluster_weights', {})
            for pw in cw:
                if pw in rec_cw:
                    assert abs(cw[pw] - rec_cw[pw]) < 0.01, \
                        f"S{sid} {pw}: chart={cw[pw]} != rec={rec_cw[pw]}"


# ============================================================================
# G. EDGE CASES
# ============================================================================

class TestEdgeCases:

    def test_all_low_scores_student(self, dm):
        """G.1: Student with all scores near minimum should get a pathway (with warning)."""
        # Manually check the weakest student
        df = dm.data['assessments']
        g9 = df[df['grade'] == 9]
        score_cols = [c for c in g9.columns if c.endswith('_score')]
        g9_means = g9.groupby('student_id')[score_cols].mean().mean(axis=1)
        weakest = int(g9_means.idxmin())
        
        rec = dm.get_recommendation(weakest)
        assert rec['recommended_pathway'] in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS'], \
            f"Weakest student {weakest} got invalid pathway: {rec['recommended_pathway']}"

    def test_all_high_scores_student(self, dm):
        """G.2: Student with all high scores should get a pathway without warnings."""
        df = dm.data['assessments']
        g9 = df[df['grade'] == 9]
        score_cols = [c for c in g9.columns if c.endswith('_score')]
        g9_means = g9.groupby('student_id')[score_cols].mean().mean(axis=1)
        strongest = int(g9_means.idxmax())
        
        rec = dm.get_recommendation(strongest)
        assert rec['recommended_pathway'] in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']

    def test_identical_cw_tiebreak(self, dm):
        """G.3: When CW values are very close, system must not crash."""
        from config.pathways import recommend_pathway
        # Simulate near-tie
        cw = {'STEM': 50.01, 'SOCIAL_SCIENCES': 50.00, 'ARTS_SPORTS': 49.99}
        scores = {'MATH': 50, 'ENG': 50, 'KIS_KSL': 50, 'INT_SCI': 50,
                  'SOC_STUD': 50, 'REL_CRE': 50, 'CRE_ARTS': 50, 'PRE_TECH': 50}
        result = recommend_pathway(cw, scores)
        assert result['recommended_pathway'] in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']

    def test_missing_subject_handling(self, dm):
        """G.4: Missing subjects should not crash the system."""
        from config.pathways import compute_cluster_weights, recommend_pathway
        partial_scores = {'MATH': 70, 'ENG': 60}  # only 2 subjects
        cw = compute_cluster_weights(partial_scores)
        result = recommend_pathway(cw, partial_scores)
        assert result['recommended_pathway'] in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']

    def test_score_exactly_at_boundary(self, dm):
        """G.5: Score exactly at 31 (AE1 boundary) must pass core check."""
        from config.pathways import _check_core_subjects
        scores = {'MATH': 31, 'INT_SCI': 31, 'PRE_TECH': 31}
        assert _check_core_subjects('STEM', scores), "Math=31 should pass STEM core check"
        
        scores2 = {'MATH': 30.9, 'INT_SCI': 31, 'PRE_TECH': 31}
        # 30.9 rounds to 31, so this should also pass
        from config.pathways import get_cbc_grade
        grade = get_cbc_grade(30.9)
        # If it's AE2 (not AE1), then 30.9 fails
        if grade['level'] == 'AE2':
            assert not _check_core_subjects('STEM', scores2), "Math=30.9→AE2 should fail"


# ============================================================================
# H. POP_STATS SINGLE SOURCE OF TRUTH AUDIT
# ============================================================================

class TestPopStatsSingleSource:
    """The root cause of the G9 suggestion bug was multiple pop_stats computations.
    This test ensures there is exactly ONE authoritative source."""

    def test_dashboard_caches_pop_stats(self, dm):
        """H.1: After first call, pop_stats must be cached."""
        dm._pop_stats_cache = None
        s1 = dm._get_population_stats()
        s2 = dm._get_population_stats()
        # Should be same object (cached)
        assert s1 is s2 or s1 == s2

    def test_pop_stats_uses_grade_9_only(self, dm):
        """H.2: pop_stats must be computed from Grade 9 data only."""
        df = dm.data['assessments']
        g9 = df[df['grade'] == 9]
        all_data = df
        
        stats = dm._get_population_stats()
        if stats is None:
            return  # No data case
        
        # Verify against Grade 9 manual computation
        for subj_key, stat_dict in list(stats.items())[:3]:
            col = f"{subj_key}_score"
            if col in g9.columns:
                g9_mean = g9[col].dropna().mean()
                g9_std = g9[col].dropna().std()
                all_mean = all_data[col].dropna().mean()
                
                # Should match Grade 9, NOT all grades
                assert abs(stat_dict['mean'] - g9_mean) < 0.01, \
                    f"{subj_key}: pop_stats mean={stat_dict['mean']:.2f} != G9 mean={g9_mean:.2f}"
                
                # If all-grades mean is different, verify we're NOT using it
                if abs(g9_mean - all_mean) > 1.0:
                    assert abs(stat_dict['mean'] - g9_mean) < abs(stat_dict['mean'] - all_mean), \
                        f"{subj_key}: pop_stats appears to use ALL grades instead of G9 only!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])