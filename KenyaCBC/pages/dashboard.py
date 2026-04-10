"""
Kenya CBC Pathway Recommendation System - Main Dashboard Module

Professional multi-page Plotly Dash application with:
- Proper RL integration using trained DQN agent
- Session state management across pages  
- Human-in-the-Loop override functionality
- Data editing capabilities for teachers

ScaDS.AI Dresden-Leipzig
"""

from dash import Dash, html, dcc, page_container
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PATHWAYS, COMPETENCIES, SUBJECT_NAMES
from config.competencies import IDEAL_PATHWAY_PROFILES
from src.rl.hitl import HITLManager


class DataManager:
    """
    Centralized data manager for the dashboard.
    
    Handles:
    - Data generation and loading
    - Model training and loading
    - RL-based recommendations with explanations
    - Human-in-the-Loop overrides (formal workflow)
    - Data editing
    """
    
    def __init__(self):
        self.data = None
        self.agent = None
        self.env = None
        
        # Persistent HITL state (survives hot-reload)
        from config.settings import DATA_DIR
        self._hitl_state_path = DATA_DIR / "hitl_state.json"
        self.hitl = HITLManager(state_path=self._hitl_state_path)
        
        self.overrides = {}  # Legacy compat: student_id -> {pathway, reason, timestamp, teacher}
        self.edits = {}  # student_id -> {field: new_value}
        self.training_history = None
        self.model_accuracy = None
        self._pop_stats_cache = None  # z-score population stats cache
        self._session_state_path = DATA_DIR / "session_state.json"
        self.coaching_model = None
        self.coaching_stats = None

        # Per-instance student feedback/preference storage (persisted to session state)
        self._student_preferences = {}  # student_id -> pathway_key
        self._student_feedback = {}     # student_id -> {'feedback': str, 'desired_pathway': str}

        # Auto-load persisted data and model on startup
        self._auto_load()
    
    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self.data is not None and 'profiles' in self.data
    
    def has_model(self) -> bool:
        """Check if model is trained."""
        return self.agent is not None
    
    def _auto_load(self):
        """Auto-load data and model from disk on startup (survives hot-reload)."""
        try:
            from src.data.cbc_data_generator import check_generated_files_exist
            if check_generated_files_exist():
                success = self.load_data()
                if success:
                    print("  ✓ Auto-loaded data from previous session")
                    # Load session state (edits, overrides, accuracy)
                    self._load_session_state()
                    # Try to load trained model
                    try:
                        from config.settings import MODELS_DIR
                        from src.rl.agent import PathwayRecommendationAgent
                        agent = PathwayRecommendationAgent()
                        agent.load()
                        self.agent = agent
                        # Recreate environment
                        from src.rl.environment import PathwayEnvironment
                        self.env = PathwayEnvironment(
                            self.data['assessments'], self.data['competencies'])
                        # Load accuracy from metadata if session state didn't have it
                        if self.model_accuracy is None:
                            meta_path = MODELS_DIR / 'model_metadata.json'
                            if meta_path.exists():
                                with open(meta_path) as f:
                                    meta = json.load(f)
                                self.model_accuracy = meta.get('accuracy')
                        print("  ✓ Auto-loaded trained model")
                    except Exception as e:
                        print(f"  ⚠ No saved model loaded ({e}) — user can train later")
        except Exception as e:
            print(f"  ⚠ Auto-load skipped: {e}")

    def _save_session_state(self):
        """Persist edits, overrides, and training metadata to JSON."""
        try:
            state = {
                'edits': {str(k): v for k, v in self.edits.items()},
                'overrides': {str(k): v for k, v in self.overrides.items()},
                'model_accuracy': self.model_accuracy,
                'student_preferences': {str(k): v for k, v in self._student_preferences.items()},
                'student_feedback': {str(k): v for k, v in self._student_feedback.items()},
            }
            self._session_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._session_state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            print(f"  ⚠ Session save failed: {e}")

    def _load_session_state(self):
        """Restore edits, overrides, and training metadata from JSON."""
        if not self._session_state_path.exists():
            return
        try:
            with open(self._session_state_path) as f:
                state = json.load(f)
            def _int_keys(d):
                result = {}
                for k, v in d.items():
                    try:
                        result[int(k)] = v
                    except (ValueError, TypeError):
                        pass
                return result
            self.edits = _int_keys(state.get('edits', {}))
            self.overrides = _int_keys(state.get('overrides', {}))
            self.model_accuracy = state.get('model_accuracy')
            self._student_preferences = _int_keys(state.get('student_preferences', {}))
            self._student_feedback = _int_keys(state.get('student_feedback', {}))
            n_edits = sum(len(v) for v in self.edits.values())
            n_overrides = len(self.overrides)
            if n_edits or n_overrides:
                print(f"  ✓ Restored session: {n_edits} edits, {n_overrides} overrides")
        except Exception as e:
            print(f"  ⚠ Session load failed: {e}")
    
    def get_student_options(self):
        """Get dropdown options for student selector."""
        if not self.has_data():
            return []
        students = sorted(self.data['profiles']['student_id'].unique())
        names = self.data.get('student_names', {})
        return [
            {'label': f"{names.get(s, '')} (#{s})" if names.get(s) else f'Student {s}', 'value': s}
            for s in students
        ]
    
    def get_student_count(self) -> int:
        """Get number of students."""
        if not self.has_data():
            return 0
        return len(self.data['profiles'])
    
    def get_student_ids(self) -> list:
        """Get list of all student IDs."""
        if not self.has_data():
            return []
        return sorted(self.data['profiles']['student_id'].unique().tolist())
    
    # ==================== RL-BASED RECOMMENDATIONS ====================
    
    def get_recommendation(self, student_id: int) -> dict:
        """
        Get pathway recommendation.
        
        Priority:
        1. Formal HITL override (approved request via HITLManager)
        2. Legacy direct override (backwards compatibility)
        3. KNEC cluster weights (primary recommendation — matches result slip)
        4. Hybrid RL/rule-based (supplementary AI suggestion)
        """
        # Check formal HITL override first
        hitl_override = self.hitl.get_active_override(student_id)
        if hitl_override:
            return {
                'recommended_pathway': hitl_override['desired_pathway'],
                'confidence': 1.0,
                'is_override': True,
                'override_reason': hitl_override.get('reason', ''),
                'override_by': hitl_override.get('resolved_by', 'Teacher'),
                'override_time': hitl_override.get('resolved_at', 'Unknown'),
                'reasoning': f"Teacher override: {hitl_override.get('resolution_justification', hitl_override.get('reason', ''))}",
                'q_values': {},
                'pathway_ranking': [hitl_override['desired_pathway']],
                'confidence_scores': {hitl_override['desired_pathway']: 1.0}
            }
        
        # Check legacy direct override
        if student_id in self.overrides:
            override = self.overrides[student_id]
            return {
                'recommended_pathway': override['pathway'],
                'confidence': 1.0,
                'is_override': True,
                'override_reason': override['reason'],
                'override_by': override.get('teacher', 'Unknown'),
                'override_time': override.get('timestamp', 'Unknown'),
                'reasoning': f"Manual override: {override['reason']}",
                'q_values': {},
                'pathway_ranking': [override['pathway']],
                'confidence_scores': {override['pathway']: 1.0}
            }
        
        # Use KNEC cluster weights as PRIMARY recommendation
        slip = self.get_knec_result_slip(student_id)
        cw = slip['cluster_weights']
        recommended = slip['recommended_pathway']
        max_cw = max(cw.values()) if cw else 0
        confidence = round(max_cw / 100.0, 3)  # percentage → 0-1
        
        # Get full recommendation result with multi-signal fusion
        from config.pathways import recommend_pathway as _rec_pw
        from src.data.real_data_loader import compute_all_cosine_similarities
        perf = self.get_performance(student_id)
        raw_scores = {s: d['score'] for s, d in perf.get('subject_scores', {}).items()}
        blended = self._blend_scores_knec(student_id, raw_scores)
        pop_stats = self._get_population_stats()
        from config.pathways import compute_cluster_weights as _cw
        full_cw = _cw(blended, pop_stats=pop_stats)
        cos_sims = compute_all_cosine_similarities(raw_scores)
        psi = self.get_psi(student_id)
        full_rec = _rec_pw(full_cw, raw_scores, cosine_sims=cos_sims, psi_scores=psi)

        is_hybrid = False
        ranking = sorted(cw.items(), key=lambda x: x[1], reverse=True)

        pw_name = PATHWAYS.get(recommended, {}).get('name', recommended)
        return {
            'recommended_pathway': recommended,
            'confidence': confidence,
            'is_override': False,
            'is_hybrid': False,
            'reasoning': f"KNEC placement (60% G9 + 20% SBA + 20% G6): {pw_name} ({cw.get(recommended, 0):.1f}%).",
            'q_values': cw,
            'pathway_ranking': [p for p, _ in ranking],
            'confidence_scores': {p: round(s / 100.0, 3) for p, s in cw.items()},
            'cluster_weights': cw,
            'below_expectations_warning': full_rec.get('below_expectations_warning', False),
            'eligible_pathways': full_rec.get('eligible_pathways', []),
            'core_check_failed': full_rec.get('core_check_failed', False),
            'low_confidence': full_rec.get('low_confidence', False),
        }
    
    def _get_subject_scores(self, student_id: int) -> dict:
        """Get subject scores for a student."""
        perf = self.get_performance(student_id)
        return {
            subj: data.get('score', 0)
            for subj, data in perf.get('subject_scores', {}).items()
        }
    
    def _get_student_state(self, student_id: int) -> np.ndarray:
        """Get competency state vector for a student (input to RL agent)."""
        if not self.has_data():
            return None
        
        # Get grade 9 competencies
        comp_df = self.data['competencies']
        student_comp = comp_df[
            (comp_df['student_id'] == student_id) &
            (comp_df['grade'] == 9)
        ]
        
        if len(student_comp) == 0:
            return None
        
        row = student_comp.iloc[0]
        state = []
        for comp in COMPETENCIES:
            col = f'{comp}_score'
            score = row.get(col, 50) / 100.0 if col in row.index else 0.5
            state.append(score)
        
        return np.array(state)
    
    def _rule_based_recommendation(self, student_id: int) -> dict:
        """Fallback rule-based recommendation when no model is trained."""
        np.random.seed(student_id)
        
        # Get competencies
        comp = self.get_competencies(student_id)
        comp_vector = np.array([comp.get(c, 50) / 100 for c in COMPETENCIES])
        
        # Calculate similarity to each pathway's ideal profile
        similarities = {}
        for pathway, ideal in IDEAL_PATHWAY_PROFILES.items():
            ideal_vector = np.array([ideal[c] for c in COMPETENCIES])
            sim = np.dot(comp_vector, ideal_vector) / (
                np.linalg.norm(comp_vector) * np.linalg.norm(ideal_vector) + 1e-8
            )
            similarities[pathway] = sim
        
        # Get best pathway
        ranking = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        recommended = ranking[0][0]
        
        # Generate reasoning
        top_comps = sorted(comp.items(), key=lambda x: x[1], reverse=True)[:3]
        comp_names = [COMPETENCIES[c]['name'] for c, _ in top_comps]
        
        return {
            'recommended_pathway': recommended,
            'confidence': similarities[recommended],
            'is_override': False,
            'reasoning': f"Based on strong {comp_names[0]} and {comp_names[1]} competencies.",
            'q_values': similarities,
            'pathway_ranking': [p for p, _ in ranking],
            'confidence_scores': {p: s for p, s in similarities.items()}
        }
    
    def get_recommendation_explanation(self, student_id: int) -> dict:
        """
        Generate detailed explanation for why a pathway was recommended.
        
        This uses the hybrid system which provides:
        - Proper confidence based on performance level
        - Subject-based scoring aligned with curriculum
        - Competency alignment analysis
        - Eligibility checks against pathway requirements
        """
        rec = self.get_recommendation(student_id)
        comp = self.get_competencies(student_id)
        perf = self.get_performance(student_id)
        
        pathway = rec['recommended_pathway']
        pathway_data = PATHWAYS[pathway]
        
        # Get explanation from hybrid system
        explanation = rec.get('explanation', {})
        
        # Build competency alignments
        ideal = IDEAL_PATHWAY_PROFILES.get(pathway, {})
        alignments = []
        gaps = []
        
        for c in COMPETENCIES:
            student_score = comp.get(c, 50)
            ideal_score = ideal.get(c, 0.7) * 100
            diff = student_score - ideal_score
            
            if diff >= 10:
                alignments.append({
                    'competency': COMPETENCIES[c]['name'],
                    'student': student_score,
                    'ideal': ideal_score,
                    'status': 'exceeds'
                })
            elif diff >= -10:
                alignments.append({
                    'competency': COMPETENCIES[c]['name'],
                    'student': student_score,
                    'ideal': ideal_score,
                    'status': 'meets'
                })
            else:
                gaps.append({
                    'competency': COMPETENCIES[c]['name'],
                    'student': student_score,
                    'ideal': ideal_score,
                    'gap': abs(diff)
                })
        
        # Get strong/weak subjects from explanation or calculate
        strong_subjects = explanation.get('strengths', [])
        weak_subjects = explanation.get('weaknesses', [])
        
        # Build response
        return {
            'pathway': pathway,
            'pathway_name': pathway_data['name'],
            'confidence': rec['confidence'],
            'is_rl_based': self.has_model() and not rec.get('is_override', False),
            'is_hybrid': rec.get('is_hybrid', False),
            'is_override': rec.get('is_override', False),
            'reasoning': explanation.get('reasoning', rec.get('reasoning', '')),
            'method': explanation.get('method', 'Hybrid System'),
            'performance_level': rec.get('performance_level', 'unknown'),
            'overall_performance': rec.get('overall_performance', perf.get('overall_average', 50)),
            'competency_alignments': alignments,
            'competency_gaps': gaps,
            'strong_subjects': strong_subjects if isinstance(strong_subjects, list) else [],
            'weak_subjects': weak_subjects if isinstance(weak_subjects, list) else [],
            'recommendations': explanation.get('recommendations', []),
            'eligibility': rec.get('eligibility', {}),
            'alternative_pathways': rec.get('pathway_ranking', [])[1:] if len(rec.get('pathway_ranking', [])) > 1 else [],
            'confidence_scores': rec.get('confidence_scores', {})
        }
    
    def get_knec_result_slip(self, student_id: int, grade_level: int = 9) -> dict:
        """
        Generate KNEC-style result slip with pathway suitability percentages.
        Uses z-score normalization when population data is available (matching KNEC methodology).
        """
        from config.pathways import KJSEA_SUBJECT_CODES, get_cbc_grade, compute_cluster_weights
        
        perf = self.get_performance(student_id)
        subject_scores = perf.get('subject_scores', {})
        
        # Build subject results with KNEC codes
        subjects = []
        raw_scores = {}
        for code_num, code_info in KJSEA_SUBJECT_CODES.items():
            subj_key = code_info['code']
            score_data = subject_scores.get(subj_key, {})
            raw_score = score_data.get('score', 0)
            grade = get_cbc_grade(max(raw_score, 1))
            subjects.append({
                'code': code_num, 'subject': code_info['name'],
                'subject_key': subj_key, 'performance_level': grade['level'],
                'points': grade['points'], 'description': grade['name'],
                'raw_score': raw_score,
            })
            raw_scores[subj_key] = raw_score
        
        # Add any subjects not in KJSEA codes but present in data
        for subj_key, score_data in subject_scores.items():
            if subj_key not in raw_scores:
                raw_scores[subj_key] = score_data.get('score', 0)
        
        # Apply official KNEC placement formula before computing cluster weights:
        #   60% Grade 9 (KJSEA) + 20% Grades 7–8 avg (SBA) + 20% Grade 6 (KPSEA)
        blended_scores = self._blend_scores_knec(student_id, raw_scores)

        # Compute cluster weights on blended scores with z-score normalization
        pop_stats = self._get_population_stats()
        cluster_weights = compute_cluster_weights(blended_scores, pop_stats=pop_stats)

        # Use recommend_pathway with multi-signal fusion for core subject validation
        from config.pathways import recommend_pathway as _rec_pw
        from src.data.real_data_loader import compute_all_cosine_similarities
        _cos_sims = compute_all_cosine_similarities(raw_scores)
        _psi = self.get_psi(student_id)
        pw_result = _rec_pw(cluster_weights, subject_scores=raw_scores,
                            cosine_sims=_cos_sims, psi_scores=_psi)
        recommended = pw_result['recommended_pathway']
        
        # Total points (sum of all KJSEA subject points, max = 8 * num_subjects)
        total_points = sum(s['points'] for s in subjects)
        max_points = 8 * len(subjects)
        
        return {
            'student_id': student_id, 'grade_level': grade_level,
            'subjects': subjects, 'cluster_weights': cluster_weights,
            'recommended_pathway': recommended,
            'total_points': total_points, 'max_points': max_points,
        }

    def _get_scores_for_grade(self, student_id: int, grade: int) -> dict:
        """Return {subj_key: raw_score} for a student at a specific grade level."""
        if not self.has_data():
            return {}
        rows = self.data['assessments'][
            (self.data['assessments']['student_id'] == student_id) &
            (self.data['assessments']['grade'] == grade)
        ]
        if len(rows) == 0:
            return {}
        row = rows.iloc[-1]
        return {
            col.replace('_score', ''): float(row[col])
            for col in row.index
            if col.endswith('_score') and col not in ['student_id', 'grade']
            and not pd.isna(row[col])
        }

    def _blend_scores_knec(self, student_id: int, g9_scores: dict) -> dict:
        """
        Blend subject scores using the official KNEC placement formula:
            60% Grade 9 (KJSEA) + 20% Grades 7–8 avg (SBA) + 20% Grade 6 (KPSEA)

        If a grade tier is absent from the data the weight is redistributed
        proportionally among the tiers that are present, so the sum always = 1.
        """
        g6 = self._get_scores_for_grade(student_id, 6)
        g7 = self._get_scores_for_grade(student_id, 7)
        g8 = self._get_scores_for_grade(student_id, 8)

        # Grade 7+8 SBA average per subject
        sba: dict = {}
        for s in g9_scores:
            vals = [v for v in [g7.get(s), g8.get(s)] if v is not None]
            if vals:
                sba[s] = float(np.mean(vals))

        blended: dict = {}
        for s, g9_val in g9_scores.items():
            sba_val = sba.get(s)
            g6_val  = g6.get(s)

            components = [(0.60, g9_val), (0.20, sba_val), (0.20, g6_val)]
            weight_sum = sum(w for w, v in components if v is not None)
            if weight_sum <= 0:
                blended[s] = g9_val
            else:
                blended[s] = sum(w * v for w, v in components if v is not None) / weight_sum

        return blended

    def _get_population_stats(self) -> dict:
        """Compute population mean/std per subject for z-score normalization."""
        if not self.has_data() or self._pop_stats_cache is not None:
            return self._pop_stats_cache

        assessments = self.data['assessments']
        g9 = assessments[assessments['grade'] == 9]
        stats = {}
        for col in g9.columns:
            if col.endswith('_score') and col not in ['student_id', 'grade']:
                key = col.replace('_score', '')
                vals = g9[col].dropna()
                if len(vals) > 1:
                    stats[key] = {'mean': float(vals.mean()), 'std': float(vals.std())}
        self._pop_stats_cache = stats if stats else None
        return self._pop_stats_cache

    def get_performance(self, student_id: int) -> dict:
        """Get performance data for a student."""
        if self.has_data():
            # Check for edits
            edited = self.edits.get(student_id, {})
            
            student_data = self.data['assessments'][
                (self.data['assessments']['student_id'] == student_id) &
                (self.data['assessments']['grade'] == 9)
            ]
            if len(student_data) > 0:
                row = student_data.iloc[-1]
                subject_scores = {}
                # Dynamically detect all _score columns
                for col in row.index:
                    if col.endswith('_score') and col not in ['student_id', 'grade']:
                        subj = col.replace('_score', '')
                        if not pd.isna(row[col]):
                            score = edited.get(col, row[col])
                            subject_scores[subj] = {
                            'score': score,
                            'level': self._score_to_level(score),
                            'edited': col in edited
                        }
                
                if subject_scores:
                    avg = np.mean([s['score'] for s in subject_scores.values()])
                    return {
                        'student_id': student_id,
                        'overall': {'score': avg, 'level': self._score_to_level(avg)},
                        'subject_scores': subject_scores
                    }
        
        # Mock data fallback (no generated data loaded)
        np.random.seed(student_id)
        subject_scores = {}
        for subj in SUBJECT_NAMES:
            score = np.random.uniform(35, 95)
            subject_scores[subj] = {'score': score, 'level': self._score_to_level(score), 'edited': False}
        
        avg = np.mean([s['score'] for s in subject_scores.values()])
        return {
            'student_id': student_id,
            'overall': {'score': avg, 'level': self._score_to_level(avg)},
            'subject_scores': subject_scores
        }
    
    def get_competencies(self, student_id: int) -> dict:
        """Get competency scores for a student."""
        if self.has_data():
            edited = self.edits.get(student_id, {})
            
            student_data = self.data['competencies'][
                (self.data['competencies']['student_id'] == student_id) &
                (self.data['competencies']['grade'] == 9)
            ]
            if len(student_data) > 0:
                row = student_data.iloc[0]
                comps = {}
                for c in COMPETENCIES:
                    col = f'{c}_score'
                    if col in row.index:
                        comps[c] = edited.get(col, row[col])
                if comps:
                    return comps
        
        np.random.seed(student_id)
        return {c: np.random.uniform(40, 90) for c in COMPETENCIES}
    
    def get_transitions(self, student_id: int) -> dict:
        """Get grade transition data for a student."""
        transitions = {}
        
        if self.has_data():
            student_data = self.data['assessments'][
                self.data['assessments']['student_id'] == student_id
            ]
            if len(student_data) > 0:
                # Dynamically discover all subject columns from the data
                score_cols = [c for c in student_data.columns if c.endswith('_score')]
                
                for col in score_cols:
                    subj = col.replace('_score', '')
                    grade_scores = {}
                    
                    for grade in student_data['grade'].unique():
                        grade_data = student_data[student_data['grade'] == grade]
                        score = grade_data[col].values[0] if len(grade_data) > 0 else None
                        if score is not None and not np.isnan(score):
                            grade_scores[int(grade)] = float(score)
                    
                    if grade_scores:  # Only include if there are actual scores
                        transitions[subj] = {'grade_scores': grade_scores}
                
                if transitions:
                    all_changes = []
                    for subj_data in transitions.values():
                        grades = sorted(subj_data['grade_scores'].keys())
                        if len(grades) >= 2:
                            change = subj_data['grade_scores'][grades[-1]] - subj_data['grade_scores'][grades[0]]
                            all_changes.append(change)
                    
                    avg_change = np.mean(all_changes) if all_changes else 0
                    trend = 'improving' if avg_change > 2 else ('declining' if avg_change < -2 else 'stable')
                    
                    return {
                        'student_id': student_id,
                        'overall_trend': {'trend': trend, 'avg_change': avg_change},
                        'subject_transitions': transitions
                    }
        
        # Mock data for when no real data exists
        np.random.seed(student_id)
        trend = np.random.choice(['improving', 'stable', 'declining'])
        
        # Subjects available in generated data (grade 4-9)
        all_subjects = ['MATH', 'ENG', 'KIS_KSL', 'SCI_TECH', 'SOC_STUD', 'CRE_ARTS', 'AGRI',
                        'REL_CRE', 'REL_IRE', 'REL_HRE']
        for subj in all_subjects:
            base = np.random.uniform(40, 60)
            growth = np.random.uniform(-1, 2)
            grade_scores = {g: max(0, min(100, base + growth * g + np.random.uniform(-5, 5))) for g in range(4, 10)}
            transitions[subj] = {'grade_scores': grade_scores}
        
        return {
            'student_id': student_id,
            'overall_trend': {'trend': trend, 'avg_change': 0},
            'subject_transitions': transitions
        }
    
    def get_sub_pathway_recommendation(self, student_id: int) -> dict:
        """Get recommended sub-pathway (track) based on subject performance."""
        rec = self.get_recommendation(student_id)
        pathway = rec['recommended_pathway']
        pathway_data = PATHWAYS[pathway]
        sub_pathways = pathway_data['sub_pathways']
        perf = self.get_performance(student_id)
        
        # Score each track based on relevant subjects
        sub_scores = {}
        for sub_key, sub_data in sub_pathways.items():
            scores = perf['subject_scores']
            key_subjects = sub_data.get('key_subjects', [])
            if key_subjects:
                vals = [scores.get(s, {}).get('score', 50) for s in key_subjects if s in scores]
                score = np.mean(vals) if vals else 50
            elif pathway == 'STEM':
                if 'pure' in sub_key.lower():
                    score = (scores.get('MATH', {}).get('score', 50) + scores.get('SCI_TECH', {}).get('score', 50)) / 2
                elif 'applied' in sub_key.lower():
                    score = (scores.get('AGRI', {}).get('score', 50) + scores.get('SCI_TECH', {}).get('score', 50)) / 2
                elif 'engineering' in sub_key.lower() or 'tech' in sub_key.lower():
                    score = (scores.get('MATH', {}).get('score', 50) + scores.get('SCI_TECH', {}).get('score', 50)) / 2 + 3
                else:
                    score = scores.get('MATH', {}).get('score', 50)
            elif pathway == 'SOCIAL_SCIENCES':
                if 'language' in sub_key.lower():
                    score = (scores.get('ENG', {}).get('score', 50) + scores.get('KIS_KSL', {}).get('score', 50)) / 2
                else:
                    score = (scores.get('SOC_STUD', {}).get('score', 50) + scores.get('ENG', {}).get('score', 50)) / 2
            else:  # ARTS_SPORTS
                if 'sport' in sub_key.lower():
                    score = scores.get('CRE_ARTS', {}).get('score', 50) + 3
                elif 'visual' in sub_key.lower():
                    score = scores.get('CRE_ARTS', {}).get('score', 50) + 1
                else:
                    score = scores.get('CRE_ARTS', {}).get('score', 50)
            
            sub_scores[sub_key] = score
        
        best_sub = max(sub_scores, key=sub_scores.get)
        
        return {
            'recommended_sub': best_sub,
            'sub_data': sub_pathways[best_sub],
            'all_scores': sub_scores,
            'all_subs': sub_pathways
        }
    
    def _score_to_level(self, score: float) -> str:
        """Convert score to performance level."""
        if score >= 75:
            return 'exceeding_expectation'
        elif score >= 50:
            return 'meeting_expectation'
        elif score >= 25:
            return 'approaching_expectation'
        return 'below_expectation'
    
    # ==================== DATA GENERATION & TRAINING ====================
    
    def generate_data(self, n_students: int = 500, seed: int = 42, save_csv: bool = True) -> bool:
        """
        Generate synthetic student data using IRT-based model.
        
        Args:
            n_students: Number of students to generate
            seed: Random seed for reproducibility
            save_csv: If True (default), save generated data to CSV files
        
        Data Generation Flow:
        1. CSV files define curriculum structure (subjects, strands, substrands, indicators)
        2. competencysubstrand_4_9.csv links substrands → competencies
        3. substrandindicator_4_9.csv links indicators → substrands (with weights)
        4. For each student: θ = α + β × (grade - min_grade) [latent competency]
        5. For each indicator: score = 100 × logistic(a·θ - b) [IRT model]
        6. Aggregate indicator scores → subject scores (wide format)
        """
        try:
            from src.data.cbc_data_generator import generate_dashboard_data
            
            print(f"\n  Using IRT Model (authentic CBC curriculum)...")
            data = generate_dashboard_data(
                n_students=n_students, 
                grades=(4, 5, 6, 7, 8, 9),
                seed=seed,
                save_csv=save_csv
            )
            
            self.data = {
                'profiles': data['students'],
                'assessments': data['assessments'],
                'competencies': data['competencies'],
                'pathways': data['pathways'],
                'indicators': data.get('indicators'),
                'theta': data.get('theta'),
                'indicator_params': data.get('indicator_params')
            }
            
            # Clear old state
            self.overrides = {}
            self.edits = {}
            self._pop_stats_cache = None
            self.hitl.clear()  # Reset HITL requests — old data = old requests
            self._save_session_state()
            
            print(f"  ✓ Generated {n_students} students")
            return True
            
        except Exception as e:
            print(f"Error generating data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_data(self) -> bool:
        """
        Load existing data from generated CSV files.
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            from src.data.cbc_data_generator import load_dashboard_data, check_generated_files_exist
            
            if not check_generated_files_exist():
                print("  ⚠ No generated data found. Please generate data first.")
                return False
            
            data = load_dashboard_data()
            
            self.data = {
                'profiles': data['students'],
                'assessments': data['assessments'],
                'competencies': data['competencies'],
                'pathways': data['pathways'],
                'indicators': data.get('indicators'),
                'theta': data.get('theta'),
                'indicator_params': data.get('indicator_params')
            }
            
            # Clear old state
            self.overrides = {}
            self.edits = {}
            self._pop_stats_cache = None
            
            n_students = len(self.data['pathways'])
            print(f"  ✓ Loaded {n_students} students from CSV files")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # =================================================================
    # REAL CSV DATA LOADING (Phase 0/4)
    # =================================================================
    
    def load_csv_data(self, filepath: str = None, simulate_g4_g6: bool = True) -> bool:
        """
        Load real student data from an uploaded CSV file.
        Supports gradess1.csv format and similar structures.
        Auto-detects subjects, grades, and pathway columns.
        """
        try:
            from src.data.real_data_loader import load_real_data_for_dashboard
            
            data = load_real_data_for_dashboard(
                filepath=filepath, simulate_g4_g6=simulate_g4_g6, seed=42
            )
            
            self.data = {
                'profiles': data['students'],
                'assessments': data['assessments'],
                'competencies': data['competencies'],
                'pathways': data['pathways'],
                'indicators': None,
                'theta': None,
                'indicator_params': None,
                'pop_stats': data.get('pop_stats', {}),
                'pathway_labels': data.get('pathway_labels', {}),
                'raw_data': data.get('raw_data'),
                'student_names': {},  # populated below
            }
            
            # Build student name lookup
            if 'name' in data['students'].columns:
                for _, r in data['students'].iterrows():
                    self.data['student_names'][r['student_id']] = r['name']
            
            # Clear old state
            self.overrides = {}
            self.edits = {}
            self._pop_stats_cache = data.get('pop_stats', None)
            self.coaching_model = None
            self.hitl.clear()
            self._save_session_state()
            
            n = len(data['pathways'])
            print(f"  ✓ Loaded {n} students from CSV")
            return True
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # =================================================================
    # DQN COACHING AGENT (Phase 3)
    # =================================================================

    def train_coaching_agent(self, n_epochs: int = 80) -> dict:
        """Train the DQN coaching agent on real transition data."""
        if not self.has_data():
            return {'success': False, 'error': 'No data available'}
        
        try:
            from src.rl.dqn_coaching import extract_transitions, train_coaching_agent
            
            transitions = extract_transitions(
                self.data['assessments'],
                self.data.get('pathways'),
                self.data.get('pop_stats', self._get_population_stats())
            )
            
            model, stats = train_coaching_agent(
                transitions, n_epochs=n_epochs, batch_size=64
            )
            
            self.coaching_model = model
            self.coaching_stats = stats
            return {'success': True, **stats}
            
        except Exception as e:
            print(f"Error training coaching agent: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def get_coaching_recommendation(self, student_id: int) -> dict:
        """Get DQN coaching recommendation for a student."""
        if self.coaching_model is None:
            return {'available': False, 'reason': 'Coaching model not trained'}
        
        try:
            from src.rl.dqn_coaching import encode_state
            from src.data.real_data_loader import compute_all_cosine_similarities
            from config.pathways import compute_cluster_weights, _core_gap_score
            
            df = self.data['assessments']
            student_data = df[df['student_id'] == student_id]
            
            if len(student_data) == 0:
                return {'available': False, 'reason': 'Student not found'}
            
            # Build scores by grade
            scores_by_grade = {}
            for _, row in student_data.iterrows():
                g = int(row['grade'])
                scores_by_grade[g] = {
                    col.replace('_score', ''): float(row[col])
                    for col in row.index
                    if col.endswith('_score') and col not in ['student_id', 'grade']
                    and pd.notna(row[col])
                }
            
            max_g = max(scores_by_grade.keys())
            g_scores = scores_by_grade[max_g]
            
            pop_stats = self.data.get('pop_stats', self._get_population_stats())
            cw = compute_cluster_weights(g_scores, pop_stats=pop_stats)
            cos = compute_all_cosine_similarities(g_scores)
            gaps = {pw: _core_gap_score(pw, g_scores)
                    for pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']}
            
            # Get student preference (from pathway labels if available)
            pref = self.data.get('pathway_labels', {}).get(student_id, None)
            
            # Get student feedback signal (Dr. Mayeku item a)
            fb = self.get_student_feedback(student_id)
            fb_signal = fb.get('feedback')  # 'satisfied' | 'wants_different' | None
            
            state = encode_state(scores_by_grade, cw, cos, gaps, pref, feedback=fb_signal)
            rec = self.coaching_model.get_recommendation(state)
            rec['available'] = True
            rec['cosine_similarities'] = cos
            return rec
            
        except Exception as e:
            return {'available': False, 'reason': str(e)}
    
    # =================================================================
    # STUDENT PREFERENCE STORAGE (3.25) + HITL ENHANCEMENTS (Phase 5)
    # =================================================================
    
    def set_student_preference(self, student_id: int, preferred_pathway: str):
        """Store student's pathway preference for DQN training."""
        self._student_preferences[student_id] = preferred_pathway
    
    def set_student_feedback(self, student_id: int, feedback: str, desired_pathway: str = None):
        """
        Store student's feedback on their recommendation (Dr. Mayeku item a).
        feedback: 'satisfied' | 'wants_different'
        desired_pathway: only used when feedback == 'wants_different'
        """
        self._student_feedback[student_id] = {
            'feedback': feedback,
            'desired_pathway': desired_pathway,
        }
        if feedback == 'wants_different' and desired_pathway:
            self._student_preferences[student_id] = desired_pathway
        self._save_session_state()
    
    def get_student_feedback(self, student_id: int) -> dict:
        """Get stored student feedback."""
        return self._student_feedback.get(student_id, {'feedback': None, 'desired_pathway': None})
    
    def get_coaching_plan(self, student_id: int) -> dict:
        """
        Get rule-based coaching plan incorporating student feedback.
        Works without DQN training — always available.
        """
        if not self.has_data():
            return {'status': 'no_data', 'message': 'Load data first.', 'focus_subjects': []}
        
        try:
            from src.rl.dqn_coaching import generate_coaching_plan
            from config.pathways import compute_cluster_weights
            
            rec = self.get_recommendation(student_id)
            perf = self.get_performance(student_id)
            scores = {k: v['score'] for k, v in perf.get('subject_scores', {}).items()}
            
            fb = self.get_student_feedback(student_id)
            pop_stats = self._get_population_stats()
            cw = compute_cluster_weights(scores, pop_stats=pop_stats)
            
            return generate_coaching_plan(
                scores=scores,
                recommended_pathway=rec['recommended_pathway'],
                student_feedback=fb.get('feedback'),
                desired_pathway=fb.get('desired_pathway'),
                cluster_weights=cw,
            )
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'focus_subjects': []}
    
    def get_student_preference(self, student_id: int) -> str:
        """Get student's stored pathway preference."""
        # Check explicit preference first, then fall back to actual labels
        if student_id in self._student_preferences:
            return self._student_preferences[student_id]
        if self.data and 'pathway_labels' in self.data:
            return self.data['pathway_labels'].get(student_id, '')
        return ''
    
    def get_teacher_review_data(self, student_id: int) -> dict:
        """
        Get comprehensive data for teacher review (Phase 5.2).
        Includes recommendation, PSI, cosine, coaching plan.
        """
        rec = self.get_recommendation(student_id)
        psi = self.get_psi(student_id)
        cos = self.get_cosine_similarities(student_id)
        coaching = self.get_coaching_recommendation(student_id)
        slip = self.get_knec_result_slip(student_id)
        
        return {
            'student_id': student_id,
            'recommendation': rec,
            'psi': psi,
            'cosine_similarities': cos,
            'coaching': coaching,
            'cluster_weights': slip['cluster_weights'],
            'total_points': slip['total_points'],
            'max_points': slip['max_points'],
        }

    # =================================================================
    # PATHWAY HISTORY & COMPARISON (Phase 1/2)
    # =================================================================
    
    def get_pathway_history(self, student_id: int) -> pd.DataFrame:
        """Get pathway suitability at each grade level."""
        if not self.has_data():
            return pd.DataFrame()
        try:
            from src.data.real_data_loader import compute_pathway_suitability_by_grade
            return compute_pathway_suitability_by_grade(
                self.data['assessments'], student_id,
                pop_stats=self._get_population_stats()
            )
        except Exception as e:
            print(f"Error getting pathway history: {e}")
            return pd.DataFrame()
    
    def get_suggested_pathway_per_grade(self, student_id: int) -> dict:
        """Get suggested pathway at each grade level for history chart.

        Earlier grades use raw grade scores (shows progression).
        Latest grade is pinned to get_recommendation() so the Grade-9 point
        in the history chart always matches the recommendation panel exactly.
        """
        if not self.has_data():
            return {}
        try:
            from src.data.real_data_loader import get_suggested_pathway_per_grade
            result = get_suggested_pathway_per_grade(
                self.data['assessments'], student_id,
                pop_stats=self._get_population_stats()
            )
            # Pin latest grade to the official recommendation (uses KNEC blending)
            if result:
                latest_grade = max(result.keys())
                rec = self.get_recommendation(student_id)
                result[latest_grade] = rec['recommended_pathway']
            return result
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Error getting per-grade suggestions: %s", e)
            return {}
    
    def get_pathway_comparison(self, student_id: int, desired_pathway: str) -> tuple:
        """Get comparison table for desired vs recommended pathway."""
        if not self.has_data():
            return pd.DataFrame(), {}
        try:
            from src.data.real_data_loader import generate_pathway_comparison
            
            # Get current scores
            perf = self.get_performance(student_id)
            subject_scores = {
                subj: data['score']
                for subj, data in perf.get('subject_scores', {}).items()
            }
            
            # Get current recommendation
            rec = self.get_recommendation(student_id)
            current_pw = rec['recommended_pathway']
            
            return generate_pathway_comparison(subject_scores, current_pw, desired_pathway)
        except Exception as e:
            print(f"Error generating comparison: {e}")
            return pd.DataFrame(), {}
    
    def get_cosine_similarities(self, student_id: int) -> dict:
        """Get cosine similarity scores for all pathways."""
        try:
            from src.data.real_data_loader import compute_all_cosine_similarities
            perf = self.get_performance(student_id)
            subject_scores = {
                subj: data['score']
                for subj, data in perf.get('subject_scores', {}).items()
            }
            return compute_all_cosine_similarities(subject_scores)
        except Exception:
            return {'STEM': 0, 'SOCIAL_SCIENCES': 0, 'ARTS_SPORTS': 0}
    
    def get_psi(self, student_id: int) -> dict:
        """Get Pathway Strength Index for all pathways."""
        try:
            from src.data.real_data_loader import compute_psi

            perf = self.get_performance(student_id)
            subject_scores = {
                subj: data['score']
                for subj, data in perf.get('subject_scores', {}).items()
            }

            # Use KNEC-blended scores for placement-aligned PSI
            blended = self._blend_scores_knec(student_id, subject_scores)
            pop_stats = self.data.get('pop_stats', self._get_population_stats())
            from config.pathways import compute_cluster_weights
            cw = compute_cluster_weights(blended, pop_stats=pop_stats)
            
            result = {}
            for pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']:
                cw_pct = min(cw.get(pw, 0) / 100.0, 1.0)
                result[pw] = compute_psi(subject_scores, pw, cw.get(pw, 0), cw_pct)
            return result
        except Exception:
            return {'STEM': 0, 'SOCIAL_SCIENCES': 0, 'ARTS_SPORTS': 0}

    def train_model(self, n_episodes: int = 500) -> dict:
        """
        Train the DQN model on the generated data.
        
        The agent includes all improvements:
        - Double DQN (reduces Q-value overestimation)
        - Dueling Architecture (separate value/advantage)
        - Prioritized Replay (efficient sampling)
        - Reward Shaping (dense learning signal)
        
        Args:
            n_episodes: Maximum number of training episodes
        
        Returns training results including accuracy and history.
        """
        if not self.has_data():
            return {'success': False, 'error': 'No data available'}
        
        try:
            from src.rl.environment import PathwayEnvironment
            from src.rl.agent import PathwayRecommendationAgent
            
            # Create environment
            self.env = PathwayEnvironment(
                self.data['assessments'],
                self.data['competencies'],
                verbose=True,
            )

            # Create and train agent
            self.agent = PathwayRecommendationAgent(verbose=True)
            
            history = self.agent.train(
                self.env,
                episodes=n_episodes,
                verbose=True,
                batch_per_episode=min(128, len(self.env.students)),
                early_stopping=False,  # Train full episodes requested by user
                patience=50,
                min_accuracy=0.95
            )
            
            # Save model
            self.agent.save()

            # Evaluate on all students
            from src.rl.trainer import evaluate_model
            from config.settings import MODELS_DIR
            results = evaluate_model(self.agent, self.data, verbose=False)
            self.model_accuracy = results['overall_accuracy']
            self._save_session_state()

            # Persist accuracy in model metadata so navbar shows correct value on cold restart
            meta_path = MODELS_DIR / 'model_metadata.json'
            with open(meta_path, 'w') as f:
                json.dump({'accuracy': self.model_accuracy,
                           'pathway_accuracy': results['pathway_accuracy']}, f)
            
            return {
                'success': True,
                'accuracy': results['overall_accuracy'],
                'pathway_accuracy': results['pathway_accuracy'],
            }
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    # ==================== HUMAN-IN-THE-LOOP ====================

    def submit_hitl_request(self, student_id: int, current_pathway: str,
                            desired_pathway: str, reason: str,
                            requested_by: str = "Student") -> str:
        """
        Submit a formal HITL pathway change request.
        Returns request_id.
        """
        return self.hitl.submit_request(
            student_id=student_id,
            current_pathway=current_pathway,
            desired_pathway=desired_pathway,
            reason=reason,
            requested_by=requested_by,
        )
    
    def approve_hitl_request(self, request_id: str, teacher_id: str,
                             justification: str = "") -> bool:
        """Approve a pending HITL request."""
        return self.hitl.approve_request(request_id, teacher_id, justification)
    
    def reject_hitl_request(self, request_id: str, teacher_id: str,
                            reason: str = "") -> bool:
        """Reject a pending HITL request."""
        return self.hitl.reject_request(request_id, teacher_id, reason)
    
    def get_pending_hitl_requests(self) -> list:
        """Get all pending HITL requests."""
        return self.hitl.get_pending_requests()
    
    def apply_override(self, student_id: int, pathway: str, reason: str, teacher: str = "Teacher") -> bool:
        """
        Apply a human override to a student's pathway recommendation.
        
        This is the Human-in-the-Loop functionality.
        """
        if pathway not in PATHWAYS:
            return False
        
        self.overrides[student_id] = {
            'pathway': pathway,
            'reason': reason,
            'teacher': teacher,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self._save_session_state()
        return True
    
    def remove_override(self, student_id: int) -> bool:
        """Remove a human override."""
        if student_id in self.overrides:
            del self.overrides[student_id]
            self._save_session_state()
            return True
        return False
    
    def get_override(self, student_id: int) -> dict:
        """Get override for a student if exists (HITL or legacy)."""
        # Check HITL approved overrides first
        hitl_override = self.hitl.get_active_override(student_id)
        if hitl_override:
            return {
                'pathway': hitl_override['desired_pathway'],
                'reason': hitl_override.get('reason', ''),
                'teacher': hitl_override.get('resolved_by', 'Teacher'),
                'timestamp': hitl_override.get('resolved_at', ''),
            }
        # Fall back to legacy direct overrides
        return self.overrides.get(student_id, None)
    
    def get_all_overrides(self) -> list:
        """Get all overrides (both HITL-approved and legacy direct)."""
        overrides = []
        # HITL approved overrides
        for sid, data in self.hitl.get_all_overrides().items():
            overrides.append({
                'student_id': sid,
                'pathway': data['desired_pathway'],
                'reason': data.get('reason', ''),
                'teacher': data.get('resolved_by', 'Teacher'),
                'timestamp': data.get('resolved_at', ''),
                'source': 'hitl',
            })
        # Legacy direct overrides
        for sid, data in self.overrides.items():
            overrides.append({
                'student_id': sid,
                **data,
                'source': 'legacy',
            })
        return overrides
    
    # ==================== DATA EDITING ====================
    
    def edit_student_data(self, student_id: int, field: str, value: float) -> bool:
        """
        Edit a student's data (for teacher corrections).
        
        Changes are stored in memory and applied when accessing data.
        """
        if student_id not in self.edits:
            self.edits[student_id] = {}
        
        self.edits[student_id][field] = value
        self._save_session_state()
        return True
    
    def get_edits(self, student_id: int) -> dict:
        """Get all edits for a student."""
        return self.edits.get(student_id, {})
    
    def clear_edits(self, student_id: int) -> bool:
        """Clear all edits for a student."""
        if student_id in self.edits:
            del self.edits[student_id]
            self._save_session_state()
            return True
        return False
    
    def save_edits_to_data(self) -> bool:
        """Apply all edits permanently to the data."""
        if not self.has_data():
            return False
        
        for student_id, edits in self.edits.items():
            for field, value in edits.items():
                # Update assessments
                mask = self.data['assessments']['student_id'] == student_id
                if field in self.data['assessments'].columns:
                    self.data['assessments'].loc[mask, field] = value
                
                # Update competencies
                mask = self.data['competencies']['student_id'] == student_id
                if field in self.data['competencies'].columns:
                    self.data['competencies'].loc[mask, field] = value
        
        self.edits = {}
        return True
    
    # ==================== HUMAN-IN-THE-LOOP RETRAINING ====================
    
    def retrain_with_feedback(self, n_episodes: int = 100) -> dict:
        """
        Retrain the model incorporating teacher feedback (overrides).
        
        Incorporates both formal HITL approved overrides and legacy overrides.
        """
        hitl_overrides = self.hitl.get_all_overrides()
        total_overrides = len(self.overrides) + len(hitl_overrides)
        
        if not self.has_model() or total_overrides == 0:
            return {'success': False, 'error': 'No model or no overrides to learn from'}

        if not self.has_data():
            return {'success': False, 'error': 'No data loaded'}

        try:
            # Always rebuild env from current data to avoid stale student IDs after data reload
            from src.rl.environment import PathwayEnvironment
            self.env = PathwayEnvironment(
                self.data['assessments'], self.data['competencies'], verbose=False)

            print(f"\n  Retraining with {total_overrides} teacher overrides...")
            
            # Map pathways to the primary DQN action for each pathway.
            # STEM→0 (strengthen_stem_math), SS→2 (strengthen_ss_languages),
            # ARTS→4 (strengthen_arts_creative) — these are the canonical GT actions.
            pathway_to_action = {'STEM': 0, 'SOCIAL_SCIENCES': 2, 'ARTS_SPORTS': 4}

            # Add legacy override experiences to replay buffer
            for student_id, override_data in self.overrides.items():
                state = self.env.reset(student_id)
                new_pathway = override_data['pathway']
                action = pathway_to_action.get(new_pathway, 0)
                self.agent.memory.push(state, action, 1.5, state, True)
                print(f"    Added legacy feedback for student {student_id}: {new_pathway}")

            # Add HITL override experiences
            for student_id, override_data in hitl_overrides.items():
                state = self.env.reset(student_id)
                new_pathway = override_data['desired_pathway']
                action = pathway_to_action.get(new_pathway, 0)
                self.agent.memory.push(state, action, 1.5, state, True)
                print(f"    Added HITL feedback for student {student_id}: {new_pathway}")
            
            # Continue training with feedback in buffer
            history = self.agent.train(
                self.env,
                episodes=n_episodes,
                verbose=True,
                batch_per_episode=min(64, len(self.env.students)),
                early_stopping=False
            )
            
            # Save updated model
            self.agent.save()
            
            # Re-evaluate
            from src.rl.trainer import evaluate_model
            results = evaluate_model(self.agent, self.data, verbose=False)
            self.model_accuracy = results['overall_accuracy']
            self._save_session_state()
            
            print(f"  ✓ Retraining complete. New accuracy: {self.model_accuracy:.2%}")
            
            return {
                'success': True,
                'accuracy': self.model_accuracy,
                'overrides_incorporated': total_overrides,
                'episodes': n_episodes
            }
            
        except Exception as e:
            print(f"Error in retraining: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    # ==================== CSV EXPORT/IMPORT ====================
    
    def export_data_to_csv(self, output_dir: str = None) -> dict:
        """
        Export all generated data to CSV files.
        
        Creates:
        - assessments.csv: Student subject scores
        - competencies.csv: Student competency scores
        - pathways.csv: Recommended pathways
        - overrides.csv: Teacher overrides
        
        Returns:
            Dictionary with file paths
        """
        if not self.has_data():
            return {'success': False, 'error': 'No data to export'}
        
        from pathlib import Path
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'data' / 'exports'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        try:
            # Export assessments
            assessments_path = output_dir / 'assessments.csv'
            self.data['assessments'].to_csv(assessments_path, index=False)
            files['assessments'] = str(assessments_path)
            
            # Export competencies
            competencies_path = output_dir / 'competencies.csv'
            self.data['competencies'].to_csv(competencies_path, index=False)
            files['competencies'] = str(competencies_path)
            
            # Export pathways
            pathways_path = output_dir / 'pathways.csv'
            self.data['pathways'].to_csv(pathways_path, index=False)
            files['pathways'] = str(pathways_path)
            
            # Export overrides if any
            if self.overrides:
                import pandas as pd
                overrides_df = pd.DataFrame([
                    {'student_id': sid, **data}
                    for sid, data in self.overrides.items()
                ])
                overrides_path = output_dir / 'overrides.csv'
                overrides_df.to_csv(overrides_path, index=False)
                files['overrides'] = str(overrides_path)
            
            print(f"  ✓ Exported {len(files)} files to {output_dir}")
            
            return {'success': True, 'files': files, 'output_dir': str(output_dir)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def import_data_from_csv(self, input_dir: str) -> dict:
        """
        Import data from CSV files.
        
        Expects:
        - assessments.csv
        - competencies.csv
        - pathways.csv (optional)
        - overrides.csv (optional)
        
        Returns:
            Import results
        """
        import pandas as pd
        from pathlib import Path
        
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            return {'success': False, 'error': f'Directory not found: {input_dir}'}
        
        try:
            # Load required files
            assessments_path = input_dir / 'assessments.csv'
            competencies_path = input_dir / 'competencies.csv'
            
            if not assessments_path.exists() or not competencies_path.exists():
                return {'success': False, 'error': 'Missing required files (assessments.csv, competencies.csv)'}
            
            self.data = {
                'assessments': pd.read_csv(assessments_path),
                'competencies': pd.read_csv(competencies_path),
                'profiles': None,
                'pathways': None
            }
            self._pop_stats_cache = None
            
            # Create profiles from assessments
            student_ids = self.data['assessments']['student_id'].unique()
            self.data['profiles'] = pd.DataFrame({'student_id': student_ids})
            
            # Load pathways if exists
            pathways_path = input_dir / 'pathways.csv'
            if pathways_path.exists():
                self.data['pathways'] = pd.read_csv(pathways_path)
            else:
                # Compute pathways from competencies
                from src.data.cbc_data_generator import determine_pathways
                self.data['pathways'] = determine_pathways(
                    self.data['competencies'],
                    self.data['assessments']
                )
            
            # Load overrides if exists
            overrides_path = input_dir / 'overrides.csv'
            if overrides_path.exists():
                overrides_df = pd.read_csv(overrides_path)
                self.overrides = {}
                for _, row in overrides_df.iterrows():
                    self.overrides[row['student_id']] = {
                        'pathway': row['pathway'],
                        'reason': row.get('reason', ''),
                        'teacher': row.get('teacher', 'Unknown'),
                        'timestamp': row.get('timestamp', '')
                    }
            
            n_students = len(student_ids)
            print(f"  ✓ Imported data for {n_students} students")
            
            return {
                'success': True,
                'n_students': n_students,
                'files_loaded': ['assessments.csv', 'competencies.csv'] + 
                               (['pathways.csv'] if pathways_path.exists() else []) +
                               (['overrides.csv'] if overrides_path.exists() else [])
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    # ==================== MULTI-SEED EVALUATION ====================
    
    def run_multi_seed_evaluation(self, seeds: list = None, episodes: int = 200) -> dict:
        """
        Run evaluation with multiple random seeds.
        
        Returns mean ± std accuracy with 95% CI.
        """
        if not self.has_data():
            return {'success': False, 'error': 'No data available'}
        
        try:
            from src.rl.evaluation import multi_seed_evaluation
            
            if seeds is None:
                seeds = [42, 123, 456, 789, 1011]
            
            results = multi_seed_evaluation(
                self.data,
                seeds=seeds,
                episodes=episodes,
                verbose=True
            )
            
            return {
                'success': True,
                'accuracy_mean': results['accuracy']['mean'],
                'accuracy_std': results['accuracy']['std'],
                'ci_lower': results['accuracy']['ci_95_lower'],
                'ci_upper': results['accuracy']['ci_95_upper'],
                'n_seeds': len(seeds),
                'full_results': results
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def run_hyperparameter_tuning(self, param_grid: dict = None, 
                                   n_trials: int = 3, episodes: int = 150) -> dict:
        """
        Run hyperparameter grid search.
        
        Returns best parameters and all results.
        """
        if not self.has_data():
            return {'success': False, 'error': 'No data available'}
        
        try:
            from src.rl.evaluation import hyperparameter_grid_search
            
            results = hyperparameter_grid_search(
                self.data,
                param_grid=param_grid,
                n_trials_per_config=n_trials,
                episodes=episodes,
                verbose=True
            )
            
            return {
                'success': True,
                'best_config': results['best_config'],
                'best_accuracy': results['best_accuracy'],
                'all_results': results['all_results'][:10]  # Top 10
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


# Global singleton instance
_data_manager = DataManager()

def get_data_manager() -> DataManager:
    """Get the global DataManager instance."""
    return _data_manager


def create_app() -> Dash:
    """Create the multi-page Dash application."""
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="dash_pages",
        external_stylesheets=[
            dbc.themes.FLATLY,  # Modern, clean theme
            dbc.icons.FONT_AWESOME,
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
        ],
        suppress_callback_exceptions=True,
        title="Kenya CBC Pathway System",
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ]
    )
    
    # Enable Dash Dev Tools (shows the toolbar at bottom with Errors, Callbacks, etc.)
    app.enable_dev_tools(
        dev_tools_ui=True,           # Show the dev tools UI
        dev_tools_props_check=True,  # Check component props
        dev_tools_serve_dev_bundles=True,  # Serve development bundles
        dev_tools_hot_reload=False,  # Disabled: prevents state loss and pytest conflicts
        dev_tools_prune_errors=True  # Prune error messages
    )
    
    # Custom CSS for navbar layout (full width content)
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                * { font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
                html, body { height: 100%; margin: 0; padding: 0; }
                #react-entry-point { height: 100%; }
                .card { border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .btn { font-weight: 500; }
                
                /* Top Navbar */
                .top-navbar {
                    background: linear-gradient(90deg, #1E3A8A 0%, #1D4ED8 100%);
                    padding: 0.75rem 1.5rem;
                    position: sticky;
                    top: 0;
                    z-index: 1000;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    flex-wrap: nowrap;
                    min-height: 56px;
                }
                
                /* Brand */
                .navbar-brand {
                    color: white;
                    font-weight: 600;
                    font-size: 1.1rem;
                    text-decoration: none;
                    white-space: nowrap;
                    pointer-events: none;
                    cursor: default;
                    margin-right: 2rem;
                }
                
                /* Nav Links Container */
                .nav-links {
                    display: flex;
                    gap: 0.5rem;
                    flex-wrap: nowrap;
                }
                
                /* Nav Links */
                .nav-links .nav-link {
                    color: rgba(255,255,255,0.7) !important;
                    padding: 0.5rem 1rem !important;
                    border-radius: 6px;
                    font-weight: 500;
                    font-size: 0.9rem;
                    text-decoration: none;
                    white-space: nowrap;
                    transition: all 0.2s;
                }
                .nav-links .nav-link:hover {
                    color: #fff !important;
                    background: rgba(255,255,255,0.1);
                }
                .nav-links .nav-link.active {
                    color: #fff !important;
                    background: rgba(96, 165, 250, 0.35);
                }
                
                /* Status Badges */
                .status-container {
                    display: flex;
                    gap: 0.5rem;
                    margin-left: auto;
                }
                .status-badge {
                    font-size: 0.75rem;
                    padding: 0.3rem 0.6rem;
                    border-radius: 4px;
                    white-space: nowrap;
                }
                
                /* Main Content */
                .main-content {
                    background: #F8FAFC;
                    min-height: calc(100vh - 56px);
                    width: 100%;
                }
                
                /* Responsive */
                @media (max-width: 768px) {
                    .top-navbar { flex-wrap: wrap; padding: 0.5rem; }
                    .nav-links { width: 100%; justify-content: center; margin-top: 0.5rem; }
                    .status-container { display: none; }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    app.layout = html.Div([
        # Session stores
        dcc.Store(id='selected-student-store', storage_type='session'),
        dcc.Store(id='data-status-store', storage_type='session'),
        dcc.Store(id='model-status-store', storage_type='session'),
        
        # Location for URL
        dcc.Location(id='url', refresh=False),
        
        # Top Navbar
        html.Nav([
            # Brand
            html.Span("🎓 Kenya CBC", className="navbar-brand"),
            
            # Navigation links
            html.Div([
                dcc.Link("Home", href="/", className="nav-link", id="nav-home"),
                dcc.Link("Analysis", href="/analysis", className="nav-link", id="nav-analysis"),
                dcc.Link("Teacher", href="/teacher", className="nav-link", id="nav-teacher"),
                dcc.Link("Advanced", href="/advanced", className="nav-link", id="nav-advanced"),
                dcc.Link("About", href="/about", className="nav-link", id="nav-about"),
            ], className="nav-links"),
            
            # Status badges (right side)
            html.Div(id='navbar-status', className="status-container")
            
        ], className="top-navbar"),
        
        # Main content area (full width)
        html.Div([
            page_container
        ], className="main-content p-4")
    ])
    
    # Register global callbacks
    _register_global_callbacks(app)
    
    return app


def _register_global_callbacks(app):
    """Register callbacks that work across all pages."""
    from dash import callback, Input, Output, State, no_update
    
    @callback(
        Output('navbar-status', 'children'),
        [Input('url', 'pathname'),
         Input('data-status-store', 'data'),
         Input('model-status-store', 'data')]
    )
    def update_navbar_status(pathname, _data_trigger, _model_trigger):
        dm = get_data_manager()
        
        badges = []
        
        if dm.has_data():
            badges.append(
                dbc.Badge([
                    html.I(className="fas fa-database me-1"),
                    f"{dm.get_student_count()}"
                ], color="success", className="status-badge me-2")
            )
        else:
            badges.append(
                dbc.Badge([
                    html.I(className="fas fa-database me-1"),
                    "No Data"
                ], color="danger", className="status-badge me-2")
            )
        
        if dm.has_model():
            acc = dm.model_accuracy or 0
            badges.append(
                dbc.Badge([
                    html.I(className="fas fa-brain me-1"),
                    f"{acc*100:.1f}%"
                ], color="success", className="status-badge me-2")
            )
        else:
            badges.append(
                dbc.Badge([
                    html.I(className="fas fa-brain me-1"),
                    "—"
                ], color="warning", className="status-badge me-2")
            )
        
        if dm.overrides or dm.hitl.approved_count > 0:
            total_overrides = len(dm.overrides) + dm.hitl.approved_count
            badges.append(
                dbc.Badge([
                    html.I(className="fas fa-user-edit me-1"),
                    f"{total_overrides}"
                ], color="info", className="status-badge")
            )
        
        if dm.hitl.pending_count > 0:
            badges.append(
                dbc.Badge([
                    html.I(className="fas fa-clock me-1"),
                    f"{dm.hitl.pending_count} pending"
                ], color="warning", className="status-badge")
            )
        
        return badges
    
    # Add active state to nav links
    @callback(
        [Output('nav-home', 'className'),
         Output('nav-analysis', 'className'),
         Output('nav-teacher', 'className'),
         Output('nav-advanced', 'className'),
         Output('nav-about', 'className')],
        Input('url', 'pathname')
    )
    def update_active_nav(pathname):
        base = "nav-link"
        active = "nav-link active"
        n = 5
        
        mapping = {
            '/': 0,
            '/analysis': 1,
            '/teacher': 2,
            '/advanced': 3,
            '/about': 4,
        }
        
        result = [base] * n
        idx = mapping.get(pathname, 0 if pathname is None else -1)
        if 0 <= idx < n:
            result[idx] = active
        return tuple(result)