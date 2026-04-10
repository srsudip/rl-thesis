"""
Real Data Loader & Backward Simulator
=====================================
Loads real student CSV data (e.g. gradess1.csv) and optionally simulates
earlier grades (G4-G6) from observed G7-G9 trajectories.

Produces the same data structures expected by the dashboard:
  - assessments (wide format with subject_score columns per student/grade)
  - competencies (derived from subject scores)
  - pathways (KNEC cluster weights + cosine similarity)

Also provides a CSV upload parser that auto-detects column structure.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

_log = logging.getLogger(__name__)

# ============================================================================
# SUBJECT MAPPING: Real CSV column names → Internal subject codes
# ============================================================================

# Maps the real CSV subject name stems to our internal codes
SUBJECT_NAME_MAP = {
    'math': 'MATH',
    'mathematics': 'MATH',
    'english': 'ENG',
    'kiswahili': 'KIS_KSL',
    'integrated': 'INT_SCI',      # "Integrated_Grade7" → INT_SCI
    'integrated_science': 'INT_SCI',
    'agriculture': 'AGRI',
    'social': 'SOC_STUD',         # "Social_Grade7" → SOC_STUD
    'social_studies': 'SOC_STUD',
    'cre': 'REL_CRE',
    'creative_arts_and_sports': 'CRE_ARTS',
    'creative_arts': 'CRE_ARTS',
    'pretechnical': 'PRE_TECH',
    'pre_technical': 'PRE_TECH',
    'science_technology': 'SCI_TECH',
    'sci_tech': 'SCI_TECH',
}

# Pathway label normalization
PATHWAY_LABEL_MAP = {
    'stem': 'STEM',
    'social science': 'SOCIAL_SCIENCES',
    'social sciences': 'SOCIAL_SCIENCES',
    'social_sciences': 'SOCIAL_SCIENCES',
    'arts and sports': 'ARTS_SPORTS',
    'arts & sports': 'ARTS_SPORTS',
    'arts_sports': 'ARTS_SPORTS',
    'arts and sports science': 'ARTS_SPORTS',
}

# Internal subject codes used by the system
ALL_SUBJECTS_JS = ['MATH', 'ENG', 'KIS_KSL', 'INT_SCI', 'AGRI', 'SOC_STUD',
                   'REL_CRE', 'CRE_ARTS', 'PRE_TECH']

# Derived subjects for Grade 7-9 (generated from base subjects)
DERIVED_SUBJECTS_JS = {
    'HEALTH_ED': ('INT_SCI', 'CRE_ARTS', 0.6, 0.4),
    'BUS_STUD': ('SOC_STUD', 'MATH', 0.6, 0.4),
    'LIFE_SKILLS': ('SOC_STUD', 'REL_CRE', 0.5, 0.5),
    'SPORTS_PE': ('CRE_ARTS', 'INT_SCI', 0.7, 0.3),
}

# Upper primary subjects (Grade 4-6) - different naming
ALL_SUBJECTS_UP = ['MATH', 'ENG', 'KIS_KSL', 'SCI_TECH', 'AGRI', 'SOC_STUD',
                   'REL_CRE', 'CRE_ARTS']


# ============================================================================
# CSV PARSER: Auto-detect column structure
# ============================================================================

def parse_csv_auto(filepath: str) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Parse a CSV file and auto-detect subject columns, grade levels, and pathway column.
    
    Supports formats like:
      - Math_Grade7, Math_Grade8, ... (gradess1.csv style)
      - MATH_score (grade in separate column)
      
    Returns:
      - df: parsed DataFrame
      - subjects_found: list of detected subject names
      - pathway_col: name of the pathway column (or None)
    """
    df = pd.read_csv(filepath)
    
    # Detect pathway column
    pathway_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in ['pathway', 'track', 'stream']):
            pathway_col = col
            break
    
    # Detect subject_grade pattern (e.g., "Math_Grade7")
    grade_pattern = re.compile(r'^(.+?)_?[Gg]rade_?(\d+)$')
    subjects_found = set()
    grades_found = set()
    
    for col in df.columns:
        match = grade_pattern.match(col)
        if match:
            subj_raw = match.group(1).strip().rstrip('_')
            grade = int(match.group(2))
            subjects_found.add(subj_raw)
            grades_found.add(grade)
    
    subjects_found = sorted(subjects_found)
    grades_found = sorted(grades_found)
    
    errors = []
    if not subjects_found:
        errors.append("No subject columns detected (expected pattern: Subject_GradeN)")
    if not grades_found:
        errors.append("No grade levels detected")
    if pathway_col and df[pathway_col].isna().all():
        errors.append(f"Pathway column '{pathway_col}' is entirely empty")
    
    # Check for missing values in score columns
    score_cols = [c for c in df.columns if grade_pattern.match(c)]
    missing = df[score_cols].isna().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        errors.append(f"Missing values in {len(cols_with_missing)} columns: {list(cols_with_missing.index[:5])}")
    
    # Score range validation
    for col in score_cols:
        vals = df[col].dropna()
        if vals.min() < 0 or vals.max() > 100:
            errors.append(f"Column {col}: scores out of range [0,100] (min={vals.min()}, max={vals.max()})")
    
    if errors:
        for e in errors:
            _log.warning("CSV parsing: %s", e)

    _log.info("Parsed: %d students, %d subjects, grades %s",
              len(df), len(subjects_found), grades_found)
    if pathway_col:
        _log.info("Pathway column: '%s' → %s",
                  pathway_col, df[pathway_col].value_counts().to_dict())
    
    return df, subjects_found, pathway_col


# ============================================================================
# CONVERT REAL CSV → INTERNAL FORMAT
# ============================================================================

def real_csv_to_internal(filepath: str, simulate_g4_g6: bool = True,
                         seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Convert a real CSV (e.g. gradess1.csv) to the internal dashboard format.
    
    Returns dict with keys: 'students', 'assessments', 'competencies', 'pathways',
                            'raw_data', 'pathway_labels'
    """
    df_raw, subjects_raw, pathway_col = parse_csv_auto(filepath)
    rng = np.random.default_rng(seed)
    
    # --- Step 1: Reshape to long format (student_id, grade, subject, score) ---
    grade_pattern = re.compile(r'^(.+?)_?[Gg]rade_?(\d+)$')

    # Pre-compute column → (internal_code, grade) mapping once (not per row)
    col_meta: Dict[str, tuple] = {}
    for col in df_raw.columns:
        m = grade_pattern.match(col)
        if m:
            subj_raw = m.group(1).strip().rstrip('_')
            grade_num = int(m.group(2))
            subj_key = subj_raw.lower().replace(' ', '_')
            col_meta[col] = (SUBJECT_NAME_MAP.get(subj_key, subj_key.upper()), grade_num)

    # Melt all score columns at once (vectorized — no per-row Python loop)
    score_cols = list(col_meta.keys())
    df_melt = df_raw[score_cols].copy()
    df_melt.index = np.arange(1, len(df_raw) + 1)   # 1-indexed student_id
    df_melt.index.name = 'student_id'
    df_melt = df_melt.reset_index().melt(id_vars='student_id', var_name='col', value_name='score')
    df_melt = df_melt.dropna(subset=['score'])
    df_melt['score'] = np.clip(df_melt['score'].astype(float), 1.0, 100.0)
    df_melt['subject'] = df_melt['col'].map(lambda c: col_meta[c][0])
    df_melt['grade'] = df_melt['col'].map(lambda c: col_meta[c][1]).astype(int)
    df_long = df_melt[['student_id', 'grade', 'subject', 'score']].reset_index(drop=True)
    
    # --- Step 2: Detect available grades ---
    available_grades = sorted(df_long['grade'].unique())
    _log.info("Available grades: %s", available_grades)
    
    # --- Step 3: Extract pathway labels ---
    pathway_labels = None
    if pathway_col:
        labels_raw = df_raw[pathway_col].astype(str).str.strip().str.lower()
        labels_mapped = labels_raw.map(lambda x: PATHWAY_LABEL_MAP.get(x, x.upper()))
        pathway_labels = dict(zip(np.arange(1, len(df_raw) + 1), labels_mapped))
    
    # --- Step 4: Simulate G4-G6 if requested and not present ---
    if simulate_g4_g6:
        missing_grades = [g for g in [4, 5, 6] if g not in available_grades]
        if missing_grades:
            _log.info("Simulating grades %s from available data...", missing_grades)
            df_simulated = simulate_earlier_grades(df_long, missing_grades, rng)
            df_long = pd.concat([df_simulated, df_long], ignore_index=True)
            df_long = df_long.sort_values(['student_id', 'grade', 'subject']).reset_index(drop=True)
    
    # --- Step 5: Convert to wide format (assessments) ---
    df_assessments = df_long.pivot_table(
        index=['student_id', 'grade'],
        columns='subject',
        values='score'
    ).reset_index()
    
    # Flatten column names
    df_assessments.columns = [
        f"{c}_score" if c not in ['student_id', 'grade'] else c
        for c in df_assessments.columns
    ]
    
    # --- Step 6: Add derived subjects for G7-9 ---
    js_mask = df_assessments['grade'].isin([7, 8, 9])
    for derived, (src_a, src_b, wa, wb) in DERIVED_SUBJECTS_JS.items():
        col_a = f"{src_a}_score"
        col_b = f"{src_b}_score"
        derived_col = f"{derived}_score"
        if derived_col not in df_assessments.columns:
            df_assessments[derived_col] = np.nan
        if col_a in df_assessments.columns and col_b in df_assessments.columns:
            vals = (df_assessments[col_a].fillna(50) * wa +
                    df_assessments[col_b].fillna(50) * wb +
                    rng.normal(0, 2.0, len(df_assessments)))
            df_assessments.loc[js_mask, derived_col] = np.clip(vals[js_mask], 1, 100)
    
    # --- Step 7: Derive competency scores from subject scores ---
    df_competencies = derive_competencies_from_subjects(df_assessments, rng)
    
    # --- Step 8: Build student list ---
    student_ids = sorted(df_long['student_id'].unique())
    df_students = pd.DataFrame({'student_id': student_ids})
    
    # Add names if available
    if 'Name' in df_raw.columns or 'name' in df_raw.columns:
        name_col = 'Name' if 'Name' in df_raw.columns else 'name'
        name_map = dict(enumerate(df_raw[name_col].astype(str), start=1))
        df_students['name'] = df_students['student_id'].map(name_map)
    
    all_grades = sorted(df_long['grade'].unique())
    _log.info("Final dataset: %d students, grades %s, %d subject columns",
              len(student_ids), all_grades, len(df_assessments.columns) - 2)
    
    return {
        'students': df_students,
        'profiles': df_students,
        'assessments': df_assessments,
        'competencies': df_competencies,
        'pathway_labels': pathway_labels,
        'raw_data': df_raw,
        'raw_long': df_long,
    }


# ============================================================================
# BACKWARD SIMULATION: Generate G4-G6 from G7-G9
# ============================================================================

def simulate_earlier_grades(df_long: pd.DataFrame, target_grades: List[int],
                            rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate earlier grade scores by backward extrapolation from observed data.
    
    Strategy:
    1. Compute per-student, per-subject growth rate from observed grades
    2. Extrapolate backward with decay + noise
    3. Ensure plausible ranges (earlier grades slightly lower on average)
    """
    students = df_long['student_id'].unique()
    subjects = df_long['subject'].unique()
    observed_grades = sorted(df_long['grade'].unique())
    min_observed = min(observed_grades)
    
    target_grades_back = [tg for tg in target_grades if tg < min_observed]
    if not target_grades_back:
        return pd.DataFrame(columns=['student_id', 'grade', 'subject', 'score'])

    records = []

    # groupby replaces the O(n) per-student DataFrame filter scan
    for (sid, subj), grp in df_long.groupby(['student_id', 'subject'], sort=False):
        grp_sorted   = grp.sort_values('grade')
        grades_obs   = grp_sorted['grade'].values.astype(float)
        scores_obs   = grp_sorted['score'].values.astype(float)
        first_score  = scores_obs[0] if len(scores_obs) > 0 else 50.0

        if len(grades_obs) >= 2:
            coeffs              = np.polyfit(grades_obs, scores_obs, 1)
            slope, intercept    = float(coeffs[0]), float(coeffs[1])
        else:
            slope, intercept    = 0.0, float(scores_obs.mean()) if len(scores_obs) > 0 else 50.0

        for tg in target_grades_back:
            distance  = min_observed - tg
            noise_std = 4.0 + distance * 1.5
            predicted = intercept + slope * tg
            sim_score = predicted + rng.normal(0, noise_std)
            sim_score = min(sim_score, first_score + rng.normal(3, 5))
            records.append({
                'student_id': sid, 'grade': tg, 'subject': subj,
                'score': float(np.clip(sim_score, 10, 95))
            })

    df_sim = pd.DataFrame(records)
    _log.info("Simulated %d records for grades %s", len(df_sim), target_grades_back)
    return df_sim


# ============================================================================
# COMPETENCY DERIVATION FROM SUBJECT SCORES
# ============================================================================

# CBC Competency → Contributing subjects (with weights)
COMPETENCY_SUBJECT_MAP = {
    'Communication_and_Collaboration': {
        'ENG': 0.4, 'KIS_KSL': 0.4, 'SOC_STUD': 0.2
    },
    'Critical_Thinking_and_Problem_Solving': {
        'MATH': 0.4, 'INT_SCI': 0.3, 'PRE_TECH': 0.3
    },
    'Creativity_and_Imagination': {
        'CRE_ARTS': 0.5, 'PRE_TECH': 0.3, 'ENG': 0.2
    },
    'Citizenship': {
        'SOC_STUD': 0.4, 'REL_CRE': 0.4, 'KIS_KSL': 0.2
    },
    'Digital_Literacy': {
        'MATH': 0.3, 'PRE_TECH': 0.4, 'INT_SCI': 0.3
    },
    'Learning_to_Learn': {
        'MATH': 0.25, 'ENG': 0.25, 'INT_SCI': 0.25, 'SOC_STUD': 0.25
    },
    'Self_Efficacy': {
        'CRE_ARTS': 0.3, 'AGRI': 0.3, 'REL_CRE': 0.2, 'SOC_STUD': 0.2
    },
}


def derive_competencies_from_subjects(df_assessments: pd.DataFrame,
                                      rng: np.random.Generator) -> pd.DataFrame:
    """
    Derive CBC competency scores from subject scores.
    Each competency is a weighted average of contributing subjects + noise.
    """
    comp_names = list(COMPETENCY_SUBJECT_MAP.keys())
    all_subjs  = sorted({s for w in COMPETENCY_SUBJECT_MAP.values() for s in w})

    # Weight matrix W: (n_comp, n_subj) — raw (unnormalized) weights
    W = np.zeros((len(comp_names), len(all_subjs)))
    for i, (comp, subj_weights) in enumerate(COMPETENCY_SUBJECT_MAP.items()):
        for subj, w in subj_weights.items():
            if subj in all_subjs:
                W[i, all_subjs.index(subj)] = w
    row_sums = W.sum(axis=1, keepdims=True)
    W_norm   = np.where(row_sums > 0, W / row_sums, 0.0)   # (n_comp, n_subj)

    # Score matrix S: (n_rows, n_subj) — missing values filled with 50 (neutral)
    score_cols = [f"{s}_score" for s in all_subjs]
    S = df_assessments.reindex(columns=['student_id', 'grade'] + score_cols)
    scores_mat = S[score_cols].fillna(50.0).values   # (n_rows, n_subj)

    # Vectorized weighted average + noise: (n_rows, n_comp)
    comp_mat = scores_mat @ W_norm.T + rng.normal(0.0, 3.0, (len(scores_mat), len(comp_names)))
    comp_mat = np.clip(comp_mat, 1.0, 100.0)

    result = df_assessments[['student_id', 'grade']].reset_index(drop=True).copy()
    for i, comp_name in enumerate(comp_names):
        result[f'{comp_name}_score'] = comp_mat[:, i]
    return result


# ============================================================================
# COSINE SIMILARITY: Student profile alignment with pathway ideals
# ============================================================================

# Ideal pathway profile vectors (relative importance weights, not scores)
PATHWAY_IDEAL_PROFILES = {
    'STEM': {
        'MATH': 1.0, 'INT_SCI': 1.0, 'PRE_TECH': 0.9,
        'AGRI': 0.5, 'ENG': 0.4, 'KIS_KSL': 0.2,
        'SOC_STUD': 0.1, 'REL_CRE': 0.1, 'CRE_ARTS': 0.1
    },
    'SOCIAL_SCIENCES': {
        'SOC_STUD': 1.0, 'ENG': 0.9, 'KIS_KSL': 0.9,
        'REL_CRE': 0.6, 'AGRI': 0.3, 'MATH': 0.3,
        'INT_SCI': 0.2, 'CRE_ARTS': 0.2, 'PRE_TECH': 0.1
    },
    'ARTS_SPORTS': {
        'CRE_ARTS': 1.0, 'KIS_KSL': 0.5, 'ENG': 0.5,
        'SOC_STUD': 0.3, 'REL_CRE': 0.3, 'AGRI': 0.2,
        'MATH': 0.1, 'INT_SCI': 0.1, 'PRE_TECH': 0.1
    },
}


def compute_cosine_similarity(student_scores: dict, pathway: str,
                               subject_order: List[str] = None) -> float:
    """
    Compute cosine similarity between a student's score vector and
    the ideal pathway profile vector.
    
    Returns value in [0, 1] where 1 = perfect alignment.
    """
    if subject_order is None:
        subject_order = ALL_SUBJECTS_JS
    
    ideal = PATHWAY_IDEAL_PROFILES.get(pathway, {})
    
    # Build vectors
    student_vec = np.array([student_scores.get(s, 0) for s in subject_order], dtype=float)
    ideal_vec = np.array([ideal.get(s, 0) for s in subject_order], dtype=float)
    
    # Weight the student scores by ideal importance (emphasize pathway-relevant subjects)
    weighted_student = student_vec * ideal_vec
    
    # Cosine similarity
    dot = np.dot(weighted_student, ideal_vec)
    norm_s = np.linalg.norm(weighted_student)
    norm_i = np.linalg.norm(ideal_vec)
    
    if norm_s == 0 or norm_i == 0:
        return 0.0
    
    return float(dot / (norm_s * norm_i))


def compute_all_cosine_similarities(student_scores: dict) -> dict:
    """Compute cosine similarity for all three pathways."""
    return {
        pw: compute_cosine_similarity(student_scores, pw)
        for pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
    }


# ============================================================================
# PATHWAY STRENGTH INDEX (PSI)
# ============================================================================

def compute_psi(student_scores: dict, pathway: str,
                cluster_weight: float, cluster_weight_percentile: float,
                growth_rates: dict = None) -> float:
    """
    Pathway Strength Index: How strong is this student within this pathway?
    
    PSI = 0.35 × CW_percentile + 0.25 × core_margin + 0.20 × cosine_sim + 0.20 × growth_trend
    
    Returns value in [0, 1].
    """
    from config.pathways import PATHWAY_CORE_SUBJECTS
    
    # Component 1: Cluster weight percentile (already 0-1)
    cw_pct = np.clip(cluster_weight_percentile, 0, 1)
    
    # Component 2: Core subject margin above AE1 (31%)
    core_cfg = PATHWAY_CORE_SUBJECTS.get(pathway, {})
    margins = []
    
    for subj in core_cfg.get('required', []):
        score = student_scores.get(subj, 0)
        margins.append(max(0, score - 31) / 69)  # normalized margin above 31
    
    for subj_list_key in ['required_one_of', 'at_least_one']:
        subj_list = core_cfg.get(subj_list_key, [])
        if subj_list:
            best = max(student_scores.get(s, 0) for s in subj_list)
            margins.append(max(0, best - 31) / 69)
    
    core_margin = np.mean(margins) if margins else 0.0
    
    # Component 3: Cosine similarity
    cos_sim = compute_cosine_similarity(student_scores, pathway)
    
    # Component 4: Growth trend (if available)
    if growth_rates:
        relevant_subjects = list(PATHWAY_IDEAL_PROFILES.get(pathway, {}).keys())[:5]
        growth_vals = [growth_rates.get(s, 0) for s in relevant_subjects]
        # Positive growth = good, normalize
        avg_growth = np.mean(growth_vals) / 30  # ~30 points max growth
        growth_score = np.clip(avg_growth + 0.5, 0, 1)  # center at 0.5
    else:
        growth_score = 0.5
    
    psi = 0.35 * cw_pct + 0.25 * core_margin + 0.20 * cos_sim + 0.20 * growth_score
    return float(np.clip(psi, 0, 1))


# ============================================================================
# PATHWAY SUITABILITY AT EACH GRADE (for history timeline)
# ============================================================================

def compute_pathway_suitability_by_grade(df_assessments: pd.DataFrame,
                                          student_id: int,
                                          pop_stats: dict = None) -> pd.DataFrame:
    """
    Compute pathway eligibility and strength at each grade level.
    Returns DataFrame with grade, pathway, eligible, cluster_weight, cosine_sim, psi.
    """
    from config.pathways import compute_cluster_weights, _check_core_subjects, PATHWAY_MIN_THRESHOLD
    
    student_data = df_assessments[df_assessments['student_id'] == student_id]
    grades = sorted(student_data['grade'].unique())
    
    # Use provided pop_stats or compute from Grade 9 (matching main recommendation)
    if pop_stats is None:
        max_grade = df_assessments['grade'].max()
        g9_data = df_assessments[df_assessments['grade'] == max_grade]
        pop_stats = {}
        for col in g9_data.columns:
            if col.endswith('_score') and col not in ['student_id', 'grade']:
                key = col.replace('_score', '')
                vals = g9_data[col].dropna()
                if len(vals) > 1:
                    pop_stats[key] = {'mean': float(vals.mean()), 'std': float(vals.std())}
    
    records = []
    prev_scores = {}
    
    for grade in grades:
        row = student_data[student_data['grade'] == grade]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        
        # Extract subject scores
        subject_scores = {}
        for col in row.index:
            if col.endswith('_score') and col not in ['student_id', 'grade']:
                key = col.replace('_score', '')
                val = row[col]
                if pd.notna(val):
                    subject_scores[key] = float(val)
        
        # Compute growth rates from previous grade
        growth_rates = {}
        if prev_scores:
            for subj in subject_scores:
                if subj in prev_scores and prev_scores[subj] > 0:
                    growth_rates[subj] = subject_scores[subj] - prev_scores[subj]
        
        # Compute cluster weights
        cw = compute_cluster_weights(subject_scores, pop_stats=pop_stats)
        
        # Cosine similarities
        cos_sims = compute_all_cosine_similarities(subject_scores)
        
        # All cluster weights for percentile computation (simplified: use this student's CW relative to thresholds)
        for pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']:
            threshold = PATHWAY_MIN_THRESHOLD.get(pw, 25)
            eligible = cw[pw] >= threshold
            
            # Check core subjects
            core_ok = _check_core_subjects(pw, subject_scores)
            
            # Simplified percentile (CW as fraction of max possible)
            cw_pct = min(cw[pw] / 100.0, 1.0)
            
            psi = compute_psi(
                subject_scores, pw,
                cluster_weight=cw[pw],
                cluster_weight_percentile=cw_pct,
                growth_rates=growth_rates
            )
            
            records.append({
                'grade': grade,
                'pathway': pw,
                'eligible': eligible and core_ok,
                'cluster_weight': round(cw[pw], 2),
                'cosine_sim': round(cos_sims[pw], 3),
                'psi': round(psi, 3),
                'core_check': core_ok,
            })
        
        prev_scores = subject_scores.copy()
    
    return pd.DataFrame(records)


# ============================================================================
# SUGGESTED PATHWAY PER GRADE (for performance history chart)
# ============================================================================

def get_suggested_pathway_per_grade(df_assessments: pd.DataFrame,
                                     student_id: int,
                                     pop_stats: dict = None) -> dict:
    """
    Returns {grade: 'STEM'/'SOCIAL_SCIENCES'/'ARTS_SPORTS'} for each available grade.
    
    CRITICAL: Must use the same logic AND same pop_stats as recommend_pathway()
    so that Grade 9 suggestion matches the actual Recommended Pathway display.
    
    Args:
        pop_stats: MUST be the same Grade-9 pop_stats used by the main recommendation.
    """
    from config.pathways import compute_cluster_weights, recommend_pathway
    
    student_data = df_assessments[df_assessments['student_id'] == student_id]
    grades = sorted(student_data['grade'].unique())
    
    # If no pop_stats provided, compute from Grade 9 only (matching main recommendation)
    if pop_stats is None:
        max_grade = df_assessments['grade'].max()
        g9_data = df_assessments[df_assessments['grade'] == max_grade]
        pop_stats = {}
        for col in g9_data.columns:
            if col.endswith('_score') and col not in ['student_id', 'grade']:
                key = col.replace('_score', '')
                vals = g9_data[col].dropna()
                if len(vals) > 1:
                    pop_stats[key] = {'mean': float(vals.mean()), 'std': float(vals.std())}
    
    result = {}
    for grade in grades:
        row = student_data[student_data['grade'] == grade]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        
        subject_scores = {}
        for col in row.index:
            if col.endswith('_score') and col not in ['student_id', 'grade']:
                key = col.replace('_score', '')
                val = row[col]
                if pd.notna(val):
                    subject_scores[key] = float(val)
        
        # Use multi-signal fusion matching the main recommendation pipeline
        # (must include PSI so Grade-9 result matches get_recommendation exactly)
        cw = compute_cluster_weights(subject_scores, pop_stats=pop_stats)
        cos_sims = compute_all_cosine_similarities(subject_scores)
        psi_scores = {
            pw: compute_psi(
                subject_scores, pw,
                cluster_weight=cw[pw],
                cluster_weight_percentile=min(cw[pw] / 100.0, 1.0),
            )
            for pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
        }
        rec = recommend_pathway(cw, subject_scores, cosine_sims=cos_sims, psi_scores=psi_scores)
        result[int(grade)] = rec['recommended_pathway']

    return result


# ============================================================================
# PATHWAY COMPARISON TABLE (for "I want a different pathway")
# ============================================================================

#  Cross-level subject aliases.  Junior-secondary CSVs (e.g. gradess1.csv)
#  carry INT_SCI but not the upper-primary SCI_TECH equivalent (and vice
#  versa for SPORTS_PE / PHE).  When checking gaps for one we transparently
#  fall back to the other so the comparison table never reports a +31 gap
#  on "Science and Technology" while the student already has Integrated
#  Science 68%.
SUBJECT_ALIASES = {
    'SCI_TECH':  'INT_SCI',
    'INT_SCI':   'SCI_TECH',
    'PHE':       'SPORTS_PE',
    'SPORTS_PE': 'PHE',
}


def _resolve_score(student_scores: dict, subj: str) -> tuple:
    """Return (score, source_subject) using cross-level aliases when needed."""
    if subj in student_scores:
        return float(student_scores[subj]), subj
    alias = SUBJECT_ALIASES.get(subj)
    if alias and alias in student_scores:
        return float(student_scores[alias]), alias
    return 0.0, subj


def generate_pathway_comparison(student_scores: dict, current_pathway: str,
                                 desired_pathway: str) -> pd.DataFrame:
    """
    Generate comparison table showing current scores vs. requirements for desired pathway.
    """
    from config.pathways import PATHWAY_CORE_SUBJECTS

    ideal = PATHWAY_IDEAL_PROFILES.get(desired_pathway, {})
    core_cfg = PATHWAY_CORE_SUBJECTS.get(desired_pathway, {})
    min_score = core_cfg.get('min_score', 31)

    #  Build a set of subject keys to display, deduping by alias so the
    #  same underlying competency is not listed twice (SCI_TECH + INT_SCI).
    def _canonical(subj: str) -> str:
        # Prefer the form that actually exists in student_scores.
        if subj in student_scores:
            return subj
        alias = SUBJECT_ALIASES.get(subj)
        if alias and alias in student_scores:
            return alias
        return subj

    relevant: dict[str, str] = {}   # canonical_key -> requirement_type

    def _add(subj: str, req_type: str):
        key = _canonical(subj)
        # 'Core' beats 'Language' beats 'Content' beats 'Supporting'.
        priority = {'Core': 3, 'Language': 2, 'Content': 2, 'Supporting': 1}
        if key not in relevant or priority[req_type] > priority[relevant[key]]:
            relevant[key] = req_type

    for subj in core_cfg.get('required', []):
        _add(subj, 'Core')

    #  required_one_of: at least one of the alternatives must clear the
    #  threshold.  Show only the alternative the student actually has
    #  (after alias resolution); if NONE of them are present we fall back
    #  to listing the first one so the user still sees the requirement.
    req_one = core_cfg.get('required_one_of', [])
    if req_one:
        present = [s for s in req_one
                   if s in student_scores or SUBJECT_ALIASES.get(s) in student_scores]
        for subj in (present or req_one[:1]):
            _add(subj, 'Language')

    for subj in core_cfg.get('at_least_one', []):
        _add(subj, 'Content')

    # Add top-weighted ideal-profile subjects (also alias-collapsed).
    for subj, _w in sorted(ideal.items(), key=lambda x: x[1], reverse=True)[:5]:
        key = _canonical(subj)
        relevant.setdefault(key, 'Supporting')

    rows = []
    for subj in sorted(relevant.keys()):
        current, _src = _resolve_score(student_scores, subj)
        requirement_type = relevant[subj]
        gap = max(0, min_score - current)
        status = '✓ Met' if current >= min_score else f'↑ Need +{gap:.0f}'
        rows.append({
            'Subject': subj,
            'Current Score': round(current, 1),
            'Required (≥AE1)': min_score,
            'Gap': round(gap, 1),
            'Status': status,
            'Requirement Type': requirement_type,
            'Pathway Weight': round(ideal.get(subj, ideal.get(SUBJECT_ALIASES.get(subj, ''), 0)), 2),
        })

    df = pd.DataFrame(rows)
    
    # Add cosine similarity comparison
    cos_current = compute_cosine_similarity(student_scores, current_pathway)
    cos_desired = compute_cosine_similarity(student_scores, desired_pathway)
    
    return df, {
        'current_pathway': current_pathway,
        'desired_pathway': desired_pathway,
        'cosine_current': round(cos_current, 3),
        'cosine_desired': round(cos_desired, 3),
    }


# ============================================================================
# MAIN ENTRY POINT: Load data for dashboard
# ============================================================================

def load_real_data_for_dashboard(filepath: str = None,
                                  simulate_g4_g6: bool = True,
                                  seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Main entry point: load real CSV data and prepare for dashboard.
    
    If filepath is None, looks for gradess1.csv in uploads or data directory.
    """
    from config.pathways import compute_cluster_weights, recommend_pathway
    
    if filepath is None:
        # Search in common locations
        candidates = [
            Path('/mnt/user-data/uploads/gradess1.csv'),
            Path(__file__).resolve().parent.parent.parent / 'data' / 'gradess1.csv',
        ]
        for c in candidates:
            if c.exists():
                filepath = str(c)
                break
    
    if filepath is None or not Path(filepath).exists():
        raise FileNotFoundError(f"No real data CSV found. Provide a filepath or place gradess1.csv in data/")
    
    _log.info("Loading real data from: %s", filepath)
    result = real_csv_to_internal(filepath, simulate_g4_g6=simulate_g4_g6, seed=seed)
    
    # --- Compute pathways using KNEC engine ---
    df_assessments = result['assessments']
    pathway_labels = result.get('pathway_labels', {})
    
    # Population stats for z-score normalization (use Grade 9)
    max_grade = df_assessments['grade'].max()
    g9_data = df_assessments[df_assessments['grade'] == max_grade]
    
    pop_stats = {}
    for col in g9_data.columns:
        if col.endswith('_score') and col not in ['student_id', 'grade']:
            key = col.replace('_score', '')
            vals = g9_data[col].dropna()
            if len(vals) > 1:
                pop_stats[key] = {'mean': float(vals.mean()), 'std': float(vals.std())}
    
    # Build score dicts per student without iterrows — use set_index + to_dict
    score_cols_g9 = [c for c in g9_data.columns
                     if c.endswith('_score') and c not in ('student_id', 'grade')]
    g9_indexed = g9_data.set_index('student_id')[score_cols_g9]

    pathway_records = []
    for sid, score_row in g9_indexed.iterrows():
        subject_scores = {
            col.replace('_score', ''): float(val)
            for col, val in score_row.items()
            if pd.notna(val)
        }

        cw = compute_cluster_weights(subject_scores, pop_stats=pop_stats)
        cos_sims = compute_all_cosine_similarities(subject_scores)
        rec = recommend_pathway(cw, subject_scores, cosine_sims=cos_sims)
        rec_pw = rec['recommended_pathway']

        pathway_records.append({
            'student_id': sid,
            'recommended_pathway': rec_pw,
            'confidence': round(rec.get('suitability_score', cw.get(rec_pw, 0)) / 100, 3),
            'STEM_score': cw['STEM'],
            'SOCIAL_SCIENCES_score': cw['SOCIAL_SCIENCES'],
            'ARTS_SPORTS_score': cw['ARTS_SPORTS'],
            'STEM_cosine': cos_sims['STEM'],
            'SS_cosine': cos_sims['SOCIAL_SCIENCES'],
            'ARTS_cosine': cos_sims['ARTS_SPORTS'],
            'actual_pathway': pathway_labels.get(sid, '') if pathway_labels else '',
        })
    
    result['pathways'] = pd.DataFrame(pathway_records)
    
    # Store pop_stats for reuse
    result['pop_stats'] = pop_stats
    
    # Log accuracy against actual labels
    if pathway_labels:
        df_pw = result['pathways']
        correct = (df_pw['recommended_pathway'] == df_pw['actual_pathway']).sum()
        total = len(df_pw)
        _log.info("KNEC recommendation accuracy vs actual labels: %d/%d = %.1f%%",
                  correct, total, 100 * correct / total)
        for pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']:
            actual = df_pw[df_pw['actual_pathway'] == pw]
            if len(actual) > 0:
                pw_correct = (actual['recommended_pathway'] == pw).sum()
                _log.info("  %s: %d/%d = %.1f%%", pw, pw_correct, len(actual),
                          100 * pw_correct / len(actual))
    
    return result


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    data = load_real_data_for_dashboard()
    print(f"\nStudents: {len(data['students'])}")
    print(f"Assessment shape: {data['assessments'].shape}")
    print(f"Competency shape: {data['competencies'].shape}")
    print(f"Pathway distribution:")
    print(data['pathways']['recommended_pathway'].value_counts())
    
    # Test cosine similarity for first student
    g9 = data['assessments'][data['assessments']['grade'] == 9]
    row = g9.iloc[0]
    scores = {col.replace('_score', ''): row[col] for col in g9.columns
              if col.endswith('_score') and col not in ['student_id', 'grade'] and pd.notna(row[col])}
    print(f"\nStudent 1 cosine sims: {compute_all_cosine_similarities(scores)}")
    
    # Test pathway history
    suit = compute_pathway_suitability_by_grade(data['assessments'], 1)
    print(f"\nStudent 1 pathway suitability by grade:")
    print(suit.to_string(index=False))
