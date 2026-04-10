"""
CBC Data Generator - IRT-based Latent Competency Model

This is the MAIN data generation module that consolidates all functionality.

Data Flow:
    Curriculum CSVs (data/) → simulate_all() → generated_*.csv (data/)
                                                      ↓
                                             load_dashboard_data()
                                                      ↓
                                                  Dashboard

IRT Model:
    - Latent competency: θ = α + β × (grade - min_grade)
    - Indicator score: score = 100 × logistic(a·θ - b)
"""

import asyncio
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

READ_KW = dict(sep=";", engine="python", encoding="utf-8-sig")

# Curriculum table cache
_curriculum_cache = {}

def _load_table(name: str) -> pd.DataFrame:
    if name not in _curriculum_cache:
        _curriculum_cache[name] = pd.read_csv(DATA_DIR / f"{name}.csv", **READ_KW)
    return _curriculum_cache[name]


# =============================================================================
# Helper Functions
# =============================================================================

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def to_assessment(score: float) -> int:
    if score < 25: return 1
    elif score < 50: return 2
    elif score < 75: return 3
    else: return 4


# =============================================================================
# Covariance Structure
# =============================================================================

def build_alpha_beta_cov(n_comp: int, sigma_alpha: float = 1.0, sigma_beta: float = 0.5,
                          corr_alpha: float = 0.3, corr_beta: float = 0.3,
                          corr_alpha_beta: float = -0.2) -> np.ndarray:
    K = n_comp
    Sigma_alpha = np.full((K, K), corr_alpha)
    np.fill_diagonal(Sigma_alpha, 1.0)
    Sigma_alpha *= sigma_alpha ** 2

    Sigma_beta = np.full((K, K), corr_beta)
    np.fill_diagonal(Sigma_beta, 1.0)
    Sigma_beta *= sigma_beta ** 2

    Sigma_ab = np.diag([corr_alpha_beta * sigma_alpha * sigma_beta] * K)
    return np.vstack([np.hstack([Sigma_alpha, Sigma_ab]), np.hstack([Sigma_ab.T, Sigma_beta])])


# =============================================================================
# Latent Competency Generation
# =============================================================================

def simulate_latent_competencies(n_students: int = 100, grades: tuple[int, ...] = (4, 5, 6),
                                   sigma_alpha: float = 0.7, sigma_beta: float = 0.35,
                                   corr_alpha: float = 0.3, corr_beta: float = 0.3,
                                   corr_alpha_beta: float = -0.2, random_state: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    comp_names = _load_table("competency_4_9")["competency_code"].tolist()
    n_comp = len(comp_names)

    Sigma = build_alpha_beta_cov(n_comp, sigma_alpha, sigma_beta, corr_alpha, corr_beta, corr_alpha_beta)
    ab = rng.multivariate_normal(np.zeros(2 * n_comp), Sigma, size=n_students)
    alpha, beta = ab[:, :n_comp], ab[:, n_comp:]  # (n_students, n_comp)

    min_grade    = min(grades)
    grade_offsets = np.array(grades) - min_grade   # (n_grades,)
    n_grades     = len(grades)

    # theta[s, g, k] = alpha[s,k] + beta[s,k] * grade_offsets[g]  — fully vectorized
    theta = (alpha[:, np.newaxis, :] +
             beta[:, np.newaxis, :] * grade_offsets[np.newaxis, :, np.newaxis])  # (n_stu, n_grades, n_comp)

    # Build flat index arrays without any Python loop
    stu_ids    = np.repeat(np.arange(1, n_students + 1), n_grades * n_comp)
    grade_arr  = np.tile(np.repeat(np.array(grades), n_comp), n_students)
    comp_arr   = np.tile(comp_names, n_students * n_grades)
    theta_flat = theta.ravel()
    alpha_flat = np.tile(alpha[:, np.newaxis, :], (1, n_grades, 1)).ravel()
    beta_flat  = np.tile(beta[:, np.newaxis, :],  (1, n_grades, 1)).ravel()

    return pd.DataFrame({
        "student_id": stu_ids,
        "grade":      grade_arr,
        "competency": comp_arr,
        "theta":      theta_flat,
        "alpha":      alpha_flat,
        "beta":       beta_flat,
    })


# =============================================================================
# Indicator Structure
# =============================================================================

def define_indicators_by_grade() -> pd.DataFrame:
    df_ind = _load_table("indicator_4_9")
    df_strand = _load_table("strand_4_9")
    df_subject = _load_table("subject_4_9")
    df_substrand = _load_table("substrand_4_9")
    df_si = _load_table("substrandindicator_4_9")
    df_cs = _load_table("competencysubstrand_4_9")
    df_comp = _load_table("competency_4_9")

    df_base = df_ind.merge(df_strand[["strand_id", "subject_id", "strand_code"]], on="strand_id", how="left")
    df_base = df_base.merge(df_subject[["subject_id", "grade", "subject_code"]], on="subject_id", how="left")

    df_si_m = df_si.merge(df_substrand[["substrand_id", "substrand_code"]], on="substrand_id", how="left")
    df_si_m = df_si_m[df_si_m["weight"] > 0].copy()

    df_main_sub = df_si_m.sort_values(["indicator_id", "substrand_id"]).groupby("indicator_id").first().reset_index()[["indicator_id", "substrand_code"]]

    df_ic = df_si_m.merge(df_cs, on="substrand_id", how="left").merge(df_comp[["competency_id", "competency_code"]], on="competency_id", how="left")
    comp_series = df_ic.dropna(subset=["competency_code"]).groupby("indicator_id")["competency_code"].apply(lambda x: sorted(set(x))).rename("competencies")

    result = df_base.merge(df_main_sub, on="indicator_id", how="left").merge(comp_series.reset_index(), on="indicator_id", how="left")
    result = result.rename(columns={"subject_code": "subject", "strand_code": "strand", "substrand_code": "substrand"})
    result["competencies"] = result["competencies"].apply(lambda x: x if isinstance(x, list) else [])
    return result


# =============================================================================
# IRT Parameter Generation
# =============================================================================

def simulate_indicator_parameters(df_ind: pd.DataFrame, a_mean: float = 1.0, a_sd: float = 0.3,
                                    b_sd: float = 0.5, random_state: int = 124) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    comp_names = _load_table("competency_4_9")["competency_code"].tolist()
    n_comp = len(comp_names)
    n_ind = len(df_ind)

    mean_diff: dict[str, float] = {
        # Upper Primary (Grade 4-6) subjects
        "MATH": 0.3, "SCI_TECH": 0.1, "ENG": 0.0, "KIS_KSL": 0.0,
        "SOC_STUD": -0.1, "CRE_ARTS": -0.2, "AGRI": -0.1,
        "REL_CRE": -0.15, "REL_IRE": -0.15, "REL_HRE": -0.15,
        "HOME_SCI": -0.1, "PHE": -0.25,
        # Junior Secondary (Grade 7-9) additional subjects
        "INT_SCI": 0.2, "PRE_TECH": 0.1, "HEALTH_ED": -0.2,
        "BUS_STUD": -0.05, "LIFE_SKILLS": -0.3, "SPORTS_PE": -0.25,
    }

    # Vectorized: build boolean mask (n_ind, n_comp) — True where competency applies
    comp_sets = [set(c) if isinstance(c, list) else set() for c in df_ind["competencies"]]
    mask = np.array([[c in cs for c in comp_names] for cs in comp_sets], dtype=bool)  # (n_ind, n_comp)

    # Generate all discrimination parameters at once, zero out non-applicable slots
    a_raw = rng.lognormal(mean=np.log(a_mean), sigma=a_sd, size=(n_ind, n_comp))
    a_mat = np.where(mask, a_raw, 0.0)  # (n_ind, n_comp)

    # Vectorized difficulty parameters — look up per-subject mean in one pass
    b_means = np.fromiter((mean_diff.get(s, 0.0) for s in df_ind["subject"]), dtype=float, count=n_ind)
    b_vec = rng.normal(b_means, b_sd)  # (n_ind,)

    df_ind = df_ind.copy()
    for k, cname in enumerate(comp_names):
        df_ind[f"a_{cname}"] = a_mat[:, k]
    df_ind["b"] = b_vec
    return df_ind


# =============================================================================
# Indicator Response Generation
# =============================================================================

def simulate_indicator_responses(df_theta: pd.DataFrame, df_ind: pd.DataFrame,
                                   score_noise_sd: float = 5.0, random_state: int = 2024) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    comp_names = _load_table("competency_4_9")["competency_code"].tolist()
    a_cols = [f"a_{c}" for c in comp_names]

    # Theta matrix: (n_students × n_grades, n_comp) — one row per (student, grade) pair
    theta_wide = (df_theta
                  .pivot_table(index=["student_id", "grade"], columns="competency", values="theta")
                  .reindex(columns=comp_names, fill_value=0.0))
    sg_index = theta_wide.index   # MultiIndex of (student_id, grade)
    Theta    = theta_wide.values  # (n_sg, n_comp)

    grade_dfs = []
    for grade in sorted(df_ind["grade"].unique()):
        ind_g = df_ind[df_ind["grade"] == grade].reset_index(drop=True)
        A = ind_g[a_cols].values  # (n_ind, n_comp)
        b = ind_g["b"].values     # (n_ind,)

        # Select theta rows for this grade
        grade_mask  = np.array([g == grade for _, g in sg_index])
        Theta_g     = Theta[grade_mask]                               # (n_stu, n_comp)
        student_ids = np.array([s for s, g in sg_index if g == grade])
        n_stu, n_ind = len(student_ids), len(ind_g)

        # Vectorized IRT: eta[i,j] = Theta_g[i] · A[j] - b[j]
        eta    = Theta_g @ A.T - b[np.newaxis, :]                    # (n_stu, n_ind)
        p      = logistic(eta)                                        # (n_stu, n_ind)
        scores = np.clip(rng.normal(100.0 * p, score_noise_sd), 1.0, 100.0)
        assess = np.where(scores < 25, 1,
                 np.where(scores < 50, 2,
                 np.where(scores < 75, 3, 4)))

        # Build DataFrame entirely from arrays — no per-row Python loop
        grade_dfs.append(pd.DataFrame({
            "student_id":     np.repeat(student_ids, n_ind),
            "grade":          grade,
            "indicator_id":   np.tile(ind_g["indicator_id"].values,   n_stu),
            "indicator_code": np.tile(ind_g["indicator_code"].values,  n_stu),
            "subject":        np.tile(ind_g["subject"].values,         n_stu),
            "strand":         np.tile(ind_g["strand"].values,          n_stu),
            "substrand":      np.tile(ind_g["substrand"].values,       n_stu),
            "p_true":         p.ravel(),
            "score":          scores.ravel(),
            "assessment":     assess.ravel(),
        }))

    return pd.concat(grade_dfs, ignore_index=True)


# =============================================================================
# Main Simulation
# =============================================================================

def simulate_all(n_students: int = 100, grades: tuple[int, ...] = (4, 5, 6, 7, 8, 9),
                 sigma_alpha: float = 0.7, sigma_beta: float = 0.35,
                 corr_alpha: float = 0.3, corr_beta: float = 0.3,
                 corr_alpha_beta: float = -0.2, score_noise_sd: float = 5.0,
                 random_state: int = 123, save_csv: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_theta = simulate_latent_competencies(n_students, grades, sigma_alpha, sigma_beta, corr_alpha, corr_beta, corr_alpha_beta, random_state)
    df_ind = simulate_indicator_parameters(define_indicators_by_grade(), random_state=random_state + 1)
    df_resp = simulate_indicator_responses(df_theta, df_ind, score_noise_sd, random_state + 2)

    if save_csv:
        df_theta.to_csv(DATA_DIR / "generated_theta.csv", index=False, encoding="utf-8-sig")
        df_ind.to_csv(DATA_DIR / "generated_ind.csv", index=False, encoding="utf-8-sig")
        df_resp.to_csv(DATA_DIR / "generated_resp.csv", index=False, encoding="utf-8-sig")

    return df_theta, df_ind, df_resp


# =============================================================================
# Pathway Weights
# =============================================================================

# Use shared config for pathway weights
from config.pathways import PATHWAY_SUBJECT_WEIGHTS, compute_cluster_weights, get_cbc_grade


# =============================================================================
# Dashboard Integration
# =============================================================================

def aggregate_to_subjects_wide(df_resp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate indicator responses to subject-level scores (wide format).

    Also derives CBC-required subjects that aren't in the curriculum CSVs
    but are needed for KJSEA result slips and pathway suitability:

    Grade 4-6 (Upper Primary) additions:
      HOME_SCI  ≈ blend of SCI_TECH + AGRI
      PHE       ≈ blend of CRE_ARTS + noise

    Grade 7-9 (Junior Secondary) additions:
      INT_SCI   = SCI_TECH renamed (Science & Tech → Integrated Science)
      HEALTH_ED ≈ blend of SCI_TECH + CRE_ARTS
      PRE_TECH  ≈ blend of SCI_TECH + MATH
      BUS_STUD  ≈ blend of SOC_STUD + MATH
      LIFE_SKILLS ≈ blend of SOC_STUD + REL_CRE
      SPORTS_PE ≈ blend of CRE_ARTS + noise
    """
    subject_means = df_resp.groupby(["student_id", "grade", "subject"])["score"].mean().reset_index()
    wide = subject_means.pivot_table(index=["student_id", "grade"], columns="subject", values="score").reset_index()
    wide.columns = [f"{c}_score" if c not in ["student_id", "grade"] else c for c in wide.columns]

    rng = np.random.default_rng(2024)

    def _noise(n: int) -> np.ndarray:
        return rng.normal(0, 3.0, n)  # type: ignore[return-value]

    def _safe(col: str) -> np.ndarray:
        return wide[col].values if col in wide.columns else np.full(len(wide), 50.0)

    def _blend(a: str, b: str, wa: float = 0.6, wb: float = 0.4) -> np.ndarray:
        return np.clip(_safe(a) * wa + _safe(b) * wb + _noise(len(wide)), 1, 100)

    # --- Grade 4-6 derived subjects ---
    up = wide['grade'].isin([4, 5, 6])
    if up.any():
        if 'HOME_SCI_score' not in wide.columns:
            wide['HOME_SCI_score'] = np.nan
        wide.loc[up, 'HOME_SCI_score'] = _blend('SCI_TECH_score', 'AGRI_score', 0.5, 0.5)[up]

        if 'PHE_score' not in wide.columns:
            wide['PHE_score'] = np.nan
        wide.loc[up, 'PHE_score'] = _blend('CRE_ARTS_score', 'SCI_TECH_score', 0.7, 0.3)[up]

    # --- Grade 7-9 derived subjects ---
    js = wide['grade'].isin([7, 8, 9])
    if js.any():
        # Integrated Science = Science & Technology (renamed at junior secondary)
        if 'INT_SCI_score' not in wide.columns:
            wide['INT_SCI_score'] = np.nan
        wide.loc[js, 'INT_SCI_score'] = _safe('SCI_TECH_score')[js]

        if 'HEALTH_ED_score' not in wide.columns:
            wide['HEALTH_ED_score'] = np.nan
        wide.loc[js, 'HEALTH_ED_score'] = _blend('SCI_TECH_score', 'CRE_ARTS_score', 0.6, 0.4)[js]

        if 'PRE_TECH_score' not in wide.columns:
            wide['PRE_TECH_score'] = np.nan
        wide.loc[js, 'PRE_TECH_score'] = _blend('SCI_TECH_score', 'MATH_score', 0.5, 0.5)[js]

        if 'BUS_STUD_score' not in wide.columns:
            wide['BUS_STUD_score'] = np.nan
        wide.loc[js, 'BUS_STUD_score'] = _blend('SOC_STUD_score', 'MATH_score', 0.6, 0.4)[js]

        # Religious ed: pick whichever variant exists
        re_col = None
        for c in ['REL_CRE_score', 'REL_IRE_score', 'REL_HRE_score']:
            if c in wide.columns:
                re_col = c
                break
        re_arr = _safe(re_col) if re_col else np.full(len(wide), 50.0)

        if 'LIFE_SKILLS_score' not in wide.columns:
            wide['LIFE_SKILLS_score'] = np.nan
        wide.loc[js, 'LIFE_SKILLS_score'] = np.clip(
            _safe('SOC_STUD_score') * 0.5 + re_arr * 0.5 + _noise(len(wide)), 1, 100)[js]

        if 'SPORTS_PE_score' not in wide.columns:
            wide['SPORTS_PE_score'] = np.nan
        wide.loc[js, 'SPORTS_PE_score'] = _blend('CRE_ARTS_score', 'SCI_TECH_score', 0.7, 0.3)[js]

    return wide

def theta_to_competencies_wide(df_theta: pd.DataFrame) -> pd.DataFrame:
    theta_wide = df_theta.pivot_table(index=["student_id", "grade"], columns="competency", values="theta").reset_index()
    result = theta_wide[["student_id", "grade"]].copy()
    for col in theta_wide.columns:
        if col not in ["student_id", "grade"]:
            result[f"{col}_score"] = np.clip((logistic(theta_wide[col]) * 100).round(1), 1.0, 100.0)
    return result

def determine_pathways(df_comp: pd.DataFrame, df_subj: pd.DataFrame) -> pd.DataFrame:
    """Determine pathways using KNEC-style pathway suitability percentages with z-score normalization."""
    max_grade = df_comp['grade'].max()
    g9 = df_subj[df_subj['grade'] == max_grade].copy().reset_index(drop=True)

    score_cols = [c for c in g9.columns
                  if c.endswith('_score') and c not in ('student_id', 'grade')]

    # Vectorized z-score normalisation across all students at once
    g9_norm = g9[score_cols].copy()
    for col in score_cols:
        vals = g9_norm[col].dropna()
        if len(vals) > 1:
            mu, sigma = vals.mean(), vals.std()
            if sigma > 0:
                g9_norm[col] = 50.0 + 10.0 * (g9_norm[col] - mu) / sigma

    # Vectorized cluster weight computation — matrix × weight vector per pathway
    cw_dict = {}
    for pw, pw_weights in PATHWAY_SUBJECT_WEIGHTS.items():
        numerator   = np.zeros(len(g9))
        denominator = np.zeros(len(g9))
        for subj, w in pw_weights.items():
            col = f"{subj}_score"
            # RE alias: fall back to INT_SCI ↔ SCI_TECH cross-level mapping
            if col not in g9_norm.columns:
                for alt in (f"SCI_TECH_score", f"INT_SCI_score"):
                    if alt in g9_norm.columns:
                        col = alt
                        break
            if col in g9_norm.columns:
                valid = g9_norm[col].notna()
                numerator   += np.where(valid, g9_norm[col].fillna(0) * w, 0)
                denominator += np.where(valid, w, 0)
        cw_dict[pw] = np.where(denominator > 0,
                               np.clip(numerator / denominator, 0, 100), 0.0)

    cw_df       = pd.DataFrame(cw_dict)           # (n_stu, 3)
    recommended = cw_df.idxmax(axis=1)
    confidence  = cw_df.max(axis=1) / 100.0

    return pd.DataFrame({
        'student_id':            g9['student_id'].values,
        'recommended_pathway':   recommended.values,
        'confidence':            confidence.round(3).values,
        'STEM_score':            cw_df['STEM'].round(2).values,
        'SOCIAL_SCIENCES_score': cw_df['SOCIAL_SCIENCES'].round(2).values,
        'ARTS_SPORTS_score':     cw_df['ARTS_SPORTS'].round(2).values,
    })


def check_generated_files_exist() -> bool:
    return (DATA_DIR / "generated_theta.csv").exists() and (DATA_DIR / "generated_resp.csv").exists()

def generate_dashboard_data(n_students: int = 100, grades: tuple[int, ...] = (4, 5, 6, 7, 8, 9),
                             seed: int = 42, save_csv: bool = True) -> dict[str, pd.DataFrame]:
    print(f"\n  Generating {n_students} students using IRT model...")
    df_theta, df_ind, df_resp = simulate_all(n_students=n_students, grades=grades, random_state=seed, save_csv=save_csv)
    if save_csv:
        print(f"    Saved to: {DATA_DIR}/generated_*.csv")

    df_assessments = aggregate_to_subjects_wide(df_resp)
    df_competencies = theta_to_competencies_wide(df_theta)
    df_pathways = determine_pathways(df_competencies, df_assessments)
    df_students = pd.DataFrame({'student_id': range(1, n_students + 1)})

    print(f"  ✓ Generated {n_students} students, {len(df_resp)} indicator assessments")
    return {'students': df_students, 'profiles': df_students, 'assessments': df_assessments,
            'competencies': df_competencies, 'pathways': df_pathways, 'indicators': df_resp,
            'theta': df_theta, 'indicator_params': df_ind}

def load_dashboard_data() -> dict[str, pd.DataFrame]:
    print("\n  Loading data from generated CSV files...")
    if not check_generated_files_exist():
        raise FileNotFoundError("Generated CSV files not found. Generate data first.")

    df_theta = pd.read_csv(DATA_DIR / "generated_theta.csv", encoding="utf-8-sig")
    df_resp = pd.read_csv(DATA_DIR / "generated_resp.csv", encoding="utf-8-sig")
    df_ind = pd.read_csv(DATA_DIR / "generated_ind.csv", encoding="utf-8-sig") if (DATA_DIR / "generated_ind.csv").exists() else None

    df_assessments = aggregate_to_subjects_wide(df_resp)
    df_competencies = theta_to_competencies_wide(df_theta)
    df_pathways = determine_pathways(df_competencies, df_assessments)
    df_students = pd.DataFrame({'student_id': df_theta['student_id'].unique()})

    print(f"  ✓ Loaded {len(df_students)} students from CSV files")
    return {'students': df_students, 'profiles': df_students, 'assessments': df_assessments,
            'competencies': df_competencies, 'pathways': df_pathways, 'indicators': df_resp,
            'theta': df_theta, 'indicator_params': df_ind}


async def generate_dashboard_data_async(
    n_students: int = 100,
    grades: tuple[int, ...] = (4, 5, 6, 7, 8, 9),
    seed: int = 42,
    save_csv: bool = True,
) -> dict[str, pd.DataFrame]:
    """Async wrapper — generates data in a thread pool so the event loop stays free."""
    return await asyncio.to_thread(generate_dashboard_data, n_students, grades, seed, save_csv)


async def load_dashboard_data_async() -> dict[str, pd.DataFrame]:
    """Async wrapper — loads CSV data in a thread pool so the event loop stays free."""
    return await asyncio.to_thread(load_dashboard_data)


if __name__ == "__main__":
    data = generate_dashboard_data(n_students=100, seed=42)
    print(f"\nPathway distribution:\n{data['pathways']['recommended_pathway'].value_counts()}")
