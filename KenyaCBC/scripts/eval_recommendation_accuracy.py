"""
Evaluate pathway recommendation accuracy on labelled gradess1.csv data.

Compares several recommender configurations against the `Actual Pathway`
column and prints a single summary table.  Used to drive Enhancements
A (filter-then-rank), B (logistic-learned weights) and C (margin filter)
in `config/pathways.recommend_pathway`.

Run:
    python scripts/eval_recommendation_accuracy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.real_data_loader import (
    real_csv_to_internal,
    compute_all_cosine_similarities,
)
from config.pathways import (
    compute_cluster_weights,
    recommend_pathway,
    PATHWAY_MIN_THRESHOLD,
    _check_core_subjects,
    _primary_key_subject_score,
    _composite_alignment_score,
)


PATHWAYS = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']


# ---------------------------------------------------------------------------
#  Per-config recommendation strategies
# ---------------------------------------------------------------------------

def _eligible(cw: dict, scores: dict) -> list[str]:
    out = []
    for pw in PATHWAYS:
        if cw.get(pw, 0) >= PATHWAY_MIN_THRESHOLD.get(pw, 25.0) and _check_core_subjects(pw, scores):
            out.append(pw)
    return out


def rec_baseline_cw_only(cw, scores, cos):
    """Pure CW ranking, no cosine, no PSI, no filter — pathological baseline."""
    return max(cw.items(), key=lambda x: x[1])[0]


def rec_current(cw, scores, cos):
    """Current production: composite (0.6 CW + 0.25 cos·100 + 0.15 PSI) over eligibles."""
    return recommend_pathway(cw, scores, cosine_sims=cos)['recommended_pathway']


def rec_bounded_cosine(cw, scores, cos):
    """Composite where cosine influence is clipped to ±5pt around CW."""
    elig = _eligible(cw, scores) or PATHWAYS
    def comp(pw):
        cw_v = cw.get(pw, 0)
        cos_v = cos.get(pw, 0) * 100
        nudge = max(-5.0, min(5.0, 0.10 * cos_v - 5.0))
        return cw_v + nudge
    return max(elig, key=comp)


def rec_filter_then_rank(cw, scores, cos):
    """Enhancement A — drop ineligibles first, then rank survivors by raw CW."""
    elig = _eligible(cw, scores)
    if not elig:
        elig = [pw for pw in PATHWAYS if _check_core_subjects(pw, scores)] or PATHWAYS
    elig.sort(key=lambda pw: cw.get(pw, 0), reverse=True)
    best = elig[0]
    if len(elig) > 1 and cw[elig[0]] - cw[elig[1]] < 5.0:
        ks0 = _primary_key_subject_score(elig[0], scores)
        ks1 = _primary_key_subject_score(elig[1], scores)
        if ks1 > ks0 + 5.0:
            best = elig[1]
    return best


CONFIGS = {
    'baseline (CW only, no filter)':         rec_baseline_cw_only,
    'current (composite 0.6/0.25/0.15)':     rec_current,
    'bounded cosine (±5pt nudge)':           rec_bounded_cosine,
    'filter-then-rank (Enh. A)':             rec_filter_then_rank,
}


# ---------------------------------------------------------------------------
#  Evaluation harness
# ---------------------------------------------------------------------------

def _eval_assessments(assessments: pd.DataFrame, labels: dict, dataset_name: str) -> pd.DataFrame:
    """Run all CONFIGS over a (assessments, labels) pair and print one summary table."""

    g9 = assessments[assessments['grade'] == assessments['grade'].max()]
    score_cols = [c for c in g9.columns if c.endswith('_score') and c not in ('student_id', 'grade')]

    pop_stats = {}
    for col in score_cols:
        vals = g9[col].dropna()
        if len(vals) > 1:
            pop_stats[col.replace('_score', '')] = {'mean': float(vals.mean()), 'std': float(vals.std())}

    g9_idx = g9.set_index('student_id')[score_cols]

    rows = []
    for sid, row in g9_idx.iterrows():
        actual = labels.get(sid)
        if not actual:
            continue
        scores = {c.replace('_score', ''): float(v) for c, v in row.items() if pd.notna(v)}
        cw = compute_cluster_weights(scores, pop_stats=pop_stats)
        cos = compute_all_cosine_similarities(scores)
        rec = {name: fn(cw, scores, cos) for name, fn in CONFIGS.items()}
        rec['_actual'] = actual
        rec['_sid'] = sid
        rows.append(rec)

    df = pd.DataFrame(rows)
    print(f"\n=== {dataset_name} ===")
    print(f"Evaluated {len(df)} labelled students")
    print(f"Label distribution: {dict(df['_actual'].value_counts())}\n")

    summary = []
    for name in CONFIGS:
        correct = (df[name] == df['_actual']).sum()
        acc = 100.0 * correct / len(df)
        per_pw = {}
        for pw in PATHWAYS:
            mask = df['_actual'] == pw
            if mask.sum() > 0:
                per_pw[pw] = round(100.0 * (df.loc[mask, name] == pw).sum() / mask.sum(), 1)
        summary.append({
            'config': name,
            'accuracy': round(acc, 1),
            'STEM_recall': per_pw.get('STEM', 0),
            'SS_recall': per_pw.get('SOCIAL_SCIENCES', 0),
            'ARTS_recall': per_pw.get('ARTS_SPORTS', 0),
            'correct': int(correct),
            'total': int(len(df)),
        })

    table = pd.DataFrame(summary)
    print(table.to_string(index=False))
    return table


def evaluate_real(filepath: str = None) -> pd.DataFrame:
    """Evaluate on the labelled gradess1.csv dataset (preference-noisy labels)."""
    if filepath is None:
        filepath = str(ROOT / 'data' / 'gradess1.csv')
    bundle = real_csv_to_internal(filepath, simulate_g4_g6=True, seed=42)
    labels = bundle.get('pathway_labels') or {}
    if not labels:
        raise RuntimeError(f"{filepath} has no actual pathway labels — cannot evaluate.")
    return _eval_assessments(bundle['assessments'], labels, f"REAL DATA — {Path(filepath).name}")


def evaluate_synthetic(n_students: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Evaluate on **theta-derived** labels from the IRT generator.

    Ground truth = argmax cosine(student_theta, IDEAL_PATHWAY_PROFILES) where
    theta is the *latent* CBC competency vector — independent of any subject
    score formula.  This avoids the circular sanity check where ground truth
    is itself a CW argmax.
    """
    import numpy as np
    from src.data.cbc_data_generator import generate_dashboard_data
    from config.competencies import IDEAL_PATHWAY_PROFILES, COMPETENCIES

    print(f"\nGenerating {n_students} synthetic students for theta-based eval...")
    data = generate_dashboard_data(n_students=n_students, seed=seed, save_csv=False)
    comp = data['competencies']
    max_g = comp['grade'].max()
    g9_comp = comp[comp['grade'] == max_g].set_index('student_id')

    comp_keys = list(COMPETENCIES.keys())
    score_cols_comp = [f'{k}_score' for k in comp_keys if f'{k}_score' in g9_comp.columns]
    comp_keys = [c.replace('_score', '') for c in score_cols_comp]

    ideal_mat = np.array([
        [IDEAL_PATHWAY_PROFILES[pw].get(k, 0) for k in comp_keys]
        for pw in PATHWAYS
    ], dtype=float)

    student_mat = g9_comp[score_cols_comp].to_numpy(dtype=float)
    s_norm = np.linalg.norm(student_mat, axis=1, keepdims=True)
    i_norm = np.linalg.norm(ideal_mat, axis=1, keepdims=True)
    cos = (student_mat @ ideal_mat.T) / (s_norm * i_norm.T + 1e-9)
    best_idx = cos.argmax(axis=1)

    labels = {int(sid): PATHWAYS[idx]
              for sid, idx in zip(g9_comp.index, best_idx)}
    return _eval_assessments(data['assessments'], labels,
                             f"SYNTHETIC IRT — theta ground truth, {n_students} students (seed={seed})")


def _reload_pathways_module():
    """Force-reload config.pathways so learned-weight env changes take effect."""
    import importlib
    import config.pathways as _p
    importlib.reload(_p)
    # rebind the module-level helpers used by CONFIGS / _eval_assessments
    global compute_cluster_weights, recommend_pathway, _check_core_subjects
    global _primary_key_subject_score, _composite_alignment_score, PATHWAY_MIN_THRESHOLD
    compute_cluster_weights      = _p.compute_cluster_weights
    recommend_pathway            = _p.recommend_pathway
    _check_core_subjects         = _p._check_core_subjects
    _primary_key_subject_score   = _p._primary_key_subject_score
    _composite_alignment_score   = _p._composite_alignment_score
    PATHWAY_MIN_THRESHOLD        = _p.PATHWAY_MIN_THRESHOLD
    return _p


def evaluate_learned(filepath: str = None, blend: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a third pass with LEARNED_WEIGHT_BLEND active to measure Enh. B."""
    import os
    os.environ['LEARNED_WEIGHT_BLEND'] = str(blend)
    _reload_pathways_module()
    print(f"\n### LEARNED_WEIGHT_BLEND = {blend}  (Enhancement B active) ###")
    real = evaluate_real(filepath)
    synth = evaluate_synthetic()
    return real, synth


def evaluate(filepath: str = None) -> dict:
    out = {}
    print("\n############################################################")
    print("###           HAND-AUTHORED KICD WEIGHTS (blend=0)       ###")
    print("############################################################")
    out['real_hand'] = evaluate_real(filepath)
    out['synthetic_hand'] = evaluate_synthetic()

    print("\n############################################################")
    print("###  BLEND=0.5 (50/50 hand + logistic-learned, Enh. B)   ###")
    print("############################################################")
    out['real_blend50'], out['synthetic_blend50'] = evaluate_learned(filepath, blend=0.5)

    print("\n############################################################")
    print("###  BLEND=1.0 (full logistic-learned, Enh. B)           ###")
    print("############################################################")
    out['real_blend100'], out['synthetic_blend100'] = evaluate_learned(filepath, blend=1.0)
    return out


if __name__ == '__main__':
    evaluate()
