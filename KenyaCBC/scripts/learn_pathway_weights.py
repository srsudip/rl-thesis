"""
Learn per-pathway subject weights from labelled student data
============================================================

Fits a one-vs-rest L2 logistic regression on standardized G9 subject scores
and the `Actual Pathway` labels in `data/gradess1.csv`, then writes the
learned per-(pathway, subject) coefficients to `config/learned_weights.json`.

These weights are consumed by `config.pathways.compute_cluster_weights`
*as a fallback overlay* on top of the hand-authored KICD weights — they do
NOT replace the KNEC formula, they only re-weight which subject contributes
how much to each pathway's cluster score.

Citation note (read before running):
    KNEC publishes ONLY the 60/20/20 placement formula and the AE1 thresholds.
    Per-subject weights are NOT prescribed by KNEC.  We learn them empirically
    from labelled data, following the standard educational data-mining practice
    described in:
        - Drachsler, Verbert, Santos & Manouselis (2015) — Recommender systems
          for learning, Springer.
        - Romero & Ventura (2020) — Educational data mining and learning
          analytics: an updated survey.  WIREs Data Mining.
        - Kotsiantis (2012) — Use of machine learning techniques for
          educational proposes.

Run:
    python scripts/learn_pathway_weights.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from src.data.real_data_loader import real_csv_to_internal


PATHWAYS = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
OUT_PATH = ROOT / 'config' / 'learned_weights.json'


def _load_labelled_g9(filepath: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X = G9 score matrix, y = pathway label, subject_keys)."""
    bundle = real_csv_to_internal(str(filepath), simulate_g4_g6=True, seed=42)
    assessments: pd.DataFrame = bundle['assessments']
    labels: dict = bundle.get('pathway_labels') or {}
    if not labels:
        raise SystemExit(f"{filepath} has no actual pathway labels — cannot learn.")

    g9 = assessments[assessments['grade'] == assessments['grade'].max()].copy()
    score_cols = [c for c in g9.columns
                  if c.endswith('_score') and c not in ('student_id', 'grade')]
    g9 = g9.dropna(subset=score_cols, how='any')

    g9['_label'] = g9['student_id'].map(labels)
    g9 = g9.dropna(subset=['_label'])
    g9 = g9[g9['_label'].isin(PATHWAYS)]

    X = g9[score_cols].to_numpy(dtype=float)
    y = g9['_label']
    subject_keys = [c.replace('_score', '') for c in score_cols]
    return X, y, subject_keys


def fit_and_dump(filepath: Path = None) -> dict:
    if filepath is None:
        filepath = ROOT / 'data' / 'gradess1.csv'

    X, y, subject_keys = _load_labelled_g9(filepath)
    print(f"Fitting on {len(X)} labelled students × {len(subject_keys)} subjects")
    print(f"Label distribution: {dict(y.value_counts())}")

    # ---- Standardize so coefficients are comparable across subjects ----
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # ---- 5-fold stratified CV accuracy (held-out) ----------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = []
    for tr, te in cv.split(Xz, y):
        m = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs').fit(Xz[tr], y.iloc[tr])
        cv_acc.append(m.score(Xz[te], y.iloc[te]))
    print(f"5-fold CV accuracy: mean={np.mean(cv_acc)*100:.1f}% std={np.std(cv_acc)*100:.1f}%")

    # ---- Final fit on all data for weight extraction -------------------
    model = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    model.fit(Xz, y)

    # model.classes_ ordered alphabetically; build per-pathway weight dicts
    learned = {}
    for cls_idx, pw in enumerate(model.classes_):
        coefs = model.coef_[cls_idx]
        # Map back to raw scale: divide by standard deviation so the weight
        # applies to the raw 0-100 score, not the z-score.
        raw_coefs = coefs / scaler.scale_
        # Clip negative coefficients to 0 — pathway weights must be non-negative
        # so they can be combined with the KICD formula.  Negatives would mean
        # "this subject argues AGAINST the pathway", which the formula has no
        # affordance for; we keep only the positive evidence.
        raw_coefs = np.clip(raw_coefs, 0.0, None)
        # Normalize so the maximum weight per pathway is 1.0 (matches KICD scale)
        if raw_coefs.max() > 0:
            raw_coefs = raw_coefs / raw_coefs.max()
        learned[pw] = {subject_keys[i]: round(float(raw_coefs[i]), 3)
                       for i in range(len(subject_keys))
                       if raw_coefs[i] > 0.05}    # drop near-zero noise

    out = {
        'source_file': str(filepath.name),
        'n_students':  int(len(X)),
        'cv_accuracy': round(float(np.mean(cv_acc)), 4),
        'cv_std':      round(float(np.std(cv_acc)), 4),
        'method':      'L2 logistic regression, one-vs-rest, standardized inputs',
        'note':        ('KNEC publishes only the 60/20/20 placement formula. '
                        'These weights are learned empirically — see '
                        'Drachsler et al. 2015, Romero & Ventura 2020, '
                        'Kotsiantis 2012.'),
        'weights':     learned,
    }

    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")
    for pw in PATHWAYS:
        if pw in learned:
            top = sorted(learned[pw].items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  {pw}: {top}")
    return out


if __name__ == '__main__':
    fit_and_dump()
