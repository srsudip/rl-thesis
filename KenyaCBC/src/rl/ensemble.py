"""
PathwayEnsemble — Beautiful Stable Soup
========================================

Combines DQN (reinforcement learning, trajectory-based) and
Transformer+LoRA (supervised, full G4-G9 sequence) into a single
prediction by entropy-weighted averaging.

Design
------
Each model outputs a probability distribution over 3 pathways.

  DQN         : cluster_weights (0-100%) normalised → probabilities
  Transformer : softmax confidence_scores already in [0,1]

The weight given to each model is inversely proportional to the
*entropy* of its output distribution:
  - A model that says "90% STEM, 5% SS, 5% ARTS" (low entropy) is
    confident — give it more weight.
  - A model that says "35% / 33% / 32%" (high entropy) is unsure —
    give it less weight.

Agreement signal
----------------
  AGREE    : both models pick the same pathway → boost confidence
  DISAGREE : models differ → reduce confidence, flag for teacher review

The final ensemble dict follows the same schema as individual model
recommend() returns so it can be dropped in anywhere the DQN result is
currently used.
"""

import numpy as np
from typing import Dict, Optional

PATHWAY_ORDER = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
_LOG2 = np.log(2)  # for normalising entropy to [0,1]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(prob_dict: dict) -> np.ndarray:
    """Return a length-3 probability array aligned to PATHWAY_ORDER."""
    raw = np.array([float(prob_dict.get(p, 0.0)) for p in PATHWAY_ORDER],
                   dtype=np.float64)
    total = raw.sum()
    if total <= 0:
        return np.ones(3) / 3.0   # uniform fallback
    return raw / total


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy (nats), clipped to avoid log(0)."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _entropy_weight(probs: np.ndarray, n_classes: int = 3) -> float:
    """
    Confidence weight in [0, 1].

    Weight = 1 - normalised_entropy
      → 1 when distribution is a one-hot (maximally confident)
      → 0 when distribution is uniform (maximally uncertain)
    """
    max_entropy = np.log(n_classes)   # entropy of uniform distribution
    h = _entropy(probs)
    return float(1.0 - h / max_entropy)


# ─────────────────────────────────────────────────────────────────────────────
#  Core ensemble logic  (stateless — no class needed)
# ─────────────────────────────────────────────────────────────────────────────

def combine(dqn_result: dict, transformer_result: dict) -> dict:
    """
    Combine one DQN recommendation dict and one Transformer recommendation
    dict into an ensemble recommendation dict.

    Parameters
    ----------
    dqn_result : dict
        Output of DataManager.get_recommendation() — must have
        'recommended_pathway' and 'confidence_scores' (or 'cluster_weights').
    transformer_result : dict
        Output of PathwayTransformerAgent.recommend() — must have
        'recommended_pathway' and 'confidence_scores'.

    Returns
    -------
    dict with keys:
        recommended_pathway  str
        confidence           float  0-1
        confidence_scores    dict   pathway → probability
        agreement            str    'AGREE' | 'DISAGREE'
        flag_for_review      bool
        dqn_pathway          str
        dqn_confidence       float
        transformer_pathway  str
        transformer_confidence float
        dqn_weight           float  (entropy-derived)
        transformer_weight   float
        attention_weights    dict   grade → weight  (from Transformer)
        model                str
    """
    # ── 1. Extract probability distributions ─────────────────────────────────
    dqn_scores = dqn_result.get('confidence_scores') or {}
    # cluster_weights are percentages; confidence_scores are already [0,1]
    # If confidence_scores came from cluster_weights, they're already normalised
    # in get_recommendation(), but normalise again for safety.
    p_dqn = _normalise(dqn_scores)

    trans_scores = transformer_result.get('confidence_scores') or {}
    p_trans = _normalise(trans_scores)

    # ── 2. Entropy-based weights ──────────────────────────────────────────────
    w_dqn   = _entropy_weight(p_dqn)
    w_trans = _entropy_weight(p_trans)

    w_total = w_dqn + w_trans
    if w_total <= 0:
        w_dqn = w_trans = 0.5
        w_total = 1.0

    w_dqn_norm   = w_dqn   / w_total
    w_trans_norm = w_trans / w_total

    # ── 3. Weighted combination ───────────────────────────────────────────────
    p_ensemble = w_dqn_norm * p_dqn + w_trans_norm * p_trans

    best_idx    = int(np.argmax(p_ensemble))
    recommended = PATHWAY_ORDER[best_idx]
    confidence  = float(p_ensemble[best_idx])

    # ── 4. Agreement check ────────────────────────────────────────────────────
    dqn_pathway   = dqn_result.get('recommended_pathway', '')
    trans_pathway = transformer_result.get('recommended_pathway', '')
    agreed        = (dqn_pathway == trans_pathway)

    if agreed:
        agreement = 'AGREE'
        # Boost confidence: interpolate toward 1.0 by 20% of remaining gap
        confidence = confidence + 0.20 * (1.0 - confidence)
        flag_for_review = False
    else:
        agreement = 'DISAGREE'
        # Penalise confidence: reduce by 15% of current value
        confidence = confidence * 0.85
        flag_for_review = True

    confidence = round(min(1.0, max(0.0, confidence)), 3)

    # ── 5. Build output ───────────────────────────────────────────────────────
    ranking = sorted(range(3), key=lambda i: p_ensemble[i], reverse=True)

    return {
        'recommended_pathway'    : recommended,
        'confidence'             : confidence,
        'confidence_scores'      : {PATHWAY_ORDER[i]: round(float(p_ensemble[i]), 3)
                                    for i in range(3)},
        'pathway_ranking'        : [PATHWAY_ORDER[i] for i in ranking],
        'agreement'              : agreement,
        'flag_for_review'        : flag_for_review,

        # Individual model outputs preserved for UI
        'dqn_pathway'            : dqn_pathway,
        'dqn_confidence'         : round(float(dqn_result.get('confidence', 0.0)), 3),
        'dqn_weight'             : round(w_dqn_norm, 3),
        'transformer_pathway'    : trans_pathway,
        'transformer_confidence' : round(float(transformer_result.get('confidence', 0.0)), 3),
        'transformer_weight'     : round(w_trans_norm, 3),
        'attention_weights'      : transformer_result.get('attention_weights', {}),

        'model'                  : 'DQN+Transformer Ensemble',
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience class (stateless — wraps combine())
# ─────────────────────────────────────────────────────────────────────────────

class PathwayEnsemble:
    """
    Thin wrapper that holds references to both agents and delegates to
    combine() for the actual merging logic.

    Usage
    -----
        ensemble = PathwayEnsemble(dqn_agent, transformer_agent)
        result   = ensemble.recommend(student_id, assessments_df)
    """

    def __init__(self, dqn_agent, transformer_agent):
        self.dqn         = dqn_agent
        self.transformer = transformer_agent

    def recommend(self, student_id: int, assessments_df) -> dict:
        """
        Run both models and return the merged ensemble recommendation.

        Falls back gracefully if either model is unavailable:
          - Only DQN available  → return DQN result as-is
          - Only Transformer    → return Transformer result as-is
          - Both available      → full entropy-weighted merge
        """
        dqn_ok   = self.dqn is not None and getattr(self.dqn, '_trained', True)
        trans_ok = (self.transformer is not None
                    and getattr(self.transformer, '_trained', False))

        if dqn_ok and trans_ok:
            try:
                dqn_res   = self.dqn.get_recommendation(student_id)
            except Exception:
                dqn_ok = False

        if trans_ok:
            try:
                trans_res = self.transformer.recommend(student_id, assessments_df)
            except Exception:
                trans_ok = False

        if dqn_ok and trans_ok:
            result = combine(dqn_res, trans_res)
            result['student_id'] = student_id
            return result

        if dqn_ok:
            r = self.dqn.get_recommendation(student_id)
            r.setdefault('agreement', 'DQN_ONLY')
            r.setdefault('flag_for_review', False)
            r.setdefault('model', 'DQN (Transformer unavailable)')
            return r

        if trans_ok:
            r = self.transformer.recommend(student_id, assessments_df)
            r.setdefault('agreement', 'TRANSFORMER_ONLY')
            r.setdefault('flag_for_review', False)
            r.setdefault('model', 'Transformer (DQN unavailable)')
            return r

        return {
            'recommended_pathway': 'STEM',
            'confidence': 0.0,
            'confidence_scores': {},
            'agreement': 'NO_MODEL',
            'flag_for_review': True,
            'model': 'No model available',
            'student_id': student_id,
        }

    def batch_recommend(self, student_ids: list, assessments_df) -> list:
        """Return a list of ensemble dicts for multiple students."""
        return [self.recommend(sid, assessments_df) for sid in student_ids]

    def disagreement_report(self, student_ids: list, assessments_df) -> dict:
        """
        Summarise agreement rates across a cohort.

        Returns
        -------
        dict with keys:
            total           int
            agreed          int
            disagreed       int
            agreement_rate  float
            flags           list of student_ids that need review
        """
        results = self.batch_recommend(student_ids, assessments_df)
        flags   = [r['student_id'] for r in results
                   if r.get('flag_for_review', False)]
        agreed  = sum(1 for r in results if r.get('agreement') == 'AGREE')

        return {
            'total'         : len(results),
            'agreed'        : agreed,
            'disagreed'     : len(results) - agreed,
            'agreement_rate': round(agreed / max(1, len(results)), 3),
            'flags'         : flags,
        }
