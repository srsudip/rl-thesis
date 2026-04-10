"""
Pathway definitions, subject mappings, and CBC grading system.

Aligned with Kenya's Competency-Based Education (CBE) system as implemented
by KNEC for the 2025 KJSEA and subsequent Grade 10 placement.

Subject list sources:
  - KICD Final CBC Task Force Recommendations
  - KNEC KJSEA 2025 result slips
  - Ministry of Education Senior School placement circulars (2025)

Grading sources:
  - KNEC 8-level grading system (Dec 2025)
  - Education CS Julius Ogamba statements on pathway thresholds

Placement formula:
  60% KJSEA (Grade 9) + 20% SBA (Grade 7-8) + 20% KPSEA (Grade 6)
"""

# =====================================================
# Official KNEC 8-Level Grading System (KJSEA 2025)
# =====================================================
CBC_GRADING_SYSTEM = {
    'EE1': {'name': 'Exceeding Expectations', 'al': 8, 'min': 90, 'max': 100, 'points': 8, 'label': 'Exceptional',       'color': '#166534'},
    'EE2': {'name': 'Exceeding Expectations', 'al': 7, 'min': 75, 'max': 89,  'points': 7, 'label': 'Very Good',         'color': '#22c55e'},
    'ME1': {'name': 'Meeting Expectations',   'al': 6, 'min': 58, 'max': 74,  'points': 6, 'label': 'Good',              'color': '#3b82f6'},
    'ME2': {'name': 'Meeting Expectations',   'al': 5, 'min': 41, 'max': 57,  'points': 5, 'label': 'Fair',              'color': '#60a5fa'},
    'AE1': {'name': 'Approaching Expectations', 'al': 4, 'min': 31, 'max': 40, 'points': 4, 'label': 'Needs Improvement', 'color': '#f59e0b'},
    'AE2': {'name': 'Approaching Expectations', 'al': 3, 'min': 21, 'max': 30, 'points': 3, 'label': 'Below Average',    'color': '#fbbf24'},
    'BE1': {'name': 'Below Expectations',     'al': 2, 'min': 11, 'max': 20,  'points': 2, 'label': 'Well Below Average', 'color': '#ef4444'},
    'BE2': {'name': 'Below Expectations',     'al': 1, 'min': 1,  'max': 10,  'points': 1, 'label': 'Minimal',           'color': '#dc2626'},
}


def get_cbc_grade(score: float) -> dict:
    """Convert a raw score (0-100) to CBC grade level."""
    import math
    # Use math.floor(x + 0.5) instead of round() to avoid Python 3 banker's rounding
    # round(30.5) = 30 in Python 3, but we need 31 (standard rounding)
    score = int(math.floor(max(1, min(100, score)) + 0.5))
    for level, info in CBC_GRADING_SYSTEM.items():
        if info['min'] <= score <= info['max']:
            return {
                'level': level, 'name': info['name'], 'points': info['points'],
                'al': info['al'], 'color': info['color'], 'label': info['label'],
                'description': info['name'],
            }
    return {'level': 'BE2', 'name': 'Below Expectations', 'points': 1, 'al': 1,
            'color': '#dc2626', 'label': 'Minimal', 'description': 'Below Expectations'}


def is_below_expectations(score: float) -> bool:
    """Check if a score falls in Below Expectations (BE1 or BE2)."""
    return score <= 20


# =====================================================
# CBC Learning Areas / Subjects (Grade 4-9)
# =====================================================

SUBJECT_NAMES = {
    # Upper Primary (Grade 4-6)
    'ENG':       'English',
    'KIS_KSL':   'Kiswahili / Kenya Sign Language',
    'MATH':      'Mathematics',
    'SCI_TECH':  'Science and Technology',
    'SOC_STUD':  'Social Studies',
    'CRE_ARTS':  'Creative Arts',
    'AGRI':      'Agriculture',
    'REL_CRE':   'Religious Education (CRE)',
    'REL_IRE':   'Religious Education (IRE)',
    'REL_HRE':   'Religious Education (HRE)',
    'HOME_SCI':  'Home Science',
    'PHE':       'Physical and Health Education',
    # Junior Secondary (Grade 7-9) additional
    'INT_SCI':   'Integrated Science',
    'HEALTH_ED': 'Health Education',
    'PRE_TECH':  'Pre-Technical and Pre-Career Education',
    'BUS_STUD':  'Business Studies',
    'LIFE_SKILLS': 'Life Skills',
    'SPORTS_PE': 'Sports and Physical Education',
}

REL_SUBJECTS = {'REL_CRE', 'REL_IRE', 'REL_HRE'}

# Upper primary subjects (Grade 4-6)
UPPER_PRIMARY_SUBJECTS = [
    'ENG', 'KIS_KSL', 'MATH', 'SCI_TECH', 'SOC_STUD',
    'CRE_ARTS', 'AGRI', 'HOME_SCI', 'PHE',
    # Each student has one of CRE/IRE/HRE
]

# Junior secondary subjects (Grade 7-9) - core
JUNIOR_SECONDARY_SUBJECTS = [
    'ENG', 'KIS_KSL', 'MATH', 'INT_SCI', 'HEALTH_ED',
    'PRE_TECH', 'SOC_STUD', 'BUS_STUD', 'AGRI',
    'LIFE_SKILLS', 'SPORTS_PE',
    # Each student has one of CRE/IRE/HRE
]


def get_religious_ed_code(student_scores: dict) -> str:
    """Return whichever RE variant the student has data for."""
    for code in REL_SUBJECTS:
        if code in student_scores:
            return code
    return 'REL_CRE'


# =====================================================
# KNEC Subject Codes
# =====================================================
KJSEA_SUBJECT_CODES = {
    901: {'code': 'ENG',        'name': 'English'},
    902: {'code': 'KIS_KSL',    'name': 'Kiswahili / Kenya Sign Language'},
    903: {'code': 'MATH',       'name': 'Mathematics'},
    904: {'code': 'INT_SCI',    'name': 'Integrated Science'},
    905: {'code': 'HEALTH_ED',  'name': 'Health Education'},
    906: {'code': 'PRE_TECH',   'name': 'Pre-Technical and Pre-Career Education'},
    907: {'code': 'SOC_STUD',   'name': 'Social Studies'},
    908: {'code': 'REL_CRE',    'name': 'Religious Education'},
    909: {'code': 'BUS_STUD',   'name': 'Business Studies'},
    910: {'code': 'AGRI',       'name': 'Agriculture'},
    911: {'code': 'LIFE_SKILLS','name': 'Life Skills'},
    912: {'code': 'SPORTS_PE',  'name': 'Sports and Physical Education'},
}

KPSEA_PAPER_CODES = {
    601: {'code': 'ENG',      'name': 'English Language',       'subjects': ['ENG']},
    602: {'code': 'KIS_KSL',  'name': 'Kiswahili Lugha',       'subjects': ['KIS_KSL']},
    603: {'code': 'MATH',     'name': 'Mathematics',            'subjects': ['MATH']},
    604: {'code': 'SCI_TECH', 'name': 'Science and Technology', 'subjects': ['SCI_TECH', 'AGRI', 'HOME_SCI']},
    605: {'code': 'SOC_CRE',  'name': 'Social Studies & RE',    'subjects': ['SOC_STUD', 'CRE_ARTS', 'REL_CRE']},
}

PLACEMENT_WEIGHTS = {'KJSEA': 0.60, 'SBA': 0.20, 'KPSEA': 0.20}

# =====================================================
# Pathway Cluster Weights
# =====================================================
PATHWAY_SUBJECT_WEIGHTS = {
    # Based on KICD CBC Senior Secondary curriculum (Aug 2025 circular)
    # Weights reflect how strongly each Junior Secondary subject predicts
    # Senior Secondary pathway suitability
    'STEM': {
        # Core indicators: Math, Science, Pre-Technical
        'MATH': 1.0,     # Core Mathematics required for STEM — NON-NEGOTIABLE
        'INT_SCI': 1.0,  # Feeds → Biology, Chemistry, Physics, General Science
        'SCI_TECH': 0.9, # Science & Technology (upper-primary proxy for INT_SCI)
        'PRE_TECH': 0.9, # Feeds → Technical Studies (Aviation, Construction, etc.)
                         # ↑ raised 0.8→0.9: KICD confirms it's a PRIMARY STEM indicator
        'AGRI': 0.3,     # STEM elective only (Applied Sciences track)
        'ENG': 0.3,      # Core but not STEM-specific
        'KIS_KSL': 0.2,  # Core but not STEM-specific
    },
    'SOCIAL_SCIENCES': {
        # Core indicators: Languages, Social Studies, Business, RE
        'SOC_STUD': 1.0,   # Feeds → History & Citizenship, Geography
        'ENG': 0.9,        # Feeds → Languages & Literature track
        'KIS_KSL': 0.9,    # Feeds → Languages & Literature track
        'BUS_STUD': 0.8,   # Feeds → Humanities & Business Studies track
        'REL_CRE': 0.7,    # Feeds → CRE/IRE/HRE
        'LIFE_SKILLS': 0.5, # Life Skills Education
        'AGRI': 0.2,       # Minimal relevance
    },
    'ARTS_SPORTS': {
        # Core indicators: Creative Arts, Sports, Health
        'CRE_ARTS': 1.0,   # Feeds → Fine Arts, Music & Dance, Theatre & Film
        'SPORTS_PE': 0.9,  # Feeds → Sports & Recreation
        'HEALTH_ED': 0.6,  # Supporting subject
        'KIS_KSL': 0.3,    # Supporting (cross-pathway flexibility)
        'ENG': 0.3,        # Supporting
        'SOC_STUD': 0.2,   # Minor relevance
    },
}

# Subjects shown on result slip per pathway
PATHWAY_RESULT_SLIP_SUBJECTS = {
    'STEM': [
        ('MATH',     'Mathematics'),
        ('INT_SCI',  'Integrated Science'),
        ('SCI_TECH', 'Science and Technology'),
        ('PRE_TECH', 'Pre-Technical and Pre-Career Education'),
        ('ENG',      'English'),
        ('AGRI',     'Agriculture'),
    ],
    'SOCIAL_SCIENCES': [
        ('SOC_STUD',   'Social Studies'),
        ('ENG',        'English'),
        ('KIS_KSL',    'Kiswahili / Kenya Sign Language'),
        ('BUS_STUD',   'Business Studies'),
        ('REL_CRE',    'Religious Education'),
        ('LIFE_SKILLS','Life Skills'),
    ],
    'ARTS_SPORTS': [
        ('CRE_ARTS',  'Creative Arts'),
        ('SPORTS_PE', 'Sports and Physical Education'),
        ('HEALTH_ED', 'Health Education'),
        ('KIS_KSL',   'Kiswahili / Kenya Sign Language'),
        ('ENG',       'English'),
        ('SOC_STUD',  'Social Studies'),
    ],
}

# Minimum pathway suitability threshold (KNEC official, CS Ogamba Dec 2025)
# If a student's cluster weight for a pathway meets this, they qualify
PATHWAY_MIN_THRESHOLD = {
    'STEM': 20.0,
    'SOCIAL_SCIENCES': 25.0,
    'ARTS_SPORTS': 25.0,
}

# Gate 2: Core subject requirements per pathway.
# Students must achieve ≥ AE1 (31%) in pathway-relevant core subjects.
# 'required' = ALL must meet threshold (hard gate)
# 'required_one_of' = at least ONE must meet threshold
PATHWAY_CORE_SUBJECTS = {
    'STEM': {
        'required': ['MATH'],                           # Math is non-negotiable for STEM
        'required_one_of': ['INT_SCI', 'SCI_TECH'],    # At least one science ≥ AE1
        'min_score': 31,
    },
    'SOCIAL_SCIENCES': {
        'required_one_of': ['ENG', 'KIS_KSL'],         # At least one language ≥ AE1
        'min_score': 31,
    },
    'ARTS_SPORTS': {
        'required_one_of': ['CRE_ARTS', 'SPORTS_PE'],  # At least one arts/sports ≥ AE1
        'min_score': 31,
    },
}

MAX_CLUSTER_WEIGHT = 100.0

# =====================================================
# Official Key Subjects per Pathway (KICD/KNEC 2025)
# Source: KICD CBC Task Force, KNEC KJSEA 2025 placement criteria,
#         Kenya Ministry of Education Senior School placement circulars
# =====================================================
# 'primary'  : subjects MOST critical for pathway success — strong performance here
#              is the clearest signal of genuine pathway suitability
# 'secondary': supporting subjects that strengthen (but do not define) pathway fit
PATHWAY_KEY_SUBJECTS = {
    'STEM': {
        # All three are primary STEM predictors per KICD:
        # Mathematics is non-negotiable; Integrated Science feeds Biology/
        # Chemistry/Physics; Pre-Technical feeds Engineering & Technical tracks
        'primary':   ['MATH', 'INT_SCI', 'PRE_TECH'],
        'secondary': ['SCI_TECH', 'AGRI'],   # SCI_TECH = upper-primary INT_SCI proxy
    },
    'SOCIAL_SCIENCES': {
        # Humanities core: Social Studies (History/Geography), Languages (ENG/KIS),
        # Business Studies — all primary for SS placement per MoE circular
        'primary':   ['SOC_STUD', 'ENG', 'KIS_KSL', 'BUS_STUD'],
        'secondary': ['REL_CRE', 'LIFE_SKILLS'],
    },
    'ARTS_SPORTS': {
        # Creative Arts & Sports is a COMBINED subject in KJSEA 2025 (code 910);
        # split internally but both are primary indicators for this pathway
        'primary':   ['CRE_ARTS', 'SPORTS_PE'],
        'secondary': ['HEALTH_ED', 'ENG', 'KIS_KSL'],
    },
}


def _check_core_subjects(pathway_key: str, subject_scores: dict) -> bool:
    """
    Check if core subject requirements are met for a pathway.

    Gate 2: A student must achieve ≥ AE1 (31%) in the pathway's core subjects.
    - 'required': ALL subjects must meet the threshold.
    - 'required_one_of': at least ONE must meet the threshold.

    Returns True if the student passes, False if they fail.
    """
    core = PATHWAY_CORE_SUBJECTS.get(pathway_key)
    if not core or not subject_scores:
        return True  # No core defined or no scores → pass by default

    min_score = core.get('min_score', 31)

    # All required subjects must meet threshold
    for subj in core.get('required', []):
        if subject_scores.get(subj, 0) < min_score:
            return False

    # At least one of the group must meet threshold
    req_one = core.get('required_one_of', [])
    if req_one:
        if not any(subject_scores.get(s, 0) >= min_score for s in req_one):
            return False

    return True



# =====================================================
# Learned weights overlay (Enhancement B)
# =====================================================
# Loaded once from `config/learned_weights.json` if present.  These are
# logistic-regression coefficients fit against labelled gradess1.csv data
# (see `scripts/learn_pathway_weights.py`).  When loaded they are blended
# with the hand-authored KICD weights via `LEARNED_WEIGHT_BLEND`:
#     final_weight = (1 - blend) * hand + blend * learned
# A blend of 0 disables the overlay; a blend of 1 fully replaces hand-authored.
# Default blend = 0.0 — set in JSON or via env to opt in.

import json as _json
import os as _os
from pathlib import Path as _Path

_LEARNED_WEIGHTS_PATH = _Path(__file__).resolve().parent / 'learned_weights.json'
LEARNED_WEIGHTS: dict = {}
LEARNED_WEIGHT_META: dict = {}
LEARNED_WEIGHT_BLEND: float = float(_os.environ.get('LEARNED_WEIGHT_BLEND', '0.0'))

if _LEARNED_WEIGHTS_PATH.exists():
    try:
        _meta = _json.loads(_LEARNED_WEIGHTS_PATH.read_text())
        LEARNED_WEIGHTS = _meta.get('weights', {}) or {}
        LEARNED_WEIGHT_META = {k: v for k, v in _meta.items() if k != 'weights'}
    except Exception:    # corrupt JSON should not crash the app
        LEARNED_WEIGHTS = {}


def _effective_pw_weights(pw_key: str) -> dict:
    """
    Return the per-subject weight dict for a pathway, blending hand-authored
    KICD weights with logistic-learned coefficients (if available).
    """
    hand = PATHWAY_SUBJECT_WEIGHTS.get(pw_key, {})
    learned = LEARNED_WEIGHTS.get(pw_key, {}) if LEARNED_WEIGHT_BLEND > 0 else {}
    if not learned:
        return hand
    # Union of subject keys
    keys = set(hand) | set(learned)
    blend = max(0.0, min(1.0, LEARNED_WEIGHT_BLEND))
    return {k: (1 - blend) * hand.get(k, 0.0) + blend * learned.get(k, 0.0) for k in keys}


def compute_cluster_weights(subject_scores: dict, pop_stats: dict = None,
                            prev_cluster_weights: dict = None) -> dict:
    """
    Compute KNEC-style pathway suitability percentages.

    Args:
        subject_scores: {subject_key: raw_score (0-100)}
        pop_stats: Optional {subject_key: {'mean': float, 'std': float}}
        prev_cluster_weights: Optional previous grade CW for trajectory stability
    Returns: {pathway_key: percentage (0-100)}
    """
    if pop_stats:
        normed = {}
        for subj, raw in subject_scores.items():
            stats = pop_stats.get(subj)
            if stats and stats.get('std', 0) > 0:
                normed[subj] = 50.0 + 10.0 * (raw - stats['mean']) / stats['std']
            else:
                normed[subj] = raw
        scores = normed
    else:
        scores = dict(subject_scores)

    # Collapse RE variants
    for rel_code in REL_SUBJECTS:
        if rel_code in scores:
            scores['REL_CRE'] = scores[rel_code]
            break

    weights = {}
    for pw_key in PATHWAY_SUBJECT_WEIGHTS:
        pw_weights = _effective_pw_weights(pw_key)
        weighted_sum = 0.0
        weight_total = 0.0
        for subj_key, w in pw_weights.items():
            raw = scores.get(subj_key, None)
            if raw is None:
                # Cross-level fallbacks
                if subj_key == 'INT_SCI' and 'SCI_TECH' in scores:
                    raw = scores['SCI_TECH']
                elif subj_key == 'SCI_TECH' and 'INT_SCI' in scores:
                    raw = scores['INT_SCI']
                elif subj_key == 'PHE' and 'SPORTS_PE' in scores:
                    raw = scores['SPORTS_PE']
                elif subj_key == 'SPORTS_PE' and 'PHE' in scores:
                    raw = scores['PHE']
                else:
                    continue
            weighted_sum += raw * w
            weight_total += w
        if weight_total > 0:
            weights[pw_key] = round(max(0, min(100, weighted_sum / weight_total)), 2)
        else:
            weights[pw_key] = 0.0

    # Trajectory stability: blend with previous CW to reward consistency
    if prev_cluster_weights:
        for pw_key in weights:
            prev = prev_cluster_weights.get(pw_key, weights[pw_key])
            stability = 1.0 - min(1.0, abs(weights[pw_key] - prev) / max(weights[pw_key], prev, 1e-6))
            # 85% current CW + 15% stability bonus (consistent = reliable)
            weights[pw_key] = round(weights[pw_key] * 0.85 + prev * 0.15 * stability
                                    + weights[pw_key] * 0.15 * (1 - stability), 2)

    return weights


def _core_gap_score(pathway_key: str, subject_scores: dict) -> float:
    """
    Calculate how far a student is from meeting a pathway's core requirements.
    Lower = closer to meeting requirements = better fallback choice.
    
    'required' subjects are hard gates — failing them adds a large penalty
    to prevent the pathway being recommended as a fallback.
    """
    core = PATHWAY_CORE_SUBJECTS.get(pathway_key)
    if not core or not subject_scores:
        return 0.0

    min_score = core.get('min_score', 31)
    gap = 0.0

    # Gap for required subjects (ALL must meet) — HARD GATE, heavy penalty
    for subj in core.get('required', []):
        score = subject_scores.get(subj, 0)
        if score < min_score:
            gap += 100 + (min_score - score)  # 100-point penalty per failed required

    # Gap for required_one_of (best subject's gap)
    req_one = core.get('required_one_of', [])
    if req_one:
        best = max(subject_scores.get(s, 0) for s in req_one)
        if best < min_score:
            gap += (min_score - best)

    # Gap for any_of (best subject's gap)
    any_of = core.get('any_of', [])
    if any_of:
        best = max(subject_scores.get(s, 0) for s in any_of)
        if best < min_score:
            gap += (min_score - best)

    return gap


def _primary_key_subject_score(pathway_key: str, subject_scores: dict) -> float:
    """
    Average score across the PRIMARY key subjects for a pathway.

    Used as a tiebreaker when composite scores are close (within 5 pts).
    This measures how genuinely prepared the student is for the pathway's
    core demands — not just the overall cluster weight.

    Returns 0.0 if no scores are available.
    """
    primary = PATHWAY_KEY_SUBJECTS.get(pathway_key, {}).get('primary', [])
    if not primary or not subject_scores:
        return 0.0
    available = [subject_scores.get(s, 0) for s in primary if s in subject_scores]
    return sum(available) / len(available) if available else 0.0


def _least_gap_pathway(sorted_pw: list, subject_scores: dict) -> str:
    """
    When all pathways fail core check, pick the one requiring least remediation.
    Ties broken by cluster weight (sorted_pw is already sorted by CW desc).
    """
    best_pw = sorted_pw[0][0]
    best_gap = float('inf')
    for pw, _ in sorted_pw:
        gap = _core_gap_score(pw, subject_scores) if subject_scores else 0
        if gap < best_gap:
            best_gap = gap
            best_pw = pw
    return best_pw


def _composite_alignment_score(pw: str, cluster_weights: dict,
                                cosine_sims: dict = None,
                                psi_scores: dict = None) -> float:
    """
    Multi-signal composite score for pathway ranking.

    Fuses three complementary alignment signals:
      - Cluster Weight (CW): KNEC weighted subject score (0-100)
      - Cosine Similarity: angle between student profile and ideal pathway vector (0-1 → 0-100)
      - Pathway Suitability Index (PSI): competency-profile closeness (0-1 → 0-100)

    Weights:
      CW alone:              100% CW
      CW + cosine:           70% CW + 30% cosine
      CW + PSI:              80% CW + 20% PSI
      CW + cosine + PSI:     60% CW + 25% cosine + 15% PSI
    """
    cw_val  = cluster_weights.get(pw, 0.0)
    cos_val = (cosine_sims.get(pw, 0.0) * 100.0) if cosine_sims else None
    psi_val = (psi_scores.get(pw, 0.0)  * 100.0) if psi_scores  else None

    if cos_val is not None and psi_val is not None:
        return 0.60 * cw_val + 0.25 * cos_val + 0.15 * psi_val
    elif cos_val is not None:
        return 0.70 * cw_val + 0.30 * cos_val
    elif psi_val is not None:
        return 0.80 * cw_val + 0.20 * psi_val
    else:
        return cw_val


def recommend_pathway(cluster_weights: dict, subject_scores: dict = None,
                      cosine_sims: dict = None, psi_scores: dict = None) -> dict:
    """
    Recommend pathway using a two-stage filter-then-rank pipeline.

    Pipeline (Enhancement A — see scripts/eval_recommendation_accuracy.py):
      Stage 1 — HARD FILTER drops any pathway that fails either gate:
        (a) Cluster weight ≥ KNEC threshold  (STEM 20 %, SS/Arts 25 %)
        (b) Core subjects ≥ AE1              (PATHWAY_CORE_SUBJECTS)
      Stage 2 — RANK survivors by raw cluster weight (NOT composite).
      Stage 3 — TIEBREAK only when the top-2 survivors are within 5 CW pts:
        (a) Primary key subject average      (PATHWAY_KEY_SUBJECTS['primary'])
        (b) If still tied, cosine similarity nudge (small ±)
      Stage 4 — LOW-CONFIDENCE FLAG (Enhancement C): when the eligible margin
                is narrow (< 3 CW pts AND < 5 key-subject pts) we set
                `low_confidence=True` so the UI can ask a human to review.

    Why this design (justified empirically — see eval script):
      The previous composite formula `0.60·CW + 0.25·cosine + 0.15·PSI` ranked
      survivors using cosine, which on theta-truth synthetic data **dropped
      accuracy from 39 % to 31.8 %** because cosine on the (small) ideal
      profile vector saturates near 1.0 for any broadly-competent student
      and overrides cluster weight differences of 5–10 pts.  Rank-by-CW
      eliminates the override; cosine survives only as a tiebreaker.

    Fallback tiers (when nothing survives Stage 1):
      Tier 2 — pathways that pass core check but not CW threshold
      Tier 3 — least-gap pathway (with `below_expectations_warning=True`)

    Args:
        cluster_weights: {pathway_key: percentage (0-100)}
        subject_scores:  Optional {subject_key: raw_score (0-100)} for core validation
        cosine_sims:     Optional {pathway_key: cosine_similarity (0-1)} — tiebreaker only
        psi_scores:      Optional {pathway_key: PSI (0-1)}                — composite_score field only
    """
    if not cluster_weights:
        return {'recommended_pathway': 'STEM', 'is_tie': False,
                'meets_threshold': False, 'below_expectations_warning': True,
                'suitability_score': 0, 'composite_score': 0, 'tie_pathways': [],
                'eligible_pathways': [], 'core_check_failed': True,
                'low_confidence': True}

    # ----- Stage 1: hard filter -----------------------------------------
    eligible: list[str] = []
    core_passing: list[str] = []
    for pw in cluster_weights:
        cw_score  = cluster_weights.get(pw, 0)
        meets_thr = cw_score >= PATHWAY_MIN_THRESHOLD.get(pw, 25.0)
        passes_core = _check_core_subjects(pw, subject_scores) if subject_scores else True
        if meets_thr and passes_core:
            eligible.append(pw)
        elif passes_core:
            core_passing.append(pw)

    # ----- Stage 2: rank survivors by RAW cluster weight ----------------
    eligible.sort(key=lambda pw: cluster_weights.get(pw, 0), reverse=True)
    core_passing.sort(key=lambda pw: cluster_weights.get(pw, 0), reverse=True)

    low_confidence = False
    if eligible:
        best_pw = eligible[0]

        # ----- Stage 3: tiebreaker on narrow CW gap ---------------------
        if len(eligible) > 1 and subject_scores:
            cw_top, cw_2nd = cluster_weights[eligible[0]], cluster_weights[eligible[1]]
            if cw_top - cw_2nd < 5.0:
                ks0 = _primary_key_subject_score(eligible[0], subject_scores)
                ks1 = _primary_key_subject_score(eligible[1], subject_scores)
                # Switch only on a clear key-subject lead (>5 pts).  This
                # remains symmetric with the previous behaviour but is now
                # measured against CW gap (the ranking signal), not composite.
                if ks1 > ks0 + 5.0:
                    best_pw = eligible[1]
                elif abs(ks1 - ks0) <= 5.0 and cosine_sims:
                    # Cosine survives only as a final symmetric nudge.
                    if cosine_sims.get(eligible[1], 0) > cosine_sims.get(eligible[0], 0) + 0.05:
                        best_pw = eligible[1]
                # ----- Stage 4: low-confidence flag (Enhancement C) -----
                low_confidence = True
    elif core_passing:
        best_pw = core_passing[0]
        low_confidence = True
    else:
        sorted_pw_cw = sorted(cluster_weights.items(), key=lambda x: x[1], reverse=True)
        best_pw = _least_gap_pathway(sorted_pw_cw, subject_scores)
        low_confidence = True

    best_score     = cluster_weights[best_pw]
    best_composite = _composite_alignment_score(best_pw, cluster_weights, cosine_sims, psi_scores)
    min_thr        = PATHWAY_MIN_THRESHOLD.get(best_pw, 25.0)
    passes_core    = _check_core_subjects(best_pw, subject_scores) if subject_scores else True

    return {
        'recommended_pathway':      best_pw,
        'suitability_score':        best_score,
        'composite_score':          round(best_composite, 2),
        'is_tie':                   False,
        'tie_pathways':             [],
        'meets_threshold':          best_score >= min_thr,
        'below_expectations_warning': best_score < min_thr or not passes_core,
        'eligible_pathways':        eligible,
        'core_check_failed':        not passes_core,
        'low_confidence':           low_confidence,
    }


# =====================================================
# Senior Secondary Pathways with Tracks
# =====================================================
PATHWAYS = {
    'STEM': {
        'name': 'STEM',
        'tracks': {
            'PURE_SCIENCES': {
                'name': 'Pure Sciences',
                'description': 'Physics, Chemistry, Biology — theoretical foundations',
                'key_subjects': ['MATH', 'INT_SCI', 'SCI_TECH'],
                'senior_electives': ['Physics', 'Chemistry', 'Biology', 'Advanced Mathematics'],
                'careers': ['Doctor', 'Researcher', 'Pharmacist', 'Data Scientist', 'Biotechnologist'],
            },
            'APPLIED_SCIENCES': {
                'name': 'Applied Sciences',
                'description': 'Agriculture, Environmental Science, Health Sciences',
                'key_subjects': ['INT_SCI', 'AGRI', 'SCI_TECH'],
                'senior_electives': ['Agriculture', 'Biology', 'Chemistry', 'Geography', 'Computer Studies', 'Home Science'],
                'careers': ['Agronomist', 'Environmentalist', 'Nutritionist', 'Veterinarian'],
            },
            'TECH_ENGINEERING': {
                'name': 'Technical and Engineering',
                'description': 'Mechanical, Electrical, Civil, Aviation technology',
                'key_subjects': ['MATH', 'INT_SCI', 'PRE_TECH'],
                'senior_electives': ['Physics', 'Building Construction', 'Electricity', 'Power Mechanics', 'Aviation Technology'],
                'careers': ['Engineer', 'Architect', 'Pilot', 'Technician'],
            },
            'CAREER_TECH_STUDIES': {
                'name': 'Career and Technology Studies',
                'description': 'Vocational technology, ICT, practical applied skills',
                'key_subjects': ['PRE_TECH', 'MATH', 'ENG'],
                'senior_electives': ['Computer Science', 'Woodwork', 'Leatherwork', 'Garment Making', 'Culinary Arts'],
                'careers': ['IT Specialist', 'Software Developer', 'Electrician', 'Mechanic'],
            },
        },
        'sub_pathways': {},
        'description': 'For students excelling in mathematics, sciences, and technical subjects',
        'color': '#2563eb', 'icon': '\U0001f52c', 'min_composite_pct': 20,
    },
    'SOCIAL_SCIENCES': {
        'name': 'Social Sciences',
        'tracks': {
            'HUMANITIES_BUSINESS': {
                'name': 'Humanities and Business Studies',
                'description': 'History, Geography, Economics, Business, Religious Studies',
                'key_subjects': ['SOC_STUD', 'ENG', 'BUS_STUD'],
                'senior_electives': ['History and Citizenship', 'Geography', 'Business Studies', 'CRE/IRE/HRE'],
                'careers': ['Lawyer', 'Economist', 'Journalist', 'Diplomat', 'Accountant'],
            },
            'LANGUAGES_LITERATURE': {
                'name': 'Languages and Literature',
                'description': 'English, Kiswahili, Foreign Languages, Literature',
                'key_subjects': ['ENG', 'KIS_KSL', 'SOC_STUD'],
                'senior_electives': ['Literature in English', 'Fasihi ya Kiswahili', 'French', 'German', 'Arabic'],
                'careers': ['Teacher', 'Translator', 'Author', 'Public Relations Specialist'],
            },
        },
        'sub_pathways': {},
        'description': 'For students strong in languages, social studies, humanities, and business',
        'color': '#10B981', 'icon': '\U0001f4da', 'min_composite_pct': 25,
    },
    'ARTS_SPORTS': {
        'name': 'Arts and Sports Science',
        'tracks': {
            'PERFORMING_ARTS': {
                'name': 'Performing Arts',
                'description': 'Music, Dance, Theatre, Film, Media Studies',
                'key_subjects': ['CRE_ARTS', 'ENG', 'KIS_KSL'],
                'senior_electives': ['Music', 'Theatre and Film', 'Fine Arts', 'Media Studies'],
                'careers': ['Musician', 'Actor', 'Film Director', 'Choreographer', 'Producer'],
            },
            'VISUAL_ARTS': {
                'name': 'Visual Arts',
                'description': 'Fine Art, Applied Art, Design, Digital Media, Crafts',
                'key_subjects': ['CRE_ARTS', 'ENG'],
                'senior_electives': ['Fine Arts', 'Applied Art', 'Crafts', 'Time Based Media'],
                'careers': ['Graphic Designer', 'Animator', 'Photographer', 'Fashion Designer'],
            },
            'SPORTS_SCIENCE': {
                'name': 'Sports Science',
                'description': 'Sports management, Physical Education, Coaching, Athletics',
                'key_subjects': ['SPORTS_PE', 'PHE', 'INT_SCI'],
                'senior_electives': ['Advanced Physical Education', 'Athletics', 'Indoor Games', 'Biology'],
                'careers': ['Athlete', 'Coach', 'Sports Manager', 'Physiotherapist'],
            },
        },
        'sub_pathways': {},
        'description': 'For students talented in creative arts, performing arts, and sports',
        'color': '#F97316', 'icon': '\U0001f3a8', 'min_composite_pct': 25,
    },
}

for _pk, _pv in PATHWAYS.items():
    _pv['sub_pathways'] = _pv['tracks']

SUB_PATHWAYS = {}
for pathway_key, pathway_data in PATHWAYS.items():
    for track_key, track_data in pathway_data['tracks'].items():
        SUB_PATHWAYS[track_key] = {**track_data, 'parent_pathway': pathway_key, 'parent_name': pathway_data['name']}


def get_pathway_name(pathway_key: str) -> str:
    """Get full display name for a pathway."""
    pw = PATHWAYS.get(pathway_key)
    return pw['name'] if pw else pathway_key


def get_pathway_index(pathway: str) -> int:
    return list(PATHWAYS.keys()).index(pathway)

def get_pathway_from_index(index: int) -> str:
    return list(PATHWAYS.keys())[index]


# =====================================================
# Additional exports for backward compatibility
# =====================================================

# Below Expectations threshold (score <= this means BE)
BELOW_EXPECTATIONS_THRESHOLD = 20

# Senior School core subjects (all pathways)
SENIOR_SCHOOL_CORE_SUBJECTS = [
    'English',
    'Kiswahili / Kenya Sign Language',
    'Community Service Learning',
    'Physical Education',
]

# Backward-compatible aliases
def get_pathway_display_name(pathway_key: str) -> str:
    """Alias for get_pathway_name."""
    return get_pathway_name(pathway_key)

def check_pathway_eligibility(cluster_weights: dict) -> dict:
    """Check which pathways a student is eligible for (KNEC minimum suitability)."""
    eligible = {}
    for pw_key, threshold in PATHWAY_MIN_THRESHOLD.items():
        score = cluster_weights.get(pw_key, 0)
        eligible[pw_key] = {
            'eligible': score >= threshold,
            'score': score,
            'threshold': threshold,
            'below_expectations': score < threshold,
        }
    return eligible

def has_pathway_tie(cluster_weights: dict, threshold: float = 2.0) -> bool:
    """Legacy compatibility — always returns False. Ties are no longer flagged."""
    return False