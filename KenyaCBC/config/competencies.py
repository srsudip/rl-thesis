"""
Competency Framework definitions (7 Core Competencies of CBC).

These are the core competencies that Kenya's CBC aims to develop
in every learner throughout their education journey (PP1 through Grade 12).

Source: Kenya Institute of Curriculum Development (KICD)

The competencies are NOT the same as subjects. They are cross-cutting
learning outcomes that are developed through multiple subjects.

In our system:
  - COMPETENCY_SUBJECT_WEIGHTS defines how much each subject contributes
    to developing each competency (used to compute competency scores
    from subject scores).
  - IDEAL_PATHWAY_PROFILES defines the target competency profile for
    each Senior Secondary pathway. These are predefined based on
    educational theory and pathway requirements — they do NOT change
    randomly. A STEM student ideally has high Critical Thinking and
    Digital Literacy; an Arts student ideally has high Creativity.
"""

# =====================================================
# 7 Core Competencies of CBC
# =====================================================
COMPETENCIES = {
    'COMM_COLLAB': {
        'name': 'Communication and Collaboration',
        'description': 'Ability to communicate effectively and work with others across diverse contexts',
        'relevant_subjects': ['ENG', 'KIS_KSL', 'SOC_STUD', 'BUS_STUD']
    },
    'SELF_EFFICACY': {
        'name': 'Self-efficacy',
        'description': 'Confidence in one\'s ability to succeed and take on challenges',
        'relevant_subjects': ['MATH', 'SCI_TECH', 'INT_SCI', 'PRE_TECH', 'SPORTS_PE']
    },
    'CRIT_THINK_PROB_SOLVE': {
        'name': 'Critical Thinking and Problem Solving',
        'description': 'Ability to analyze information, reason logically, and solve complex problems',
        'relevant_subjects': ['MATH', 'SCI_TECH', 'INT_SCI', 'PRE_TECH', 'SOC_STUD']
    },
    'CREATIVITY_IMAGINATION': {
        'name': 'Creativity and Imagination',
        'description': 'Ability to think creatively, innovate, and express ideas in new ways',
        'relevant_subjects': ['CRE_ARTS', 'ENG', 'PRE_TECH', 'HOME_SCI']
    },
    'CITIZENSHIP': {
        'name': 'Citizenship',
        'description': 'Understanding civic responsibilities, national values, and social cohesion',
        'relevant_subjects': ['SOC_STUD', 'REL_CRE', 'REL_IRE', 'REL_HRE', 'LIFE_SKILLS']
    },
    'DIGITAL_LITERACY': {
        'name': 'Digital Literacy',
        'description': 'Ability to use digital technologies effectively and responsibly',
        'relevant_subjects': ['PRE_TECH', 'SCI_TECH', 'INT_SCI', 'MATH']
    },
    'LEARNING_TO_LEARN': {
        'name': 'Learning to Learn',
        'description': 'Ability to acquire new knowledge independently and reflect on learning',
        'relevant_subjects': ['MATH', 'ENG', 'SCI_TECH', 'INT_SCI', 'SOC_STUD']
    }
}

# =====================================================
# Competency-Subject Weight Matrix
#
# How much each subject contributes to each competency.
# Weights sum to ~1.0 per competency. Subjects not listed
# contribute 0.
#
# These weights are used to derive competency scores from
# subject-level assessment data.
# =====================================================
COMPETENCY_SUBJECT_WEIGHTS = {
    'COMM_COLLAB': {
        'ENG': 0.30, 'KIS_KSL': 0.25, 'SOC_STUD': 0.15,
        'CRE_ARTS': 0.10, 'BUS_STUD': 0.10, 'LIFE_SKILLS': 0.10,
    },
    'SELF_EFFICACY': {
        'MATH': 0.25, 'SCI_TECH': 0.20, 'INT_SCI': 0.20,
        'PRE_TECH': 0.10, 'AGRI': 0.10, 'SPORTS_PE': 0.10, 'PHE': 0.05,
    },
    'CRIT_THINK_PROB_SOLVE': {
        'MATH': 0.30, 'SCI_TECH': 0.20, 'INT_SCI': 0.20,
        'PRE_TECH': 0.15, 'SOC_STUD': 0.10, 'BUS_STUD': 0.05,
    },
    'CREATIVITY_IMAGINATION': {
        'CRE_ARTS': 0.35, 'ENG': 0.15, 'PRE_TECH': 0.15,
        'HOME_SCI': 0.10, 'KIS_KSL': 0.10, 'SPORTS_PE': 0.10, 'PHE': 0.05,
    },
    'CITIZENSHIP': {
        'SOC_STUD': 0.30, 'REL_CRE': 0.25, 'ENG': 0.10,
        'KIS_KSL': 0.10, 'LIFE_SKILLS': 0.15, 'HEALTH_ED': 0.10,
    },
    'DIGITAL_LITERACY': {
        'PRE_TECH': 0.35, 'SCI_TECH': 0.25, 'INT_SCI': 0.15,
        'MATH': 0.15, 'BUS_STUD': 0.10,
    },
    'LEARNING_TO_LEARN': {
        'MATH': 0.20, 'ENG': 0.20, 'SCI_TECH': 0.15,
        'INT_SCI': 0.15, 'SOC_STUD': 0.15, 'AGRI': 0.10, 'LIFE_SKILLS': 0.05,
    }
}

# =====================================================
# Ideal Pathway Competency Profiles (predefined)
#
# These represent the TARGET competency distribution
# for each pathway. They are derived from educational
# theory — NOT from random data.
#
# Values are normalized (0-1) and indicate how important
# each competency is for success in the pathway.
#
# For example, STEM requires very high Critical Thinking
# (1.0) and Digital Literacy (0.95), while Arts & Sports
# requires very high Creativity (1.0).
# =====================================================
IDEAL_PATHWAY_PROFILES = {
    'STEM': {
        'COMM_COLLAB': 0.6,
        'SELF_EFFICACY': 0.9,
        'CRIT_THINK_PROB_SOLVE': 1.0,
        'CREATIVITY_IMAGINATION': 0.7,
        'CITIZENSHIP': 0.5,
        'DIGITAL_LITERACY': 0.95,
        'LEARNING_TO_LEARN': 0.85
    },
    'SOCIAL_SCIENCES': {
        'COMM_COLLAB': 1.0,
        'SELF_EFFICACY': 0.7,
        'CRIT_THINK_PROB_SOLVE': 0.75,
        'CREATIVITY_IMAGINATION': 0.6,
        'CITIZENSHIP': 0.95,
        'DIGITAL_LITERACY': 0.5,
        'LEARNING_TO_LEARN': 0.9
    },
    'ARTS_SPORTS': {
        'COMM_COLLAB': 0.8,
        'SELF_EFFICACY': 0.85,
        'CRIT_THINK_PROB_SOLVE': 0.5,
        'CREATIVITY_IMAGINATION': 1.0,
        'CITIZENSHIP': 0.6,
        'DIGITAL_LITERACY': 0.4,
        'LEARNING_TO_LEARN': 0.7
    }
}


def get_competency_names() -> list:
    """Return list of competency codes."""
    return list(COMPETENCIES.keys())


def get_competency_display_name(code: str) -> str:
    """Get display name for a competency code."""
    return COMPETENCIES.get(code, {}).get('name', code)
