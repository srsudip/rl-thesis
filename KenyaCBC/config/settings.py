"""
General settings and directory configuration.
"""
from pathlib import Path

# =====================================================
# Directory Configuration
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = DATA_DIR / "processed"
PAGES_DIR = BASE_DIR / "pages"

# =====================================================
# Grade Configuration
# =====================================================
LOWER_PRIMARY_GRADES = (1, 2, 3)
UPPER_PRIMARY_GRADES = (4, 5, 6)
JUNIOR_SECONDARY_GRADES = (7, 8, 9)
ALL_GRADES = LOWER_PRIMARY_GRADES + UPPER_PRIMARY_GRADES + JUNIOR_SECONDARY_GRADES

# =====================================================
# Assessment Weights for Pathway Placement
# =====================================================
# 20% Grade 6 KPSEA + 20% School-based G7&8 + 60% Grade 9 KJSEA
PLACEMENT_WEIGHTS = {
    'kpsea_grade6': 0.20,
    'school_based_g7_g8': 0.20,
    'kjsea_grade9': 0.60
}

# =====================================================
# Performance Classification Thresholds
# =====================================================
PERFORMANCE_LEVELS = {
    'below_expectation': (0, 25),
    'approaching_expectation': (25, 50),
    'meeting_expectation': (50, 75),
    'exceeding_expectation': (75, 100)
}

# =====================================================
# Dashboard Configuration
# =====================================================
DASHBOARD_CONFIG = {
    'theme': 'plotly_white',
    'colors': {
        'STEM': '#1f77b4',
        'SOCIAL_SCIENCES': '#ff7f0e',
        'ARTS_SPORTS': '#2ca02c',
        'below_expectation': '#d62728',
        'approaching_expectation': '#ff7f0e',
        'meeting_expectation': '#2ca02c',
        'exceeding_expectation': '#1f77b4'
    },
    'plot_height': 400,
    'plot_width': 600
}

# =====================================================
# Utility Functions
# =====================================================
def get_grade_level(grade: int) -> str:
    """Return the level category for a given grade."""
    if grade in LOWER_PRIMARY_GRADES:
        return 'lower_primary'
    elif grade in UPPER_PRIMARY_GRADES:
        return 'upper_primary'
    elif grade in JUNIOR_SECONDARY_GRADES:
        return 'junior_secondary'
    else:
        raise ValueError(f"Invalid grade: {grade}")


def score_to_performance_level(score: float) -> str:
    """Convert a score (0-100) to a performance level."""
    for level, (low, high) in PERFORMANCE_LEVELS.items():
        if low <= score < high:
            return level
    return 'exceeding_expectation' if score >= 75 else 'below_expectation'
