"""Config package - exports pathway definitions, grading, RL config, and settings."""

# -- Pathways, grading, subjects --
from config.pathways import (
    PATHWAYS, SUB_PATHWAYS, SUBJECT_NAMES, PATHWAY_SUBJECT_WEIGHTS,
    CBC_GRADING_SYSTEM, PLACEMENT_WEIGHTS, MAX_CLUSTER_WEIGHT,
    PATHWAY_MIN_THRESHOLD, BELOW_EXPECTATIONS_THRESHOLD,
    PATHWAY_CORE_SUBJECTS, PATHWAY_KEY_SUBJECTS,
    UPPER_PRIMARY_SUBJECTS, JUNIOR_SECONDARY_SUBJECTS,
    PATHWAY_RESULT_SLIP_SUBJECTS, SENIOR_SCHOOL_CORE_SUBJECTS,
    REL_SUBJECTS,
    get_cbc_grade, compute_cluster_weights, recommend_pathway,
    get_pathway_index, get_pathway_from_index, get_pathway_name,
    get_pathway_display_name, check_pathway_eligibility, has_pathway_tie,
    is_below_expectations, get_religious_ed_code,
    KJSEA_SUBJECT_CODES, KPSEA_PAPER_CODES,
)

# -- Settings (directories, grade ranges, performance levels) --
from config.settings import (
    BASE_DIR, CONFIG_DIR, DATA_DIR, DB_DIR, MODELS_DIR, OUTPUT_DIR, PAGES_DIR,
    LOWER_PRIMARY_GRADES, UPPER_PRIMARY_GRADES, JUNIOR_SECONDARY_GRADES,
    ALL_GRADES,
    PERFORMANCE_LEVELS, DASHBOARD_CONFIG,
    get_grade_level, score_to_performance_level,
)

# -- RL hyperparameters --
from config.rl_config import (
    RL_CONFIG, REWARD_CONFIG, CONSISTENCY_CONFIG, TRAINING_CONFIG,
)

# -- Competencies (7 Core CBC Competencies) --
from config.competencies import (
    COMPETENCIES, COMPETENCY_SUBJECT_WEIGHTS, IDEAL_PATHWAY_PROFILES,
    get_competency_names, get_competency_display_name,
)

__all__ = [
    # Pathways
    'PATHWAYS', 'SUB_PATHWAYS', 'SUBJECT_NAMES', 'PATHWAY_SUBJECT_WEIGHTS',
    'CBC_GRADING_SYSTEM', 'PLACEMENT_WEIGHTS', 'MAX_CLUSTER_WEIGHT',
    'PATHWAY_MIN_THRESHOLD', 'BELOW_EXPECTATIONS_THRESHOLD', 'PATHWAY_KEY_SUBJECTS',
    'UPPER_PRIMARY_SUBJECTS', 'JUNIOR_SECONDARY_SUBJECTS',
    'PATHWAY_RESULT_SLIP_SUBJECTS', 'SENIOR_SCHOOL_CORE_SUBJECTS',
    'REL_SUBJECTS', 'KJSEA_SUBJECT_CODES', 'KPSEA_PAPER_CODES',
    'get_cbc_grade', 'compute_cluster_weights', 'recommend_pathway',
    'get_pathway_index', 'get_pathway_from_index', 'get_pathway_name',
    'get_pathway_display_name', 'check_pathway_eligibility', 'has_pathway_tie',
    'is_below_expectations', 'get_religious_ed_code',
    # Settings
    'BASE_DIR', 'CONFIG_DIR', 'DATA_DIR', 'DB_DIR', 'MODELS_DIR',
    'OUTPUT_DIR', 'PAGES_DIR',
    'LOWER_PRIMARY_GRADES', 'UPPER_PRIMARY_GRADES', 'JUNIOR_SECONDARY_GRADES',
    'ALL_GRADES', 'PERFORMANCE_LEVELS', 'DASHBOARD_CONFIG',
    'get_grade_level', 'score_to_performance_level',
    # RL
    'RL_CONFIG', 'REWARD_CONFIG', 'CONSISTENCY_CONFIG', 'TRAINING_CONFIG',
    # Competencies
    'COMPETENCIES', 'COMPETENCY_SUBJECT_WEIGHTS', 'IDEAL_PATHWAY_PROFILES',
    'get_competency_names', 'get_competency_display_name',
]
