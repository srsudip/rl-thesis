"""
Data Generation Module - Kenya CBC System

Supports both:
  1. IRT-based synthetic data (cbc_data_generator.py)
  2. Real CSV data loading (real_data_loader.py)
"""

from .cbc_data_generator import (
    generate_dashboard_data,
    load_dashboard_data,
    check_generated_files_exist,
    simulate_all,
    DATA_DIR
)

from .real_data_loader import (
    load_real_data_for_dashboard,
    compute_all_cosine_similarities,
    compute_psi,
    compute_pathway_suitability_by_grade,
    get_suggested_pathway_per_grade,
    generate_pathway_comparison,
)

__all__ = [
    'generate_dashboard_data',
    'load_dashboard_data',
    'check_generated_files_exist',
    'simulate_all',
    'DATA_DIR',
    'load_real_data_for_dashboard',
    'compute_all_cosine_similarities',
    'compute_psi',
    'compute_pathway_suitability_by_grade',
    'get_suggested_pathway_per_grade',
    'generate_pathway_comparison',
]
