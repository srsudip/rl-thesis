"""
Shared pytest fixtures for the Kenya CBC test suite.

Session-scoped fixtures are generated once and reused across all tests
that request them, keeping the suite fast.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path for all test modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dashboard_data():
    """30-student dataset generated once for the whole test session."""
    from src.data.cbc_data_generator import generate_dashboard_data
    return generate_dashboard_data(n_students=30, seed=42, save_csv=False)


@pytest.fixture(scope="session")
def large_dashboard_data():
    """200-student dataset for audit / score-consistency tests."""
    from src.data.cbc_data_generator import generate_dashboard_data
    return generate_dashboard_data(n_students=200, seed=42, save_csv=False)


# ---------------------------------------------------------------------------
# RL fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def env(dashboard_data):
    """PathwayEnvironment built once per session."""
    from src.rl.environment import PathwayEnvironment
    return PathwayEnvironment(
        dashboard_data["assessments"],
        dashboard_data["competencies"],
        verbose=False,
    )


@pytest.fixture(scope="session")
def agent():
    """Untrained PathwayRecommendationAgent (lightweight — just checks structure)."""
    from src.rl.agent import PathwayRecommendationAgent
    return PathwayRecommendationAgent(verbose=False)


# ---------------------------------------------------------------------------
# HITL fixture (function-scoped so each test gets a clean manager)
# ---------------------------------------------------------------------------

@pytest.fixture()
def hitl_manager():
    from src.rl.hitl import HITLManager
    return HITLManager()
