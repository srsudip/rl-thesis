"""
Pages package for Kenya CBC Pathway System.

Multi-page Plotly Dash application with:
- Home: Data generation and model training
- Analysis: Pathway recommendations with AI explanations
- Transitions: Grade transition analysis
- Settings: Data editing and model management
"""

from pages.dashboard import create_app, get_data_manager, DataManager

__all__ = ['create_app', 'get_data_manager', 'DataManager']
