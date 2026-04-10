"""
About Page - System Information, CBC Structure, Methodology
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

dash.register_page(__name__, path='/about', name='About',
                   title='Kenya CBC - About', order=5)


layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4([html.I(className="fas fa-info-circle me-2"), "About This System"], className="mb-0"),
            html.Small("Kenya CBC Pathway Recommendation System", className="text-muted"),
        ])
    ], className="mb-4 pt-2"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-graduation-cap me-2"), "Kenya CBC Structure"]),
                dbc.CardBody([
                    html.P("2-6-3-3-3 Education System:", className="fw-bold mb-2"),
                    html.Ul([
                        html.Li("2 years Pre-Primary"),
                        html.Li("6 years Primary (Grades 1-6) — assessed by KPSEA"),
                        html.Li("3 years Junior Secondary (Grades 7-9) — assessed by KJSEA"),
                        html.Li("3 years Senior Secondary (Grades 10-12)"),
                        html.Li("3 years Tertiary"),
                    ], className="small mb-3"),

                    html.P("Senior School Pathways:", className="fw-bold mb-2"),
                    dbc.Row([
                        dbc.Col([html.Div([
                            html.Span("🔬", style={'fontSize': '1.5rem'}),
                            html.H6("STEM", className="mt-1 mb-0"),
                            html.Small("Pure Sciences, Applied Sciences, Technical & Engineering, Careers & Technology",
                                      className="text-muted"),
                        ], className="text-center p-2 bg-light rounded")], md=4),
                        dbc.Col([html.Div([
                            html.Span("📚", style={'fontSize': '1.5rem'}),
                            html.H6("Social Sciences", className="mt-1 mb-0"),
                            html.Small("Humanities & Business, Languages & Literature",
                                      className="text-muted"),
                        ], className="text-center p-2 bg-light rounded")], md=4),
                        dbc.Col([html.Div([
                            html.Span("🎨", style={'fontSize': '1.5rem'}),
                            html.H6("Arts & Sports", className="mt-1 mb-0"),
                            html.Small("Performing Arts, Visual Arts, Sports & Recreation",
                                      className="text-muted"),
                        ], className="text-center p-2 bg-light rounded")], md=4),
                    ], className="mb-3"),

                    html.P("KNEC Grading Scale (KJSEA 2025):", className="fw-bold mb-2"),
                    dbc.Table([
                        html.Thead(html.Tr([html.Th("Level"), html.Th("Points"), html.Th("Range"), html.Th("Label")])),
                        html.Tbody([
                            html.Tr([html.Td("EE1"), html.Td("8"), html.Td("90-100%"), html.Td("Exceptional")]),
                            html.Tr([html.Td("EE2"), html.Td("7"), html.Td("75-89%"), html.Td("Very Good")]),
                            html.Tr([html.Td("ME1"), html.Td("6"), html.Td("58-74%"), html.Td("Good")]),
                            html.Tr([html.Td("ME2"), html.Td("5"), html.Td("41-57%"), html.Td("Fair")]),
                            html.Tr([html.Td("AE1"), html.Td("4"), html.Td("31-40%"), html.Td("Needs Improvement")]),
                            html.Tr([html.Td("AE2"), html.Td("3"), html.Td("21-30%"), html.Td("Below Average")]),
                            html.Tr([html.Td("BE1"), html.Td("2"), html.Td("11-20%"), html.Td("Well Below Avg")]),
                            html.Tr([html.Td("BE2"), html.Td("1"), html.Td("1-10%"), html.Td("Minimal")]),
                        ])
                    ], bordered=True, size="sm"),

                    html.P(["Placement: ", html.Strong("KJSEA (60%) + SBA (20%) + KPSEA (20%)")],
                          className="small text-muted"),
                ])
            ], className="shadow-sm")
        ], lg=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-microchip me-2"), "System Architecture"]),
                dbc.CardBody([
                    html.H6("Recommendation Engine"),
                    html.Ul([
                        html.Li([html.Strong("Primary: "), "KNEC pathway suitability — weighted average of subject scores per pathway (%)"]),
                        html.Li([html.Strong("Thresholds: "), "STEM ≥ 20%, Social Sciences / Arts & Sports ≥ 25% (KNEC composite) + core subjects ≥ AE1 (31%)"]),
                        html.Li([html.Strong("AI Supplement: "), "DQN (Double + Dueling + PER) trained on student data"]),
                        html.Li([html.Strong("HITL: "), "Student requests pathway change → teacher approves/rejects"]),
                    ], className="small mb-3"),

                    html.H6("Technical Stack"),
                    html.Ul([
                        html.Li("Python / NumPy (no PyTorch dependency)"),
                        html.Li("Plotly Dash / Bootstrap 5"),
                        html.Li("IRT-based synthetic data generation"),
                        html.Li("Jordan et al. (2020) statistical benchmarks"),
                    ], className="small mb-3"),

                    html.H6("Data Generation"),
                    html.Ul([
                        html.Li("2-Parameter IRT model for item responses"),
                        html.Li("7 CBC Core Competencies as latent abilities"),
                        html.Li("10 subjects × 6 grades (4-9) = 60 assessment points"),
                        html.Li("Subject scores aggregated from strand/sub-strand indicators"),
                    ], className="small mb-3"),

                    html.H6("Evaluation"),
                    html.Ul([
                        html.Li("20 automated tests across 6 categories"),
                        html.Li("Multi-seed evaluation with 95% confidence intervals"),
                        html.Li("Baseline comparisons (Random, Rule-based, RandomForest)"),
                        html.Li("DKW bands, PBP-t convergence analysis"),
                    ], className="small"),
                ])
            ], className="shadow-sm mb-3"),

            dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-university me-2"), "Research"]),
                dbc.CardBody([
                    html.P([html.Strong("ScaDS.AI Dresden-Leipzig"), html.Br(),
                           "Center for Scalable Data Analytics and Artificial Intelligence"], className="small mb-2"),
                    html.P([html.Strong("Universität Leipzig"), html.Br(),
                           "Bachelor's Thesis — Computer Science"], className="small mb-0"),
                ])
            ], className="shadow-sm"),
        ], lg=6),
    ]),
], fluid=True)
