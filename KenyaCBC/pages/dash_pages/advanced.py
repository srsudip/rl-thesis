"""
Advanced Evaluation & Benchmarks Page

Provides:
- Jordan et al. (2020) benchmark runner
- Test suite runner
- Multi-seed evaluation
- Baseline comparisons
- Hyperparameter tuning
- HITL retraining
"""

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

dash.register_page(__name__, path='/advanced', name='Advanced',
                   order=4, title='Advanced | Kenya CBC')


# ─────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────

layout = dbc.Container([
    html.H4([html.I(className="fas fa-flask me-2"), "Advanced Evaluation & Benchmarks"],
            className="mb-4 mt-2"),
    html.P("Run benchmarks, tests, and advanced evaluation tools.",
           className="text-muted mb-4"),

    # ── Row 1: Benchmarks & Tests ──
    dbc.Row([
        # Jordan et al. Benchmarks
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Jordan et al. (2020) Benchmarks"
                ], className="fw-bold"),
                dbc.CardBody([
                    html.P(["Run DKW confidence bands, PBP-t convergence analysis, "
                           "and per-pathway accuracy with 95% ",
                           html.Span("CIs", id="adv-ci-tooltip-1",
                                     style={'textDecoration': 'underline dotted', 'cursor': 'help'}),
                           "."],
                           className="text-muted small mb-3"),
                    dbc.Tooltip(
                        "CI = Confidence Interval. A 95% CI means we are 95% confident "
                        "the true accuracy lies within this range. Narrower intervals "
                        "indicate more reliable estimates.",
                        target="adv-ci-tooltip-1", placement="top"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Trials", className="small"),
                            dbc.Input(id='adv-bm-trials', type='number',
                                      value=3, min=1, max=30, size="sm"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Episodes", className="small"),
                            dbc.Input(id='adv-bm-episodes', type='number',
                                      value=200, min=50, max=1000, size="sm"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Students", className="small"),
                            dbc.Input(id='adv-bm-students', type='number',
                                      value=100, min=20, max=500, size="sm"),
                        ], width=4),
                    ], className="mb-3"),
                    dbc.Button([
                        html.I(className="fas fa-play me-1"),
                        "Run Benchmarks"
                    ], id='adv-run-benchmarks-btn', color="primary", className="w-100"),
                    dcc.Loading(
                        html.Div(id='adv-benchmark-results', className="mt-3"),
                        type="circle",
                    ),
                ]),
            ], className="h-100"),
        ], md=6, className="mb-4"),

        # Test Suite
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-vial me-2"),
                    "Test Suite (20 Tests)"
                ], className="fw-bold"),
                dbc.CardBody([
                    html.P("Run automated tests across 6 categories: "
                           "data generator, environment, agent, trainer, HITL, benchmarks.",
                           className="text-muted small mb-3"),
                    dbc.Button([
                        html.I(className="fas fa-play me-1"),
                        "Run All Tests"
                    ], id='adv-run-tests-btn', color="success", className="w-100 mb-3"),
                    dcc.Loading(
                        html.Div(id='adv-test-results', className="mt-2"),
                        type="circle",
                    ),
                ]),
            ], className="h-100"),
        ], md=6, className="mb-4"),
    ]),

    # ── Row 2: Multi-seed, Baselines, Hyperparameters ──
    dbc.Row([
        # Multi-seed Evaluation
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-random me-2"),
                    "Multi-Seed Evaluation"
                ], className="fw-bold"),
                dbc.CardBody([
                    html.P("Train with multiple random seeds and report mean ± std.",
                           className="text-muted small mb-2"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Seeds", className="small"),
                        dbc.Input(id='adv-num-seeds', type='number',
                                  value=5, min=2, max=10, size="sm"),
                        dbc.Button("Run", id='adv-multiseed-btn',
                                   color="primary", outline=True, size="sm"),
                    ], size="sm", className="mb-2"),
                    dcc.Loading(
                        html.Div(id='adv-multiseed-result', className="small"),
                        type="dot",
                    ),
                ]),
            ]),
        ], md=4, className="mb-4"),

        # Baseline Comparisons
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-balance-scale me-2"),
                    "Baseline Comparisons"
                ], className="fw-bold"),
                dbc.CardBody([
                    html.P("Compare DQN against random, rule-based, and ML baselines.",
                           className="text-muted small mb-2"),
                    dbc.Button([
                        html.I(className="fas fa-chart-line me-1"),
                        "Compare All"
                    ], id='adv-baselines-btn', color="info", outline=True,
                       size="sm", className="w-100 mb-2"),
                    dcc.Loading(
                        html.Div(id='adv-baselines-result', className="small"),
                        type="dot",
                    ),
                ]),
            ]),
        ], md=4, className="mb-4"),

        # Hyperparameter Tuning
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-sliders-h me-2"),
                    "Hyperparameter Tuning"
                ], className="fw-bold"),
                dbc.CardBody([
                    html.P(["Grid search over ",
                           html.Span("learning rate", id="adv-lr-tooltip",
                                     style={'textDecoration': 'underline dotted', 'cursor': 'help'}),
                           ", hidden dim, gamma."],
                           className="text-muted small mb-2"),
                    dbc.Tooltip(
                        "LR = Learning Rate. Controls how much the model updates its weights "
                        "each training step. Too high → unstable training. Too low → slow convergence. "
                        "Typical range: 0.0001 – 0.01.",
                        target="adv-lr-tooltip", placement="top"),
                    dbc.Button([
                        html.I(className="fas fa-search me-1"),
                        "Grid Search"
                    ], id='adv-hyperparam-btn', color="warning", outline=True,
                       size="sm", className="w-100 mb-2"),
                    dcc.Loading(
                        html.Div(id='adv-hyperparam-result', className="small"),
                        type="dot",
                    ),
                ]),
            ]),
        ], md=4, className="mb-4"),
    ]),

    # ── Row 3: HITL Retraining ──
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-sync me-2"),
                    "Human-in-the-Loop Retraining"
                ], className="fw-bold"),
                dbc.CardBody([
                    html.P("Retrain the model incorporating approved teacher overrides "
                           "as high-reward experiences.", className="text-muted small mb-2"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Episodes", className="small"),
                        dbc.Input(id='adv-retrain-episodes', type='number',
                                  value=100, min=10, max=500, size="sm"),
                        dbc.Button([
                            html.I(className="fas fa-graduation-cap me-1"),
                            "Retrain"
                        ], id='adv-retrain-btn', color="success", outline=True, size="sm"),
                    ], size="sm"),
                    dcc.Loading(
                        html.Div(id='adv-retrain-result', className="small mt-2"),
                        type="dot",
                    ),
                ]),
            ]),
        ], md=8, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-file-csv me-2"),
                    "Data Export"
                ], className="fw-bold"),
                dbc.CardBody([
                    dbc.Button([html.I(className="fas fa-download me-1"), "Export CSV"],
                               id='adv-export-btn', color="secondary", outline=True,
                               size="sm", className="w-100 mb-2"),
                    html.Div(id='adv-export-result', className="small"),
                ]),
            ]),
        ], md=4, className="mb-4"),
    ]),

], fluid=True)


# ─────────────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────────────

@callback(
    Output('adv-benchmark-results', 'children'),
    Input('adv-run-benchmarks-btn', 'n_clicks'),
    State('adv-bm-trials', 'value'),
    State('adv-bm-episodes', 'value'),
    State('adv-bm-students', 'value'),
    prevent_initial_call=True,
)
def run_benchmarks(n_clicks, n_trials, n_episodes, n_students):
    if not n_clicks:
        return no_update
    try:
        from benchmarks.benchmark import RLBenchmark
        bm = RLBenchmark(
            n_students=n_students or 100,
            n_episodes=n_episodes or 200,
            n_trials=n_trials or 3,
            verbose=True,
        )
        results = bm.run_all()
        s = results['summary']
        pp = results['per_pathway']
        conv = results['convergence']

        rows = []
        for pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']:
            p = pp[pw]
            rows.append(html.Tr([
                html.Td(pw, className="small"),
                html.Td(f"{p['mean_accuracy']:.1%}", className="small fw-bold"),
                html.Td(f"[{p['ci_lower']:.1%}, {p['ci_upper']:.1%}]", className="small"),
            ]))
        o = pp['OVERALL']
        rows.append(html.Tr([
            html.Td("OVERALL", className="small fw-bold"),
            html.Td(f"{o['mean_accuracy']:.1%}", className="small fw-bold text-primary"),
            html.Td(f"[{o['ci_lower']:.1%}, {o['ci_upper']:.1%}]", className="small"),
        ]))

        final_ep_acc = conv['mean_accuracy'][-1] if conv['mean_accuracy'] else 0

        return html.Div([
            dbc.Alert(f"✓ {s['n_trials']} trials completed in {s['total_time_s']}s",
                      color="success", className="py-2 small"),
            html.P(f"DKW ε = {results['performance_distribution']['dkw_epsilon']:.4f} | "
                   f"50% acc at episode {conv.get('first_50pct_episode', '—')} | "
                   f"Final smoothed acc: {final_ep_acc:.1%}",
                   className="small text-muted"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Pathway", className="small"),
                    html.Th("Accuracy", className="small"),
                    html.Th([
                        "95% CI ",
                        html.I(className="fas fa-info-circle text-muted",
                               id="adv-ci-tooltip-2", style={'cursor': 'help', 'fontSize': '0.7rem'}),
                    ], className="small"),
                ])),
                html.Tbody(rows),
            ], bordered=True, size="sm", className="mb-0"),
            dbc.Tooltip(
                "95% Confidence Interval — the range within which the true accuracy "
                "likely falls. Computed from multiple independent training trials.",
                target="adv-ci-tooltip-2", placement="top"),
        ])
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger", className="small")


@callback(
    Output('adv-test-results', 'children'),
    Input('adv-run-tests-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def run_tests(n_clicks):
    if not n_clicks:
        return no_update
    try:
        import subprocess
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests/test_all.py', '-v', '--tb=short'],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        output = result.stdout + result.stderr
        # Count pass/fail
        passed = output.count(' PASSED')
        failed = output.count(' FAILED')
        errors = output.count(' ERROR')
        total = passed + failed + errors

        if failed == 0 and errors == 0:
            color = "success"
            icon = "✓"
        else:
            color = "danger"
            icon = "✗"

        return html.Div([
            dbc.Alert(f"{icon} {passed}/{total} tests passed", color=color, className="py-2 small"),
            html.Pre(output[-2000:] if len(output) > 2000 else output,
                     className="small bg-dark text-light p-2 rounded",
                     style={'maxHeight': '300px', 'overflow': 'auto', 'fontSize': '0.75rem'}),
        ])
    except subprocess.TimeoutExpired:
        return dbc.Alert("Tests timed out (>120s)", color="warning", className="small")
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger", className="small")


@callback(
    Output('adv-multiseed-result', 'children'),
    Input('adv-multiseed-btn', 'n_clicks'),
    State('adv-num-seeds', 'value'),
    prevent_initial_call=True,
)
def run_multiseed(n_clicks, n_seeds):
    if not n_clicks:
        return no_update
    from pages.dashboard import get_data_manager
    dm = get_data_manager()
    if not dm.has_data():
        return dbc.Alert("Generate data first", color="warning", className="small py-1")
    try:
        from src.rl.evaluation import multi_seed_evaluation
        seeds = list(range(42, 42 + (n_seeds or 5)))
        results = multi_seed_evaluation(dm.data, seeds=seeds, episodes=200, verbose=True)
        acc = results['accuracy']
        return html.Div([
            html.Strong(f"{acc['mean']:.1%} ± {acc['std']:.1%}", className="text-primary"),
            html.Br(),
            html.Span([
                "95% CI: ",
                html.Span(f"[{acc['ci_95_lower']:.1%}, {acc['ci_95_upper']:.1%}]",
                          id="adv-ci-tooltip-3", style={'cursor': 'help'}),
            ]),
            dbc.Tooltip(
                "The range of accuracy values we'd expect 95% of the time "
                "if we repeated training with different random seeds.",
                target="adv-ci-tooltip-3", placement="bottom"),
        ])
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger", className="small py-1")


@callback(
    Output('adv-baselines-result', 'children'),
    Input('adv-baselines-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def run_baselines(n_clicks):
    if not n_clicks:
        return no_update
    from pages.dashboard import get_data_manager
    dm = get_data_manager()
    if not dm.has_data():
        return dbc.Alert("Generate data first", color="warning", className="small py-1")
    try:
        import math
        from src.rl.evaluation import run_all_baseline_comparisons
        results = run_all_baseline_comparisons(dm.data, seeds=[42, 123, 456],
                                                episodes=200, verbose=True)

        def safe_pct(val, default="—"):
            """Format a value as percentage, handling None/NaN."""
            if val is None:
                return default
            try:
                if math.isnan(val):
                    return default
                return f"{val:.1%}"
            except (TypeError, ValueError):
                return default

        rows = []
        # DQN row
        dqn = results.get('dqn_results', {})
        dqn_mean = safe_pct(dqn.get('mean'))
        dqn_std = dqn.get('std', 0)
        dqn_str = dqn_mean
        if dqn_std and not (isinstance(dqn_std, float) and math.isnan(dqn_std)):
            dqn_str += f" ± {dqn_std:.1%}"

        rows.append(html.Tr([
            html.Td("DQN (Ours)", className="small fw-bold"),
            html.Td(dqn_str, className="small fw-bold text-primary"),
            html.Td("—", className="small text-muted"),
            html.Td("—", className="small text-muted"),
        ]))

        # Baseline rows
        comparisons = results.get('statistical_comparisons', {})
        for name, br in results.get('baseline_results', {}).items():
            bl_mean = safe_pct(br.get('mean'))
            bl_std = br.get('std', 0)
            bl_str = bl_mean
            if bl_std and not (isinstance(bl_std, float) and math.isnan(bl_std)):
                bl_str += f" ± {bl_std:.1%}"

            comp = comparisons.get(name, {})
            # Extract p-value — handle both nested and flat structures
            p_val = comp.get('paired_ttest', {}).get('p_value', None)
            is_sig = comp.get('paired_ttest', {}).get('significant', False)
            effect = comp.get('effect_size', {}).get('interpretation', '')
            sig_mark = " ✓" if is_sig else ""

            p_str = f"p={p_val:.3f}{sig_mark}" if p_val is not None and not math.isnan(p_val) else "—"

            rows.append(html.Tr([
                html.Td(name, className="small"),
                html.Td(bl_str, className="small"),
                html.Td(p_str, className="small"),
                html.Td(effect or "—", className="small"),
            ]))

        return dbc.Table([
            html.Thead(html.Tr([html.Th("Method", className="small"),
                                html.Th("Accuracy", className="small"),
                                html.Th("Significance", className="small"),
                                html.Th("Effect Size", className="small")])),
            html.Tbody(rows),
        ], bordered=True, size="sm", hover=True) if rows else html.Span("No results")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Error: {e}", color="danger", className="small py-1")


@callback(
    Output('adv-hyperparam-result', 'children'),
    Input('adv-hyperparam-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def run_hyperparams(n_clicks):
    if not n_clicks:
        return no_update
    from pages.dashboard import get_data_manager
    dm = get_data_manager()
    if not dm.has_data():
        return dbc.Alert("Generate data first", color="warning", className="small py-1")
    try:
        from src.rl.evaluation import hyperparameter_grid_search
        results = hyperparameter_grid_search(dm.data, n_trials_per_config=2,
                                              episodes=150, verbose=True)
        best = results.get('best_config', {})
        best_acc = results.get('best_accuracy', 0)
        all_res = results.get('all_results', [])
        
        # Best config as readable badges
        param_labels = {
            'learning_rate': 'LR',
            'hidden_dim': 'Hidden',
            'epsilon_decay': 'ε Decay',
            'gamma': 'γ',
        }
        config_badges = [
            dbc.Badge(f"{param_labels.get(k, k)}: {v}", color="light",
                      text_color="dark", className="me-1 mb-1")
            for k, v in best.items()
        ]
        
        # Top 5 results table
        top_rows = []
        for i, r in enumerate(all_res[:5]):
            cfg = r['config']
            is_best = (i == 0)
            cls = "small fw-bold text-success" if is_best else "small"
            top_rows.append(html.Tr([
                html.Td(f"{'★' if is_best else ''} #{i+1}", className=cls),
                html.Td(f"{r['mean_accuracy']:.1%} ± {r['std_accuracy']:.1%}", className=cls),
                html.Td(str(cfg.get('learning_rate', '—')), className="small"),
                html.Td(str(cfg.get('hidden_dim', '—')), className="small"),
                html.Td(str(cfg.get('epsilon_decay', '—')), className="small"),
            ]))
        
        return html.Div([
            html.Div([
                html.I(className="fas fa-trophy text-success me-2"),
                html.Strong(f"Best: {best_acc:.1%}", className="text-success"),
            ], className="mb-2"),
            html.Div(config_badges, className="mb-2"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("#", className="small"),
                                    html.Th("Accuracy", className="small"),
                                    html.Th([
                                        "LR ",
                                        html.I(className="fas fa-info-circle text-muted",
                                               id="adv-lr-tooltip-2",
                                               style={'cursor': 'help', 'fontSize': '0.7rem'}),
                                    ], className="small"),
                                    html.Th("Hidden", className="small"),
                                    html.Th("Decay", className="small")])),
                html.Tbody(top_rows),
            ], bordered=True, size="sm", hover=True) if top_rows else None,
            dbc.Tooltip(
                "Learning Rate — controls update step size during training. "
                "Smaller values train slowly but stably; larger values converge "
                "faster but risk overshooting.",
                target="adv-lr-tooltip-2", placement="top") if top_rows else None,
        ])
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger", className="small py-1")


@callback(
    Output('adv-retrain-result', 'children'),
    Input('adv-retrain-btn', 'n_clicks'),
    State('adv-retrain-episodes', 'value'),
    prevent_initial_call=True,
)
def retrain_with_feedback(n_clicks, episodes):
    if not n_clicks:
        return no_update
    from pages.dashboard import get_data_manager
    dm = get_data_manager()
    result = dm.retrain_with_feedback(n_episodes=episodes or 100)
    if result.get('success'):
        return dbc.Alert(
            f"✓ Retrained with {result['overrides_incorporated']} overrides. "
            f"New accuracy: {result['accuracy']:.1%}",
            color="success", className="small py-1")
    return dbc.Alert(f"Failed: {result.get('error', 'Unknown')}", color="danger", className="small py-1")


@callback(
    Output('adv-export-result', 'children'),
    Input('adv-export-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def export_csv(n_clicks):
    if not n_clicks:
        return no_update
    from pages.dashboard import get_data_manager
    dm = get_data_manager()
    result = dm.export_data_to_csv()
    if result.get('success'):
        return dbc.Alert(f"✓ Exported to {result['output_dir']}", color="success", className="small py-1")
    return dbc.Alert(f"Failed: {result.get('error')}", color="danger", className="small py-1")


