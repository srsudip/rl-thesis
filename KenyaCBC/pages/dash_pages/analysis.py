"""
Analysis Page — Reimagined
Layout: Recommendation → Slip + Subjects → Suitability Radar + History → Comparison + Override
"""
from dash import html, dcc, callback, Input, Output, State, register_page
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PATHWAYS, SUBJECT_NAMES
from config.pathways import get_cbc_grade, PATHWAY_MIN_THRESHOLD, _check_core_subjects, PATHWAY_KEY_SUBJECTS

register_page(__name__, path="/analysis", name="Analysis", title="Kenya CBC - Analysis")
from pages.dashboard import get_data_manager

PW_COLORS = {'STEM': '#2563EB', 'SOCIAL_SCIENCES': '#10B981', 'ARTS_SPORTS': '#F97316'}
PW_NAMES = {'STEM': 'STEM', 'SOCIAL_SCIENCES': 'Social Sciences', 'ARTS_SPORTS': 'Arts & Sports Science'}
CHART_FONT = dict(family='IBM Plex Sans, sans-serif', color='#0F172A')

SCORE_GRADIENT = [
    (10, '#FEE2E2'), (20, '#FECACA'), (30, '#FED7AA'), (40, '#FDE68A'),
    (57, '#A7F3D0'), (74, '#34D399'), (89, '#60A5FA'), (100, '#2563EB'),
]

def score_color(s):
    for t, c in SCORE_GRADIENT:
        if s <= t: return c
    return '#2563EB'

def score_text(s):
    return '#0F172A' if s < 57 else '#fff'

def bar_color(v):
    if v >= 75: return '#2563EB'
    if v >= 58: return '#60A5FA'
    if v >= 41: return '#10B981'
    if v >= 31: return '#F59E0B'
    return '#EF4444'


# ============================================================================
# LAYOUT
# ============================================================================
layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H4("Student Analysis", className="mb-0"), md=3),
        dbc.Col(dcc.Dropdown(id='analysis-student-selector', placeholder="Search student...",
                            searchable=True, clearable=True), md=6),
        dbc.Col(html.Div(id='analysis-quick-info', className="text-end"), md=3),
    ], className="mb-3 pt-2 align-items-center"),

    html.Div(id='analysis-no-student-warning'),

    # Row 1: Pathway Recommendation | KJSEA Slip
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-robot me-1"),
                " Pathway Recommendation",
            ], className="fw-bold"),
            dbc.CardBody(html.Div(id='analysis-recommendation'), className="p-2"),
        ]), md=5),
        dbc.Col(dbc.Card([
            dbc.CardHeader("KJSEA Result Slip"),
            dbc.CardBody(html.Div(id='analysis-result-slip')),
        ]), md=7),
    ], className="mb-3"),

    # Row 2: Your Feedback | Coaching Plan
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Your Feedback"),
            dbc.CardBody([
                html.P("Satisfied with this recommendation?",
                       className="mb-1", style={'fontSize': '0.83rem'}),
                dbc.RadioItems(
                    id='analysis-feedback-radio',
                    options=[
                        {'label': ' Yes', 'value': 'satisfied'},
                        {'label': ' No — prefer different', 'value': 'wants_different'},
                    ],
                    inline=True,
                    className="mb-2", style={'fontSize': '0.83rem'},
                ),
                html.Div(id='analysis-feedback-desired-wrapper', children=[
                    dcc.Dropdown(
                        id='analysis-feedback-desired',
                        options=[{'label': PATHWAYS[p]['name'], 'value': p}
                                for p in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']],
                        placeholder="Select preferred pathway...",
                        className="mb-2",
                        style={'fontSize': '0.83rem'},
                    ),
                ], style={'display': 'none'}),
                dbc.Button("Submit", id='analysis-feedback-submit',
                          color="dark", size="sm", className="mb-2"),
                html.Div(id='analysis-feedback-result', className="mb-1"),
            ]),
        ]), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Coaching Plan"),
            dbc.CardBody(html.Div(id='analysis-coaching-plan')),
        ]), md=8),
    ], className="mb-3"),

    # Row 3: Subject Scores + Pathway Alignment
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Subject Scores"),
            dbc.CardBody(dcc.Graph(id='analysis-subject-bar', style={'height': '360px'},
                                  config={'displayModeBar': False}))
        ]), md=7),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Pathway Alignment"),
            dbc.CardBody(dcc.Graph(id='analysis-pathway-radar', style={'height': '360px'},
                                  config={'displayModeBar': False}))
        ]), md=5),
    ], className="mb-3"),

    # Row 4: Performance Trajectory (full width)
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Performance Trajectory (Grade 4\u20139)"),
            dbc.CardBody([
                dcc.Graph(id='analysis-history-chart', style={'height': '420px'},
                         config={'displayModeBar': False}),
                html.Div(id='analysis-grade-detail', className="mt-2"),
            ])
        ]), md=12),
    ], className="mb-3"),

    # Row 5: Comparison + Request Change
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Pathway Comparison"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id='analysis-compare-pathway',
                        options=[{'label': PATHWAYS[p]['name'], 'value': p}
                                for p in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']],
                        placeholder="Compare with...", className="mb-2"), md=8),
                    dbc.Col(dbc.Button("Compare", id='analysis-compare-btn',
                                      color="dark", size="sm", className="w-100"), md=4),
                ]),
                html.Div(id='analysis-comparison-result'),
            ])
        ]), md=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Request Change"),
            dbc.CardBody([
                html.Div(id='analysis-override-status'),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id='analysis-change-pathway',
                        options=[{'label': PATHWAYS[p]['name'], 'value': p}
                                for p in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']],
                        placeholder="Pathway...", className="mb-2"), md=6),
                    dbc.Col(dbc.Input(id='analysis-change-by', placeholder="Name",
                                    size="sm", className="mb-2"), md=6),
                ]),
                dbc.Textarea(id='analysis-change-reason', placeholder="Reason...",
                            className="mb-2", style={'height': '50px', 'fontSize': '0.85rem'}),
                dbc.Button("Submit", id='analysis-submit-change',
                          color="warning", size="sm", className="w-100"),
                html.Div(id='analysis-change-result', className="mt-2"),
            ])
        ]), md=6),
    ], className="mb-3"),
], fluid=True)


# ============================================================================
# CALLBACKS
# ============================================================================

@callback([Output('analysis-student-selector', 'options'),
           Output('analysis-student-selector', 'value')],
          Input('selected-student-store', 'data'))
def populate(store):
    dm = get_data_manager()
    opts = dm.get_student_options() if dm.has_data() else []
    val = store if isinstance(store, (int, float)) and store else None
    return opts, val

@callback(Output('selected-student-store', 'data', allow_duplicate=True),
          Input('analysis-student-selector', 'value'), prevent_initial_call=True)
def sync(sid): return sid

@callback(Output('analysis-quick-info', 'children'), Input('analysis-student-selector', 'value'))
def quick(sid):
    if not sid: return ""
    dm = get_data_manager()
    if not dm.has_data(): return ""
    rec = dm.get_recommendation(sid)
    pw = PATHWAYS[rec['recommended_pathway']]
    return dbc.Badge([pw['icon'], " ", pw['name']], style={'backgroundColor': pw['color']}, className="fs-6")

@callback(Output('analysis-no-student-warning', 'children'), Input('analysis-student-selector', 'value'))
def warn(sid):
    if sid: return ""
    return dbc.Alert("Select a student above to view their analysis.", color="light",
                    className="py-2 text-center", style={'color': '#6b7280'})

@callback([Output('analysis-feedback-radio', 'value'),
           Output('analysis-feedback-desired', 'value'),
           Output('analysis-feedback-result', 'children')],
          Input('analysis-student-selector', 'value'))
def reset_feedback(sid):
    """Reset feedback form on student change; load existing feedback if any."""
    if not sid:
        return None, None, ""
    dm = get_data_manager()
    fb = dm.get_student_feedback(int(sid))
    if fb.get('feedback'):
        return fb['feedback'], fb.get('desired_pathway'), ""
    return None, None, ""


# ============================================================================
# MAIN UPDATE
# ============================================================================

@callback(
    [Output('analysis-result-slip', 'children'),
     Output('analysis-recommendation', 'children'),
     Output('analysis-subject-bar', 'figure'),
     Output('analysis-pathway-radar', 'figure'),
     Output('analysis-history-chart', 'figure'),
     Output('analysis-override-status', 'children')],
    Input('analysis-student-selector', 'value')
)
def update(sid):
    from dash import no_update
    dm = get_data_manager()

    empty_fig = go.Figure()
    empty_fig.update_layout(
        annotations=[{'text': 'Select a student', 'showarrow': False,
                      'font': {'color': '#bbb', 'size': 13, 'family': 'IBM Plex Sans, sans-serif'}}],
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=CHART_FONT)

    empty_card = html.Div(
        html.P("Select a student above to continue.",
               className="text-muted text-center", style={'fontSize': '0.82rem', 'marginTop': '1rem'}))

    if not sid or not dm.has_data():
        return empty_card, empty_card, empty_fig, empty_fig, empty_fig, ""

    try:
        return _update_inner(sid, dm, empty_fig)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        err = dbc.Alert(f"Error: {exc}", color="danger", className="small py-1")
        return err, err, empty_fig, empty_fig, empty_fig, ""


def _update_inner(sid, dm, empty_fig):

    slip = dm.get_knec_result_slip(sid)
    rec = dm.get_recommendation(sid)
    perf = dm.get_performance(sid)
    pathway = rec['recommended_pathway']
    pw_data = PATHWAYS[pathway]
    cw = slip['cluster_weights']

    # ---- RECOMMENDATION (with detailed reasoning) ----
    cos = dm.get_cosine_similarities(sid)
    psi = dm.get_psi(sid)
    psi_val = psi.get(pathway, 0)
    psi_pct = int(psi_val * 100)
    psi_label = "Excellent" if psi_val > 0.8 else "Strong" if psi_val > 0.6 else "Moderate" if psi_val > 0.3 else "Developing"
    psi_color = "success" if psi_val > 0.6 else "warning" if psi_val > 0.3 else "danger"

    override_badge = dbc.Badge("Override", color="warning", className="ms-2") if rec.get('is_override') else ""

    # Get subject scores and pathway weights for reasoning
    from config.pathways import PATHWAY_SUBJECT_WEIGHTS
    subj_scores = perf.get('subject_scores', {})
    pw_weights = PATHWAY_SUBJECT_WEIGHTS.get(pathway, {})

    # Top contributing subjects for this pathway (score × weight, sorted)
    contributions = []
    for s_key, weight in sorted(pw_weights.items(), key=lambda x: x[1], reverse=True):
        score = subj_scores.get(s_key, {}).get('score', 0)
        contributions.append({
            'subject': SUBJECT_NAMES.get(s_key, s_key),
            'score': score,
            'weight': weight,
            'contribution': score * weight,
            'grade': get_cbc_grade(score),
        })

    # Top 3 strengths and weaknesses for this pathway
    top_strengths = [c for c in contributions if c['score'] >= 58][:3]
    top_gaps = [c for c in reversed(contributions) if c['score'] < 41 and c['weight'] >= 0.5][:2]

    # Pathway strength score: avg of official PRIMARY key subjects (KICD/KNEC 2025)
    # These are the subjects that most directly predict success in the pathway
    _primary_keys = PATHWAY_KEY_SUBJECTS.get(pathway, {}).get('primary', [])
    _secondary_keys = PATHWAY_KEY_SUBJECTS.get(pathway, {}).get('secondary', [])
    _primary_scores = [subj_scores.get(sk, {}).get('score', 0) for sk in _primary_keys
                       if sk in subj_scores]
    pathway_strength_score = (sum(_primary_scores) / len(_primary_scores)
                              if _primary_scores else 0.0)

    # Per-primary-subject breakdown for display
    _primary_breakdown = []
    for _sk in _primary_keys:
        _sc = subj_scores.get(_sk, {}).get('score', None)
        if _sc is not None:
            _primary_breakdown.append({
                'key': _sk,
                'name': SUBJECT_NAMES.get(_sk, _sk),
                'score': _sc,
                'grade': get_cbc_grade(_sc),
                'gap': max(0, 70 - _sc),   # gap to ME1 ideal
            })

    pathway_strength_label = (
        "Excellent" if pathway_strength_score >= 75 else
        "Strong"    if pathway_strength_score >= 58 else
        "Moderate"  if pathway_strength_score >= 41 else
        "Developing"
    )
    pathway_strength_color = (
        "#2563EB" if pathway_strength_score >= 75 else
        "#10B981" if pathway_strength_score >= 58 else
        "#F59E0B" if pathway_strength_score >= 41 else
        "#EF4444"
    )

    # Runner-up pathway
    cw_sorted = sorted(cw.items(), key=lambda x: x[1], reverse=True)
    runner_up = None
    for pk, pv in cw_sorted:
        if pk != pathway:
            runner_up = (pk, pv)
            break
    margin = cw.get(pathway, 0) - (runner_up[1] if runner_up else 0)

    # Build reasoning text
    reason_parts = []
    if len(top_strengths) >= 2:
        strength_names = ' and '.join(c['subject'] for c in top_strengths[:2])
        reason_parts.append(f"Strong performance in {strength_names} contributes significantly to this pathway's cluster weight.")
    if runner_up:
        ru_name = PW_NAMES.get(runner_up[0], runner_up[0])
        if margin > 10:
            reason_parts.append(f"This pathway leads {ru_name} by {margin:.1f} points — a clear preference.")
        elif margin > 3:
            reason_parts.append(f"This pathway leads {ru_name} by {margin:.1f} points.")
        else:
            reason_parts.append(f"Close match with {ru_name} ({runner_up[1]:.1f}%) — only {margin:.1f}pt difference.")
    if top_gaps:
        gap_names = ', '.join(c['subject'] for c in top_gaps)
        reason_parts.append(f"Improving {gap_names} would further strengthen this recommendation.")

    # Eligibility list — recommended pathway highlighted
    raw_scores = {k: v.get('score', 0) for k, v in subj_scores.items()} if subj_scores else {}
    elig_items = []
    for pk, pv in cw_sorted:
        thr = PATHWAY_MIN_THRESHOLD.get(pk, 25)
        chosen = pk == pathway
        meets_cw = pv >= thr
        passes_core = _check_core_subjects(pk, raw_scores)
        fully_eligible = meets_cw and passes_core

        if chosen:
            label = "\u2605 Recommended"
        elif fully_eligible:
            label = "Eligible"
        elif meets_cw and not passes_core:
            label = "Core below AE1"
        else:
            label = f"Below {thr:.0f}%"

        label_col = '#10B981' if fully_eligible or chosen else '#F59E0B' if meets_cw else '#EF4444'
        pw_col = PW_COLORS.get(pk, '#666')

        if chosen:
            elig_items.append(html.Div([
                html.Div([
                    html.Span(PW_NAMES[pk], style={'fontWeight': '700', 'fontSize': '1rem', 'color': pw_col}),
                    html.Span(f" {pv:.1f}%", style={'fontFamily': 'IBM Plex Mono, monospace',
                              'fontWeight': '600', 'fontSize': '0.95rem', 'color': '#0F172A', 'marginLeft': '6px'}),
                ]),
                html.Span(label, style={'color': '#10B981', 'fontWeight': '600', 'fontSize': '0.82rem'}),
            ], style={'backgroundColor': '#F8FAFC', 'padding': '0.6rem 0.8rem',
                     'borderRadius': '8px', 'borderLeft': f'4px solid {pw_col}',
                     'marginBottom': '0.5rem'}))
        else:
            elig_items.append(html.Div([
                html.Small(PW_NAMES[pk], style={'color': pw_col, 'fontWeight': '400'}),
                html.Small(f" {pv:.1f}% ", style={'fontFamily': 'IBM Plex Mono, monospace', 'color': '#4a4a5a'}),
                html.Small(label, style={'color': label_col, 'fontSize': '0.73rem'}),
            ], className="mb-1", style={'paddingLeft': '0.8rem'}))

    # Top subject contributions mini-table
    contrib_rows = []
    for c in contributions[:5]:
        sc = c['score']
        contrib_rows.append(html.Div([
            html.Span(c['subject'], style={'fontSize': '0.78rem', 'width': '140px', 'display': 'inline-block'}),
            html.Span(f"{sc:.0f}%", className="score-cell",
                     style={'backgroundColor': score_color(sc), 'color': score_text(sc)}),
            html.Span(f" ×{c['weight']:.1f}", style={'fontSize': '0.7rem', 'color': '#7a7a8a',
                      'fontFamily': 'IBM Plex Mono, monospace', 'marginLeft': '4px'}),
        ], className="mb-1"))

    # ---- KJSEA RESULT SLIP (full — code + subject + score + level + pts) ----
    # Official key subjects per pathway (KICD/KNEC 2025) — not a weight threshold
    # PRIMARY: most critical for pathway success; SECONDARY: supporting signals
    rec_primary   = set(PATHWAY_KEY_SUBJECTS.get(pathway, {}).get('primary',   []))
    rec_secondary = set(PATHWAY_KEY_SUBJECTS.get(pathway, {}).get('secondary', []))

    # Per-subject: which OTHER pathways also list it as a primary key subject?
    # Used to show multi-pathway indicator badges in the slip
    _PW_SHORT = {'STEM': 'S', 'SOCIAL_SCIENCES': 'SS', 'ARTS_SPORTS': 'A'}
    _PW_COLOR = PW_COLORS  # {'STEM': '#2563EB', 'SOCIAL_SCIENCES': '#10B981', 'ARTS_SPORTS': '#F97316'}
    subj_other_pathways: dict = {}   # subject_key → list of non-recommended pathway keys
    for _pk, _pkd in PATHWAY_KEY_SUBJECTS.items():
        if _pk == pathway:
            continue
        for _sk in _pkd.get('primary', []):
            subj_other_pathways.setdefault(_sk, []).append(_pk)

    # Gap data from coaching plan (subject_key → {target, gap, priority})
    _coaching = dm.get_coaching_plan(int(sid))
    _name_to_key = {v: k for k, v in SUBJECT_NAMES.items()}
    gap_lookup = {}
    for _f in _coaching.get('focus_subjects', []):
        _sk = _name_to_key.get(_f.get('subject_name', ''), '')
        if _sk:
            gap_lookup[_sk] = {'target': _f.get('target', 0),
                                'gap': _f.get('gap', 0),
                                'priority': _f.get('priority', '')}
    slip_rows = []
    for s in sorted(slip['subjects'], key=lambda x: x.get('code', '')):
        sc = s.get('raw_score', 0)
        s_key = s.get('subject_key', '')
        is_primary   = s_key in rec_primary    # Key for RECOMMENDED pathway
        is_secondary = s_key in rec_secondary  # Supporting for recommended pathway

        # Row highlight: primary key subjects get a coloured left border
        row_style = {
            'backgroundColor': f"{pw_data['color']}18",
            'borderLeft': f"3px solid {pw_data['color']}",
        } if is_primary else (
            {'borderLeft': f"1px dashed {pw_data['color']}88"} if is_secondary else {}
        )

        _gap_info = gap_lookup.get(s_key)
        if _gap_info and _gap_info['gap'] > 0:
            _gap_txt = f"+{_gap_info['gap']:.0f}"
            _gap_col = '#EF4444' if _gap_info['priority'] == 'critical' else \
                       '#F59E0B' if _gap_info['priority'] == 'high' else '#3B82F6'
        elif _gap_info:
            _gap_txt, _gap_col = "✓", '#10B981'
        else:
            _gap_txt, _gap_col = "—", '#bbb'

        # Build subject cell: name + recommended-pathway badge + other-pathway badges
        subj_badges = []
        if is_primary:
            subj_badges.append(dbc.Badge(
                "★ key", color="light",
                style={'fontSize': '0.55rem', 'color': pw_data['color'],
                       'border': f"1px solid {pw_data['color']}",
                       'marginLeft': '4px', 'verticalAlign': 'middle'},
            ))
        elif is_secondary:
            subj_badges.append(dbc.Badge(
                "support", color="light",
                style={'fontSize': '0.52rem', 'color': pw_data['color'] + 'aa',
                       'border': f"1px dashed {pw_data['color']}88",
                       'marginLeft': '4px', 'verticalAlign': 'middle'},
            ))
        # Badges for other pathways where this subject is a primary key subject
        for _opk in subj_other_pathways.get(s_key, []):
            subj_badges.append(dbc.Badge(
                _PW_SHORT[_opk], color="light",
                title=f"Also key for {PW_NAMES[_opk]}",
                style={'fontSize': '0.50rem', 'color': _PW_COLOR[_opk],
                       'border': f"1px solid {_PW_COLOR[_opk]}",
                       'marginLeft': '3px', 'verticalAlign': 'middle', 'opacity': '0.7'},
            ))

        slip_rows.append(html.Tr([
            html.Td(s.get('code', ''), className="text-muted", style={'fontSize': '0.73rem'}),
            html.Td([s.get('subject', '')] + subj_badges,
                    style={'fontSize': '0.83rem',
                           'fontWeight': '600' if is_primary else '500'}),
            html.Td(html.Span(f"{sc:.0f}%", className="score-cell",
                    style={'backgroundColor': score_color(sc), 'color': score_text(sc)})),
            html.Td(s.get('performance_level', ''), style={'fontSize': '0.78rem', 'color': '#6b7280'}),
            html.Td(str(s.get('points', '')), style={'fontWeight': '700', 'fontSize': '0.88rem'}),
            html.Td(_gap_txt, style={'textAlign': 'center', 'fontWeight': '600',
                                     'fontSize': '0.8rem', 'color': _gap_col}),
        ], style=row_style))

    slip_table = html.Div([
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Code"), html.Th("Subject"),
                html.Th("Score"), html.Th("Level"), html.Th("Pts"),
                html.Th("Gap", style={'textAlign': 'center', 'fontSize': '0.7rem',
                                      'color': '#7a7a8a'}),
            ])),
            html.Tbody(slip_rows),
        ], bordered=False, hover=True, size="sm", className="mb-2 slip-table"),
        html.Div([
            html.Strong(f"Total: {slip['total_points']}/{slip['max_points']} points",
                       style={'fontSize': '0.88rem'}),
        ], className="text-end mb-2"),
        # Per-pathway key subject legend (all 3 pathways)
        html.Div([
            html.Div("Key subjects by pathway:", style={'fontSize': '0.68rem',
                     'color': '#6b7280', 'marginBottom': '3px', 'fontWeight': '600'}),
            *[html.Div([
                html.Span(
                    f"{_PW_SHORT[_pk]} ",
                    style={'fontWeight': '700', 'color': _PW_COLOR[_pk],
                           'fontSize': '0.65rem', 'fontFamily': 'IBM Plex Mono, monospace'}),
                html.Span(PW_NAMES[_pk] + ": ",
                          style={'fontSize': '0.65rem', 'color': '#374151', 'fontWeight': '500'}),
                html.Span(
                    ", ".join(SUBJECT_NAMES.get(s, s) for s in PATHWAY_KEY_SUBJECTS[_pk]['primary']),
                    style={'fontSize': '0.62rem', 'color': '#6b7280'}),
            ], style={'marginBottom': '1px'}) for _pk in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']],
            html.Div([
                html.Span("★ key", style={'fontSize': '0.6rem', 'color': pw_data['color'],
                           'border': f"1px solid {pw_data['color']}", 'borderRadius': '3px',
                           'padding': '0 3px', 'marginRight': '6px'}),
                html.Span("= primary key subject for recommended pathway  |  ",
                          style={'fontSize': '0.6rem', 'color': '#9ca3af'}),
                html.Span("S/SS/A", style={'fontSize': '0.6rem', 'color': '#6b7280',
                           'border': '1px solid #9ca3af', 'borderRadius': '3px',
                           'padding': '0 3px', 'marginRight': '6px'}),
                html.Span("= also key for that pathway  |  ",
                          style={'fontSize': '0.6rem', 'color': '#9ca3af'}),
                html.Span("support", style={'fontSize': '0.6rem', 'color': '#9ca3af',
                           'border': '1px dashed #9ca3af', 'borderRadius': '3px',
                           'padding': '0 3px', 'marginRight': '6px'}),
                html.Span("= supporting subject for recommended pathway",
                          style={'fontSize': '0.6rem', 'color': '#9ca3af'}),
            ], style={'marginTop': '4px'}),
        ], style={'backgroundColor': '#F8FAFC', 'border': '1px solid #E2E8F0',
                  'borderRadius': '6px', 'padding': '6px 10px', 'marginBottom': '6px'}),
        html.Div([
            *[html.Span(f" {lbl} ", className="badge me-1",
                        style={'backgroundColor': c, 'color': tc, 'fontSize': '0.6rem',
                               'fontWeight': '500', 'borderRadius': '4px'})
              for lbl, c, tc in [
                  ('BE2', '#FEE2E2', '#991B1B'), ('BE1', '#FECACA', '#991B1B'),
                  ('AE2', '#FED7AA', '#92400E'), ('AE1', '#F59E0B', '#fff'),
                  ('ME2', '#6EE7B7', '#065F46'), ('ME1', '#10B981', '#fff'),
                  ('EE2', '#60A5FA', '#fff'),    ('EE1', '#2563EB', '#fff'),
              ]],
        ]),
    ])

    # ---- SUBJECT BAR (horizontal, sorted by score) ----
    subj = perf.get('subject_scores', {})
    sorted_s = sorted(subj.keys(), key=lambda k: subj[k]['score'])
    names = [SUBJECT_NAMES.get(k, k) for k in sorted_s]
    vals = [subj[k]['score'] for k in sorted_s]

    subj_fig = go.Figure(data=[go.Bar(
        y=names, x=vals, orientation='h',
        marker_color=[bar_color(v) for v in vals],
        text=[f"{v:.0f}%" for v in vals], textposition='outside', textfont=dict(size=10),
    )])
    subj_fig.update_layout(margin=dict(l=10, r=40, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(range=[0, 105]),
                          yaxis=dict(automargin=True), height=360, font=CHART_FONT)

    # ---- PATHWAY ALIGNMENT BAR (just CW percentages) ----
    pw_keys = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
    pw_labels = [PW_NAMES[pk] for pk in pw_keys]
    cw_vals = [cw.get(pk, 0) for pk in pw_keys]
    pw_colors = [PW_COLORS[pk] for pk in pw_keys]
    thresholds = [PATHWAY_MIN_THRESHOLD.get(pk, 25) for pk in pw_keys]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Bar(
        x=pw_labels, y=cw_vals, marker_color=pw_colors,
        text=[f"{v:.1f}%" for v in cw_vals], textposition='outside',
        textfont=dict(size=12, family='IBM Plex Mono, monospace'),
    ))
    # Threshold annotation
    radar_fig.add_annotation(
        text="Thresholds: STEM \u2265 20%  \u00b7  SS / Arts \u2265 25%",
        xref="paper", yref="paper", x=0.5, y=1.05,
        showarrow=False, font=dict(size=8, color='#7a7a8a', family='IBM Plex Sans, sans-serif'))
    radar_fig.update_layout(
        showlegend=False, bargap=0.4,
        yaxis=dict(range=[0, max(cw_vals) * 1.25 if cw_vals else 100], gridcolor='#e8ebe4'),
        xaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=35, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', height=280, font=CHART_FONT,
    )

    # ---- PERFORMANCE HISTORY (ENHANCED) ----
    trans       = dm.get_transitions(sid)
    per_grade_pw = dm.get_suggested_pathway_per_grade(sid)
    subj_trans  = trans.get('subject_transitions', {})
    all_grades  = sorted(set(
        g for d in subj_trans.values() for g in d.get('grade_scores', {}).keys()))
    grade_labels = [f"G{g}" for g in all_grades]

    avg_scores, max_scores, min_scores = [], [], []
    for g in all_grades:
        scores = [d['grade_scores'][g]
                  for d in subj_trans.values() if g in d.get('grade_scores', {})]
        avg_scores.append(float(np.mean(scores)) if scores else 0.0)
        max_scores.append(float(max(scores)) if scores else 0.0)
        min_scores.append(float(min(scores)) if scores else 0.0)

    # Trend summary
    if len(avg_scores) >= 2:
        _delta = avg_scores[-1] - avg_scores[0]
        _half  = len(avg_scores) // 2
        _slope_recent = avg_scores[-1] - avg_scores[_half]
        if _delta > 5:
            trend_str, trend_col = f"↗ +{_delta:.0f} pts overall", '#10B981'
        elif _delta < -5:
            trend_str, trend_col = f"↘ {_delta:.0f} pts overall", '#EF4444'
        else:
            trend_str, trend_col = "→ Stable trajectory", '#F59E0B'
        if abs(_slope_recent) > 4:
            recent_note = (f"  ·  recently {'improving' if _slope_recent > 0 else 'declining'}"
                           f" ({_slope_recent:+.0f} pts)")
        else:
            recent_note = ""
    else:
        trend_str, trend_col, recent_note = "—", '#888', ""

    hist_fig = go.Figure()

    # ── Layer 1: performance band background zones ────────────────────
    _bands = [
        (0,  30,  '#FEE2E2', 'BE'),
        (30, 57,  '#FEF3C7', 'AE'),
        (57, 74,  '#D1FAE5', 'ME'),
        (74, 100, '#DBEAFE', 'EE'),
    ]
    for _y0, _y1, _fc, _lbl in _bands:
        hist_fig.add_shape(type='rect',
            x0=0, x1=1, xref='paper', y0=_y0, y1=_y1, yref='y',
            fillcolor=_fc, opacity=0.38, line_width=0, layer='below')
        hist_fig.add_annotation(
            x=1.01, y=(_y0 + _y1) / 2,
            xref='paper', yref='y',
            text=f"<b>{_lbl}</b>",
            showarrow=False,
            font=dict(size=8, color='#9a9aaa', family='IBM Plex Mono, monospace'),
            xanchor='left',
        )

    # ── Layer 3: subject score range envelope (min–max spread) ────────
    hist_fig.add_trace(go.Scatter(
        x=grade_labels + grade_labels[::-1],
        y=max_scores + min_scores[::-1],
        fill='toself',
        fillcolor='rgba(40,28,89,0.06)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Subject score range',
        hoverinfo='skip',
        showlegend=True,
    ))

    # ── Layer 4: Suitability lines for all 3 pathways ────────────────────
    # Solid lines = Cluster Weight (KNEC score-based, primary signal)
    # Dashed lines = PSI (competency-based, secondary signal)
    pw_history = dm.get_pathway_history(sid)
    # Track recommended pathway per grade for switch-point detection
    _rec_per_grade: dict = per_grade_pw  # {grade: pathway_key}
    if len(pw_history) > 0:
        for _pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']:
            _col_pw = PW_COLORS.get(_pw, '#666')
            _is_rec = _pw == pathway
            _rows_pw = pw_history[pw_history['pathway'] == _pw].sort_values('grade')
            if len(_rows_pw) == 0:
                continue
            _x = [f"G{int(g)}" for g in _rows_pw['grade']]

            # ── Cluster Weight line (primary — solid) ─────────────────
            _cw_y = [round(float(v), 1) for v in _rows_pw['cluster_weight']]
            hist_fig.add_trace(go.Scatter(
                x=_x, y=_cw_y,
                mode='lines+markers+text' if _is_rec else 'lines+markers',
                name=f'{PW_NAMES[_pw]} suitability',
                line=dict(width=2.8 if _is_rec else 1.4,
                          dash='solid',
                          color=_col_pw),
                marker=dict(size=8 if _is_rec else 4, color=_col_pw,
                            symbol='diamond' if _is_rec else 'circle-open',
                            line=dict(width=1.5 if _is_rec else 1,
                                      color=_col_pw)),
                opacity=1.0 if _is_rec else 0.55,
                text=[f"<b>{v:.0f}%</b>" for v in _cw_y] if _is_rec else None,
                textposition='top center',
                textfont=dict(size=8, color=_col_pw,
                              family='IBM Plex Mono, monospace'),
                hovertemplate=(f'<b>{PW_NAMES[_pw]}</b> Suitability (CW)<br>'
                               '%{x}: %{y:.1f}%<extra></extra>'),
                legendgroup=_pw,
                legendgrouptitle_text=None,
            ))

            # ── PSI line (secondary — dashed, no text) ────────────────
            _psi_y = [round(float(v) * 100, 1) for v in _rows_pw['psi']]
            hist_fig.add_trace(go.Scatter(
                x=_x, y=_psi_y,
                mode='lines',
                name=f'{PW_NAMES[_pw]} PSI',
                line=dict(width=1.0, dash='dot', color=_col_pw),
                opacity=0.4 if not _is_rec else 0.6,
                hovertemplate=(f'<b>{PW_NAMES[_pw]}</b> PSI<br>'
                               '%{x}: %{y:.1f}%<extra></extra>'),
                legendgroup=_pw,
                showlegend=False,
            ))

        # ── Switch-point annotations: mark grades where pathway changed ──
        _sorted_grades = sorted(_rec_per_grade.keys())
        for _i in range(1, len(_sorted_grades)):
            _g_prev = _sorted_grades[_i - 1]
            _g_curr = _sorted_grades[_i]
            if _rec_per_grade.get(_g_curr) != _rec_per_grade.get(_g_prev):
                _new_pw  = _rec_per_grade[_g_curr]
                _new_col = PW_COLORS.get(_new_pw, '#666')
                _pw_short = {'STEM': 'STEM', 'SOCIAL_SCIENCES': 'SS', 'ARTS_SPORTS': 'Arts'}
                hist_fig.add_vline(
                    x=f"G{_g_curr}",
                    line=dict(color=_new_col, width=1.5, dash='dot'),
                    opacity=0.6,
                )
                hist_fig.add_annotation(
                    x=f"G{_g_curr}", y=105,
                    xref='x', yref='y',
                    text=f"→{_pw_short.get(_new_pw, _new_pw[:4])}",
                    showarrow=False,
                    font=dict(size=7.5, color=_new_col,
                              family='IBM Plex Mono, monospace'),
                    bgcolor='rgba(255,255,255,0.75)',
                    borderpad=2,
                )

    # ── Layer 5: average score — main series ──────────────────────────
    _marker_colors = [bar_color(v) for v in avg_scores]
    hist_fig.add_trace(go.Scatter(
        x=grade_labels,
        y=avg_scores,
        mode='lines+markers+text',
        name='Average score',
        line=dict(width=3, color='#1E3A8A'),
        marker=dict(
            size=13,
            color=_marker_colors,
            line=dict(width=2.5, color='white'),
            symbol='circle',
        ),
        fill='tozeroy',
        fillcolor='rgba(40,28,89,0.07)',
        text=[f"<b>{s:.0f}</b>" for s in avg_scores],
        textposition='top center',
        textfont=dict(size=10, color='#1e1e2e',
                      family='IBM Plex Mono, monospace'),
        hovertemplate='<b>%{x}</b><br>Average: %{y:.1f}%<extra></extra>',
    ))

    # ── Layer 6: pathway suggestion labels under each grade ───────────
    _pw_short = {'STEM': 'STEM', 'SOCIAL_SCIENCES': 'SS', 'ARTS_SPORTS': 'Arts'}
    for _g in all_grades:
        _pw_at = per_grade_pw.get(_g, '')
        if _pw_at:
            _c = PW_COLORS.get(_pw_at, '#666')
            hist_fig.add_annotation(
                x=f"G{_g}", y=-11,
                xref='x', yref='y',
                text=f"<b>{_pw_short.get(_pw_at, _pw_at[:4])}</b>",
                showarrow=False,
                font=dict(size=8, color=_c),
            )

    # ── Layout ────────────────────────────────────────────────────────
    hist_fig.update_layout(
        margin=dict(l=15, r=55, t=48, b=55),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            range=[-18, 112],
            title="Score (%)",
            gridcolor='#e8ebe4',
            gridwidth=0.5,
            zeroline=False,
            tickfont=dict(size=9),
        ),
        xaxis=dict(
            title="",
            gridcolor='rgba(0,0,0,0)',
            tickfont=dict(size=11, family='IBM Plex Sans, sans-serif', color='#1e1e2e'),
        ),
        legend=dict(
            orientation='h',
            yanchor='top', y=-0.12,
            xanchor='center', x=0.5,
            font=dict(size=8),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#e0e4e8',
            borderwidth=1,
        ),
        title=dict(
            text=(f"<b>Grade 4–9 Trajectory</b>"
                  f"  <span style='color:{trend_col};font-size:11px'>"
                  f"{trend_str}{recent_note}</span>"),
            font=dict(size=12, family='IBM Plex Sans, sans-serif'),
            x=0.01, y=0.97,
        ),
        font=CHART_FONT,
        hovermode='x unified',
    )

    dqn_conf = rec.get('confidence', 0)
    dqn_cw_score = cw.get(pathway, 0)
    reasoning_text = rec.get('reasoning', '')
    reasoning_color = "light"

    # ====================================================================
    # HELPER: inline CSS bar (avoids extra Plotly figures)
    # ====================================================================
    def _bar(value, max_val, color, height='10px', border_radius='4px'):
        pct = min(100, 100 * value / max_val) if max_val else 0
        return html.Div(style={
            'width': f'{pct:.1f}%', 'height': height,
            'backgroundColor': color, 'borderRadius': border_radius,
            'display': 'inline-block', 'verticalAlign': 'middle',
            'minWidth': '3px',
        })

    # ====================================================================
    # TAB 1 — SUMMARY
    # ====================================================================

    # Reasoning box
    reasoning_box = (dbc.Alert(
        html.Small(reasoning_text, style={'lineHeight': '1.5'}),
        color=reasoning_color, className="py-2 px-3 mb-2",
        style={'fontSize': '0.75rem'})
        if reasoning_text else "")


    # Strengths and gaps
    strength_badges = [
        html.Span([
            html.Span(c['subject'],
                      style={'fontWeight': '500', 'fontSize': '0.72rem'}),
            html.Span(f" {c['score']:.0f}%",
                      style={'fontFamily': 'IBM Plex Mono, monospace',
                             'fontSize': '0.7rem', 'marginLeft': '2px'}),
        ], className="badge me-1 mb-1",
           style={'backgroundColor': score_color(c['score']),
                  'color': score_text(c['score']), 'padding': '4px 7px',
                  'borderRadius': '6px', 'fontWeight': '400'})
        for c in (top_strengths or contributions[:3])
    ]
    gap_badges = [
        html.Span([
            html.Span(c['subject'], style={'fontWeight': '500', 'fontSize': '0.72rem'}),
            html.Span(f" {c['score']:.0f}%",
                      style={'fontFamily': 'IBM Plex Mono, monospace',
                             'fontSize': '0.7rem', 'marginLeft': '2px'}),
        ], className="badge me-1 mb-1",
           style={'backgroundColor': score_color(c['score']),
                  'color': score_text(c['score']), 'padding': '4px 7px',
                  'borderRadius': '6px', 'fontWeight': '400'})
        for c in top_gaps
    ]

    # ====================================================================
    # BUILD COMPONENTS FOR MERGED PANEL
    # ====================================================================
    pathway_rows = []
    for pk in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']:
        v      = cw.get(pk, 0)
        col    = PW_COLORS.get(pk, '#666')
        is_pw  = pk == pathway
        thr    = PATHWAY_MIN_THRESHOLD.get(pk, 25)
        pw_d   = PATHWAYS.get(pk, {})

        if is_pw:
            pathway_rows.append(html.Div([
                # Name + icon + score on one line
                html.Div([
                    html.Span(pw_d.get('icon', ''),
                              style={'fontSize': '1.1rem', 'marginRight': '6px'}),
                    html.Span(PW_NAMES[pk],
                              style={'fontWeight': '700', 'fontSize': '0.95rem', 'color': col}),
                    html.Span(f" {v:.1f}%",
                              style={'fontFamily': 'IBM Plex Mono, monospace',
                                     'fontWeight': '700', 'fontSize': '0.88rem',
                                     'color': '#1e1e2e', 'marginLeft': '6px'}),
                    html.Span(f"  ·  {dqn_conf:.0%} confidence",
                              style={'fontSize': '0.7rem', 'color': '#6b7280',
                                     'fontFamily': 'IBM Plex Mono, monospace'}),
                    dbc.Badge("★ Recommended", color="light",
                              style={'fontSize': '0.65rem', 'color': col,
                                     'border': f'1px solid {col}',
                                     'marginLeft': '8px', 'verticalAlign': 'middle'}),
                ], className="mb-1 d-flex align-items-center flex-wrap"),
                # Confidence bar
                html.Div([
                    _bar(v, 100, col, height='10px', border_radius='5px'),
                    html.Div(style={
                        'position': 'absolute', 'left': f'{thr}%',
                        'top': '0', 'bottom': '0', 'width': '2px',
                        'backgroundColor': '#888', 'opacity': '0.4',
                    }),
                ], style={'position': 'relative', 'marginTop': '2px'}),
            ], style={
                'backgroundColor': f'{col}18',
                'border': f'1px solid {col}50',
                'borderLeft': f'4px solid {col}',
                'borderRadius': '8px', 'padding': '0.55rem 0.7rem',
                'marginBottom': '0.4rem',
            }))
        else:
            pathway_rows.append(html.Div([
                html.Div(PW_NAMES[pk],
                         style={'fontSize': '0.76rem', 'width': '130px', 'flexShrink': '0',
                                'color': col}),
                html.Div([
                    _bar(v, 100, '#c8d8dc', height='8px'),
                    html.Div(style={
                        'position': 'absolute', 'left': f'{thr}%',
                        'top': '0', 'bottom': '0', 'width': '2px',
                        'backgroundColor': '#888', 'opacity': '0.35',
                    }),
                ], style={'flex': '1', 'position': 'relative',
                          'display': 'flex', 'alignItems': 'center'}),
                html.Div(f"{v:.1f}%",
                         style={'fontSize': '0.7rem', 'fontFamily': 'IBM Plex Mono, monospace',
                                'width': '38px', 'textAlign': 'right', 'color': '#4a4a5a'}),
            ], className="mb-1 d-flex align-items-center gap-2",
               style={'paddingLeft': '0.4rem'}))


    # ====================================================================
    # MERGED RECOMMENDATION PANEL
    # ====================================================================
    rec_section = html.Div([
        # Pathway scores — recommended highlighted, others compact
        html.Small("PATHWAY SCORES  (▏= eligibility threshold)",
                   className="fw-bold d-block mb-2",
                   style={'fontSize': '0.62rem', 'letterSpacing': '0.05em', 'color': '#7a7a8a'}),
        html.Div(pathway_rows, className="mb-2"),

        reasoning_box,

        # Warnings
        *([] if not rec.get('core_check_failed') else [
            dbc.Alert("Core subject(s) below AE1 — assigned to next eligible pathway.",
                      color="warning", className="py-1 mb-2", style={'fontSize': '0.75rem'})]),
        *([] if not rec.get('below_expectations_warning') or rec.get('core_check_failed') else [
            dbc.Alert("No pathway meets minimum threshold — assigned to highest score.",
                      color="warning", className="py-1 mb-2", style={'fontSize': '0.75rem'})]),
        # Enhancement C — borderline-margin flag for HITL review
        *([] if not rec.get('low_confidence') or rec.get('core_check_failed')
              or rec.get('below_expectations_warning') else [
            dbc.Alert(
                ["Borderline match — top pathways are within 5 cluster-weight points. ",
                 html.Strong("Teacher review recommended"), " before final placement."],
                color="info", className="py-1 mb-2",
                style={'fontSize': '0.75rem',
                       'backgroundColor': '#e0f2fe', 'border': '1px solid #38bdf8',
                       'color': '#075985'})]),

        # Strengths / gaps
        html.Small("YOUR STRENGTHS", className="fw-bold d-block mb-1",
                   style={'fontSize': '0.65rem', 'letterSpacing': '0.06em', 'color': '#7a7a8a'}),
        html.Div(strength_badges or [html.Small("—", className="text-muted")],
                 className="mb-2"),

        *([] if not gap_badges else [
            html.Small("TO DEVELOP", className="fw-bold d-block mb-1",
                       style={'fontSize': '0.65rem', 'letterSpacing': '0.06em', 'color': '#7a7a8a'}),
            html.Div(gap_badges, className="mb-2"),
        ]),

        # PSI
        html.Small("Pathway Suitability Index (PSI)",
                   className="fw-bold d-block mb-1",
                   style={'fontSize': '0.65rem', 'letterSpacing': '0.05em', 'color': '#7a7a8a'}),
        dbc.Progress(value=psi_pct, color=psi_color, style={'height': '12px'},
                     label=f"{psi_pct}% — {psi_label}", className="mb-1"),

        # Pathway strength score (based on official KICD primary key subjects)
        html.Small("STRENGTH WITHIN PATHWAY",
                   className="fw-bold d-block mt-2 mb-1",
                   style={'fontSize': '0.65rem', 'letterSpacing': '0.06em', 'color': '#7a7a8a'}),
        html.Div([
            html.Span(f"{pathway_strength_score:.0f}%",
                      style={'fontFamily': 'IBM Plex Mono, monospace',
                             'fontWeight': '700', 'fontSize': '1.15rem',
                             'color': pathway_strength_color}),
            html.Span(f"  {pathway_strength_label}",
                      style={'fontSize': '0.75rem', 'color': '#4a4a5a',
                             'fontWeight': '500', 'marginLeft': '4px'}),
            html.Small(" · primary key subjects",
                       style={'color': '#9a9aaa', 'fontSize': '0.68rem'}),
        ], className="mb-1"),
        dbc.Progress(
            value=int(pathway_strength_score),
            style={'height': '8px', 'backgroundColor': '#e8ebe4'},
            color="info" if pathway_strength_score >= 75 else
                  "success" if pathway_strength_score >= 58 else
                  "warning" if pathway_strength_score >= 41 else "danger",
            className="mb-1",
        ),
        # Per-primary-subject breakdown
        *([html.Div([
            html.Div([
                html.Span(_bd['name'],
                          style={'fontSize': '0.72rem', 'width': '120px',
                                 'display': 'inline-block', 'color': '#4a4a5a'}),
                html.Span(f"{_bd['score']:.0f}%",
                          className="score-cell",
                          style={'backgroundColor': score_color(_bd['score']),
                                 'color': score_text(_bd['score']),
                                 'fontSize': '0.68rem'}),
                *([html.Span(f" ↑{70-_bd['score']:.0f} to ME1",
                             style={'fontSize': '0.62rem', 'color': '#f59e0b',
                                    'marginLeft': '4px'})]
                  if _bd['gap'] > 0 else
                  [html.Span(" ✓",
                             style={'fontSize': '0.65rem', 'color': '#10b981',
                                    'marginLeft': '4px'})]),
            ], className="mb-1")
            for _bd in _primary_breakdown
        ], className="mb-2")] if _primary_breakdown else []),

        # Reasoning narrative
        *([] if not reason_parts else [
            html.P(" ".join(reason_parts),
                   style={'fontSize': '0.74rem', 'lineHeight': '1.5', 'color': '#4a4a5a'},
                   className="mt-2 mb-0")]),

        # Override badge
        *([] if not rec.get('is_override') else [
            dbc.Badge("Override active — teacher has manually assigned this pathway.",
                      color="warning", className="d-block mt-2 small")]),
    ], style={'fontSize': '0.82rem'})

    # Override status
    ovr = ""
    if rec.get('is_override'):
        ovr = dbc.Alert(f"Override active: {pw_data['name']}", color="warning", className="py-1 mb-2")

    return slip_table, rec_section, subj_fig, radar_fig, hist_fig, ovr

# ---- Reset comparison and grade detail on student change ----
@callback([Output('analysis-comparison-result', 'children', allow_duplicate=True),
           Output('analysis-grade-detail', 'children', allow_duplicate=True),
           Output('analysis-change-result', 'children', allow_duplicate=True),
           Output('analysis-compare-pathway', 'value', allow_duplicate=True),
           Output('analysis-change-pathway', 'value', allow_duplicate=True),
           Output('analysis-change-by', 'value', allow_duplicate=True),
           Output('analysis-change-reason', 'value', allow_duplicate=True)],
          Input('analysis-student-selector', 'value'),
          prevent_initial_call=True)
def reset_on_student_change(sid):
    return ("", html.Small("Click a grade point for details.", className="text-muted"),
            "", None, None, "", "")


# ---- Grade detail click ----
@callback(Output('analysis-grade-detail', 'children', allow_duplicate=True),
          Input('analysis-history-chart', 'clickData'),
          State('analysis-student-selector', 'value'),
          prevent_initial_call=True)
def grade_detail(click, sid):
    if not click or not sid: return html.Small("Click a grade point for details.", className="text-muted")
    dm = get_data_manager()
    if not dm.has_data(): return ""
    try:
        grade = int(click['points'][0].get('x', '').replace('G', ''))
    except: return ""

    # ── Per-pathway suitability at this grade ─────────────────────────────
    pw_history = dm.get_pathway_history(sid)
    suitability_row = {}
    if len(pw_history) > 0:
        grade_rows = pw_history[pw_history['grade'] == grade]
        for _, r in grade_rows.iterrows():
            suitability_row[r['pathway']] = {
                'cw': round(r.get('cluster_weight', 0), 1),
                'psi': round(r.get('psi', 0) * 100, 1),
            }

    # ── Trajectory label per pathway: compare this grade to the previous ────
    # Build CW trajectory: {pathway: {grade: cw}} for trend arrow
    _cw_by_grade: dict = {}
    if len(pw_history) > 0:
        for _, _r in pw_history.iterrows():
            _cw_by_grade.setdefault(_r['pathway'], {})[int(_r['grade'])] = float(_r['cluster_weight'])

    def _trend_arrow(pw_key: str, g: int) -> tuple:
        """Return (arrow, color) comparing CW at grade g vs grade g-1."""
        _grades = sorted(_cw_by_grade.get(pw_key, {}).keys())
        _idx = _grades.index(g) if g in _grades else -1
        if _idx <= 0:
            return "—", '#94a3b8'
        _prev_cw = _cw_by_grade[pw_key].get(_grades[_idx - 1], 0)
        _curr_cw = _cw_by_grade[pw_key].get(g, 0)
        _delta = _curr_cw - _prev_cw
        if _delta > 3:   return f"↑ +{_delta:.0f}pt", '#10b981'
        if _delta < -3:  return f"↓ {_delta:.0f}pt",  '#ef4444'
        return "→ stable", '#f59e0b'

    suit_badges = []
    for _pw in ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']:
        if _pw in suitability_row:
            cw_val   = suitability_row[_pw]['cw']
            psi_val  = suitability_row[_pw]['psi']
            _arrow, _arrow_col = _trend_arrow(_pw, grade)
            suit_badges.append(html.Div([
                html.Span(PW_NAMES.get(_pw, _pw),
                          style={'fontWeight': '700', 'fontSize': '0.72rem',
                                 'color': PW_COLORS.get(_pw, '#666'), 'minWidth': '110px',
                                 'display': 'inline-block'}),
                html.Span(f"CW {cw_val:.0f}%",
                          style={'fontFamily': 'IBM Plex Mono, monospace', 'fontSize': '0.7rem',
                                 'color': '#334155', 'marginRight': '6px'}),
                html.Span(f"PSI {psi_val:.0f}%",
                          style={'fontFamily': 'IBM Plex Mono, monospace', 'fontSize': '0.68rem',
                                 'color': '#64748b', 'marginRight': '8px'}),
                html.Span(_arrow,
                          style={'fontSize': '0.68rem', 'fontWeight': '600',
                                 'color': _arrow_col}),
            ], className="mb-1",
               style={'padding': '3px 8px', 'borderRadius': '5px',
                      'borderLeft': f"3px solid {PW_COLORS.get(_pw, '#666')}",
                      'backgroundColor': f"{PW_COLORS.get(_pw,'#666')}11"}))

    # ── Subject scores at this grade ──────────────────────────────────────
    trans = dm.get_transitions(sid)
    badges = []
    for s, d in sorted(trans.get('subject_transitions', {}).items(),
                       key=lambda x: x[1].get('grade_scores', {}).get(grade, 0), reverse=True):
        if grade in d.get('grade_scores', {}):
            sc = d['grade_scores'][grade]
            gl = get_cbc_grade(sc)
            badges.append(html.Span([
                html.Span(f"{SUBJECT_NAMES.get(s,s)}", style={'fontWeight': '400'}),
                html.Span(f" {sc:.0f}%", style={'fontWeight': '600', 'fontFamily': 'IBM Plex Mono, monospace'}),
                html.Span(f" {gl['level']}", style={'fontSize': '0.62rem', 'opacity': '0.7'}),
            ], className="badge me-1 mb-1",
               style={'backgroundColor': score_color(sc), 'color': score_text(sc),
                      'fontSize': '0.72rem'}))

    pw = dm.get_suggested_pathway_per_grade(sid).get(grade, '')
    pw_badge = dbc.Badge(f"→ {PW_NAMES.get(pw, pw)}", style={'backgroundColor': PW_COLORS.get(pw, '#666')},
                        className="ms-2") if pw else ""
    return html.Div([
        html.Small(f"Grade {grade} ", className="fw-bold"), pw_badge,
        html.Div(suit_badges, className="mt-1") if suit_badges else None,
        html.Div(badges, className="mt-1"),
    ])


# ---- Comparison ----
@callback(Output('analysis-comparison-result', 'children', allow_duplicate=True),
          Input('analysis-compare-btn', 'n_clicks'),
          [State('analysis-student-selector', 'value'), State('analysis-compare-pathway', 'value')],
          prevent_initial_call=True)
def compare(n, sid, desired):
    dm = get_data_manager()
    if not sid or not desired or not dm.has_data():
        return html.Small("Select student and pathway.", className="text-muted")
    try:
        df, meta = dm.get_pathway_comparison(sid, desired)
        if len(df) == 0: return ""
        rows = []
        for _, r in df.iterrows():
            gap = r.get('Gap', 0)
            rows.append(html.Tr([
                html.Td(SUBJECT_NAMES.get(r['Subject'], r['Subject']), style={'fontSize': '0.82rem'}),
                html.Td(html.Span(f"{r['Current Score']:.0f}%", className="score-cell",
                        style={'backgroundColor': score_color(r['Current Score']),
                               'color': score_text(r['Current Score'])}), style={'textAlign': 'center'}),
                html.Td(f"+{gap:.0f}" if gap > 0 else "—",
                        style={'textAlign': 'center', 'color': '#EF4444' if gap > 0 else '#10B981',
                               'fontWeight': '600', 'fontSize': '0.85rem'}),
            ]))
        return html.Div([
            html.Div([
                dbc.Badge(f"Current: {meta.get('cosine_current',0):.0%}",
                         style={'backgroundColor': PW_COLORS.get(meta.get('current_pathway',''), '#666')}, className="me-1"),
                dbc.Badge(f"Desired: {meta.get('cosine_desired',0):.0%}",
                         style={'backgroundColor': PW_COLORS.get(desired, '#666')}),
            ], className="mb-2"),
            dbc.Table([html.Thead(html.Tr([html.Th("Subject"), html.Th("Score"), html.Th("Gap")])),
                      html.Tbody(rows)], bordered=False, hover=True, size="sm", style={'fontSize': '0.82rem'}),
        ])
    except Exception as e:
        return html.Small(f"Error: {e}", className="text-danger")


# ---- Override ----
@callback(Output('analysis-change-result', 'children', allow_duplicate=True),
          Input('analysis-submit-change', 'n_clicks'),
          [State('analysis-student-selector', 'value'), State('analysis-change-pathway', 'value'),
           State('analysis-change-reason', 'value'), State('analysis-change-by', 'value')],
          prevent_initial_call=True)
def override(n, sid, pw, reason, by):
    dm = get_data_manager()
    if not sid: return dbc.Alert("No student", color="warning", className="py-1")
    if not pw or not reason: return dbc.Alert("Fill all fields", color="warning", className="py-1")
    rec = dm.get_recommendation(sid)
    if pw == rec['recommended_pathway']: return dbc.Alert("Already assigned", color="info", className="py-1")
    try:
        dm.hitl.submit_request(student_id=int(sid), current_pathway=rec['recommended_pathway'],
                               desired_pathway=pw, reason=reason, requested_by=by or "Student")
        return dbc.Alert("Request submitted.", color="success", className="py-1")
    except ValueError as e:
        return dbc.Alert(str(e), color="danger", className="py-1")


# ---- Feedback: show/hide desired pathway dropdown ----
@callback(Output('analysis-feedback-desired-wrapper', 'style'),
          Input('analysis-feedback-radio', 'value'))
def toggle_desired(val):
    if val == 'wants_different':
        return {'display': 'block'}
    return {'display': 'none'}


# ---- Feedback: submit ----
@callback([Output('analysis-feedback-result', 'children', allow_duplicate=True),
           Output('analysis-coaching-plan', 'children', allow_duplicate=True)],
          Input('analysis-feedback-submit', 'n_clicks'),
          [State('analysis-student-selector', 'value'),
           State('analysis-feedback-radio', 'value'),
           State('analysis-feedback-desired', 'value')],
          prevent_initial_call=True)
def submit_feedback(n, sid, feedback, desired):
    from dash import no_update
    dm = get_data_manager()
    if not sid or not feedback:
        return dbc.Alert("Select your feedback first.", color="light", className="py-1"), no_update
    if feedback == 'wants_different' and not desired:
        return dbc.Alert("Please select which pathway you'd prefer.", color="warning", className="py-1"), no_update
    
    dm.set_student_feedback(int(sid), feedback, desired)
    
    msg = "Thanks! Your feedback has been recorded." if feedback == 'satisfied' \
        else f"Got it — coaching plan updated for {PATHWAYS.get(desired, {}).get('name', desired)}."
    
    return dbc.Alert(msg, color="success", className="py-1"), _build_coaching_card(sid)


# ---- Coaching plan: display on student select ----
@callback(Output('analysis-coaching-plan', 'children'),
          Input('analysis-student-selector', 'value'))
def show_coaching(sid):
    if not sid:
        return html.P("Select a student to see their coaching plan.",
                      className="text-muted text-center py-3")
    return _build_coaching_card(sid)


def _build_coaching_card(sid):
    """Build the coaching plan display."""
    dm = get_data_manager()
    if not dm.has_data():
        return html.P("Load data first.", className="text-muted text-center py-3")
    
    plan = dm.get_coaching_plan(int(sid))
    if plan.get('status') == 'error':
        return html.Small(plan.get('message', 'Error'), className="text-danger")
    
    # Status badge
    status_map = {
        'satisfied': ('Satisfied', 'success'),
        'coaching_to_desired': ('Coaching to Desired Pathway', 'warning'),
        'strengthen_current': ('Strengthening Current Pathway', 'primary'),
    }
    label, color = status_map.get(plan.get('status', ''), ('', 'secondary'))
    
    # Focus subjects table
    focus = plan.get('focus_subjects', [])
    rows = []
    priority_badges = {
        'critical': ('Critical', '#EF4444'),
        'high': ('High', '#F59E0B'),
        'medium': ('Medium', '#3B82F6'),
        'maintain': ('Maintain', '#10B981'),
    }
    for f in focus:
        p_label, p_color = priority_badges.get(f['priority'], ('', '#666'))
        rows.append(html.Tr([
            html.Td(f['subject_name'], style={'fontSize': '0.83rem'}),
            html.Td(html.Span(f"{f['current']:.0f}%", className="score-cell",
                    style={'backgroundColor': score_color(f['current']),
                           'color': score_text(f['current'])}),
                    style={'textAlign': 'center'}),
            html.Td(f"{f['target']}%", style={'textAlign': 'center', 'fontWeight': '600',
                                               'fontSize': '0.83rem'}),
            html.Td(f"+{f['gap']:.0f}" if f['gap'] > 0 else "—",
                    style={'textAlign': 'center', 'color': '#EF4444' if f['gap'] > 5 else '#10B981',
                           'fontWeight': '600', 'fontSize': '0.83rem'}),
            html.Td(dbc.Badge(p_label, style={'backgroundColor': p_color}, className=""),
                    style={'textAlign': 'center'}),
        ]))

    target_name = plan.get('target_pathway_name', '')
    target_pw = plan.get('target_pathway', '')
    pw_color = PW_COLORS.get(target_pw, '#666')

    return html.Div([
        # Header with status
        html.Div([
            dbc.Badge(label, color=color, className="me-2"),
            html.Strong(f"Target: {target_name}", style={'color': pw_color}),
        ], className="mb-2"),
        # Message
        html.P(plan.get('message', ''), style={'fontSize': '0.88rem', 'lineHeight': '1.5'},
               className="mb-2"),
        # Focus subjects
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Subject"), html.Th("Now", style={'textAlign': 'center'}),
                html.Th("Target", style={'textAlign': 'center'}),
                html.Th("Gap", style={'textAlign': 'center'}),
                html.Th("Priority", style={'textAlign': 'center'}),
            ])),
            html.Tbody(rows)
        ], bordered=False, hover=True, size="sm") if rows else "",
    ])