"""
Home Page — Data-first science dashboard.
Top indicators → Chart + Controls → Student lookup.
"""
from dash import html, dcc, callback, Input, Output, State, register_page
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64, tempfile, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PATHWAYS

register_page(__name__, path="/", name="Home", title="Kenya CBC - Home")
from pages.dashboard import get_data_manager

STEM_C = '#2563EB'
SOC_C = '#10B981'
ARTS_C = '#F97316'
CHART_FONT = dict(family='IBM Plex Sans, sans-serif', color='#0F172A')


def _indicator(value, label, color=None):
    style = {'color': color} if color else {}
    return html.Div([
        html.Div(str(value), className="stat-value", style=style),
        html.Div(label, className="stat-label"),
    ], className="stat-card")


layout = dbc.Container([
    # ── Title row ──
    dbc.Row([
        dbc.Col([
            html.H5("Kenya CBC Pathway Recommendation", className="mb-0",
                    style={'fontWeight': '700', 'letterSpacing': '-0.01em'}),
            html.Small("Deep Q-Learning pathway alignment · Grade 9 → Senior Secondary",
                      className="text-muted", style={'fontSize': '0.78rem'}),
        ], md=8, className="pt-2 pb-1"),
        dbc.Col([
            html.Div(id='home-status-message'),
        ], md=4),
    ], className="mb-2"),

    # Hidden outputs for callbacks that write to stats (kept for compatibility)
    html.Div(id='home-data-stats', style={'display': 'none'}),
    html.Div(id='home-model-stats', style={'display': 'none'}),

    # ── Controls row ──
    dbc.Row([
        # Data source
        dbc.Col(dbc.Card([
            dbc.CardHeader("Data"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Students", style={'fontSize': '0.72rem', 'fontWeight': '600'}),
                        dbc.Input(id='home-num-students', type='number', value=500,
                                 min=1, max=100000, size="sm"),
                    ], width=7),
                    dbc.Col([
                        html.Label("Seed", style={'fontSize': '0.72rem', 'fontWeight': '600'}),
                        dbc.Input(id='home-seed', type='number', value=42,
                                 min=1, max=99999, size="sm"),
                    ], width=5),
                ], className="mb-2"),
                dbc.Button("Generate", id='home-generate-btn',
                          color="dark", className="w-100 mb-2", size="sm"),
                dcc.Upload(
                    id='home-csv-upload',
                    children=dbc.Button("Upload CSV", color="outline-dark",
                                       size="sm", className="w-100"),
                    multiple=False, accept='.csv'
                ),
            ], style={'padding': '0.8rem'}),
        ]), md=4, className="mb-2"),

        # DQN Model
        dbc.Col(dbc.Card([
            dbc.CardHeader("DQN Model"),
            dbc.CardBody([
                html.Label("Episodes", style={'fontSize': '0.72rem', 'fontWeight': '600'}),
                dbc.Input(id='home-num-episodes', type='number', value=500,
                         min=1, max=100000, size="sm", className="mb-2"),
                dbc.Button("Train", id='home-train-btn',
                          color="dark", className="w-100", size="sm"),
                html.Div(id='home-model-accuracy', className="text-center mt-2"),
            ], style={'padding': '0.8rem'}),
        ]), md=4, className="mb-2"),

        # Student lookup
        dbc.Col(dbc.Card([
            dbc.CardHeader("Student Lookup"),
            dbc.CardBody([
                dcc.Dropdown(id='home-student-select', placeholder="Search by name or ID...",
                            searchable=True, clearable=True, className="mb-2"),
                html.Div(id='home-student-info',
                    children=html.Small("Select a student to preview",
                                       className="text-muted", style={'fontSize': '0.78rem'})),
                dcc.Link(dbc.Button("Open Analysis →", color="dark", className="w-100 mt-2", size="sm"),
                        href="/analysis"),
            ], style={'padding': '0.8rem'}),
        ]), md=4, className="mb-2"),
    ]),

    # ── Pathway Distribution — bottom ──
    dbc.Card([
        dbc.CardHeader("Pathway Distribution"),
        dbc.CardBody([
            dcc.Graph(id='home-pathway-chart', style={'height': '320px'},
                     config={'displayModeBar': False}),
        ])
    ], className="mb-3"),

], fluid=True)


# ==================== CHARTS ====================

def _empty_dist_chart():
    fig = go.Figure()
    fig.add_annotation(text="Generate or upload data", showarrow=False,
                      font=dict(color='#bbb', size=12, family='IBM Plex Sans, sans-serif'))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=CHART_FONT)
    return fig


def _build_dist_chart(dm):
    if not dm.has_data():
        return _empty_dist_chart()

    counts = {'STEM': 0, 'SOCIAL_SCIENCES': 0, 'ARTS_SPORTS': 0}
    for sid in dm.get_student_ids():
        rec = dm.get_recommendation(sid)
        pw = rec.get('recommended_pathway', '')
        if pw in counts:
            counts[pw] += 1

    total = sum(counts.values()) or 1
    keys = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
    names = ['STEM', 'Social Sciences', 'Arts & Sports']
    vals = [counts[k] for k in keys]
    pcts = [v / total * 100 for v in vals]
    colors = [STEM_C, SOC_C, ARTS_C]

    fig = go.Figure()
    # Main bars
    fig.add_trace(go.Bar(
        x=names, y=vals, marker_color=colors, marker_line_width=0,
        text=[f"<b>{v}</b><br><span style='font-size:10px'>{p:.1f}%</span>"
              for v, p in zip(vals, pcts)],
        textposition='outside', textfont=dict(size=12, **CHART_FONT),
    ))
    # Threshold lines as annotations
    fig.add_annotation(text="STEM ≥ 20%  ·  Social Sci / Arts ≥ 25%  ·  Core ≥ AE1",
                      xref="paper", yref="paper", x=0.5, y=1.06,
                      showarrow=False, font=dict(size=9, color='#7a7a8a', family='IBM Plex Sans, sans-serif'))

    fig.update_layout(
        margin=dict(l=45, r=15, t=35, b=35),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title="Students", gridcolor='#e8ebe4', zeroline=False),
        xaxis=dict(tickfont=dict(size=11)),
        font=CHART_FONT, height=340,
        bargap=0.35,
    )
    return fig


# ==================== CALLBACKS ====================

@callback(
    [Output('home-status-message', 'children'),
     Output('home-data-stats', 'children'),
     Output('home-student-select', 'options'),
     Output('home-pathway-chart', 'figure'),
     Output('data-status-store', 'data')],
    [Input('home-generate-btn', 'n_clicks')],
    [State('home-num-students', 'value'), State('home-seed', 'value')],
    prevent_initial_call=False
)
def generate_data(n_clicks, n_students, seed):
    from dash import ctx
    dm = get_data_manager()
    status = None

    if ctx.triggered_id == 'home-generate-btn' and n_clicks:
        try: n_students = max(1, int(abs(n_students or 500)))
        except: n_students = 500
        try: seed = int(abs(seed or 42))
        except: seed = 42
        success = dm.generate_data(n_students=n_students, seed=seed)
        status = dbc.Alert(f"✓ Generated {n_students} students",
                          color="success", dismissable=True, duration=4000,
                          style={'fontSize': '0.82rem', 'padding': '0.4rem 0.8rem'}) if success \
                 else dbc.Alert("Generation failed", color="danger", dismissable=True)

    if dm.has_data():
        n = dm.get_student_count()
        return status, _indicator(n, "Students"), dm.get_student_options(), _build_dist_chart(dm), n

    return status, _indicator("—", "No Data", "#ccc"), [], _empty_dist_chart(), 0


@callback(
    [Output('home-model-accuracy', 'children'),
     Output('home-model-stats', 'children'),
     Output('model-status-store', 'data')],
    [Input('home-train-btn', 'n_clicks')],
    [State('home-num-episodes', 'value')],
    prevent_initial_call=True
)
def train_model(n_clicks, n_episodes):
    from dash import ctx, no_update
    dm = get_data_manager()
    if ctx.triggered_id != 'home-train-btn' or not n_clicks:
        return no_update, no_update, no_update
    try: n_episodes = max(1, int(abs(n_episodes or 500)))
    except: n_episodes = 500
    results = dm.train_model(n_episodes=n_episodes)
    if results['success']:
        acc = results['accuracy']
        return (dbc.Badge(f"{acc*100:.1f}%", color="success", style={'fontSize': '0.78rem'}),
                _indicator(f"{acc*100:.1f}%", "Accuracy"),
                acc)
    return (dbc.Badge("Failed", color="danger"), _indicator("—", "Not Trained", "#ccc"), None)


@callback(
    [Output('home-model-accuracy', 'children', allow_duplicate=True),
     Output('home-model-stats', 'children', allow_duplicate=True)],
    Input('url', 'pathname'), prevent_initial_call=True)
def init_model(pathname):
    dm = get_data_manager()
    if dm.has_model():
        acc = dm.model_accuracy or 0
        return (dbc.Badge(f"{acc*100:.1f}%", color="success", style={'fontSize': '0.78rem'}),
                _indicator(f"{acc*100:.1f}%", "Accuracy"))
    return (html.Small("Train after data", className="text-muted", style={'fontSize': '0.75rem'}),
            _indicator("—", "DQN Accuracy", "#ccc"))


@callback(Output('home-student-select', 'options', allow_duplicate=True),
          Input('url', 'pathname'), prevent_initial_call=True)
def load_opts(pathname):
    return get_data_manager().get_student_options()


@callback(Output('selected-student-store', 'data'),
          Input('home-student-select', 'value'), prevent_initial_call=True)
def store_student(sid): return sid


@callback(Output('home-student-select', 'value'),
          Input('url', 'pathname'), State('selected-student-store', 'data'),
          prevent_initial_call=False)
def load_student(pathname, stored): return stored


@callback(Output('home-student-info', 'children'),
          Input('home-student-select', 'value'))
def show_info(sid):
    if not sid:
        return html.Small("Select a student to preview their pathway",
                         className="text-muted", style={'fontSize': '0.78rem'})
    dm = get_data_manager()
    if not dm.has_data():
        return html.Small("Load data first", className="text-muted")
    rec = dm.get_recommendation(sid)
    pw = PATHWAYS[rec['recommended_pathway']]
    cw = rec.get('cluster_weights', {})
    pw_key = rec['recommended_pathway']
    pw_colors = {'STEM': STEM_C, 'SOCIAL_SCIENCES': SOC_C, 'ARTS_SPORTS': ARTS_C}
    return html.Div([
        html.Span(pw['icon'], style={'fontSize': '1rem'}, className="me-1"),
        html.Strong(pw['name'], style={'color': pw_colors.get(pw_key, '#555'), 'fontSize': '0.88rem'}),
        html.Span(f" {cw.get(pw_key, 0):.1f}%",
                 style={'color': '#8a8a8a', 'fontSize': '0.82rem', 'fontFamily': 'IBM Plex Mono, monospace'},
                 className="ms-1"),
    ])


# CSV Upload
@callback(
    [Output('home-status-message', 'children', allow_duplicate=True),
     Output('home-data-stats', 'children', allow_duplicate=True),
     Output('home-student-select', 'options', allow_duplicate=True),
     Output('home-pathway-chart', 'figure', allow_duplicate=True),
     Output('data-status-store', 'data', allow_duplicate=True)],
    Input('home-csv-upload', 'contents'),
    State('home-csv-upload', 'filename'),
    prevent_initial_call=True
)
def upload_csv(contents, filename):
    from dash import no_update
    dm = get_data_manager()
    if contents is None:
        return no_update, no_update, no_update, no_update, no_update
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        tmp.write(decoded); tmp.close()
        success = dm.load_csv_data(filepath=tmp.name, simulate_g4_g6=True)
        os.unlink(tmp.name)
        if success:
            n = dm.get_student_count()
            return (dbc.Alert(f"✓ Loaded {n} students from {filename}", color="success",
                            dismissable=True, duration=4000,
                            style={'fontSize': '0.82rem', 'padding': '0.4rem 0.8rem'}),
                    _indicator(n, f"From CSV"), dm.get_student_options(), _build_dist_chart(dm), n)
        return (dbc.Alert(f"Failed to load {filename}", color="danger", dismissable=True),
                no_update, no_update, no_update, no_update)
    except Exception as e:
        return (dbc.Alert(f"Error: {e}", color="danger", dismissable=True),
                no_update, no_update, no_update, no_update)