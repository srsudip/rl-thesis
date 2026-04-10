"""
Teacher Review Page - HITL Workflow + Student Data Editing

Sections:
1. Pending Pathway Change Requests (card-based, no ID typing)
2. Edit Student Data (teacher corrections)
3. Audit Log
"""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PATHWAYS, SUBJECT_NAMES, COMPETENCIES

dash.register_page(__name__, path='/teacher', name='Teacher Review',
                   title='Kenya CBC - Teacher Review', order=3)
from pages.dashboard import get_data_manager


layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4([html.I(className="fas fa-user-shield me-2"), "Teacher Review"], className="mb-0"),
            html.Small("Approve/reject pathway requests and edit student data", className="text-muted"),
        ])
    ], className="mb-3 pt-2"),

    # Section 1: Pending Requests
    dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-clock me-2"), "Pending Requests",
            html.Span(id='teacher-pending-badge', className="ms-2"),
        ]),
        dbc.CardBody([
            html.Div(id='teacher-pending-requests'),
            html.Div(id='teacher-action-result', className="mt-2"),
        ])
    ], className="shadow-sm mb-3"),

    dcc.Store(id='teacher-refresh-trigger', data=0),

    # Section 2: Edit Student Data
    dbc.Card([
        dbc.CardHeader([html.I(className="fas fa-edit me-2"), "Edit Student Data"]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Student:", className="small fw-bold"),
                    dcc.Dropdown(id='teacher-edit-student', placeholder="Select student...",
                                searchable=True, clearable=True),
                ], md=3),
                dbc.Col([
                    html.Label("Field:", className="small fw-bold"),
                    dcc.Dropdown(id='teacher-edit-field', placeholder="Select field..."),
                ], md=3),
                dbc.Col([
                    html.Label("New Value (0-100):", className="small fw-bold"),
                    dbc.Input(id='teacher-edit-value', type='number', min=0, max=100, size="sm"),
                ], md=2),
                dbc.Col([
                    html.Label("\u00a0", className="small fw-bold"),
                    dbc.Button([html.I(className="fas fa-save me-1"), "Apply"],
                              id='teacher-edit-apply', color="warning", size="sm", className="w-100"),
                ], md=2),
                dbc.Col([
                    html.Label("\u00a0", className="small fw-bold"),
                    dbc.Button([html.I(className="fas fa-undo me-1"), "Clear"],
                              id='teacher-edit-clear', color="outline-secondary", size="sm",
                              className="w-100"),
                ], md=2),
            ]),
            html.Div(id='teacher-edit-result', className="mt-2"),
            html.Div(id='teacher-current-data', className="mt-2"),
        ])
    ], className="shadow-sm mb-3"),

    # Section 3: Audit Log
    dbc.Card([
        dbc.CardHeader([html.I(className="fas fa-history me-2"), "Audit Log"]),
        dbc.CardBody(html.Div(id='teacher-audit-log'))
    ], className="shadow-sm"),
], fluid=True)


# ==================== CALLBACKS ====================

@callback(
    [Output('teacher-pending-requests', 'children'),
     Output('teacher-pending-badge', 'children')],
    [Input('url', 'pathname'), Input('teacher-refresh-trigger', 'data')]
)
def load_pending(pathname, trigger):
    dm = get_data_manager()
    pending = dm.get_pending_hitl_requests()

    if not pending:
        return html.P("No pending requests.", className="text-muted text-center py-3"), \
               dbc.Badge("0", color="secondary")

    cards = []
    for req in pending:
        sid = req['student_id']
        current = req.get('current_pathway', '?')
        desired = req.get('desired_pathway', '?')
        cur_pw = PATHWAYS.get(current, {})
        des_pw = PATHWAYS.get(desired, {})
        reason = req.get('reason', '')
        by = req.get('requested_by', 'Unknown')
        req_id = req.get('request_id', '')

        # Get student info for context
        try:
            slip = dm.get_knec_result_slip(sid)
            cw = slip['cluster_weights']
            pts_info = f"STEM: {cw.get('STEM', 0):.1f}% | SS: {cw.get('SOCIAL_SCIENCES', 0):.1f}% | Arts: {cw.get('ARTS_SPORTS', 0):.1f}%"
        except Exception:
            pts_info = ""

        cards.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        # Left: student info
                        dbc.Col([
                            html.Div([
                                html.Strong(f"Student {sid}", className="me-2"),
                                dbc.Badge(by, color="info", className="me-1"),
                            ], className="mb-1"),
                            html.Div([
                                html.Span(f"{cur_pw.get('icon', '')} {cur_pw.get('name', current)}",
                                         className="text-muted me-1"),
                                html.Span("\u2192", className="mx-1"),
                                html.Strong(f"{des_pw.get('icon', '')} {des_pw.get('name', desired)}",
                                           style={'color': des_pw.get('color', '#333')}),
                            ], className="mb-1"),
                            html.Small(f'"{reason}"', className="text-muted d-block"),
                            html.Small(pts_info, className="text-muted") if pts_info else None,
                        ], md=6),
                        # Right: action
                        dbc.Col([
                            dbc.Input(id={'type': 'teacher-justification', 'index': req_id},
                                     placeholder="Justification...", size="sm", className="mb-2"),
                            dbc.ButtonGroup([
                                dbc.Button([html.I(className="fas fa-check me-1"), "Approve"],
                                          id={'type': 'teacher-approve', 'index': req_id},
                                          color="success", size="sm"),
                                dbc.Button([html.I(className="fas fa-times me-1"), "Reject"],
                                          id={'type': 'teacher-reject', 'index': req_id},
                                          color="danger", size="sm", outline=True),
                            ]),
                        ], md=6, className="d-flex flex-column justify-content-center"),
                    ]),
                ], className="py-2")
            ], className="mb-2 border-start border-warning border-3")
        )

    return html.Div(cards), dbc.Badge(str(len(pending)), color="warning")


@callback(
    [Output('teacher-action-result', 'children'),
     Output('teacher-refresh-trigger', 'data')],
    [Input({'type': 'teacher-approve', 'index': dash.ALL}, 'n_clicks'),
     Input({'type': 'teacher-reject', 'index': dash.ALL}, 'n_clicks')],
    [State({'type': 'teacher-justification', 'index': dash.ALL}, 'value'),
     State('teacher-refresh-trigger', 'data')],
    prevent_initial_call=True
)
def handle_action(approve_clicks, reject_clicks, justifications, trigger):
    from dash import ctx
    if not ctx.triggered_id:
        return no_update, no_update

    # CRITICAL GUARD: Dash pattern-matching ALL callbacks fire when new 
    # buttons are first rendered (n_clicks=None). We must verify an actual
    # human click happened.
    
    # Quick bail: if ALL click counts are None or 0, no one clicked anything
    all_clicks = (approve_clicks or []) + (reject_clicks or [])
    if not any(c and c > 0 for c in all_clicks):
        return no_update, no_update

    triggered_type = ctx.triggered_id.get('type', '') if isinstance(ctx.triggered_id, dict) else ''
    triggered_idx = ctx.triggered_id.get('index', '') if isinstance(ctx.triggered_id, dict) else ''
    
    if not triggered_type or not triggered_idx:
        return no_update, no_update

    # Verify the specific triggered button was actually clicked
    actual_click = False
    if triggered_type == 'teacher-approve':
        for i, prop_id in enumerate(ctx.inputs_list[0]):
            if prop_id['id']['index'] == triggered_idx:
                if approve_clicks[i] and approve_clicks[i] > 0:
                    actual_click = True
                break
    elif triggered_type == 'teacher-reject':
        for i, prop_id in enumerate(ctx.inputs_list[1]):
            if prop_id['id']['index'] == triggered_idx:
                if reject_clicks[i] and reject_clicks[i] > 0:
                    actual_click = True
                break

    if not actual_click:
        return no_update, no_update

    req_id = triggered_idx
    action = triggered_type
    dm = get_data_manager()

    just_text = ""
    for i, prop_id in enumerate(ctx.states_list[0]):
        if prop_id['id']['index'] == req_id:
            just_text = justifications[i] or ""
            break

    try:
        if action == 'teacher-approve':
            dm.approve_hitl_request(req_id, teacher_id="Teacher",
                                   justification=just_text or "Approved")
            return (dbc.Alert([html.I(className="fas fa-check me-2"), f"Approved request for student."],
                            color="success", className="py-1", duration=3000),
                    (trigger or 0) + 1)
        else:
            dm.reject_hitl_request(req_id, teacher_id="Teacher",
                                  reason=just_text or "Rejected")
            return (dbc.Alert([html.I(className="fas fa-times me-2"), f"Rejected request."],
                            color="danger", className="py-1", duration=3000),
                    (trigger or 0) + 1)
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger", className="py-1"), no_update


# ---- Edit Student Data ----
@callback(Output('teacher-edit-student', 'options'), Input('url', 'pathname'))
def load_students(p):
    return get_data_manager().get_student_options()


@callback(
    [Output('teacher-edit-field', 'options'),
     Output('teacher-current-data', 'children')],
    Input('teacher-edit-student', 'value')
)
def show_student_data(sid):
    if not sid:
        return [], ""
    dm = get_data_manager()
    perf = dm.get_performance(sid)
    subjects = perf.get('subject_scores', {})

    field_opts = [{'label': SUBJECT_NAMES.get(s, s), 'value': f"{s}_score"} for s in subjects]
    for c in COMPETENCIES:
        cname = COMPETENCIES[c]['name'] if isinstance(COMPETENCIES[c], dict) else c
        field_opts.append({'label': f"\u2295 {cname}", 'value': f"{c}_score"})

    badges = []
    for s, d in subjects.items():
        edited = " \u270f\ufe0f" if d.get('edited') else ""
        color = "success" if d['score'] >= 60 else ("warning" if d['score'] >= 40 else "danger")
        badges.append(dbc.Badge(f"{SUBJECT_NAMES.get(s, s)}: {d['score']:.0f}{edited}",
                               color=color, className="me-1 mb-1"))

    return field_opts, html.Div([html.Small("Scores: ", className="fw-bold")] + badges)


@callback(
    Output('teacher-edit-result', 'children'),
    [Input('teacher-edit-apply', 'n_clicks'), Input('teacher-edit-clear', 'n_clicks')],
    [State('teacher-edit-student', 'value'),
     State('teacher-edit-field', 'value'),
     State('teacher-edit-value', 'value')],
    prevent_initial_call=True
)
def apply_edit(apply_clicks, clear_clicks, sid, field, value):
    from dash import ctx
    dm = get_data_manager()
    if not sid:
        return dbc.Alert("Select a student", color="warning", className="py-1")

    if ctx.triggered_id == 'teacher-edit-clear':
        dm.clear_edits(sid)
        return dbc.Alert("Edits cleared", color="info", className="py-1", duration=2000)

    if ctx.triggered_id == 'teacher-edit-apply':
        if not field or value is None:
            return dbc.Alert("Select field and enter value", color="warning", className="py-1")
        dm.edit_student_data(sid, field, float(value))
        return dbc.Alert(f"Updated {field} = {value}", color="success", className="py-1", duration=2000)

    return no_update


# ---- Audit Log ----
@callback(
    Output('teacher-audit-log', 'children'),
    [Input('url', 'pathname'), Input('teacher-refresh-trigger', 'data')]
)
def show_audit(pathname, trigger):
    dm = get_data_manager()
    all_reqs = dm.hitl.get_all_requests()

    if not all_reqs:
        return html.P("No requests yet.", className="text-muted text-center py-3")

    rows = []
    for req in reversed(all_reqs[-20:]):
        status = req.get('status', 'unknown')
        color = {'pending': 'warning', 'approved': 'success', 'rejected': 'danger'}.get(status, 'secondary')
        cur = PATHWAYS.get(req.get('current_pathway', ''), {}).get('name', '?')
        des = PATHWAYS.get(req.get('desired_pathway', ''), {}).get('name', '?')
        rows.append(html.Tr([
            html.Td(f"Student {req.get('student_id', '?')}", className="small"),
            html.Td(f"{cur} \u2192 {des}", className="small"),
            html.Td(dbc.Badge(status, color=color)),
            html.Td(req.get('reason', '')[:40], className="small text-muted"),
        ]))

    return dbc.Table([
        html.Thead(html.Tr([html.Th("Student"), html.Th("Change"),
                            html.Th("Status"), html.Th("Reason")])),
        html.Tbody(rows)
    ], bordered=True, hover=True, size="sm")
