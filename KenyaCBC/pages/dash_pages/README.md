# pages/dash_pages/ — Dashboard Pages

Five Plotly Dash page modules. Each exports a `layout` object (or callable) and registers its callbacks via `register_callbacks(app)`.

## Pages

| File | Route | Purpose |
|------|-------|---------|
| `home.py` | `/` | Generate data, train DQN model, system status |
| `analysis.py` | `/analysis` | Per-student pathway recommendation, KJSEA result slip, history |
| `teacher_review.py` | `/teacher` | HITL: review pending requests, approve/reject, audit log |
| `advanced.py` | `/advanced` | Benchmarks, multi-seed evaluation, hyperparameter search |
| `about.py` | `/about` | CBC structure, grading system, model architecture explanation |

## analysis.py in detail

The most complex page. Renders in four layers for a selected student:

1. **KJSEA Result Slip** — Official KNEC format, 12 subjects (codes 901–912), 8-level grades (EE1–BE2), composite KNEC score
2. **Pathway Recommendation Panel** — Multi-signal fusion result (60% cluster weight + 25% cosine similarity + 15% PSI), two-gate eligibility check, confidence bar, strength-within-pathway score
3. **Pathway Suitability History** — Line chart: recommended pathway shown as solid line with percentage annotations at each grade point (4–9); competing pathways shown as faint dotted lines
4. **DQN Coaching Card** — Supplementary AI suggestion with XAI feature importances; only visible when RL agent disagrees with KNEC recommendation

### Multi-signal Fusion (implemented in this page's recommendation call)

```
composite = 0.60 × cluster_weight + 0.25 × cosine_similarity + 0.15 × PSI
```

Falls back gracefully when cosine/PSI signals are absent (weights redistributed to cluster weight).

### Pathway Strength Score

Computed from the KNEC cluster weight contributions of the top-3 core subjects in the recommended pathway. Displayed as a percentage with a Plotly gauge.

## teacher_review.py in detail

HITL workflow page:
- **Pending requests** table: student ID, requested pathway, counselor notes
- **Approve** button: logs decision to `data/hitl_state.json`, marks override active
- **Reject** button: logs rejection, no pathway change
- **Audit log** table: full decision history with timestamps and teacher ID
- **Student data editor**: inline editing of individual subject scores for data correction

## Callback Pattern

All callbacks follow the same pattern:

```python
@app.callback(Output(...), Input(...), State('student-store', 'data'))
def update(trigger, student_id):
    dm = get_data_manager()
    ...
```

`get_data_manager()` returns the singleton `DataManager` from `pages/dashboard.py`.
