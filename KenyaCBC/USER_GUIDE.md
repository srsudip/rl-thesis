# Kenya CBC Pathway Recommendation System — User Guide

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open your browser at **http://127.0.0.1:8050**

---

## Pages Overview

The dashboard has five pages accessible from the navigation bar:

### 1. Home

The landing page. Click **Generate Data** to create a synthetic student population (default: 500 students). This simulates Grade 4–9 assessment data using an Item Response Theory (IRT) model aligned with the CBC curriculum. Once generated, you will see a summary of pathway distribution and can click any student card to jump to their analysis.

### 2. Analysis

The core student view. Select a student from the dropdown to see:

- **KJSEA Result Slip** — Mimics the official KNEC Grade 9 result slip with 12 subjects (codes 901–912), 8-level grading (EE1 → BE2), and total points.
- **Recommended Pathway** — The system's recommendation based on KNEC cluster weights and core subject validation. A student must have Mathematics ≥ AE1 (31%) to be eligible for STEM. If no pathway meets its minimum suitability threshold (STEM 20%, SS/Arts 25%), a warning is shown.
- **Pathway Suitability** — Bar chart comparing suitability percentages across STEM, Social Sciences, and Arts and Sports Science. Dashed lines mark KNEC minimums (STEM 20%, SS/Arts 25%).
- **Recommended Track** — Within the recommended pathway, shows the best-matching track (e.g., Pure Sciences, Technical and Engineering) with suggested electives and career options.
- **Subject Scores** — Horizontal bar chart of all subject scores for the selected student.
- **Competency Scores** — Separate chart showing the 7 CBC core competencies (Communication, Critical Thinking, Creativity, Digital Literacy, Learning to Learn, Citizenship, Self-efficacy).
- **Performance History** — Line chart tracking scores from Grade 4 to Grade 9. Click any data point to see a detailed breakdown for that grade.
- **Request Pathway Change** — Students or counselors can submit a request to change the recommended pathway. This goes to the Teacher page for review.

### 3. Teacher

The teacher review panel. Here teachers can:

- **Review pending requests** — See all pathway change requests submitted from the Analysis page. Each request shows the student, current pathway, desired pathway, and reason.
- **Approve or reject** — With a justification. Approved overrides immediately take effect: the student's recommended pathway, key subjects, and track update accordingly on the Analysis page.
- **Edit student data** — Modify individual subject scores if corrections are needed.

### 4. Advanced

For researchers and technical users:

- **Train the DQN Agent** — Train the reinforcement learning model that provides supplementary AI-based recommendations.
- **Run Benchmarks** — Performance Distribution (DKW), Convergence Analysis, and Per-Pathway Accuracy with confidence intervals.
- **Hyperparameter Search** — Grid search over learning rate, gamma, and epsilon decay.
- **Multi-Seed Evaluation** — Run multiple independent training trials to assess reliability.

### 5. About

Overview of the system architecture, CBC framework, and technical details.

---

## Key Concepts

### Pathway Eligibility

A student is recommended for a pathway only if:

1. **Suitability ≥ KNEC minimum** — STEM ≥ 20%, Social Sciences/Arts & Sports ≥ 25% cluster weight.
2. **Core subjects ≥ AE1 (31%)** — STEM requires Mathematics ≥ 31% plus at least one science subject ≥ 31%. Social Sciences requires at least one of Social Studies, English, Kiswahili, or Business Studies ≥ 31%. Arts and Sports requires at least one of Creative Arts, Sports/PE, or Health Education ≥ 31%.

If a student fails both checks for all pathways, the system flags that academic support is needed.

### The Three Pathways

| Pathway | Key Subjects | Tracks |
|---------|-------------|--------|
| STEM | Mathematics, Integrated Science, Science & Technology, Pre-Technical Education | Pure Sciences, Applied Sciences, Technical & Engineering, Career & Technology |
| Social Sciences | Social Studies, English, Kiswahili, Business Studies, Religious Education | Humanities & Business, Languages & Literature |
| Arts and Sports Science | Creative Arts, Sports/PE, Health Education | Performing Arts, Visual Arts, Sports Science |

### KNEC 8-Level Grading

| Level | Range | Category |
|-------|-------|----------|
| EE1 | 90–100% | Exceeding Expectations |
| EE2 | 75–89% | Exceeding Expectations |
| ME1 | 58–74% | Meeting Expectations |
| ME2 | 41–57% | Meeting Expectations |
| AE1 | 31–40% | Approaching Expectations |
| AE2 | 21–30% | Approaching Expectations |
| BE1 | 11–20% | Below Expectations |
| BE2 | 1–10% | Below Expectations |

### Human-in-the-Loop (HITL)

The system supports a formal override process:

1. A student or counselor submits a pathway change request from the Analysis page.
2. A teacher reviews and approves or rejects the request on the Teacher page.
3. If approved, the student's pathway, key subjects, and track recommendation all update to reflect the new pathway.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Page is blank | Click "Generate Data" on the Home page first |
| Student shows no data | Ensure data generation completed (check terminal for errors) |
| Pathway change not reflected | Verify the request was approved on the Teacher page |
| Charts not loading | Refresh the browser; ensure `plotly` is installed |

---

## Technical Requirements

- Python 3.9+
- Dependencies: `dash`, `dash-bootstrap-components`, `plotly`, `numpy`, `pandas`
- All dependencies listed in `requirements.txt`
- No database required — data is generated in-memory and persisted as CSV files in the `data/` directory
