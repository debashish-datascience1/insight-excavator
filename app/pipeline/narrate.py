"""
Narrator — Phase 6.

Receives ONLY verified numeric facts. Phrases them in plain English (one sentence per finding).
Also builds a Plotly chart per finding.

Anti-hallucination contract: the LLM receives no raw data — only p-value, effect size,
n, and summary_numbers. It is forbidden from adding any claim not present in its input.
"""

import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import linregress
from app.models import Finding, RunState
from app.llm import call_llm

_SYSTEM_PROMPT = """You are a data insight narrator. You receive statistically verified findings.
Your ONLY job: phrase each finding as ONE clear sentence in plain English.

Hard rules:
1. Do NOT add, infer, speculate, or claim anything not present in the inputs.
2. State the direction and magnitude concretely — use the numbers.
3. One sentence per finding. Start with the most surprising fact.
4. Never say "appears", "seems", "may suggest" — state what the numbers show.
5. Mention column names naturally (e.g. "customers on the premium plan spend...").

Respond with ONLY:
{"insights": [{"finding_id": "...", "insight_text": "..."}]}"""


def _build_chart(finding: Finding, df: pd.DataFrame) -> str:
    cols = finding.hypothesis.columns
    try:
        htype = finding.hypothesis.type

        if htype == "correlation" and len(cols) == 2:
            sample = df[cols].dropna().sample(min(500, len(df)), random_state=42)
            x_vals = sample[cols[0]].values.astype(float)
            y_vals = sample[cols[1]].values.astype(float)
            slope, intercept, *_ = linregress(x_vals, y_vals)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="markers", name="data",
                marker=dict(size=5, opacity=0.5, color="#4C72B0"),
            ))
            fig.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept,
                mode="lines", name="trend", line=dict(color="#DD4444", width=2),
            ))
            fig.update_layout(title=f"Correlation: {cols[0]} vs {cols[1]}",
                              xaxis_title=cols[0], yaxis_title=cols[1])

        elif htype == "group_difference" and len(cols) == 2:
            data = df[cols].dropna()
            fig = px.box(data, x=cols[1], y=cols[0], color=cols[1],
                         title=f"{cols[0]} by {cols[1]}")
            fig.update_layout(showlegend=False)

        elif htype == "association" and len(cols) == 2:
            ct = pd.crosstab(df[cols[0]], df[cols[1]])
            ct = ct.head(10)  # top 10 rows for readability
            fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues",
                            title=f"Association: {cols[0]} vs {cols[1]}")

        elif htype == "trend" and len(cols) == 2:
            data = df[cols].dropna().sort_values(cols[1])
            fig = px.line(data, x=cols[1], y=cols[0],
                          title=f"Trend: {cols[0]} over time")

        elif htype == "anomaly" and len(cols) >= 1:
            col = cols[0]
            vals = df[col].dropna().values.astype(float)
            lo = finding.summary_numbers.get("lower_fence")
            hi = finding.summary_numbers.get("upper_fence")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=vals, name=col, marker_color="#4C72B0", opacity=0.75))
            if lo is not None:
                fig.add_vline(x=lo, line_dash="dash", line_color="red",
                              annotation_text="lower fence", annotation_position="top left")
            if hi is not None:
                fig.add_vline(x=hi, line_dash="dash", line_color="red",
                              annotation_text="upper fence", annotation_position="top right")
            fig.update_layout(title=f"Anomaly distribution: {col}", xaxis_title=col)

        else:
            fig = go.Figure()
            fig.add_annotation(text="Chart not available for this finding type",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=13),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig.to_json()

    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Chart could not be generated",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig.to_json()


def run_narrate(state: RunState, df: pd.DataFrame) -> RunState:
    t0 = time.time()
    state.stage = "narrating"

    if not state.verified_findings:
        state.stage = "done"
        state.stage_timings["narrate"] = 0.0
        return state

    # Build payload — ONLY verified numbers, no raw data rows
    payload = [
        {
            "finding_id": f.hypothesis.id,
            "type": f.hypothesis.type,
            "columns": f.hypothesis.columns,
            "test": f.test_name,
            "stat": round(f.stat, 4),
            "effect_size": round(f.effect_size, 4),
            "effect_size_label": f.effect_size_label,
            "p_value": round(f.p_value, 6),
            "n": f.n,
            "summary_numbers": f.summary_numbers,
        }
        for f in state.verified_findings
    ]

    result = call_llm(_SYSTEM_PROMPT, f"Findings:\n{json.dumps(payload, indent=2)}")
    insight_map = {item["finding_id"]: item["insight_text"]
                   for item in result.get("insights", [])}

    for f in state.verified_findings:
        f.insight_text = insight_map.get(f.hypothesis.id, f.hypothesis.description)
        f.chart_json = _build_chart(f, df)

    state.stage = "done"
    state.stage_timings["narrate"] = round(time.time() - t0, 2)
    return state
