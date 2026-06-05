"""
Streamlit UI — Insight Excavator
Upload → profile → clean → verify → insight cards + charts → download report
"""

import sys
import os
import uuid
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow running from repo root or ui/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.ingest import load_file, run_ingest, profile_dataframe
from app.pipeline.clean import run_clean
from app.pipeline.hypothesize import run_hypothesize
from app.pipeline.verify import run_verify
from app.pipeline.gate import run_gate
from app.pipeline.narrate import run_narrate
from app.models import RunState
import app.config as cfg

# ─── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Insight Excavator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .insight-card { border-left: 4px solid #4C72B0; padding: 0.5rem 1rem; margin-bottom: 0.5rem; }
  .stat-badge { background: #f0f2f6; border-radius: 4px; padding: 2px 8px;
                font-size: 0.82rem; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("Insight Excavator")
st.caption("Most data tools guess. **This one proves.**")

# ─── Sidebar — config info ────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown(
        "An agentic pipeline that **statistically verifies** every insight before showing it. "
        "No hallucinated analytics.\n\n"
        "**Verify loop:** LLM proposes → stat test confirms → significance gate keeps only real findings."
    )
    st.divider()
    st.caption(f"Rounds: {cfg.VERIFY_LOOP_ROUNDS} · α = {cfg.SIGNIFICANCE_THRESHOLD}")

# ─── Upload ───────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload a dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="The pipeline will profile, clean, and discover verified insights automatically.",
)

if not uploaded:
    st.info("Upload a CSV or Excel file to begin. Try the sample dataset in `data/sample/`.")
    st.stop()

# ─── Run pipeline (cached per file) ───────────────────────────

file_key = f"{uploaded.name}_{uploaded.size}"

if st.session_state.get("file_key") != file_key:
    st.session_state.file_key = file_key
    st.session_state.result = None
    st.session_state.df_clean = None

if st.session_state.result is None:
    content = uploaded.read()
    try:
        df_raw = load_file(content, uploaded.name)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    status_box = st.empty()
    progress_bar = st.progress(0)

    state = RunState(run_id=str(uuid.uuid4()), file_name=uploaded.name)

    def _update(msg: str, pct: int) -> None:
        status_box.info(f"**{msg}**")
        progress_bar.progress(pct)

    # Stage 1 — Profile
    _update("Profiling dataset…", 5)
    state = run_ingest(state, df_raw)
    df = df_raw.copy()

    # Stage 2 — Clean
    _update("Cleaning data (LLM proposes ops → deterministic executor)…", 15)
    try:
        state, df = run_clean(state, df)
        state.profile = profile_dataframe(df)
    except Exception as e:
        st.warning(f"Cleaning skipped (LLM unavailable): {e}")

    # Verify rounds
    for r in range(1, cfg.VERIFY_LOOP_ROUNDS + 1):
        pct_base = 30 + (r - 1) * 25
        _update(f"Generating hypotheses — round {r}…", pct_base)
        try:
            prior = state.verified_findings if r > 1 else None
            state = run_hypothesize(state, df, round_number=r, prior_findings=prior)
        except Exception as e:
            st.warning(f"Hypothesis generation skipped (LLM unavailable): {e}")
            break

        _update(f"Verifying hypotheses statistically — round {r}…", pct_base + 12)
        state = run_verify(state, df)
        state = run_gate(state)

    # Narrate
    _update("Narrating verified findings…", 85)
    try:
        state = run_narrate(state, df)
    except Exception as e:
        st.warning(f"Narration skipped (LLM unavailable): {e}")
        state.stage = "done"

    progress_bar.progress(100)
    n_insights = len(state.verified_findings)
    status_box.success(
        f"Done in {sum(state.stage_timings.values()):.1f}s — "
        f"**{n_insights} verified insight{'s' if n_insights != 1 else ''}** found."
    )

    st.session_state.result = state
    st.session_state.df_clean = df

state: RunState = st.session_state.result
df: pd.DataFrame = st.session_state.df_clean

# ─── Summary metrics ──────────────────────────────────────────

st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows analysed", f"{state.profile.row_count:,}")
c2.metric("Columns", state.profile.col_count)
c3.metric("Hypotheses tested", len(state.findings))
c4.metric("Verified insights", len(state.verified_findings))
if state.stage_timings:
    c5.metric("Total time", f"{sum(state.stage_timings.values()):.1f}s")

if state.cleaning_result:
    cr = state.cleaning_result
    st.caption(f"**Cleaning:** {cr.diff_summary}")

# ─── Stage timings (architecture evidence) ────────────────────

with st.expander("Stage timings", expanded=False):
    for stage, secs in state.stage_timings.items():
        st.text(f"  {stage:<30} {secs:.2f}s")

# ─── No insights ──────────────────────────────────────────────

if not state.verified_findings:
    st.warning(
        "No statistically significant insights found. "
        "Try a larger or more varied dataset (aim for > 200 rows and a mix of numeric + categorical columns)."
    )
    st.stop()

# ─── Insight cards ────────────────────────────────────────────

st.divider()
st.subheader(f"Verified Insights ({len(state.verified_findings)})")
st.caption("Every finding below has passed a real statistical test. p-value, effect size, and n are shown on each card.")

for i, finding in enumerate(state.verified_findings):
    header = finding.insight_text or finding.hypothesis.description
    with st.expander(f"#{i + 1} — {header}", expanded=(i == 0)):
        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.markdown(f"**{header}**")
            st.markdown("---")
            st.markdown(
                f"**Test:** `{finding.test_name}`  \n"
                f"**Effect ({finding.effect_size_label}):** "
                f"`{finding.effect_size:.4f}`  \n"
                f"**p-value:** `{finding.p_value:.4e}`  \n"
                f"**Sample size (n):** `{finding.n:,}`  \n"
                f"**Surprise score:** `{finding.surprise_score:.4f}`"
            )
            with st.expander("Detailed stats"):
                st.json(finding.summary_numbers)

        with col_right:
            if finding.chart_json:
                try:
                    fig = go.Figure(json.loads(finding.chart_json))
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
                except Exception:
                    st.caption("Chart unavailable")

# ─── Download report ──────────────────────────────────────────

st.divider()

report_lines = [
    "# Insight Excavator — Analysis Report\n",
    f"**Dataset:** {state.file_name}",
    f"**Rows:** {state.profile.row_count:,} | **Columns:** {state.profile.col_count}",
    f"**Hypotheses tested:** {len(state.findings)} | **Verified insights:** {len(state.verified_findings)}\n",
    "---\n",
    "## Verified Insights\n",
]
for i, f in enumerate(state.verified_findings):
    report_lines += [
        f"### Insight {i + 1}",
        f"{f.insight_text}\n",
        f"- **Test:** {f.test_name}",
        f"- **Effect size ({f.effect_size_label}):** {f.effect_size:.4f}",
        f"- **p-value:** {f.p_value:.4e}",
        f"- **n:** {f.n:,}",
        f"- **Surprise score:** {f.surprise_score:.4f}\n",
    ]

st.download_button(
    label="Download report (Markdown)",
    data="\n".join(report_lines),
    file_name="insight_excavator_report.md",
    mime="text/markdown",
)
