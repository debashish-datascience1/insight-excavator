"""
Pipeline controller — orchestrates all stages in order via a single RunState object.
Each stage is a pure-ish function: stage(state, ...) -> state.
Stage timings are logged for progress UI and architecture evidence.
"""

import time
import uuid
import pandas as pd
from app.models import RunState
from app.pipeline.ingest import run_ingest, profile_dataframe, maybe_sample_for_llm
from app.pipeline.clean import run_clean
from app.pipeline.hypothesize import run_hypothesize
from app.pipeline.verify import run_verify
from app.pipeline.gate import run_gate
from app.pipeline.narrate import run_narrate
import app.config as cfg


def run_pipeline(
    df: pd.DataFrame,
    file_name: str = "upload.csv",
    progress_callback=None,  # optional callable(stage_name: str, pct: int)
) -> RunState:
    """Run the full pipeline. Returns a completed RunState."""

    def _progress(stage: str, pct: int) -> None:
        if progress_callback:
            progress_callback(stage, pct)

    state = RunState(run_id=str(uuid.uuid4()), file_name=file_name)

    try:
        # Stage 1 — Profile (deterministic, no LLM)
        _progress("Profiling dataset…", 5)
        state = run_ingest(state, df)

        # Stage 2 — Clean (LLM proposes → deterministic executor)
        _progress("Cleaning data…", 15)
        # For LLM stages, sample large datasets to keep prompts fast and within token limits.
        # Stat tests (verify) always run on the full dataframe for accuracy.
        llm_df, was_sampled = maybe_sample_for_llm(df)
        if was_sampled:
            state.data_sample_json = llm_df.head(20).to_json(orient="records", date_format="iso")
        state, df = run_clean(state, df)

        # Re-profile after cleaning so hypotheses see the clean schema
        state.profile = profile_dataframe(df)
        llm_df, _ = maybe_sample_for_llm(df)

        # Rounds of hypothesis generation + statistical verification
        for round_num in range(1, cfg.VERIFY_LOOP_ROUNDS + 1):
            base_pct = 30 + (round_num - 1) * 25

            _progress(f"Generating hypotheses (round {round_num})…", base_pct)
            prior = state.verified_findings if round_num > 1 else None
            # Hypothesize uses the (possibly sampled) llm_df for prompt context only
            state = run_hypothesize(state, llm_df, round_number=round_num, prior_findings=prior)

            _progress(f"Verifying hypotheses (round {round_num})…", base_pct + 10)
            # Verify always uses the full df for statistical accuracy
            state = run_verify(state, df)
            state = run_gate(state)

        # Stage 6 — Narrate verified findings + build charts
        _progress("Narrating verified insights…", 85)
        state = run_narrate(state, df)

        _progress("Done", 100)

    except Exception as exc:
        state.stage = "error"
        state.error = str(exc)

    return state
