"""
Cleaning agent — Phase 3.

LLM proposes operations drawn from a FIXED vocabulary.
A deterministic executor applies each op. LLM-written raw code is never executed.
"""

import time
import json
import pandas as pd
from app.models import Operation, CleaningResult, RunState, Profile
from app.llm import call_llm

ALLOWED_OPS = frozenset({
    "dedupe", "coerce_type", "trim_strings", "parse_dates",
    "fill_nulls", "drop_nulls", "normalize_strings",
})

_SYSTEM_PROMPT = """You are a data cleaning assistant. Inspect the dataset profile and propose cleaning operations.

You MUST only use operations from this EXACT vocabulary (no other ops will run):
  dedupe            — remove exact-duplicate rows (column: null)
  coerce_type       — convert a column dtype  (params: {"target_type": "int"|"float"|"str"})
  trim_strings      — strip leading/trailing whitespace from a string column
  normalize_strings — lowercase and strip a categorical column
  parse_dates       — parse a column as datetime (params: {"format": "optional strftime string"})
  fill_nulls        — fill missing values  (params: {"strategy": "mean"|"median"|"mode"|"constant", "value": <for constant>})
  drop_nulls        — drop rows where a column is null

Rules:
1. Only propose ops genuinely needed for this data.
2. Always propose dedupe first if duplicates are plausible.
3. Prioritize operations that improve downstream statistical analysis.
4. Avoid coercing columns that already have the correct type.

Respond with ONLY a JSON object:
{"operations": [{"op": "...", "column": "..." or null, "params": {}, "reason": "one line"}]}"""


def _propose_operations(profile: Profile, sample_json: str) -> list[Operation]:
    summary = {
        "row_count": profile.row_count,
        "columns": [
            {
                "name": c.name, "kind": c.kind, "dtype": c.dtype,
                "null_pct": c.null_pct, "unique_count": c.unique_count,
                "sample_values": c.sample_values[:3],
            }
            for c in profile.columns
        ],
    }
    result = call_llm(
        _SYSTEM_PROMPT,
        f"Profile:\n{json.dumps(summary, indent=2)}\n\nSample rows:\n{sample_json}",
    )
    ops = []
    for item in result.get("operations", []):
        if item.get("op") not in ALLOWED_OPS:
            continue
        try:
            ops.append(Operation(
                op=item["op"],
                column=item.get("column"),
                params=item.get("params", {}),
                reason=item.get("reason", ""),
            ))
        except Exception:
            continue
    return ops


def _execute_op(df: pd.DataFrame, op: Operation) -> tuple[pd.DataFrame, int]:
    """Execute one cleaning op deterministically. Returns (df, cells_changed)."""
    if op.op == "dedupe":
        df = df.drop_duplicates().reset_index(drop=True)
        return df, 0

    col = op.column
    if col is None or col not in df.columns:
        return df, 0

    before_col = df[col].astype(str).copy()

    if op.op == "trim_strings":
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()

    elif op.op == "normalize_strings":
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.lower().str.strip()

    elif op.op == "coerce_type":
        target = op.params.get("target_type", "str")
        try:
            if target == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif target == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(str).replace("nan", pd.NA)
        except Exception:
            pass

    elif op.op == "parse_dates":
        fmt = op.params.get("format") or None
        try:
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
        except Exception:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    elif op.op == "fill_nulls":
        strategy = op.params.get("strategy", "mode")
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].mean()
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
        elif strategy == "constant":
            fill_val = op.params.get("value", 0)
        else:
            mode = df[col].mode()
            fill_val = mode.iloc[0] if len(mode) else None
        if fill_val is not None:
            df[col] = df[col].fillna(fill_val)

    elif op.op == "drop_nulls":
        df = df.dropna(subset=[col]).reset_index(drop=True)
        return df, 0

    changed = int((before_col != df[col].astype(str)).sum())
    return df, changed


def run_clean(state: RunState, df: pd.DataFrame) -> tuple[RunState, pd.DataFrame]:
    t0 = time.time()
    state.stage = "cleaning"

    sample_json = df.head(10).to_json(orient="records", date_format="iso")
    operations = _propose_operations(state.profile, sample_json)

    rows_before = len(df)
    total_changed = 0
    applied: list[Operation] = []

    for op in operations:
        df, changed = _execute_op(df, op)
        total_changed += changed
        applied.append(op)

    rows_after = len(df)
    state.cleaning_result = CleaningResult(
        operations_applied=applied,
        rows_before=rows_before,
        rows_after=rows_after,
        rows_removed=rows_before - rows_after,
        cells_changed=total_changed,
        diff_summary=(
            f"Applied {len(applied)} operations. "
            f"Removed {rows_before - rows_after} rows, modified {total_changed} cells."
        ),
    )
    state.stage_timings["clean"] = round(time.time() - t0, 2)
    return state, df
