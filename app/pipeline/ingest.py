"""
Ingest and profile a dataset. Pure code, no LLM.
"""

import io
import time
import pandas as pd
import numpy as np
from app.models import Profile, ColumnProfile, RunState

SAMPLE_VALUES_N = 5
CATEGORICAL_MAX_UNIQUE = 50
CATEGORICAL_MAX_UNIQUE_RATIO = 0.2


def _infer_kind(col: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"
    if pd.api.types.is_numeric_dtype(col):
        return "numeric"
    if col.dtype == object:
        sample = col.dropna().head(30)
        try:
            pd.to_datetime(sample, infer_datetime_format=True)
            return "datetime"
        except Exception:
            pass
    n_unique = col.nunique()
    n_total = len(col.dropna())
    if n_total == 0:
        return "text"
    ratio = n_unique / max(n_total, 1)
    if n_unique <= CATEGORICAL_MAX_UNIQUE and ratio <= CATEGORICAL_MAX_UNIQUE_RATIO:
        return "categorical"
    return "text"


def _col_stats(col: pd.Series, kind: str) -> dict:
    if kind == "numeric":
        vals = col.dropna()
        if len(vals) == 0:
            return {}
        return {
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "median": round(float(vals.median()), 4),
            "q25": round(float(vals.quantile(0.25)), 4),
            "q75": round(float(vals.quantile(0.75)), 4),
        }
    if kind == "categorical":
        top = col.value_counts().head(5)
        return {"top_values": {str(k): int(v) for k, v in top.items()}}
    if kind == "datetime":
        vals = pd.to_datetime(col, errors="coerce").dropna()
        if len(vals) == 0:
            return {}
        return {"min": str(vals.min()), "max": str(vals.max())}
    return {}


def profile_dataframe(df: pd.DataFrame) -> Profile:
    columns, numeric_cols, categorical_cols, datetime_cols, text_cols = [], [], [], [], []

    for col_name in df.columns:
        col = df[col_name]
        kind = _infer_kind(col)
        null_count = int(col.isna().sum())
        null_pct = round(float(null_count / max(len(col), 1)), 4)
        unique_count = int(col.nunique())
        sample_values = [str(v) for v in col.dropna().head(SAMPLE_VALUES_N).tolist()]

        columns.append(ColumnProfile(
            name=col_name,
            dtype=str(col.dtype),
            kind=kind,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            sample_values=sample_values,
            stats=_col_stats(col, kind),
        ))

        if kind == "numeric":
            numeric_cols.append(col_name)
        elif kind == "categorical":
            categorical_cols.append(col_name)
        elif kind == "datetime":
            datetime_cols.append(col_name)
        else:
            text_cols.append(col_name)

    return Profile(
        row_count=len(df),
        col_count=len(df.columns),
        columns=columns,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        text_cols=text_cols,
    )


def load_file(content: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    raise ValueError(f"Unsupported file type: {filename}. Upload a CSV or Excel file.")


def run_ingest(state: RunState, df: pd.DataFrame) -> RunState:
    t0 = time.time()
    state.stage = "profiling"
    state.profile = profile_dataframe(df)
    state.data_sample_json = df.head(20).to_json(orient="records", date_format="iso")
    state.stage_timings["ingest"] = round(time.time() - t0, 2)
    return state
