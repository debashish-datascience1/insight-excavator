from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field
import uuid


class HypothesisType(str, Enum):
    CORRELATION = "correlation"
    GROUP_DIFFERENCE = "group_difference"
    ASSOCIATION = "association"
    TREND = "trend"
    ANOMALY = "anomaly"
    MISSINGNESS = "missingness"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any]
    stats: dict[str, Any] = Field(default_factory=dict)


class Profile(BaseModel):
    row_count: int
    col_count: int
    columns: list[ColumnProfile]
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    text_cols: list[str]


class Operation(BaseModel):
    op: str
    column: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str


class CleaningResult(BaseModel):
    operations_applied: list[Operation]
    rows_before: int
    rows_after: int
    rows_removed: int
    cells_changed: int
    diff_summary: str


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str
    columns: list[str]
    description: str
    round_number: int = 1


class Finding(BaseModel):
    hypothesis: Hypothesis
    test_name: str
    stat: float
    effect_size: float
    effect_size_label: str
    p_value: float
    n: int
    significant: bool
    summary_numbers: dict[str, Any] = Field(default_factory=dict)
    insight_text: str = ""
    chart_json: str = ""
    surprise_score: float = 0.0


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str = ""
    stage: str = "idle"
    profile: Optional[Profile] = None
    cleaning_result: Optional[CleaningResult] = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    verified_findings: list[Finding] = Field(default_factory=list)
    error: Optional[str] = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    data_sample_json: Optional[str] = None
