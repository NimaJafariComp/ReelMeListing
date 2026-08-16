"""Typed data contracts for Phase 2 input-quality analysis."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FindingSeverity(StrEnum):
    WARNING = "warning"
    REJECT = "reject"


class InputQualityDecision(StrEnum):
    ACCEPTED = "accepted"
    WARNING = "warning"
    REJECTED = "rejected"


class InputQualityConfig(BaseModel):
    """Calibratable thresholds for deterministic source-photo checks."""

    model_config = ConfigDict(extra="forbid")

    analysis_max_dimension: int = Field(default=1024, ge=128, le=4096)
    blur_warning_variance: float = Field(default=80.0, gt=0)
    blur_reject_variance: float = Field(default=25.0, gt=0)
    highlight_clip_warning_fraction: float = Field(default=0.08, ge=0, le=1)
    highlight_clip_reject_fraction: float = Field(default=0.18, ge=0, le=1)
    shadow_clip_warning_fraction: float = Field(default=0.12, ge=0, le=1)
    shadow_clip_reject_fraction: float = Field(default=0.30, ge=0, le=1)
    color_cast_warning_score: float = Field(default=0.12, ge=0)
    color_cast_reject_score: float = Field(default=0.22, ge=0)
    vertical_line_warning_degrees: float = Field(default=5.0, ge=0, le=45)
    vertical_line_reject_degrees: float = Field(default=10.0, ge=0, le=45)
    vertical_line_reject_enabled: bool = False
    min_vertical_lines: int = Field(default=2, ge=1, le=100)

    @model_validator(mode="after")
    def validate_threshold_ordering(self) -> InputQualityConfig:
        if self.blur_reject_variance >= self.blur_warning_variance:
            raise ValueError("Blur rejection threshold must be lower than warning threshold.")
        for warning, rejection, name in (
            (
                self.highlight_clip_warning_fraction,
                self.highlight_clip_reject_fraction,
                "Highlight clipping",
            ),
            (
                self.shadow_clip_warning_fraction,
                self.shadow_clip_reject_fraction,
                "Shadow clipping",
            ),
            (self.color_cast_warning_score, self.color_cast_reject_score, "Color cast"),
            (
                self.vertical_line_warning_degrees,
                self.vertical_line_reject_degrees,
                "Vertical-line error",
            ),
        ):
            if rejection <= warning:
                raise ValueError(f"{name} rejection threshold must exceed warning threshold.")
        return self


class InputQualityMetrics(BaseModel):
    width: int
    height: int
    exif_orientation: int | None
    blur_laplacian_variance: float
    highlight_clip_fraction: float
    shadow_clip_fraction: float
    color_cast_score: float
    vertical_line_error_degrees: float | None
    vertical_line_count: int


class QualityFinding(BaseModel):
    code: str
    severity: FindingSeverity
    message: str
    metric_value: float | int | None = None


class InputQualityReport(BaseModel):
    """Auditable deterministic analysis result for one source image."""

    report_id: str
    evaluator_version: str = "input-quality-v1"
    created_at: datetime
    source_path: str
    source_sha256: str
    metrics: InputQualityMetrics
    findings: list[QualityFinding]
    reason_codes: list[str]
    decision: InputQualityDecision
    configuration: InputQualityConfig
    diagnostic_overlay_path: str | None


class InputQualityFile(BaseModel):
    """YAML root schema for an input-quality configuration file."""

    model_config = ConfigDict(extra="forbid")

    input_quality: InputQualityConfig
