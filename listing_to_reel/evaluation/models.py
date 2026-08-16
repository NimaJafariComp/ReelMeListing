"""Typed, auditable contracts for Phase 4 image-candidate evaluation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateDecision(StrEnum):
    REJECTED = "rejected"
    QUEUED_FOR_HUMAN_REVIEW = "queued_for_human_review"
    ACCEPTED_BY_HUMAN = "accepted_by_human"
    REJECTED_BY_HUMAN = "rejected_by_human"


class RunDecision(StrEnum):
    RETRY_RECOMMENDED = "retry_recommended"
    QUEUED_FOR_HUMAN_REVIEW = "queued_for_human_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class EvaluationConfig(BaseModel):
    """Calibratable deterministic thresholds, not a truthfulness classifier."""

    model_config = ConfigDict(extra="forbid")

    analysis_max_dimension: int = Field(default=768, ge=128, le=4096)
    edge_f1_warning: float = Field(default=0.40, ge=0, le=1)
    edge_f1_reject: float = Field(default=0.25, ge=0, le=1)
    blur_ratio_warning: float = Field(default=0.55, ge=0, le=10)
    blur_ratio_reject: float = Field(default=0.25, ge=0, le=10)
    black_pixel_reject_fraction: float = Field(default=0.25, ge=0, le=1)
    vertical_line_delta_warning_degrees: float = Field(default=4.0, ge=0, le=45)
    min_edge_pixels: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> EvaluationConfig:
        if self.edge_f1_reject >= self.edge_f1_warning:
            raise ValueError("edge_f1_reject must be lower than edge_f1_warning.")
        if self.blur_ratio_reject >= self.blur_ratio_warning:
            raise ValueError("blur_ratio_reject must be lower than blur_ratio_warning.")
        return self


class EvaluationFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation: EvaluationConfig


class CandidateMetrics(BaseModel):
    edge_f1: float | None = Field(default=None, ge=0, le=1)
    source_edge_pixels: int = Field(ge=0)
    candidate_edge_pixels: int = Field(ge=0)
    blur_ratio: float | None = Field(default=None, ge=0)
    source_blur_variance: float = Field(ge=0)
    candidate_blur_variance: float = Field(ge=0)
    black_pixel_fraction: float = Field(ge=0, le=1)
    mean_luminance_delta: float = Field(ge=0)
    vertical_line_delta_degrees: float | None = Field(default=None, ge=0, le=45)


class CandidateEvaluation(BaseModel):
    candidate_index: int = Field(ge=1)
    candidate_path: str
    candidate_sha256: str
    metrics: CandidateMetrics
    reason_codes: list[str]
    score: float = Field(ge=0, le=1)
    decision: CandidateDecision


class EvaluationReport(BaseModel):
    report_id: str
    phase: str = "phase_4_evaluation"
    evaluator_version: str = "candidate-evaluation-v1"
    created_at: datetime
    edit_run_manifest_path: str
    edit_run_id: str
    source_path: str
    source_sha256: str
    configuration: EvaluationConfig
    candidates: list[CandidateEvaluation]
    recommended_candidate_index: int | None
    run_decision: RunDecision
    reason_codes: list[str]
    review_worksheet_path: str | None = None


class HumanReviewDecision(BaseModel):
    blind_candidate_id: str
    reviewer: str = Field(min_length=1, max_length=200)
    decision: CandidateDecision
    notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_human_decision(self) -> HumanReviewDecision:
        if self.decision not in {
            CandidateDecision.ACCEPTED_BY_HUMAN,
            CandidateDecision.REJECTED_BY_HUMAN,
        }:
            raise ValueError("Human review decision must accept or reject a candidate.")
        return self


class FinalDecisionRecord(BaseModel):
    phase: str = "phase_4_final_decision"
    created_at: datetime
    evaluation_report_path: str
    edit_run_id: str
    decision: RunDecision
    selected_candidate_index: int | None
    reviewer: str
    notes: str
    reason_codes: list[str]
