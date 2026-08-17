"""Typed contracts for constrained, auditable hero-video generation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from listing_to_reel.core.config import RuntimeProfile
from listing_to_reel.core.environment import EnvironmentSnapshot
from listing_to_reel.media.models import VideoMetadata


class VideoDecision(StrEnum):
    REJECTED = "rejected"
    QUEUED_FOR_HUMAN_REVIEW = "queued_for_human_review"
    ACCEPTED_BY_HUMAN = "accepted_by_human"
    REJECTED_BY_HUMAN = "rejected_by_human"


class PropertyShotRole(StrEnum):
    """Truthful, source-backed roles for a property reel shot."""

    WIDE_EXTERIOR = "wide_exterior"
    SECOND_EXTERIOR = "second_exterior"
    BACKYARD = "backyard"
    POOL_OR_PATIO = "pool_or_patio"
    ARCHITECTURAL_DETAIL = "architectural_detail"
    CLOSING_HERO = "closing_hero"


class VideoGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "diffusers"
    adapter: str = "stable_video_diffusion"
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    model_revision: str | None = None
    num_frames: int = Field(default=25, ge=14, le=25)
    model_fps: int = Field(default=7, ge=1, le=30)
    output_fps: float = Field(default=6.25, gt=0, le=30)
    motion_bucket_id: int = Field(default=32, ge=1, le=255)
    noise_aug_strength: float = Field(default=0.02, ge=0, le=0.2)
    decode_chunk_size: int = Field(default=2, ge=1, le=25)
    width: int = Field(default=1024, ge=256, multiple_of=8)
    height: int = Field(default=576, ge=256, multiple_of=8)
    output_width: int = Field(default=1080, ge=64, multiple_of=2)
    output_height: int = Field(default=1920, ge=64, multiple_of=2)
    crf: int = Field(default=18, ge=0, le=51)

    @model_validator(mode="after")
    def validate_video_config(self) -> VideoGeneratorConfig:
        if self.provider != "diffusers" or self.adapter != "stable_video_diffusion":
            raise ValueError("Phase 5 supports only the Diffusers Stable Video Diffusion adapter.")
        if abs(self.duration_seconds - 4.0) > 0.001:
            raise ValueError("Phase 5 hero clips must be exactly four seconds.")
        return self

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.output_fps


class VideoGeneratorFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_generator: VideoGeneratorConfig


class HeroVideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_decision_path: Path
    seed: int = Field(default=1, ge=0, le=2**63 - 1)
    output_dir: Path = Path("runs/videos")
    runtime_profile_name: str
    runtime_profile: RuntimeProfile
    configuration: VideoGeneratorConfig
    treatment: str = (
        "Smooth stabilized lateral gimbal glide with subtle parallax; deliberate framing; "
        "slow premium real-estate cinematography; preserve all visible property features."
    )


class MultiShotInput(BaseModel):
    """One approved source view that will become one independently generated shot."""

    model_config = ConfigDict(extra="forbid")

    role: PropertyShotRole
    final_decision_path: Path


class MultiShotVideoRequest(BaseModel):
    """Plan only; LTX rendering is intentionally a separate later operation."""

    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(min_length=1, max_length=120)
    shots: list[MultiShotInput] = Field(min_length=4, max_length=6)
    output_dir: Path = Path("runs/video-plans")

    @model_validator(mode="after")
    def validate_shots(self) -> MultiShotVideoRequest:
        roles = [shot.role for shot in self.shots]
        paths = [shot.final_decision_path for shot in self.shots]
        if len(set(roles)) != len(roles):
            raise ValueError("Each multi-shot role may appear only once.")
        if len(set(paths)) != len(paths):
            raise ValueError("Each planned shot must use a distinct approved source view.")
        if PropertyShotRole.WIDE_EXTERIOR not in roles:
            raise ValueError("A multi-shot plan requires a wide_exterior source view.")
        if PropertyShotRole.CLOSING_HERO not in roles:
            raise ValueError("A multi-shot plan requires a closing_hero source view.")
        return self


class PlannedLtxShot(BaseModel):
    index: int = Field(ge=1)
    role: PropertyShotRole
    final_decision_path: str
    evaluation_report_path: str
    source_image_path: str
    source_image_sha256: str
    duration_seconds: float = Field(ge=1.5, le=2.0)
    treatment: str


class MultiShotVideoPlanManifest(BaseModel):
    run_id: str
    phase: str = "phase_5_ltx_multishot_plan"
    created_at: datetime
    property_id: str
    provider: str = "comfyui"
    adapter: str = "ltx_video_2b_distilled"
    shots: list[PlannedLtxShot]
    total_duration_seconds: float = Field(ge=8, le=10)
    source_coverage: dict[str, str]


class HeroVideoManifest(BaseModel):
    run_id: str
    phase: str = "phase_5_hero_video"
    created_at: datetime
    final_decision_path: str
    evaluation_report_path: str
    edit_run_id: str
    selected_candidate_index: int
    hero_image_path: str
    hero_image_sha256: str
    treatment: str
    seed: int
    configuration: VideoGeneratorConfig
    resolved_model_revision: str
    runtime_profile_name: str
    runtime_profile: RuntimeProfile
    environment: EnvironmentSnapshot
    frames_directory: str
    frame_sha256: list[str]
    output_path: str
    output_sha256: str
    video: VideoMetadata
    source_coverage: dict[str, str]


class InterpolationConfig(BaseModel):
    """FFmpeg motion-compensated interpolation settings for delivery playback."""

    model_config = ConfigDict(extra="forbid")

    target_fps: int = Field(default=30, ge=24, le=60)
    crf: int = Field(default=18, ge=0, le=51)


class InterpolatedVideoManifest(BaseModel):
    run_id: str
    phase: str = "phase_5_frame_interpolation"
    created_at: datetime
    parent_video_manifest_path: str
    parent_video_run_id: str
    hero_image_path: str
    input_path: str
    input_sha256: str
    configuration: InterpolationConfig
    output_path: str
    output_sha256: str
    video: VideoMetadata


class TemporalMetrics(BaseModel):
    frame_count: int = Field(ge=1)
    duration_seconds: float = Field(ge=0)
    mean_frame_difference: float = Field(ge=0)
    frame_difference_cv: float = Field(ge=0)
    minimum_edge_f1_to_hero: float = Field(ge=0, le=1)
    maximum_black_pixel_fraction: float = Field(ge=0, le=1)


class VideoQualityReport(BaseModel):
    report_id: str
    phase: str = "phase_5_temporal_qa"
    created_at: datetime
    video_manifest_path: str
    video_run_id: str
    metrics: TemporalMetrics
    reason_codes: list[str]
    decision: VideoDecision
    review_worksheet_path: str | None = None
