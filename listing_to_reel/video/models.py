"""Typed contracts for constrained, auditable hero-video generation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

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


class LtxMotionTreatment(StrEnum):
    LATERAL_GIMBAL = "slow_lateral_gimbal_glide"
    DOLLY_IN = "gentle_dolly_in"


class LtxComfyUiConfig(BaseModel):
    """Pinned native-landscape LTX/ComfyUI settings for the CUDA renderer."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str = "http://127.0.0.1:8188"
    comfyui_root: Path
    model_revision: str
    checkpoint: str = "ltxv-2b-0.9.8-distilled-fp8.safetensors"
    text_encoder: str = "text_encoders\\t5xxl_fp8_e4m3fn.safetensors"
    vae: str = "vae\\LTXV-13B-0.9.8-dev-VAE.safetensors"
    width: int = Field(default=1024, ge=256, multiple_of=32)
    height: int = Field(default=576, ge=256, multiple_of=32)
    frames: int = Field(default=89, ge=17)
    fps: int = Field(default=30, ge=24, le=60)
    steps: int = Field(default=8, ge=1, le=30)
    seed: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_native_landscape(self) -> LtxComfyUiConfig:
        if self.width * 9 != self.height * 16:
            raise ValueError("LTX source clips must use native 16:9 landscape framing.")
        if (self.frames - 1) % 8:
            raise ValueError("LTX frame count must be 8n+1.")
        return self

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.fps


class LtxSourceView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source_path: Path
    treatment: LtxMotionTreatment


class LtxRenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(min_length=1)
    source_views: list[LtxSourceView] = Field(min_length=1, max_length=6)
    bridge_candidate_ids: list[str] = Field(default_factory=list)
    bridge_duration_seconds: dict[str, float] = Field(default_factory=dict)
    configuration: LtxComfyUiConfig
    output_dir: Path = Path("runs/ltx-videos")

    @model_validator(mode="after")
    def validate_distinct_source_views(self) -> LtxRenderRequest:
        paths = [view.source_path for view in self.source_views]
        if len(set(paths)) != len(paths):
            raise ValueError("Each LTX source view must be distinct.")
        if len(set(self.bridge_candidate_ids)) != len(self.bridge_candidate_ids):
            raise ValueError("Each selected LTX bridge candidate must be distinct.")
        if set(self.bridge_duration_seconds) - set(self.bridge_candidate_ids):
            raise ValueError(
                "Bridge duration can be supplied only for a selected bridge candidate."
            )
        return self


class LtxGeneratedClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source_path: str
    source_sha256: str
    source_coverage: str
    treatment: LtxMotionTreatment
    prompt: str
    workflow: dict[str, object]
    generated_path: str
    generated_sha256: str
    video: VideoMetadata
    decision: VideoDecision = VideoDecision.QUEUED_FOR_HUMAN_REVIEW


class LtxBridgeKind(StrEnum):
    SPATIAL = "spatial_overlap"
    LIGHTING_ONLY = "lighting_only"


class LtxBridgeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    kind: LtxBridgeKind
    from_view: str
    to_view: str
    duration_seconds: float = Field(default=3.0, ge=2.0, le=4.0)
    prompt: str
    decision: VideoDecision = VideoDecision.QUEUED_FOR_HUMAN_REVIEW
    reason: str


class LtxGeneratedBridge(LtxBridgeCandidate):
    """A user-selected, multi-conditioned LTX transition awaiting human approval."""

    from_source_path: str
    from_source_sha256: str
    to_source_path: str
    to_source_sha256: str
    workflow: dict[str, object]
    generated_path: str
    generated_sha256: str
    video: VideoMetadata


class LtxRenderManifest(BaseModel):
    run_id: str
    phase: str = "phase_5_ltx_comfyui_render"
    created_at: datetime
    property_id: str
    generator: str = "comfyui_ltx_video_only"
    configuration: LtxComfyUiConfig
    clips: list[LtxGeneratedClip]
    bridge_candidates: list[LtxBridgeCandidate]
    bridges: list[LtxGeneratedBridge] = Field(default_factory=list)
    source_coverage: dict[str, str]
    qa_status: VideoDecision = VideoDecision.QUEUED_FOR_HUMAN_REVIEW


class LtxClipQualityReport(BaseModel):
    name: str
    generated_path: str
    metrics: TemporalMetrics
    reason_codes: list[str]
    decision: VideoDecision


class LtxBridgeQualityReport(BaseModel):
    candidate_id: str
    generated_path: str
    metrics: TemporalMetrics
    endpoint_edge_f1: float
    reason_codes: list[str]
    decision: VideoDecision


class LtxQualityReport(BaseModel):
    report_id: str
    phase: str = "phase_5_ltx_comfyui_qa"
    created_at: datetime
    render_manifest_path: str
    clips: list[LtxClipQualityReport]
    bridge_candidates: list[LtxBridgeCandidate]
    bridges: list[LtxBridgeQualityReport] = Field(default_factory=list)
    decision: VideoDecision
    review_worksheet_path: str


class LtxReelManifest(BaseModel):
    run_id: str
    phase: str = "phase_5_ltx_portrait_assembly"
    created_at: datetime
    render_manifest_path: str
    accepted_clip_names: list[str]
    source_coverage: dict[str, str]
    output_path: str
    output_sha256: str
    video: VideoMetadata
    foreground_treatment: str = (
        "Complete native 16:9 landscape foreground centered over a blurred portrait background; "
        "foreground property is never cropped."
    )


class LtxTimedReelItem(BaseModel):
    """One source-backed shot or selected LTX bridge in a delivery timeline."""

    kind: Literal["ltx_bridge", "source_clip"]
    name: str
    input_path: str
    source_duration_seconds: float = Field(gt=0)
    delivery_duration_seconds: float = Field(gt=0)
    playback_speed: float = Field(gt=0)
    transition_before: Literal["opening", "continuous", "intentional_cut"]


class LtxTimedReelPlan(BaseModel):
    phase: str = "phase_5_ltx_timed_reel_plan"
    run_id: str
    created_at: datetime
    render_manifest_path: str
    requested_total_duration_seconds: float = Field(ge=8.0, le=120.0)
    requested_bridge_duration_seconds: float = Field(ge=2.0, le=4.0)
    items: list[LtxTimedReelItem] = Field(min_length=2)
    output_duration_seconds: float = Field(gt=0)
    optimization_notes: list[str]
