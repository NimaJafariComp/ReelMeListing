"""Typed contracts for Phase 3 image-editing runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from listing_to_reel.core.config import RuntimeProfile
from listing_to_reel.core.environment import EnvironmentSnapshot


class ImageEditorConfig(BaseModel):
    """Configuration for a swappable instruction-based editing adapter."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "diffusers"
    adapter: str = "instruct_pix2pix"
    model_id: str = "timbrooks/instruct-pix2pix"
    model_revision: str | None = None
    scheduler: str = "euler_ancestral"
    torch_dtype: str = "float16"
    num_inference_steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    image_guidance_scale: float = Field(default=1.5, ge=1.0, le=10.0)
    candidate_count: int = Field(default=2, ge=1, le=4)
    max_dimension: int = Field(default=512, ge=256, le=2048)
    output_quality: int = Field(default=95, ge=1, le=100)

    @model_validator(mode="after")
    def validate_adapter(self) -> ImageEditorConfig:
        if self.provider != "diffusers" or self.adapter != "instruct_pix2pix":
            raise ValueError(
                "Phase 3 supports only the Diffusers InstructPix2Pix baseline adapter."
            )
        if self.scheduler != "euler_ancestral":
            raise ValueError("Phase 3 supports the euler_ancestral scheduler only.")
        if self.torch_dtype not in {"float16", "float32"}:
            raise ValueError("torch_dtype must be float16 or float32.")
        if self.max_dimension % 8 != 0:
            raise ValueError("max_dimension must be divisible by 8 for the diffusion VAE.")
        return self


class ImageEditingFile(BaseModel):
    """YAML root schema for baseline image-editing configuration."""

    model_config = ConfigDict(extra="forbid")

    image_editor: ImageEditorConfig


class EditRequest(BaseModel):
    """Fully resolved, reproducible request for one source image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_path: Path
    input_quality_report_path: Path
    instruction: str = Field(min_length=10, max_length=1000)
    seed: int = Field(default=1, ge=0, le=2**63 - 1)
    output_dir: Path = Path("runs/edits")
    runtime_profile_name: str
    runtime_profile: RuntimeProfile
    configuration: ImageEditorConfig


class GeneratedCandidate(BaseModel):
    candidate_index: int = Field(ge=1)
    seed: int = Field(ge=0)
    artifact_path: str
    sha256: str
    wall_clock_seconds: float = Field(ge=0)
    peak_device_memory_bytes: int | None = Field(default=None, ge=0)


class EditRunManifest(BaseModel):
    """Lineage record emitted before Phase 4 decides whether a candidate is usable."""

    run_id: str
    phase: str = "phase_3_baseline_image_editing"
    created_at: datetime
    source_path: str
    source_sha256: str
    input_quality_report_path: str
    instruction: str
    configuration: ImageEditorConfig
    requested_model_revision: str | None
    resolved_model_revision: str
    runtime_profile_name: str
    runtime_profile: RuntimeProfile
    environment: EnvironmentSnapshot
    candidates: list[GeneratedCandidate]
    acceptance_decision: str = "pending_phase_4_evaluation"
