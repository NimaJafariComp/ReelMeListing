"""Typed contracts for the deterministic Phase 1 reel pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ReelSettings(BaseModel):
    """Fixed visual and encoding settings recorded for every reel run."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=1080, ge=64, le=4096)
    height: int = Field(default=1920, ge=64, le=4096)
    fps: int = Field(default=30, ge=1, le=120)
    target_duration_seconds: float = Field(default=12.0, ge=8.0, le=20.0)
    transition_seconds: float = Field(default=0.5, gt=0.0, le=5.0)
    zoom_increment_per_frame: float = Field(default=0.0006, gt=0.0, le=0.01)
    zoom_max: float = Field(default=1.06, gt=1.0, le=1.25)
    crf: int = Field(default=20, ge=0, le=51)
    preset: str = "medium"


class ReelRequest(BaseModel):
    """Ordered inputs and output location for a deterministic reel."""

    model_config = ConfigDict(extra="forbid")

    source_paths: list[Path] = Field(min_length=2, max_length=12)
    output_dir: Path = Path("runs")
    settings: ReelSettings = Field(default_factory=ReelSettings)

    @field_validator("source_paths")
    @classmethod
    def source_paths_must_exist(cls, paths: list[Path]) -> list[Path]:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"Source images do not exist: {', '.join(missing)}")
        if len(set(paths)) != len(paths):
            raise ValueError("Source images must be unique and ordered.")
        return paths

    @model_validator(mode="after")
    def transitions_must_fit(self) -> ReelRequest:
        clip_duration = self.clip_duration_seconds
        if self.settings.transition_seconds >= clip_duration:
            raise ValueError("Transition duration must be shorter than each source clip.")
        return self

    @property
    def clip_duration_seconds(self) -> float:
        image_count = len(self.source_paths)
        transition_count = image_count - 1
        return (
            self.settings.target_duration_seconds
            + transition_count * self.settings.transition_seconds
        ) / image_count


class VideoMetadata(BaseModel):
    codec_name: str
    width: int
    height: int
    pixel_format: str
    frame_rate: str
    duration_seconds: float


class ReelRunManifest(BaseModel):
    """Lineage record for a reel run; source hashes keep inputs immutable."""

    run_id: str
    phase: str = "phase_1_deterministic_reel"
    created_at: datetime
    source_paths: list[str]
    source_sha256: dict[str, str]
    normalized_paths: list[str]
    settings: ReelSettings
    ffmpeg_command: list[str]
    output_path: str
    output_sha256: str
    video: VideoMetadata
