"""Typed runtime configuration with explicit device-selection policies."""

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeviceTarget(StrEnum):
    MPS = "mps"
    CUDA = "cuda"


class ImageEditorMode(StrEnum):
    PREVIEW_ONLY = "preview_only"
    EVALUATION = "evaluation"


class RuntimeProfile(BaseModel):
    """Execution settings that must be persisted with every future run."""

    model_config = ConfigDict(extra="forbid")

    device: DeviceTarget
    image_resolution: int = Field(ge=256, le=4096)
    batch_size: int = Field(ge=1, le=8)
    attention_slicing: bool
    image_editor_mode: ImageEditorMode
    video_generation_enabled: bool
    benchmark_authority: bool

    @model_validator(mode="after")
    def validate_device_policy(self) -> "RuntimeProfile":
        if self.device is DeviceTarget.MPS:
            if self.video_generation_enabled or self.benchmark_authority:
                raise ValueError(
                    "MPS profiles cannot enable video generation or benchmark authority."
                )
            if self.image_editor_mode is not ImageEditorMode.PREVIEW_ONLY:
                raise ValueError("MPS profiles must use preview_only image editor mode.")
        if (
            self.device is DeviceTarget.CUDA
            and self.image_editor_mode is not ImageEditorMode.EVALUATION
        ):
            raise ValueError("CUDA profiles must use evaluation image editor mode.")
        return self


class RuntimeConfig(BaseModel):
    """Named runtime profiles loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    runtime_profiles: dict[str, RuntimeProfile] = Field(min_length=1)


def load_runtime_config(path: Path) -> RuntimeConfig:
    """Load a YAML profile and fail clearly for missing or malformed documents."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if raw_config is None:
        raise ValueError(f"Runtime configuration is empty: {path}")
    return RuntimeConfig.model_validate(raw_config)
