"""Typed, auditable contracts for the Phase 8 LoRA readiness decision."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ImageRole(StrEnum):
    DAYLIGHT_SOURCE = "daylight_source"
    DUSK_TARGET = "dusk_target"


class TrainingAsset(BaseModel):
    """One licensed source or target image in a property-grouped dataset."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1, max_length=160)
    listing_group_id: str = Field(min_length=1, max_length=160)
    split: DatasetSplit
    role: ImageRole
    source_uri: str = Field(min_length=1, max_length=2000)
    rights_basis: str = Field(min_length=1, max_length=160)
    rights_evidence_ref: str = Field(min_length=1, max_length=2000)
    derivative_use_allowed: bool
    training_use_allowed: bool
    portfolio_display_allowed: bool


class DayToDuskPair(BaseModel):
    """A same-property, same-view daylight-to-dusk supervision pair."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str = Field(min_length=1, max_length=160)
    source_asset_id: str = Field(min_length=1, max_length=160)
    target_asset_id: str = Field(min_length=1, max_length=160)
    same_camera_view: bool
    geometry_verified: bool


class LoRADatasetManifest(BaseModel):
    """Data contract intentionally designed to prevent rights and split shortcuts."""

    model_config = ConfigDict(extra="forbid")

    treatment: str = Field(
        default=(
            "Natural premium exterior dusk conversion that preserves building geometry and "
            "avoids oversaturated skies."
        )
    )
    assets: list[TrainingAsset] = Field(min_length=1)
    pairs: list[DayToDuskPair] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> LoRADatasetManifest:
        by_id = {asset.asset_id: asset for asset in self.assets}
        if len(by_id) != len(self.assets):
            raise ValueError("Asset IDs must be unique.")
        if len({pair.pair_id for pair in self.pairs}) != len(self.pairs):
            raise ValueError("Pair IDs must be unique.")
        for pair in self.pairs:
            if pair.source_asset_id not in by_id or pair.target_asset_id not in by_id:
                raise ValueError(f"Pair {pair.pair_id} references an unknown asset.")
            source, target = by_id[pair.source_asset_id], by_id[pair.target_asset_id]
            if (
                source.role is not ImageRole.DAYLIGHT_SOURCE
                or target.role is not ImageRole.DUSK_TARGET
            ):
                raise ValueError(f"Pair {pair.pair_id} must map daylight_source to dusk_target.")
            if source.listing_group_id != target.listing_group_id or source.split != target.split:
                raise ValueError(
                    f"Pair {pair.pair_id} must stay within one property group and split."
                )
        return self


class ReadinessEvidence(BaseModel):
    """References to completed baseline evidence, not unsupported self-attestations."""

    model_config = ConfigDict(extra="forbid")

    geometry_gate_report: str | None = None
    structure_conditioning_comparison: str | None = None
    reproducible_video_qa_report: str | None = None
    baseline_comparison_report: str | None = None


class LoRAPilotConfig(BaseModel):
    """Reproducible, deliberately narrow image-editing adapter configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "diffusers"
    adapter: str = "instruct_pix2pix_lora"
    base_model_id: str = "timbrooks/instruct-pix2pix"
    base_model_revision: str | None = None
    base_model_frozen: bool = True
    target: str = "image_editing_adapter_only"
    resolution: int = Field(default=512, ge=256, le=1024, multiple_of=8)
    rank: int = Field(default=16, ge=1, le=128)
    alpha: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.01)
    max_train_steps: int = Field(default=1200, ge=1, le=100_000)
    train_batch_size: int = Field(default=1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_scope(self) -> LoRAPilotConfig:
        if self.provider != "diffusers" or self.adapter != "instruct_pix2pix_lora":
            raise ValueError("Phase 8 supports only a Diffusers InstructPix2Pix LoRA pilot.")
        if not self.base_model_frozen or self.target != "image_editing_adapter_only":
            raise ValueError("Phase 8 freezes the base model and never trains a video model.")
        return self


class LoRAReadinessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: LoRADatasetManifest
    evidence: ReadinessEvidence
    configuration: LoRAPilotConfig = Field(default_factory=LoRAPilotConfig)


class LoRAReadinessReport(BaseModel):
    report_id: str
    phase: str = "phase_8_lora_readiness"
    created_at: datetime
    decision: str
    reason_codes: list[str]
    pair_counts: dict[str, int]
    property_counts: dict[str, int]
    configuration: LoRAPilotConfig
    training_permitted: bool
