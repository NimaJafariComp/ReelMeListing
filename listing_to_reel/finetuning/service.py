"""Evaluate Phase 8 training readiness without beginning a training run."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from listing_to_reel.finetuning.models import (
    DatasetSplit,
    LoRAPilotConfig,
    LoRAReadinessReport,
    LoRAReadinessRequest,
)


def load_lora_pilot_config(path: Path) -> LoRAPilotConfig:
    """Load a public-safe, frozen-base pilot configuration."""
    with path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    return LoRAPilotConfig.model_validate(raw["lora_pilot"])


def assess_lora_readiness(request: LoRAReadinessRequest) -> LoRAReadinessReport:
    """Return an auditable go/no-go decision; this function never trains a model."""
    assets = {asset.asset_id: asset for asset in request.dataset.assets}
    pairs_by_split = {split.value: 0 for split in DatasetSplit}
    properties_by_split: dict[str, set[str]] = {split.value: set() for split in DatasetSplit}
    reasons: list[str] = []

    for asset in request.dataset.assets:
        properties_by_split[asset.split.value].add(asset.listing_group_id)
        if not (
            asset.derivative_use_allowed
            and asset.training_use_allowed
            and asset.portfolio_display_allowed
        ):
            reasons.append("asset_rights_incomplete")

    for pair in request.dataset.pairs:
        source, target = assets[pair.source_asset_id], assets[pair.target_asset_id]
        pairs_by_split[source.split.value] += 1
        if not pair.same_camera_view:
            reasons.append("pair_not_same_camera_view")
        if not pair.geometry_verified:
            reasons.append("pair_geometry_unverified")
        if source.listing_group_id != target.listing_group_id:
            reasons.append("pair_crosses_property_group")

    split_sets = list(properties_by_split.values())
    has_property_leakage = any(
        left & right for index, left in enumerate(split_sets) for right in split_sets[index + 1 :]
    )
    if has_property_leakage:
        reasons.append("property_split_leakage")
    for split, count in pairs_by_split.items():
        if count == 0:
            reasons.append(f"missing_{split}_pairs")
    for field, value in request.evidence:
        if not value:
            reasons.append(f"missing_{field}")

    reasons = sorted(set(reasons))
    payload = request.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    report_id = f"lora-readiness-{digest[:16]}"
    return LoRAReadinessReport(
        report_id=report_id,
        created_at=datetime.now(UTC),
        decision="training_permitted" if not reasons else "not_justified",
        reason_codes=reasons,
        pair_counts=pairs_by_split,
        property_counts={split: len(groups) for split, groups in properties_by_split.items()},
        configuration=request.configuration,
        training_permitted=not reasons,
    )
