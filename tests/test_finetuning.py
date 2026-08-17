from listing_to_reel.finetuning.models import LoRAReadinessRequest
from listing_to_reel.finetuning.service import assess_lora_readiness


def _asset(asset_id: str, property_id: str, split: str, role: str) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "listing_group_id": property_id,
        "split": split,
        "role": role,
        "source_uri": f"https://example.test/{asset_id}.jpg",
        "rights_basis": "written_permission",
        "rights_evidence_ref": f"release-{property_id}",
        "derivative_use_allowed": True,
        "training_use_allowed": True,
        "portfolio_display_allowed": True,
    }


def _request() -> dict[str, object]:
    assets = []
    pairs = []
    for split, property_id in [
        ("train", "home-a"),
        ("validation", "home-b"),
        ("test", "home-c"),
    ]:
        source, target = f"{split}-day", f"{split}-dusk"
        assets.extend(
            [
                _asset(source, property_id, split, "daylight_source"),
                _asset(target, property_id, split, "dusk_target"),
            ]
        )
        pairs.append(
            {
                "pair_id": split,
                "source_asset_id": source,
                "target_asset_id": target,
                "same_camera_view": True,
                "geometry_verified": True,
            }
        )
    return {
        "dataset": {"assets": assets, "pairs": pairs},
        "evidence": {
            "geometry_gate_report": "runs/evaluations/geometry.json",
            "structure_conditioning_comparison": "runs/evaluations/controlnet.json",
            "reproducible_video_qa_report": "runs/ltx-quality/report.json",
            "baseline_comparison_report": "runs/evaluations/baseline.json",
        },
    }


def test_readiness_permits_only_complete_property_grouped_evidence() -> None:
    report = assess_lora_readiness(LoRAReadinessRequest.model_validate(_request()))
    assert report.training_permitted is True
    assert report.pair_counts == {"train": 1, "validation": 1, "test": 1}


def test_readiness_rejects_missing_evidence_and_property_split_leakage() -> None:
    payload = _request()
    payload["dataset"]["assets"][2]["listing_group_id"] = "home-a"
    payload["dataset"]["assets"][3]["listing_group_id"] = "home-a"
    payload["evidence"]["geometry_gate_report"] = None
    report = assess_lora_readiness(LoRAReadinessRequest.model_validate(payload))
    assert report.training_permitted is False
    assert "property_split_leakage" in report.reason_codes
    assert "missing_geometry_gate_report" in report.reason_codes
