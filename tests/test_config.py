from pathlib import Path

import pytest
from pydantic import ValidationError

from listing_to_reel.core.config import RuntimeConfig, load_runtime_config


def test_local_mps_profile_is_valid() -> None:
    config = load_runtime_config(Path("configs/local_mps.yaml"))

    profile = config.runtime_profiles["local_mps"]
    assert profile.device == "mps"
    assert profile.image_editor_mode == "preview_only"
    assert profile.video_generation_enabled is False
    assert profile.benchmark_authority is False


def test_mps_profile_rejects_video_generation() -> None:
    with pytest.raises(ValidationError, match="cannot enable video generation"):
        RuntimeConfig.model_validate(
            {
                "runtime_profiles": {
                    "unsafe_mps": {
                        "device": "mps",
                        "image_resolution": 512,
                        "batch_size": 1,
                        "attention_slicing": True,
                        "image_editor_mode": "preview_only",
                        "video_generation_enabled": True,
                        "benchmark_authority": False,
                    }
                }
            }
        )


def test_empty_configuration_is_rejected(tmp_path: Path) -> None:
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="configuration is empty"):
        load_runtime_config(empty_config)
