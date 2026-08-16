from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from listing_to_reel.analysis.input_quality import (
    _findings,
    analyze_input_image,
    load_input_quality_config,
)
from listing_to_reel.analysis.models import (
    FindingSeverity,
    InputQualityConfig,
    InputQualityDecision,
    InputQualityMetrics,
)


def _write_image(path: Path, image: np.ndarray) -> Path:
    Image.fromarray(image.astype(np.uint8)).save(path)
    return path


def test_input_quality_rejects_blurry_and_clipped_image(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "bad.jpg", np.full((160, 160, 3), 255))

    report = analyze_input_image(image_path, InputQualityConfig())

    assert report.decision is InputQualityDecision.REJECTED
    assert "blur_rejected" in report.reason_codes
    assert "highlight_clipping_rejected" in report.reason_codes


def test_input_quality_detects_straight_vertical_lines_and_persists_diagnostics(
    tmp_path: Path,
) -> None:
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    cv2.line(image, (70, 20), (70, 220), (255, 255, 255), 6)
    cv2.line(image, (180, 20), (180, 220), (255, 255, 255), 6)
    image_path = _write_image(tmp_path / "verticals.jpg", image)

    report = analyze_input_image(image_path, InputQualityConfig(), tmp_path / "reports")

    assert report.metrics.vertical_line_count >= 2
    assert report.metrics.vertical_line_error_degrees == pytest.approx(0.0, abs=0.5)
    assert Path(report.diagnostic_overlay_path or "").is_file()
    assert (tmp_path / "reports" / report.report_id / "report.json").is_file()


def test_quality_configuration_rejects_inverted_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "input_quality:\n  blur_warning_variance: 20\n  blur_reject_variance: 50\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Blur rejection threshold"):
        load_input_quality_config(config_path)


def test_vertical_line_proxy_is_warning_only_by_default() -> None:
    metrics = InputQualityMetrics(
        width=1000,
        height=750,
        exif_orientation=None,
        blur_laplacian_variance=500.0,
        highlight_clip_fraction=0.0,
        shadow_clip_fraction=0.0,
        color_cast_score=0.0,
        vertical_line_error_degrees=15.0,
        vertical_line_count=4,
    )

    findings = _findings(metrics, InputQualityConfig())

    assert any(finding.code == "vertical_line_error_warning" for finding in findings)
    assert not any(finding.severity is FindingSeverity.REJECT for finding in findings)
