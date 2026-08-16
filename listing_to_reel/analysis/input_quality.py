"""Phase 2 deterministic source-photo analysis and quality-gate reporting."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

from listing_to_reel.analysis.models import (
    FindingSeverity,
    InputQualityConfig,
    InputQualityDecision,
    InputQualityFile,
    InputQualityMetrics,
    InputQualityReport,
    QualityFinding,
)


def load_input_quality_config(path: Path) -> InputQualityConfig:
    """Load and validate a quality-gate YAML file."""
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    return InputQualityFile.model_validate(raw_config).input_quality


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_for_analysis(image: np.ndarray, maximum_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, maximum_dimension / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _vertical_lines(gray: np.ndarray) -> tuple[list[tuple[int, int, int, int]], float | None]:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minimum_length = max(20, round(min(gray.shape[:2]) * 0.12))
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=minimum_length,
        maxLineGap=12,
    )
    if raw_lines is None:
        return [], None

    candidates: list[tuple[tuple[int, int, int, int], float, float]] = []
    for x1, y1, x2, y2 in raw_lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        error = abs(90.0 - angle)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if error <= 30.0:
            candidates.append(((int(x1), int(y1), int(x2), int(y2)), error, length))

    if not candidates:
        return [], None
    candidates.sort(key=lambda item: item[1])
    weighted_errors = np.repeat(
        np.array([item[1] for item in candidates]),
        np.maximum(1, np.round(np.array([item[2] for item in candidates]) / 10).astype(int)),
    )
    return [item[0] for item in candidates], float(np.median(weighted_errors))


def _findings(metrics: InputQualityMetrics, config: InputQualityConfig) -> list[QualityFinding]:
    findings: list[QualityFinding] = []

    def add_threshold_finding(
        value: float,
        warning: float,
        rejection: float,
        code: str,
        label: str,
        lower_is_worse: bool = False,
    ) -> None:
        rejected = value <= rejection if lower_is_worse else value >= rejection
        warned = value <= warning if lower_is_worse else value >= warning
        if rejected:
            findings.append(
                QualityFinding(
                    code=f"{code}_rejected",
                    severity=FindingSeverity.REJECT,
                    message=f"{label} exceeds the rejection threshold.",
                    metric_value=value,
                )
            )
        elif warned:
            findings.append(
                QualityFinding(
                    code=f"{code}_warning",
                    severity=FindingSeverity.WARNING,
                    message=f"{label} exceeds the warning threshold.",
                    metric_value=value,
                )
            )

    add_threshold_finding(
        metrics.blur_laplacian_variance,
        config.blur_warning_variance,
        config.blur_reject_variance,
        "blur",
        "Blur estimate",
        lower_is_worse=True,
    )
    add_threshold_finding(
        metrics.highlight_clip_fraction,
        config.highlight_clip_warning_fraction,
        config.highlight_clip_reject_fraction,
        "highlight_clipping",
        "Highlight clipping",
    )
    add_threshold_finding(
        metrics.shadow_clip_fraction,
        config.shadow_clip_warning_fraction,
        config.shadow_clip_reject_fraction,
        "shadow_clipping",
        "Shadow clipping",
    )
    add_threshold_finding(
        metrics.color_cast_score,
        config.color_cast_warning_score,
        config.color_cast_reject_score,
        "color_cast",
        "Color-cast proxy",
    )

    if (
        metrics.vertical_line_error_degrees is None
        or metrics.vertical_line_count < config.min_vertical_lines
    ):
        findings.append(
            QualityFinding(
                code="vertical_lines_not_detected",
                severity=FindingSeverity.WARNING,
                message="Too few reliable vertical lines were found for a perspective check.",
                metric_value=metrics.vertical_line_count,
            )
        )
    else:
        if config.vertical_line_reject_enabled:
            add_threshold_finding(
                metrics.vertical_line_error_degrees,
                config.vertical_line_warning_degrees,
                config.vertical_line_reject_degrees,
                "vertical_line_error",
                "Vertical-line error",
            )
        elif metrics.vertical_line_error_degrees >= config.vertical_line_warning_degrees:
            findings.append(
                QualityFinding(
                    code="vertical_line_error_warning",
                    severity=FindingSeverity.WARNING,
                    message="Vertical-line proxy requires manual perspective review.",
                    metric_value=metrics.vertical_line_error_degrees,
                )
            )
    return findings


def _decision(findings: list[QualityFinding]) -> InputQualityDecision:
    if any(finding.severity is FindingSeverity.REJECT for finding in findings):
        return InputQualityDecision.REJECTED
    if findings:
        return InputQualityDecision.WARNING
    return InputQualityDecision.ACCEPTED


def analyze_input_image(
    source_path: Path,
    config: InputQualityConfig,
    output_dir: Path | None = None,
) -> InputQualityReport:
    """Evaluate an image, optionally persist a diagnostic overlay and JSON report."""
    source_hash = _sha256(source_path)
    with Image.open(source_path) as image_file:
        exif_orientation = image_file.getexif().get(274)
        original_width, original_height = image_file.size
        rgb = np.asarray(ImageOps.exif_transpose(image_file).convert("RGB"))

    analyzed_rgb = _resize_for_analysis(rgb, config.analysis_max_dimension)
    gray = cv2.cvtColor(analyzed_rgb, cv2.COLOR_RGB2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    highlight_fraction = float(np.mean(gray >= 250))
    shadow_fraction = float(np.mean(gray <= 5))
    channel_means = analyzed_rgb.astype(np.float64).mean(axis=(0, 1))
    color_cast_score = float(np.std(channel_means) / max(float(channel_means.mean()), 1.0))
    vertical_lines, vertical_error = _vertical_lines(gray)
    metrics = InputQualityMetrics(
        width=original_width,
        height=original_height,
        exif_orientation=exif_orientation,
        blur_laplacian_variance=blur_variance,
        highlight_clip_fraction=highlight_fraction,
        shadow_clip_fraction=shadow_fraction,
        color_cast_score=color_cast_score,
        vertical_line_error_degrees=vertical_error,
        vertical_line_count=len(vertical_lines),
    )
    findings = _findings(metrics, config)
    report_id = f"input-{source_hash[:16]}"
    overlay_path: Path | None = None

    if output_dir is not None:
        report_dir = output_dir / report_id
        report_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = report_dir / "vertical-lines.jpg"
        overlay = cv2.cvtColor(analyzed_rgb, cv2.COLOR_RGB2BGR)
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

    report = InputQualityReport(
        report_id=report_id,
        created_at=datetime.now(UTC),
        source_path=str(source_path),
        source_sha256=source_hash,
        metrics=metrics,
        findings=findings,
        reason_codes=[finding.code for finding in findings],
        decision=_decision(findings),
        configuration=config,
        diagnostic_overlay_path=str(overlay_path) if overlay_path else None,
    )
    if output_dir is not None:
        report_path = output_dir / report_id / "report.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
