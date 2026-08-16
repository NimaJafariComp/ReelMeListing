import hashlib
from pathlib import Path

import pytest
from PIL import Image

from listing_to_reel.analysis.models import (
    InputQualityConfig,
    InputQualityDecision,
    InputQualityMetrics,
    InputQualityReport,
)
from listing_to_reel.core.config import RuntimeProfile
from listing_to_reel.editing.models import EditRequest, GeneratedCandidate, ImageEditorConfig
from listing_to_reel.editing.service import generate_edit_candidates


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_quality_report(source_path: Path, destination: Path) -> Path:
    report = InputQualityReport(
        report_id="input-test",
        created_at="2026-08-16T00:00:00Z",
        source_path=str(source_path),
        source_sha256=_sha256(source_path),
        metrics=InputQualityMetrics(
            width=320,
            height=180,
            exif_orientation=None,
            blur_laplacian_variance=500.0,
            highlight_clip_fraction=0.0,
            shadow_clip_fraction=0.0,
            color_cast_score=0.0,
            vertical_line_error_degrees=0.0,
            vertical_line_count=4,
        ),
        findings=[],
        reason_codes=[],
        decision=InputQualityDecision.ACCEPTED,
        configuration=InputQualityConfig(),
        diagnostic_overlay_path=None,
    )
    destination.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return destination


class FakeEditor:
    def generate(self, request: EditRequest, candidate_dir: Path):
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        for index in range(request.configuration.candidate_count):
            artifact_path = candidate_dir / f"candidate-{index + 1:02d}.jpg"
            Image.new("RGB", (64, 64), color=(index, 0, 0)).save(artifact_path)
            candidates.append(
                GeneratedCandidate(
                    candidate_index=index + 1,
                    seed=request.seed + index,
                    artifact_path=str(artifact_path),
                    sha256=_sha256(artifact_path),
                    wall_clock_seconds=0.01,
                    peak_device_memory_bytes=None,
                )
            )
        return "deadbeef", candidates


def _request(source_path: Path, quality_report: Path, output_dir: Path) -> EditRequest:
    return EditRequest(
        source_path=source_path,
        input_quality_report_path=quality_report,
        instruction="Convert this exterior to a natural premium golden-hour treatment.",
        seed=42,
        output_dir=output_dir / "runs",
        runtime_profile_name="local_mps",
        runtime_profile=RuntimeProfile(
            device="mps",
            image_resolution=512,
            batch_size=1,
            attention_slicing=True,
            image_editor_mode="preview_only",
            video_generation_enabled=False,
            benchmark_authority=False,
        ),
        configuration=ImageEditorConfig(candidate_count=2),
    )


def test_edit_generation_persists_candidates_and_provenance(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    Image.new("RGB", (320, 180), color=(20, 40, 60)).save(source_path)
    quality_report = _accepted_quality_report(source_path, tmp_path / "quality-report.json")
    request = _request(source_path, quality_report, tmp_path)

    manifest = generate_edit_candidates(request, FakeEditor())

    assert manifest.resolved_model_revision == "deadbeef"
    assert [candidate.seed for candidate in manifest.candidates] == [42, 43]
    assert all(Path(candidate.artifact_path).is_file() for candidate in manifest.candidates)
    assert Path(request.output_dir / manifest.run_id / "manifest.json").is_file()
    assert manifest.acceptance_decision == "pending_phase_4_evaluation"


def test_edit_generation_rejects_non_accepted_source_quality_report(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    Image.new("RGB", (320, 180), color=(20, 40, 60)).save(source_path)
    quality_report = _accepted_quality_report(source_path, tmp_path / "quality-report.json")
    report_payload = InputQualityReport.model_validate_json(
        quality_report.read_text(encoding="utf-8")
    )
    rejected_report = report_payload.model_copy(update={"decision": InputQualityDecision.WARNING})
    quality_report.write_text(rejected_report.model_dump_json(indent=2), encoding="utf-8")
    request = _request(source_path, quality_report, tmp_path)

    with pytest.raises(ValueError, match="requires an accepted"):
        generate_edit_candidates(request, FakeEditor())
