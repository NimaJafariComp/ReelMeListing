import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from listing_to_reel.core.config import RuntimeProfile
from listing_to_reel.core.environment import DeviceCapabilities, EnvironmentSnapshot
from listing_to_reel.editing.models import EditRunManifest, GeneratedCandidate, ImageEditorConfig
from listing_to_reel.evaluation.models import CandidateDecision, RunDecision
from listing_to_reel.evaluation.service import (
    evaluate_edit_run,
    import_human_review,
    load_evaluation_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _architectural_image(path: Path, black: bool = False) -> None:
    image = Image.new("RGB", (320, 240), color=(0, 0, 0) if black else (180, 200, 220))
    if not black:
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((80, 40, 240, 210), fill=(230, 230, 220), outline=(20, 20, 20), width=5)
        for x in (110, 160, 210):
            drawing.rectangle(
                (x, 75, x + 20, 150),
                fill=(70, 130, 160),
                outline=(20, 20, 20),
                width=3,
            )
    image.save(path)


def _manifest(tmp_path: Path, second_black: bool = False) -> Path:
    source = tmp_path / "source.jpg"
    candidate_one = tmp_path / "candidate-01.jpg"
    candidate_two = tmp_path / "candidate-02.jpg"
    _architectural_image(source)
    _architectural_image(candidate_one)
    _architectural_image(candidate_two, black=second_black)
    runtime = RuntimeProfile(
        device="mps",
        image_resolution=512,
        batch_size=1,
        attention_slicing=True,
        image_editor_mode="preview_only",
        video_generation_enabled=False,
        benchmark_authority=False,
    )
    manifest = EditRunManifest(
        run_id="edit-test",
        created_at=datetime.now(UTC),
        source_path=str(source),
        source_sha256=_sha256(source),
        input_quality_report_path="input-report.json",
        instruction="Natural premium golden-hour exterior treatment.",
        configuration=ImageEditorConfig(candidate_count=2),
        requested_model_revision=None,
        resolved_model_revision="deadbeef",
        runtime_profile_name="local_mps",
        runtime_profile=runtime,
        environment=EnvironmentSnapshot(
            python_version="3.12",
            platform="test",
            machine="arm64",
            git_commit_sha=None,
            torch_version=None,
            capabilities=DeviceCapabilities(mps_available=True, cuda_available=False),
        ),
        candidates=[
            GeneratedCandidate(
                candidate_index=1,
                seed=1,
                artifact_path=str(candidate_one),
                sha256=_sha256(candidate_one),
                wall_clock_seconds=1.0,
            ),
            GeneratedCandidate(
                candidate_index=2,
                seed=2,
                artifact_path=str(candidate_two),
                sha256=_sha256(candidate_two),
                wall_clock_seconds=1.0,
            ),
        ],
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def test_evaluation_queues_viable_candidate_and_rejects_black_artifact(tmp_path: Path) -> None:
    report = evaluate_edit_run(
        _manifest(tmp_path, second_black=True),
        load_evaluation_config(Path("configs/evaluation.yaml")),
        tmp_path / "evaluations",
    )

    assert report.run_decision is RunDecision.QUEUED_FOR_HUMAN_REVIEW
    assert report.recommended_candidate_index == 1
    assert report.candidates[0].decision is CandidateDecision.QUEUED_FOR_HUMAN_REVIEW
    assert report.candidates[1].decision is CandidateDecision.REJECTED
    assert "artifact_black_frame_rejected" in report.candidates[1].reason_codes
    assert Path(report.review_worksheet_path or "").is_file()


def test_human_review_import_records_selected_candidate(tmp_path: Path) -> None:
    report = evaluate_edit_run(
        _manifest(tmp_path),
        load_evaluation_config(Path("configs/evaluation.yaml")),
        tmp_path / "evaluations",
    )
    worksheet = Path(report.review_worksheet_path or "")
    rows = list(csv.DictReader(worksheet.open(encoding="utf-8")))
    rows[0].update(
        {"decision": "accepted_by_human", "reviewer": "reviewer@example.com", "notes": "Pass"}
    )
    with worksheet.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    decision = import_human_review(
        tmp_path / "evaluations" / report.report_id / "report.json",
        worksheet,
        tmp_path / "decision",
    )

    assert decision.decision is RunDecision.ACCEPTED
    assert decision.selected_candidate_index == 1
    assert json.loads((tmp_path / "decision" / "final-decision.json").read_text())["reviewer"] == (
        "reviewer@example.com"
    )
