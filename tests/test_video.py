import hashlib
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from listing_to_reel.core.config import RuntimeProfile
from listing_to_reel.evaluation.models import (
    CandidateDecision,
    CandidateEvaluation,
    CandidateMetrics,
    EvaluationConfig,
    EvaluationReport,
    FinalDecisionRecord,
    RunDecision,
)
from listing_to_reel.video.models import (
    HeroVideoRequest,
    InterpolationConfig,
    MultiShotInput,
    MultiShotVideoRequest,
    PropertyShotRole,
    VideoGeneratorConfig,
)
from listing_to_reel.video.service import (
    evaluate_hero_video,
    generate_hero_video,
    interpolate_hero_video,
    plan_multishot_ltx_video,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeVideoGenerator:
    def generate(self, request: HeroVideoRequest, hero_image_path: Path, frames_dir: Path):
        frames_dir.mkdir(parents=True)
        source = Image.open(hero_image_path).convert("RGB")
        frames = []
        for index in range(request.configuration.num_frames):
            frame = source.copy()
            ImageDraw.Draw(frame).rectangle((index % 8, 0, 10 + index % 8, 10), fill="white")
            path = frames_dir / f"frame-{index:03d}.png"
            frame.save(path)
            frames.append(path)
        return "deadbeef", frames


def _approved_decision(tmp_path: Path, color: str = "#d8b084") -> Path:
    hero = tmp_path / "hero.jpg"
    image = Image.new("RGB", (320, 180), color)
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((80, 30, 240, 170), outline="black", width=5)
    image.save(hero)
    evaluation = EvaluationReport(
        report_id="quality-test",
        created_at=datetime.now(UTC),
        edit_run_manifest_path="edit.json",
        edit_run_id="edit-test",
        source_path="source.jpg",
        source_sha256="source",
        configuration=EvaluationConfig(),
        candidates=[
            CandidateEvaluation(
                candidate_index=1,
                candidate_path=str(hero),
                candidate_sha256=_sha256(hero),
                metrics=CandidateMetrics(
                    source_edge_pixels=1,
                    candidate_edge_pixels=1,
                    source_blur_variance=1,
                    candidate_blur_variance=1,
                    black_pixel_fraction=0,
                    mean_luminance_delta=0,
                ),
                reason_codes=[],
                score=1,
                decision=CandidateDecision.QUEUED_FOR_HUMAN_REVIEW,
            )
        ],
        recommended_candidate_index=1,
        run_decision=RunDecision.QUEUED_FOR_HUMAN_REVIEW,
        reason_codes=[],
    )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(evaluation.model_dump_json(), encoding="utf-8")
    decision = FinalDecisionRecord(
        created_at=datetime.now(UTC),
        evaluation_report_path=str(evaluation_path),
        edit_run_id="edit-test",
        decision=RunDecision.ACCEPTED,
        selected_candidate_index=1,
        reviewer="owner",
        notes="approved",
        reason_codes=[],
    )
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(decision.model_dump_json(), encoding="utf-8")
    return decision_path


def _approved_decisions(tmp_path: Path, count: int) -> list[Path]:
    decisions = []
    for index in range(count):
        directory = tmp_path / f"view-{index}"
        directory.mkdir()
        decisions.append(_approved_decision(directory, f"#{index + 1:02x}b084"))
    return decisions


def test_mocked_video_generation_and_temporal_qa(tmp_path: Path) -> None:
    request = HeroVideoRequest(
        final_decision_path=_approved_decision(tmp_path),
        seed=3,
        output_dir=tmp_path / "runs",
        runtime_profile_name="remote_cuda",
        runtime_profile=RuntimeProfile(
            device="cuda",
            image_resolution=1024,
            batch_size=1,
            attention_slicing=False,
            image_editor_mode="evaluation",
            video_generation_enabled=True,
            benchmark_authority=True,
        ),
        configuration=VideoGeneratorConfig(),
    )
    manifest = generate_hero_video(request, FakeVideoGenerator())
    report = evaluate_hero_video(
        request.output_dir / manifest.run_id / "manifest.json", tmp_path / "quality"
    )

    assert manifest.video.duration_seconds == 4.0
    assert manifest.video.width == 1080
    assert manifest.video.height == 1920
    assert len(manifest.frame_sha256) == 25
    assert report.metrics.frame_count == 25
    assert Path(report.review_worksheet_path or "").is_file()


def test_interpolation_preserves_duration_and_outputs_30fps(tmp_path: Path) -> None:
    request = HeroVideoRequest(
        final_decision_path=_approved_decision(tmp_path),
        seed=3,
        output_dir=tmp_path / "runs",
        runtime_profile_name="remote_cuda",
        runtime_profile=RuntimeProfile(
            device="cuda",
            image_resolution=1024,
            batch_size=1,
            attention_slicing=False,
            image_editor_mode="evaluation",
            video_generation_enabled=True,
            benchmark_authority=True,
        ),
        configuration=VideoGeneratorConfig(),
    )
    source = generate_hero_video(request, FakeVideoGenerator())
    manifest = interpolate_hero_video(
        request.output_dir / source.run_id / "manifest.json",
        InterpolationConfig(),
        tmp_path / "interpolated",
    )

    assert manifest.video.duration_seconds == 4.0
    assert manifest.video.frame_rate == "30/1"
    assert (
        evaluate_hero_video(
            tmp_path / "interpolated" / manifest.run_id / "manifest.json", tmp_path / "qa"
        ).metrics.frame_count
        == 120
    )


def test_multishot_ltx_plan_requires_approved_distinct_views(tmp_path: Path) -> None:
    decisions = _approved_decisions(tmp_path, 4)
    plan = plan_multishot_ltx_video(
        MultiShotVideoRequest(
            property_id="demo-property-001",
            output_dir=tmp_path / "plans",
            shots=[
                MultiShotInput(
                    role=PropertyShotRole.WIDE_EXTERIOR, final_decision_path=decisions[0]
                ),
                MultiShotInput(role=PropertyShotRole.BACKYARD, final_decision_path=decisions[1]),
                MultiShotInput(
                    role=PropertyShotRole.ARCHITECTURAL_DETAIL,
                    final_decision_path=decisions[2],
                ),
                MultiShotInput(
                    role=PropertyShotRole.CLOSING_HERO, final_decision_path=decisions[3]
                ),
            ],
        )
    )

    assert plan.adapter == "ltx_video_2b_distilled"
    assert plan.total_duration_seconds == 8.0
    assert len(plan.source_coverage) == 4
    assert (tmp_path / "plans" / plan.run_id / "manifest.json").is_file()
