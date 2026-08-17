from datetime import UTC, datetime
from pathlib import Path

import pytest

from listing_to_reel.benchmarking.models import BenchmarkStage, CudaTelemetry
from listing_to_reel.benchmarking.service import compare_ltx_benchmarks, record_ltx_benchmark
from listing_to_reel.core.config import ImageEditorMode, RuntimeProfile
from listing_to_reel.core.environment import DeviceCapabilities, EnvironmentSnapshot
from listing_to_reel.video.models import (
    LtxComfyUiConfig,
    LtxQualityReport,
    LtxRenderManifest,
    VideoDecision,
)


def _profile() -> RuntimeProfile:
    return RuntimeProfile(
        device="cuda",
        image_resolution=1024,
        batch_size=1,
        attention_slicing=False,
        image_editor_mode=ImageEditorMode.EVALUATION,
        video_generation_enabled=True,
        benchmark_authority=True,
    )


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.12.0",
        platform="Windows",
        machine="AMD64",
        git_commit_sha="test",
        torch_version="2.6.0+cu126",
        capabilities=DeviceCapabilities(mps_available=False, cuda_available=True),
    )


def _telemetry() -> CudaTelemetry:
    return CudaTelemetry(
        device_name="NVIDIA GeForce RTX 4070 SUPER",
        total_memory_mib=12282,
        allocated_memory_mib=8000,
        reserved_memory_mib=9000,
        cuda_runtime_version="12.6",
    )


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    config = LtxComfyUiConfig(
        comfyui_root=tmp_path,
        model_revision="test-revision",
        width=1024,
        height=576,
        frames=89,
        fps=30,
    )
    render = LtxRenderManifest(
        run_id="ltx-test",
        created_at=datetime.now(UTC),
        property_id="synthetic-home",
        configuration=config,
        clips=[],
        bridge_candidates=[],
        source_coverage={"source.png": "included_as_native_landscape_ltx_source_view"},
    )
    render_path = tmp_path / "render.json"
    render_path.write_text(render.model_dump_json(), encoding="utf-8")
    quality = LtxQualityReport(
        report_id="quality-test",
        created_at=datetime.now(UTC),
        render_manifest_path=str(render_path),
        clips=[],
        bridge_candidates=[],
        decision=VideoDecision.QUEUED_FOR_HUMAN_REVIEW,
        review_worksheet_path=str(tmp_path / "review.csv"),
    )
    quality_path = tmp_path / "quality.json"
    quality_path.write_text(quality.model_dump_json(), encoding="utf-8")
    return render_path, quality_path


def test_records_and_compares_same_workload_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    render_path, quality_path = _artifacts(tmp_path)
    monkeypatch.setattr(
        "listing_to_reel.benchmarking.service._torch_module", lambda: object()
    )
    monkeypatch.setattr(
        "listing_to_reel.benchmarking.service.assert_device_available", lambda *_: None
    )
    baseline = record_ltx_benchmark(
        label="baseline",
        cohort="rtx-4070-super-cuda-12.6",
        render_manifest_path=render_path,
        quality_report_path=quality_path,
        runtime_profile_name="remote_cuda",
        runtime_profile=_profile(),
        stages=[
            BenchmarkStage(name="generation", duration_seconds=100),
            BenchmarkStage(name="qa", duration_seconds=10),
        ],
        output_dir=tmp_path / "benchmarks",
        environment=_environment(),
        cuda=_telemetry(),
    )
    candidate = record_ltx_benchmark(
        label="optimized",
        cohort="rtx-4070-super-cuda-12.6",
        render_manifest_path=render_path,
        quality_report_path=quality_path,
        runtime_profile_name="remote_cuda",
        runtime_profile=_profile(),
        stages=[
            BenchmarkStage(name="generation", duration_seconds=80),
            BenchmarkStage(name="qa", duration_seconds=8),
        ],
        output_dir=tmp_path / "benchmarks",
        environment=_environment(),
        cuda=_telemetry(),
    )
    comparison = compare_ltx_benchmarks(
        tmp_path / "benchmarks" / baseline.run_id / "record.json",
        tmp_path / "benchmarks" / candidate.run_id / "record.json",
        tmp_path / "benchmarks",
    )
    assert comparison.generation_speedup == 1.25
    assert comparison.optimization_accepted is True


def test_benchmark_requires_generation_stage(tmp_path: Path) -> None:
    render_path, quality_path = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="generation stage"):
        record_ltx_benchmark(
            label="invalid",
            cohort="rtx-4070-super-cuda-12.6",
            render_manifest_path=render_path,
            quality_report_path=quality_path,
            runtime_profile_name="remote_cuda",
            runtime_profile=_profile(),
            stages=[BenchmarkStage(name="qa", duration_seconds=1)],
            output_dir=tmp_path / "benchmarks",
            environment=_environment(),
            cuda=_telemetry(),
        )
