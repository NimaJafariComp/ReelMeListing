"""Capture comparable CUDA LTX benchmark evidence without fabricating timings."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from listing_to_reel.benchmarking.models import (
    BenchmarkComparison,
    BenchmarkQualitySummary,
    BenchmarkStage,
    CudaTelemetry,
    LtxBenchmarkRecord,
)
from listing_to_reel.core.config import DeviceTarget, RuntimeProfile
from listing_to_reel.core.environment import (
    EnvironmentSnapshot,
    assert_device_available,
    capabilities_from_torch,
    collect_environment_snapshot,
)
from listing_to_reel.video.models import LtxQualityReport, LtxRenderManifest, VideoDecision


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _torch_module() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def collect_cuda_telemetry(torch_module: Any | None = None) -> CudaTelemetry:
    """Capture live CUDA allocation data at benchmark-record time."""
    torch_module = _torch_module() if torch_module is None else torch_module
    if torch_module is None or not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA telemetry requires an available CUDA-enabled PyTorch installation."
        )
    properties = torch_module.cuda.get_device_properties(0)
    total_memory = int(properties.total_memory / (1024 * 1024))
    return CudaTelemetry(
        device_name=str(torch_module.cuda.get_device_name(0)),
        total_memory_mib=total_memory,
        allocated_memory_mib=int(torch_module.cuda.memory_allocated(0) / (1024 * 1024)),
        reserved_memory_mib=int(torch_module.cuda.memory_reserved(0) / (1024 * 1024)),
        cuda_runtime_version=getattr(getattr(torch_module, "version", None), "cuda", None),
    )


def _quality_summary(report: LtxQualityReport) -> BenchmarkQualitySummary:
    clip_edges = [item.metrics.minimum_edge_f1_to_hero for item in report.clips]
    bridge_edges = [item.endpoint_edge_f1 for item in report.bridges]
    return BenchmarkQualitySummary(
        decision=report.decision,
        rejected_clip_count=sum(item.decision is VideoDecision.REJECTED for item in report.clips),
        rejected_bridge_count=sum(
            item.decision is VideoDecision.REJECTED for item in report.bridges
        ),
        minimum_clip_edge_f1=min(clip_edges) if clip_edges else None,
        minimum_bridge_endpoint_edge_f1=min(bridge_edges) if bridge_edges else None,
    )


def _workload_fingerprint(manifest: LtxRenderManifest) -> str:
    payload = {
        "model_revision": manifest.configuration.model_revision,
        "width": manifest.configuration.width,
        "height": manifest.configuration.height,
        "frames": manifest.configuration.frames,
        "fps": manifest.configuration.fps,
        "sources": sorted((clip.name, clip.source_sha256) for clip in manifest.clips),
        "bridges": sorted(
            (bridge.candidate_id, bridge.from_source_sha256, bridge.to_source_sha256)
            for bridge in manifest.bridges
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def record_ltx_benchmark(
    *,
    label: str,
    cohort: str,
    render_manifest_path: Path,
    quality_report_path: Path,
    runtime_profile_name: str,
    runtime_profile: RuntimeProfile,
    stages: list[BenchmarkStage],
    output_dir: Path,
    environment: EnvironmentSnapshot | None = None,
    cuda: CudaTelemetry | None = None,
) -> LtxBenchmarkRecord:
    """Persist one authoritative CUDA benchmark with caller-measured stage durations."""
    if runtime_profile.device is not DeviceTarget.CUDA or not runtime_profile.benchmark_authority:
        raise ValueError(
            "Phase 6 authoritative benchmarks require a CUDA benchmark-authority profile."
        )
    if len({stage.name for stage in stages}) != len(stages):
        raise ValueError("Benchmark stage names must be distinct.")
    if "generation" not in {stage.name for stage in stages}:
        raise ValueError("A benchmark must include a measured generation stage.")

    manifest = LtxRenderManifest.model_validate_json(
        render_manifest_path.read_text(encoding="utf-8")
    )
    quality = LtxQualityReport.model_validate_json(quality_report_path.read_text(encoding="utf-8"))
    if Path(quality.render_manifest_path).resolve() != render_manifest_path.resolve():
        raise ValueError("Quality report must belong to the supplied LTX render manifest.")
    snapshot = collect_environment_snapshot() if environment is None else environment
    torch_module = _torch_module()
    assert_device_available(DeviceTarget.CUDA, capabilities_from_torch(torch_module))
    telemetry = collect_cuda_telemetry(torch_module) if cuda is None else cuda
    fingerprint = _workload_fingerprint(manifest)
    payload = {
        "label": label,
        "cohort": cohort,
        "render_sha": _sha256(render_manifest_path),
        "quality_sha": _sha256(quality_report_path),
        "stages": [stage.model_dump() for stage in stages],
    }
    run_id = "benchmark-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    record = LtxBenchmarkRecord(
        run_id=run_id,
        created_at=datetime.now(UTC),
        label=label,
        cohort=cohort,
        render_manifest_path=str(render_manifest_path),
        render_manifest_sha256=_sha256(render_manifest_path),
        quality_report_path=str(quality_report_path),
        quality_report_sha256=_sha256(quality_report_path),
        workload_fingerprint=fingerprint,
        source_coverage=manifest.source_coverage,
        configuration={
            "model_revision": manifest.configuration.model_revision,
            "checkpoint": manifest.configuration.checkpoint,
            "width": manifest.configuration.width,
            "height": manifest.configuration.height,
            "frames": manifest.configuration.frames,
            "fps": manifest.configuration.fps,
            "steps": manifest.configuration.steps,
        },
        runtime_profile_name=runtime_profile_name,
        runtime_profile=runtime_profile,
        environment=snapshot,
        cuda=telemetry,
        stages=stages,
        total_measured_seconds=sum(stage.duration_seconds for stage in stages),
        quality=_quality_summary(quality),
    )
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def _stage(record: LtxBenchmarkRecord, name: str) -> float:
    return next(stage.duration_seconds for stage in record.stages if stage.name == name)


def _quality_not_worse(
    baseline: BenchmarkQualitySummary, candidate: BenchmarkQualitySummary
) -> bool:
    if candidate.rejected_clip_count > baseline.rejected_clip_count:
        return False
    if candidate.rejected_bridge_count > baseline.rejected_bridge_count:
        return False
    if (
        baseline.minimum_clip_edge_f1 is not None
        and candidate.minimum_clip_edge_f1 is not None
        and candidate.minimum_clip_edge_f1 < baseline.minimum_clip_edge_f1 - 0.02
    ):
        return False
    if (
        baseline.minimum_bridge_endpoint_edge_f1 is not None
        and candidate.minimum_bridge_endpoint_edge_f1 is not None
        and candidate.minimum_bridge_endpoint_edge_f1
        < baseline.minimum_bridge_endpoint_edge_f1 - 0.02
    ):
        return False
    return True


def compare_ltx_benchmarks(
    baseline_path: Path, candidate_path: Path, output_dir: Path
) -> BenchmarkComparison:
    """Compare an optimization against the same sources and delivery workload."""
    baseline = LtxBenchmarkRecord.model_validate_json(baseline_path.read_text(encoding="utf-8"))
    candidate = LtxBenchmarkRecord.model_validate_json(candidate_path.read_text(encoding="utf-8"))
    if baseline.workload_fingerprint != candidate.workload_fingerprint:
        raise ValueError(
            "Benchmarks must use the same model, delivery shape, sources, and bridges."
        )
    if baseline.cohort != candidate.cohort:
        raise ValueError("Benchmarks must be compared within the same hardware cohort.")
    quality_not_worse = _quality_not_worse(baseline.quality, candidate.quality)
    generation_speedup = _stage(baseline, "generation") / _stage(candidate, "generation")
    total_speedup = baseline.total_measured_seconds / candidate.total_measured_seconds
    accepted = (
        generation_speedup > 1.0
        and quality_not_worse
        and candidate.quality.decision is not VideoDecision.REJECTED
    )
    notes = [
        "Comparison holds model revision, delivery shape, source hashes, and bridge endpoints "
        "fixed.",
        "Optimization requires faster generation, no quality regression, and a non-rejected "
        "candidate.",
    ]
    if candidate.quality.decision is VideoDecision.REJECTED:
        notes.append("Candidate remains rejected by QA; do not promote it despite timing results.")
    payload = {"baseline": baseline.run_id, "candidate": candidate.run_id}
    comparison = BenchmarkComparison(
        run_id="benchmark-compare-"
        + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16],
        created_at=datetime.now(UTC),
        baseline_path=str(baseline_path),
        candidate_path=str(candidate_path),
        workload_fingerprint=baseline.workload_fingerprint,
        generation_speedup=generation_speedup,
        total_speedup=total_speedup,
        memory_delta_mib=(candidate.cuda.allocated_memory_mib or 0)
        - (baseline.cuda.allocated_memory_mib or 0),
        quality_not_worse=quality_not_worse,
        optimization_accepted=accepted,
        decision="accepted" if accepted else "not_accepted",
        notes=notes,
    )
    run_dir = output_dir / comparison.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "comparison.json").write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    return comparison
