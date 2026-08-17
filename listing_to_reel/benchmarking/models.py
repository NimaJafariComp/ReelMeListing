"""Typed records for hardware-specific, quality-linked LTX benchmarks."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from listing_to_reel.core.config import RuntimeProfile
from listing_to_reel.core.environment import EnvironmentSnapshot
from listing_to_reel.video.models import VideoDecision


class BenchmarkStage(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    duration_seconds: float = Field(gt=0)


class CudaTelemetry(BaseModel):
    device_name: str
    total_memory_mib: int = Field(ge=0)
    allocated_memory_mib: int | None = Field(default=None, ge=0)
    reserved_memory_mib: int | None = Field(default=None, ge=0)
    cuda_runtime_version: str | None = None


class BenchmarkQualitySummary(BaseModel):
    decision: VideoDecision
    rejected_clip_count: int = Field(ge=0)
    rejected_bridge_count: int = Field(ge=0)
    minimum_clip_edge_f1: float | None = Field(default=None, ge=0, le=1)
    minimum_bridge_endpoint_edge_f1: float | None = Field(default=None, ge=0, le=1)


class LtxBenchmarkRecord(BaseModel):
    run_id: str
    phase: str = "phase_6_ltx_benchmark"
    created_at: datetime
    label: str
    cohort: str
    render_manifest_path: str
    render_manifest_sha256: str
    quality_report_path: str
    quality_report_sha256: str
    workload_fingerprint: str
    source_coverage: dict[str, str]
    configuration: dict[str, object]
    runtime_profile_name: str
    runtime_profile: RuntimeProfile
    environment: EnvironmentSnapshot
    cuda: CudaTelemetry
    stages: list[BenchmarkStage] = Field(min_length=1)
    total_measured_seconds: float = Field(gt=0)
    quality: BenchmarkQualitySummary


class BenchmarkComparison(BaseModel):
    run_id: str
    phase: str = "phase_6_ltx_benchmark_comparison"
    created_at: datetime
    baseline_path: str
    candidate_path: str
    workload_fingerprint: str
    generation_speedup: float = Field(gt=0)
    total_speedup: float = Field(gt=0)
    memory_delta_mib: int | None = None
    quality_not_worse: bool
    optimization_accepted: bool
    decision: str
    notes: list[str]
