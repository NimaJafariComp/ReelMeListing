"""Stable Video Diffusion orchestration, vertical encoding, and temporal QA."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

from listing_to_reel.core.config import DeviceTarget
from listing_to_reel.core.environment import (
    assert_device_available,
    capabilities_from_torch,
    collect_environment_snapshot,
)
from listing_to_reel.evaluation.models import EvaluationReport, FinalDecisionRecord
from listing_to_reel.media.ffmpeg import probe_video, require_ffmpeg, run_ffmpeg
from listing_to_reel.video.models import (
    HeroVideoManifest,
    HeroVideoRequest,
    TemporalMetrics,
    VideoDecision,
    VideoGeneratorConfig,
    VideoGeneratorFile,
    VideoQualityReport,
)


class VideoGenerator(Protocol):
    def generate(
        self, request: HeroVideoRequest, hero_image_path: Path, frames_dir: Path
    ) -> tuple[str, list[Path]]: ...


def load_video_generator_config(path: Path) -> VideoGeneratorConfig:
    with path.open("r", encoding="utf-8") as config_file:
        return VideoGeneratorFile.model_validate(yaml.safe_load(config_file)).video_generator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(request: HeroVideoRequest, hero_hash: str) -> str:
    payload = json.dumps(
        {
            "hero_hash": hero_hash,
            "seed": request.seed,
            "config": request.configuration.model_dump(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"video-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _approved_hero(
    final_decision_path: Path,
) -> tuple[FinalDecisionRecord, EvaluationReport, Path, str]:
    decision = FinalDecisionRecord.model_validate_json(
        final_decision_path.read_text(encoding="utf-8")
    )
    if decision.decision.value != "accepted" or decision.selected_candidate_index is None:
        raise ValueError("Phase 5 requires a human-approved Phase 4 hero-image decision.")
    report = EvaluationReport.model_validate_json(
        Path(decision.evaluation_report_path).read_text(encoding="utf-8")
    )
    candidate = next(
        item
        for item in report.candidates
        if item.candidate_index == decision.selected_candidate_index
    )
    hero = Path(candidate.candidate_path)
    hero_hash = _sha256(hero)
    if hero_hash != candidate.candidate_sha256:
        raise ValueError("Approved hero image hash no longer matches the Phase 4 report.")
    return decision, report, hero, hero_hash


class StableVideoDiffusionGenerator:
    """CUDA-only SVD-XT adapter tuned for an RTX 4070 SUPER-class card."""

    def generate(
        self, request: HeroVideoRequest, hero_image_path: Path, frames_dir: Path
    ) -> tuple[str, list[Path]]:
        try:
            import torch
            from diffusers import StableVideoDiffusionPipeline
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError(
                "Install the GPU extra before running Stable Video Diffusion."
            ) from error
        if request.runtime_profile.device is not DeviceTarget.CUDA:
            raise ValueError(
                "Stable Video Diffusion generation is CUDA-only; MPS supports QA only."
            )
        assert_device_available(DeviceTarget.CUDA, capabilities_from_torch(torch))
        revision = (
            HfApi()
            .model_info(
                request.configuration.model_id,
                revision=request.configuration.model_revision or "main",
            )
            .sha
        )
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            request.configuration.model_id,
            revision=revision,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.enable_model_cpu_offload()
        pipe.unet.enable_forward_chunking()
        with Image.open(hero_image_path) as image_file:
            image = (
                ImageOps.exif_transpose(image_file)
                .convert("RGB")
                .resize(
                    (request.configuration.width, request.configuration.height),
                    Image.Resampling.LANCZOS,
                )
            )
        frames = pipe(
            image,
            generator=torch.manual_seed(request.seed),
            num_frames=request.configuration.num_frames,
            fps=request.configuration.model_fps,
            motion_bucket_id=request.configuration.motion_bucket_id,
            noise_aug_strength=request.configuration.noise_aug_strength,
            decode_chunk_size=request.configuration.decode_chunk_size,
        ).frames[0]
        frames_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, frame in enumerate(frames):
            path = frames_dir / f"frame-{index:03d}.png"
            frame.save(path)
            paths.append(path)
        return revision, paths


def _encode_vertical_video(
    frame_dir: Path, output_path: Path, config: VideoGeneratorConfig
) -> None:
    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{config.num_frames}/4",
        "-i",
        str(frame_dir / "frame-%03d.png"),
        "-filter_complex",
        f"[0:v]scale={config.output_width}:{config.output_height}:force_original_aspect_ratio=increase,boxblur=20:10,crop={config.output_width}:{config.output_height}[bg];"
        f"[0:v]scale={config.output_width}:{config.output_height}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2",
        "-r",
        f"{config.num_frames}/4",
        "-frames:v",
        str(config.num_frames),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(config.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_ffmpeg(command)


def generate_hero_video(request: HeroVideoRequest, generator: VideoGenerator) -> HeroVideoManifest:
    decision, report, hero, hero_hash = _approved_hero(request.final_decision_path)
    run_id = _run_id(request, hero_hash)
    run_dir = request.output_dir / run_id
    revision, frames = generator.generate(request, hero, run_dir / "frames")
    if len(frames) != request.configuration.num_frames:
        raise ValueError("Video generator returned an unexpected frame count.")
    output = run_dir / "hero.mp4"
    _encode_vertical_video(run_dir / "frames", output, request.configuration)
    metadata = probe_video(output)
    if abs(metadata.duration_seconds - 4.0) > 0.05:
        raise ValueError("Encoded hero video is not four seconds long.")
    manifest = HeroVideoManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        final_decision_path=str(request.final_decision_path),
        evaluation_report_path=decision.evaluation_report_path,
        edit_run_id=report.edit_run_id,
        selected_candidate_index=decision.selected_candidate_index,
        hero_image_path=str(hero),
        hero_image_sha256=hero_hash,
        treatment=request.treatment,
        seed=request.seed,
        configuration=request.configuration,
        resolved_model_revision=revision,
        runtime_profile_name=request.runtime_profile_name,
        runtime_profile=request.runtime_profile,
        environment=collect_environment_snapshot(),
        frames_directory=str(run_dir / "frames"),
        frame_sha256=[_sha256(frame) for frame in frames],
        output_path=str(output),
        output_sha256=_sha256(output),
        video=metadata,
        source_coverage={
            str(hero): "included_as_hero_image; no unseen property views are generated"
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _edge_f1(reference: np.ndarray, frame: np.ndarray) -> float:
    reference = cv2.resize(
        reference, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA
    )
    first, second = cv2.Canny(reference, 50, 150), cv2.Canny(frame, 50, 150)
    first_d, second_d = (
        cv2.dilate(first, np.ones((3, 3), np.uint8)),
        cv2.dilate(second, np.ones((3, 3), np.uint8)),
    )
    if not np.any(first) or not np.any(second):
        return 0.0
    precision, recall = np.mean(first_d[second > 0] > 0), np.mean(second_d[first > 0] > 0)
    return float(2 * precision * recall / max(precision + recall, 1e-9))


def evaluate_hero_video(video_manifest_path: Path, output_dir: Path) -> VideoQualityReport:
    manifest = HeroVideoManifest.model_validate_json(
        video_manifest_path.read_text(encoding="utf-8")
    )
    capture = cv2.VideoCapture(manifest.output_path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    if len(frames) != manifest.configuration.num_frames:
        raise ValueError("Decoded frame count differs from the generation manifest.")
    with Image.open(manifest.hero_image_path) as hero_file:
        hero = cv2.cvtColor(np.asarray(hero_file.convert("RGB")), cv2.COLOR_RGB2GRAY)
    differences = [
        float(np.mean(cv2.absdiff(first, second))) for first, second in zip(frames, frames[1:])
    ]
    edge_scores = [_edge_f1(hero, frame) for frame in frames]
    metrics = TemporalMetrics(
        frame_count=len(frames),
        duration_seconds=manifest.video.duration_seconds,
        mean_frame_difference=float(np.mean(differences)),
        frame_difference_cv=float(np.std(differences) / max(np.mean(differences), 1e-6)),
        minimum_edge_f1_to_hero=min(edge_scores),
        maximum_black_pixel_fraction=max(float(np.mean(frame <= 5)) for frame in frames),
    )
    reasons = []
    if metrics.minimum_edge_f1_to_hero < 0.45:
        reasons.append("geometry_drift_rejected")
    if metrics.maximum_black_pixel_fraction > 0.25:
        reasons.append("black_frame_rejected")
    if metrics.frame_difference_cv > 1.2:
        reasons.append("temporal_flicker_review")
    decision = (
        VideoDecision.REJECTED
        if any(code.endswith("rejected") for code in reasons)
        else VideoDecision.QUEUED_FOR_HUMAN_REVIEW
    )
    report_id = f"video-quality-{manifest.run_id.removeprefix('video-')}"
    report_dir = output_dir / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    worksheet = report_dir / "review.csv"
    with worksheet.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["video_path", "decision", "reviewer", "notes"])
        writer.writeheader()
        if decision is VideoDecision.QUEUED_FOR_HUMAN_REVIEW:
            writer.writerow(
                {"video_path": manifest.output_path, "decision": "", "reviewer": "", "notes": ""}
            )
    report = VideoQualityReport(
        report_id=report_id,
        created_at=datetime.now(UTC),
        video_manifest_path=str(video_manifest_path),
        video_run_id=manifest.run_id,
        metrics=metrics,
        reason_codes=reasons,
        decision=decision,
        review_worksheet_path=str(worksheet),
    )
    (report_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
