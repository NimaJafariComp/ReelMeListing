"""Native-landscape LTX image-to-video rendering through ComfyUI.

This module deliberately treats every generated clip and bridge as a review
candidate.  It never auto-accepts property geometry or transitions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from listing_to_reel.media.ffmpeg import probe_video, require_ffmpeg, run_ffmpeg
from listing_to_reel.video.models import (
    LtxBridgeCandidate,
    LtxBridgeKind,
    LtxBridgeQualityReport,
    LtxClipQualityReport,
    LtxComfyUiConfig,
    LtxGeneratedBridge,
    LtxGeneratedClip,
    LtxMotionTreatment,
    LtxQualityReport,
    LtxReelManifest,
    LtxRenderManifest,
    LtxRenderRequest,
    LtxTimedReelItem,
    LtxTimedReelManifest,
    LtxTimedReelPlan,
    TemporalMetrics,
    VideoDecision,
)


class ComfyUiClient(Protocol):
    def submit_and_wait(self, workflow: dict[str, object]) -> Path: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HttpComfyUiClient:
    """Small stdlib-only ComfyUI API client, kept behind a testable protocol."""

    def __init__(self, endpoint: str, output_dir: Path, poll_seconds: float = 1.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.output_dir = output_dir
        self.poll_seconds = poll_seconds

    def _json(self, url: str, body: dict[str, object] | None = None) -> dict[str, object]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"} if payload else {},
            method="POST" if payload else "GET",
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    def submit_and_wait(self, workflow: dict[str, object]) -> Path:
        queued = self._json(f"{self.endpoint}/prompt", {"prompt": workflow})
        prompt_id = str(queued["prompt_id"])
        for _ in range(600):
            history = self._json(f"{self.endpoint}/history/{prompt_id}")
            completed = history.get(prompt_id)
            if completed:
                for output in completed["outputs"].values():
                    images = output.get("images", [])
                    if images:
                        image = images[0]
                        return self.output_dir / image.get("subfolder", "") / image["filename"]
                raise RuntimeError(f"ComfyUI job {prompt_id} completed without a saved video.")
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"ComfyUI job {prompt_id} did not complete within ten minutes.")


def load_ltx_comfyui_config(path: Path) -> LtxComfyUiConfig:
    with path.open("r", encoding="utf-8") as config_file:
        return LtxComfyUiConfig.model_validate(yaml.safe_load(config_file))


def _prompt(treatment: LtxMotionTreatment) -> str:
    movement = {
        LtxMotionTreatment.LATERAL_GIMBAL: "slow stabilized lateral gimbal glide",
        LtxMotionTreatment.DOLLY_IN: "gentle 5–10% stabilized dolly-in",
    }[treatment]
    return (
        "Continuous native 16:9 landscape real-estate shot of the provided source view. "
        f"Use a {movement} with subtle natural parallax only. Preserve every visible roof, "
        "window, garage, patio column, lawn edge, block wall, tree, driveway, material, "
        "and proportion. Keep the complete visible property and sky in frame. No handheld "
        "shake, fast zoom, dramatic orbit, crop, invented surroundings, people, vehicles, "
        "structural additions, removals, morphing, warping, flicker, text, or watermark."
    )


def build_ltx_workflow(
    image_name: str, config: LtxComfyUiConfig, treatment: LtxMotionTreatment, seed: int, prefix: str
) -> dict[str, object]:
    """Build the verified native ComfyUI LTX node graph without external custom nodes."""
    prompt = _prompt(treatment)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": config.checkpoint, "weight_dtype": "fp8_e4m3fn"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": config.text_encoder, "type": "ltxv", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": config.vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": (
                    "crop, vertical framing, camera shake, fast movement, morphing, warped "
                    "architecture, distorted windows, changed roof, changed garage, changed "
                    "patio, changed lawn, invented surroundings, flicker, text, watermark"
                ),
                "clip": ["2", 0],
            },
        },
        "7": {
            "class_type": "LTXVImgToVideo",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["3", 0],
                "image": ["4", 0],
                "width": config.width,
                "height": config.height,
                "length": config.frames,
                "batch_size": 1,
                "strength": 1.0,
            },
        },
        "8": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["7", 0], "negative": ["7", 1], "frame_rate": float(config.fps)},
        },
        "9": {
            "class_type": "ModelSamplingLTXV",
            "inputs": {
                "model": ["1", 0],
                "max_shift": 2.05,
                "base_shift": 0.95,
                "latent": ["7", 2],
            },
        },
        "10": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["9", 0], "conditioning": ["8", 0]},
        },
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": config.steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["7", 2],
            },
        },
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["11", 0],
                "guider": ["10", 0],
                "sampler": ["12", 0],
                "sigmas": ["13", 0],
                "latent_image": ["7", 2],
            },
        },
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {
            "class_type": "SaveWEBM",
            "inputs": {
                "images": ["15", 0],
                "filename_prefix": prefix,
                "codec": "vp9",
                "fps": float(config.fps),
                "crf": 20.0,
            },
        },
    }


def _bridge_prompt(candidate: LtxBridgeCandidate) -> str:
    duration = f"{candidate.duration_seconds:g}-second"
    if candidate.kind is LtxBridgeKind.LIGHTING_ONLY:
        return (
            f"{duration.capitalize()} continuous native 16:9 architectural time-lapse "
            "between the two "
            "provided views of the same front elevation. Lock the camera position, framing, "
            "house, driveway, windows, roof, landscaping, and all geometry exactly in place. "
            "Change only sky color, ambient daylight, and practical or interior lighting. No "
            "camera motion, object motion, morphing, warping, additions, removals, flicker, "
            "text, watermark, or crop."
        )
    return (
        f"{duration.capitalize()} continuous native 16:9 luxury real-estate camera bridge "
        "between the two "
        "provided views of the same synthetic property. Invent only a restrained, stabilized "
        "slow forward or lateral gimbal move that connects their shared visible area. Preserve "
        "rooflines, windows, garage, patio columns, lawn edges, walls, trees, driveway, "
        "perspective, materials, and layout. No crossfade, slideshow, static morph, handheld "
        "shake, fast zoom, orbit, people, vehicles, invented surroundings, structural changes, "
        "warping, flicker, text, watermark, or crop."
    )


def _bridge_latent_frames(config: LtxComfyUiConfig, duration_seconds: float) -> int:
    """Map a user-selected 2–6 second delivery duration to LTX's 8n+1 latent length."""
    desired_delivery_frames = round(duration_seconds * config.fps)
    # LTXVCropGuides removes the two guide slots after sampling, so the requested
    # latent duration is the delivery duration rather than the longer working duration.
    desired_latent_delivery = max(17, desired_delivery_frames)
    lower = ((desired_latent_delivery - 1) // 8) * 8 + 1
    upper = lower + 8
    return lower if desired_latent_delivery - lower <= upper - desired_latent_delivery else upper


def build_ltx_bridge_workflow(
    from_image_name: str,
    to_image_name: str,
    config: LtxComfyUiConfig,
    candidate: LtxBridgeCandidate,
    seed: int,
    prefix: str,
) -> dict[str, object]:
    """Build a two-image, guide-conditioned LTX camera transition at native 16:9."""
    prompt = _bridge_prompt(candidate)
    latent_frames = _bridge_latent_frames(config, candidate.duration_seconds)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": config.checkpoint, "weight_dtype": "fp8_e4m3fn"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": config.text_encoder, "type": "ltxv", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": config.vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": from_image_name}},
        "5": {"class_type": "LoadImage", "inputs": {"image": to_image_name}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": (
                    "crossfade, slideshow, static image morph, warped architecture, changed "
                    "house, changed landscaping, perspective shift, flicker, text, watermark"
                ),
                "clip": ["2", 0],
            },
        },
        "8": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": config.width,
                "height": config.height,
                "length": latent_frames,
                "batch_size": 1,
            },
        },
        "9": {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "vae": ["3", 0],
                "latent": ["8", 0],
                "image": ["4", 0],
                "frame_idx": 0,
                "strength": 1.0,
            },
        },
        "10": {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": ["9", 0],
                "negative": ["9", 1],
                "vae": ["3", 0],
                "latent": ["9", 2],
                "image": ["5", 0],
                "frame_idx": latent_frames - 1,
                "strength": 1.0,
            },
        },
        "11": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["10", 0],
                "negative": ["10", 1],
                "frame_rate": float(config.fps),
            },
        },
        "12": {
            "class_type": "ModelSamplingLTXV",
            "inputs": {
                "model": ["1", 0],
                "max_shift": 2.05,
                "base_shift": 0.95,
                "latent": ["10", 2],
            },
        },
        "13": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["12", 0], "conditioning": ["11", 0]},
        },
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "16": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": config.steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["10", 2],
            },
        },
        "17": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["14", 0],
                "guider": ["13", 0],
                "sampler": ["15", 0],
                "sigmas": ["16", 0],
                "latent_image": ["10", 2],
            },
        },
        "18": {
            "class_type": "LTXVCropGuides",
            "inputs": {
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent": ["17", 0],
            },
        },
        "19": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["18", 2], "vae": ["3", 0]},
        },
        "20": {
            "class_type": "SaveWEBM",
            "inputs": {
                "images": ["19", 0],
                "filename_prefix": prefix,
                "codec": "vp9",
                "fps": float(config.fps),
                "crf": 20.0,
            },
        },
    }


def _bridges() -> list[LtxBridgeCandidate]:
    return [
        LtxBridgeCandidate(
            candidate_id="front-left-to-front-wide",
            kind=LtxBridgeKind.SPATIAL,
            from_view="03-front-day-left",
            to_view="05-front-day-wide",
            duration_seconds=3.0,
            prompt=(
                "User-selected compatible front pair. Generate a restrained lateral or forward "
                "camera continuation through the shared front geometry."
            ),
            reason=(
                "Human-review candidate only: reject if roof, windows, driveway, trees, or wall "
                "geometry bends or shifts."
            ),
        ),
        LtxBridgeCandidate(
            candidate_id="patio-to-backyard",
            kind=LtxBridgeKind.SPATIAL,
            from_view="02-covered-patio",
            to_view="04-backyard-patio",
            duration_seconds=3.0,
            prompt=(
                "User-selected adjacent patio/backyard pair. Generate a restrained camera move "
                "only through visibly compatible outdoor geometry."
            ),
            reason=(
                "Human-review candidate only: reject unless overlap is obvious and all geometry "
                "remains coherent; use a deliberate cut otherwise."
            ),
        ),
        LtxBridgeCandidate(
            candidate_id="front-day-to-twilight",
            kind=LtxBridgeKind.LIGHTING_ONLY,
            from_view="05-front-day-wide",
            to_view="01-front-twilight",
            duration_seconds=3.0,
            prompt=(
                "User-selected lighting-only pair. Freeze the front framing and geometry; change "
                "only daylight, sky color, and practical/interior lighting."
            ),
            reason=(
                "Human-review candidate only: reject if the house, driveway, trees, windows, "
                "or patio shift."
            ),
        ),
    ]


def render_ltx_views(
    request: LtxRenderRequest, client: ComfyUiClient | None = None
) -> LtxRenderManifest:
    config = request.configuration
    input_dir = config.comfyui_root / "input"
    output_dir = config.comfyui_root / "output"
    if client is None:
        client = HttpComfyUiClient(config.endpoint, output_dir)
    identity = {
        "sources": [
            (view.name, _sha256(view.source_path), view.treatment.value)
            for view in request.source_views
        ],
        "bridge_ids": request.bridge_candidate_ids,
        "bridge_durations": request.bridge_duration_seconds,
        "custom_bridges": [
            candidate.model_dump(mode="json") for candidate in request.bridge_candidates
        ],
        "configuration": config.model_dump(mode="json"),
    }
    run_id = "ltx-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    run_dir = request.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    clips: list[LtxGeneratedClip] = []
    coverage: dict[str, str] = {}
    input_images: dict[str, str] = {}
    for index, view in enumerate(request.source_views):
        if not view.source_path.is_file():
            raise FileNotFoundError(view.source_path)
        source_hash = _sha256(view.source_path)
        image_name = f"{request.property_id}-{index + 1}-{view.source_path.name}"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(view.source_path, input_dir / image_name)
        input_images[view.name] = image_name
        workflow = build_ltx_workflow(
            image_name, config, view.treatment, config.seed + index, f"{run_id}/{view.name}"
        )
        generated = client.submit_and_wait(workflow)
        if not generated.is_file():
            raise FileNotFoundError(f"ComfyUI reported a missing output: {generated}")
        retained = run_dir / generated.name
        shutil.copy2(generated, retained)
        metadata = probe_video(retained)
        if metadata.width != config.width or metadata.height != config.height:
            raise ValueError("LTX clip is not native configured landscape resolution.")
        if metadata.width * 9 != metadata.height * 16:
            raise ValueError("LTX clip is not 16:9 landscape.")
        clips.append(
            LtxGeneratedClip(
                name=view.name,
                source_path=str(view.source_path),
                source_sha256=source_hash,
                source_coverage=(
                    "native_landscape_source_anchor; complete visible foreground retained"
                ),
                treatment=view.treatment,
                prompt=_prompt(view.treatment),
                workflow=workflow,
                generated_path=str(retained),
                generated_sha256=_sha256(retained),
                video=metadata,
            )
        )
        coverage[str(view.source_path)] = "included_as_native_landscape_ltx_source_view"
    candidates = {candidate.candidate_id: candidate for candidate in _bridges()}
    candidates.update(
        {candidate.candidate_id: candidate for candidate in request.bridge_candidates}
    )
    unknown_candidates = sorted(set(request.bridge_candidate_ids) - set(candidates))
    if unknown_candidates:
        raise ValueError(f"Unknown LTX bridge candidate(s): {', '.join(unknown_candidates)}")
    views = {view.name: view for view in request.source_views}
    bridges: list[LtxGeneratedBridge] = []
    for bridge_index, candidate_id in enumerate(request.bridge_candidate_ids):
        candidate = candidates[candidate_id].model_copy(
            update={
                "duration_seconds": request.bridge_duration_seconds.get(
                    candidate_id, candidates[candidate_id].duration_seconds
                )
            }
        )
        if candidate.from_view not in views or candidate.to_view not in views:
            raise ValueError(
                f"Bridge {candidate.candidate_id} requires selected source views "
                f"{candidate.from_view} and {candidate.to_view}."
            )
        from_view = views[candidate.from_view]
        to_view = views[candidate.to_view]
        workflow = build_ltx_bridge_workflow(
            input_images[candidate.from_view],
            input_images[candidate.to_view],
            config,
            candidate,
            config.seed + len(request.source_views) + bridge_index,
            f"{run_id}/bridges/{candidate.candidate_id}",
        )
        generated = client.submit_and_wait(workflow)
        if not generated.is_file():
            raise FileNotFoundError(f"ComfyUI reported a missing bridge output: {generated}")
        bridge_dir = run_dir / "bridges"
        bridge_dir.mkdir(exist_ok=True)
        retained = bridge_dir / generated.name
        shutil.copy2(generated, retained)
        metadata = probe_video(retained)
        if metadata.width != config.width or metadata.height != config.height:
            raise ValueError("LTX bridge is not native configured landscape resolution.")
        bridges.append(
            LtxGeneratedBridge(
                **candidate.model_dump(),
                from_source_path=str(from_view.source_path),
                from_source_sha256=_sha256(from_view.source_path),
                to_source_path=str(to_view.source_path),
                to_source_sha256=_sha256(to_view.source_path),
                workflow=workflow,
                generated_path=str(retained),
                generated_sha256=_sha256(retained),
                video=metadata,
            )
        )
    manifest = LtxRenderManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        property_id=request.property_id,
        configuration=config,
        clips=clips,
        bridge_candidates=list(candidates.values()),
        bridges=bridges,
        source_coverage=coverage,
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _edge_f1(first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None) -> float:
    first_edges = cv2.Canny(first, 80, 160) > 0
    second_edges = cv2.Canny(second, 80, 160) > 0
    if mask is not None:
        first_edges &= mask
        second_edges &= mask
    overlap = np.logical_and(first_edges, second_edges).sum()
    return float((2 * overlap) / max(first_edges.sum() + second_edges.sum(), 1))


def _camera_aligned_edge_f1(source: np.ndarray, frame: np.ndarray) -> float:
    """Measure geometry after removing the allowed stabilized camera translation."""
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(
            source,
            frame,
            warp,
            cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5),
        )
    except cv2.error:
        return _edge_f1(source, frame)
    aligned = cv2.warpAffine(
        frame,
        warp,
        (source.shape[1], source.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    source_mask = np.full(source.shape, 255, dtype=np.uint8)
    mask = (
        cv2.warpAffine(
            source_mask,
            warp,
            (source.shape[1], source.shape[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        > 0
    )
    return _edge_f1(source, aligned, mask)


def _read_grayscale_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    if not frames:
        raise ValueError(f"No decodable frames in {path}")
    return frames


def _temporal_metrics(frames: list[np.ndarray], duration_seconds: float) -> TemporalMetrics:
    differences = [float(np.mean(cv2.absdiff(one, two))) for one, two in zip(frames, frames[1:])]
    return TemporalMetrics(
        frame_count=len(frames),
        duration_seconds=duration_seconds,
        mean_frame_difference=float(np.mean(differences)) if differences else 0.0,
        frame_difference_cv=float(np.std(differences) / max(np.mean(differences), 1e-6))
        if differences
        else 0.0,
        minimum_edge_f1_to_hero=0.0,
        maximum_black_pixel_fraction=max(float(np.mean(frame <= 5)) for frame in frames),
    )


def _has_tail_transition_spike(frames: list[np.ndarray]) -> bool:
    """Reject a late ghost/rollback burst instead of hiding it with repeated frames."""
    differences = np.array(
        [float(np.mean(cv2.absdiff(one, two))) for one, two in zip(frames, frames[1:])]
    )
    if len(differences) < 12:
        return False
    tail = differences[-min(18, len(differences)) :]
    stable_tail_level = max(float(np.median(tail)), 1e-6)
    return float(np.max(tail)) > 2.5 * stable_tail_level


def evaluate_ltx_render(manifest_path: Path, output_dir: Path) -> LtxQualityReport:
    manifest = LtxRenderManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    reports: list[LtxClipQualityReport] = []
    for clip in manifest.clips:
        frames = _read_grayscale_frames(Path(clip.generated_path))
        source = cv2.imread(clip.source_path, cv2.IMREAD_GRAYSCALE)
        source = cv2.resize(source, (frames[0].shape[1], frames[0].shape[0]))
        metrics = _temporal_metrics(frames, clip.video.duration_seconds)
        metrics = metrics.model_copy(
            update={
                "minimum_edge_f1_to_hero": min(
                    _camera_aligned_edge_f1(source, frame) for frame in frames
                )
            }
        )
        reasons = []
        if metrics.minimum_edge_f1_to_hero < 0.45:
            reasons.append("structural_edge_drift_rejected")
        if metrics.maximum_black_pixel_fraction > 0.25:
            reasons.append("black_frame_rejected")
        if _has_tail_transition_spike(frames):
            reasons.append("tail_ghost_or_rollback_rejected")
        if metrics.frame_difference_cv > 1.2:
            reasons.append("temporal_instability_review")
        decision = (
            VideoDecision.REJECTED
            if any(reason.endswith("rejected") for reason in reasons)
            else VideoDecision.QUEUED_FOR_HUMAN_REVIEW
        )
        reports.append(
            LtxClipQualityReport(
                name=clip.name,
                generated_path=clip.generated_path,
                metrics=metrics,
                reason_codes=reasons,
                decision=decision,
            )
        )
    bridge_reports: list[LtxBridgeQualityReport] = []
    for bridge in manifest.bridges:
        frames = _read_grayscale_frames(Path(bridge.generated_path))
        from_source = cv2.imread(bridge.from_source_path, cv2.IMREAD_GRAYSCALE)
        to_source = cv2.imread(bridge.to_source_path, cv2.IMREAD_GRAYSCALE)
        from_source = cv2.resize(from_source, (frames[0].shape[1], frames[0].shape[0]))
        to_source = cv2.resize(to_source, (frames[-1].shape[1], frames[-1].shape[0]))
        endpoint_edge_f1 = min(
            _camera_aligned_edge_f1(from_source, frames[0]),
            _camera_aligned_edge_f1(to_source, frames[-1]),
        )
        metrics = _temporal_metrics(frames, bridge.video.duration_seconds)
        reasons = []
        if endpoint_edge_f1 < 0.45:
            reasons.append("bridge_endpoint_structure_rejected")
        if metrics.maximum_black_pixel_fraction > 0.25:
            reasons.append("black_frame_rejected")
        if _has_tail_transition_spike(frames):
            reasons.append("tail_ghost_or_rollback_rejected")
        if metrics.frame_difference_cv > 1.2:
            reasons.append("temporal_instability_review")
        decision = (
            VideoDecision.REJECTED
            if any(reason.endswith("rejected") for reason in reasons)
            else VideoDecision.QUEUED_FOR_HUMAN_REVIEW
        )
        bridge_reports.append(
            LtxBridgeQualityReport(
                candidate_id=bridge.candidate_id,
                generated_path=bridge.generated_path,
                metrics=metrics,
                endpoint_edge_f1=endpoint_edge_f1,
                reason_codes=reasons,
                decision=decision,
            )
        )
    report_id = f"ltx-quality-{manifest.run_id.removeprefix('ltx-')}"
    report_dir = output_dir / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    worksheet = report_dir / "review.csv"
    with worksheet.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["kind", "name", "path_or_candidate", "decision", "reviewer", "notes"]
        )
        writer.writeheader()
        for clip in reports:
            writer.writerow(
                {
                    "kind": "clip",
                    "name": clip.name,
                    "path_or_candidate": clip.generated_path,
                    "decision": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
        for bridge in bridge_reports:
            writer.writerow(
                {
                    "kind": "bridge",
                    "name": bridge.candidate_id,
                    "path_or_candidate": bridge.generated_path,
                    "decision": "",
                    "reviewer": "",
                    "notes": "Review endpoint geometry and temporal continuity before approval.",
                }
            )
    overall = (
        VideoDecision.REJECTED
        if any(item.decision is VideoDecision.REJECTED for item in reports + bridge_reports)
        else VideoDecision.QUEUED_FOR_HUMAN_REVIEW
    )
    report = LtxQualityReport(
        report_id=report_id,
        created_at=datetime.now(UTC),
        render_manifest_path=str(manifest_path),
        clips=reports,
        bridge_candidates=manifest.bridge_candidates,
        bridges=bridge_reports,
        decision=overall,
        review_worksheet_path=str(worksheet),
    )
    (report_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def assemble_ltx_portrait_reel(
    manifest_path: Path, accepted_clip_names: list[str], output_dir: Path
) -> LtxReelManifest:
    """Assemble explicitly human-accepted landscape candidates without foreground cropping."""
    manifest = LtxRenderManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    clips_by_name = {clip.name: clip for clip in manifest.clips}
    if not accepted_clip_names:
        raise ValueError("At least one explicitly human-accepted LTX clip is required.")
    if len(set(accepted_clip_names)) != len(accepted_clip_names):
        raise ValueError("Accepted LTX clip names must be distinct.")
    missing = sorted(set(accepted_clip_names) - set(clips_by_name))
    if missing:
        raise ValueError(f"Unknown accepted LTX clips: {', '.join(missing)}")
    clips = [clips_by_name[name] for name in accepted_clip_names]
    payload = json.dumps(
        {"render": manifest.run_id, "clips": accepted_clip_names},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    run_id = f"ltx-reel-{hashlib.sha256(payload).hexdigest()[:16]}"
    run_dir = output_dir / run_id
    output = run_dir / "reel.mp4"
    require_ffmpeg()
    run_dir.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for clip in clips:
        command.extend(["-i", clip.generated_path])
    filters: list[str] = []
    foregrounds: list[str] = []
    for index, _clip in enumerate(clips):
        filters.extend(
            [
                f"[{index}:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,boxblur=20:1[bg{index}]",
                f"[{index}:v]scale=1080:608[fg{index}]",
                f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2[v{index}]",
            ]
        )
        foregrounds.append(f"[v{index}]")
    filters.append(f"{''.join(foregrounds)}concat=n={len(clips)}:v=1:a=0[video]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[video]",
            "-r",
            "30",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    run_ffmpeg(command)
    reel = LtxReelManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        render_manifest_path=str(manifest_path),
        accepted_clip_names=accepted_clip_names,
        source_coverage={clip.source_path: clip.source_coverage for clip in clips},
        output_path=str(output),
        output_sha256=_sha256(output),
        video=probe_video(output),
    )
    (run_dir / "manifest.json").write_text(reel.model_dump_json(indent=2), encoding="utf-8")
    return reel


def plan_ltx_timed_reel(
    render_manifest_path: Path,
    bridge_candidate_ids: list[str],
    total_duration_seconds: float,
    bridge_duration_seconds: float | dict[str, float],
    output_dir: Path,
    scene_fade_seconds: float = 0.45,
) -> LtxTimedReelPlan:
    """Allocate a delivery length across user-selected bridges and their follow-on shots."""
    manifest = LtxRenderManifest.model_validate_json(
        render_manifest_path.read_text(encoding="utf-8")
    )
    if not 8.0 <= total_duration_seconds <= 120.0:
        raise ValueError("Total reel duration must be between 8 and 120 seconds.")
    if not 0.2 <= scene_fade_seconds <= 1.0:
        raise ValueError("Cinematic scene fades must be between 0.2 and 1 second.")
    if isinstance(bridge_duration_seconds, dict):
        default_bridge_duration = 3.0
        bridge_duration_by_id = {
            candidate_id: bridge_duration_seconds.get(candidate_id, default_bridge_duration)
            for candidate_id in bridge_candidate_ids
        }
        overrides = dict(bridge_duration_seconds)
    else:
        default_bridge_duration = bridge_duration_seconds
        bridge_duration_by_id = {
            candidate_id: bridge_duration_seconds for candidate_id in bridge_candidate_ids
        }
        overrides = {}
    invalid_durations = [
        duration
        for duration in bridge_duration_by_id.values()
        if not 2.0 <= duration <= 6.0
    ]
    if invalid_durations:
        raise ValueError("Each requested bridge duration must be between 2 and 6 seconds.")
    if not bridge_candidate_ids:
        raise ValueError("Select at least one compatible LTX bridge.")
    if len(set(bridge_candidate_ids)) != len(bridge_candidate_ids):
        raise ValueError("Selected LTX bridges must be distinct.")
    bridges = {bridge.candidate_id: bridge for bridge in manifest.bridges}
    missing = sorted(set(bridge_candidate_ids) - set(bridges))
    if missing:
        raise ValueError(
            "Requested LTX bridge output is missing: "
            f"{', '.join(missing)}. Render it with --bridge-candidate first."
        )
    clips = {clip.name: clip for clip in manifest.clips}
    selected_bridges = [bridges[candidate_id] for candidate_id in bridge_candidate_ids]
    follow_on_names = [bridge.to_view for bridge in selected_bridges]
    missing_clips = sorted(set(follow_on_names) - set(clips))
    if missing_clips:
        raise ValueError(f"Bridge follow-on source clip is missing: {', '.join(missing_clips)}")
    scene_dissolve_count = sum(
        previous.to_view != following.from_view
        for previous, following in zip(selected_bridges, selected_bridges[1:])
    )
    shot_budget = (
        total_duration_seconds
        + scene_dissolve_count * scene_fade_seconds
        - sum(bridge_duration_by_id.values())
    )
    if shot_budget < 0.75 * len(follow_on_names):
        raise ValueError(
            "Requested duration leaves less than 0.75 seconds per follow-on shot. "
            "Use fewer bridges or increase the total reel duration."
        )
    allocated_shot_duration = shot_budget / len(follow_on_names)
    items: list[LtxTimedReelItem] = []
    for index, bridge in enumerate(selected_bridges):
        bridge_duration = bridge_duration_by_id[bridge.candidate_id]
        if index == 0:
            transition_before = "opening"
        elif selected_bridges[index - 1].to_view == bridge.from_view:
            transition_before = "continuous"
        else:
            transition_before = "cinematic_dissolve"
        items.append(
            LtxTimedReelItem(
                kind="ltx_bridge",
                name=bridge.candidate_id,
                input_path=bridge.generated_path,
                source_duration_seconds=bridge.video.duration_seconds,
                delivery_duration_seconds=bridge_duration,
                playback_speed=bridge.video.duration_seconds / bridge_duration,
                transition_before=transition_before,
            )
        )
        clip = clips[bridge.to_view]
        items.append(
            LtxTimedReelItem(
                kind="source_clip",
                name=clip.name,
                input_path=clip.generated_path,
                source_duration_seconds=clip.video.duration_seconds,
                delivery_duration_seconds=allocated_shot_duration,
                playback_speed=clip.video.duration_seconds / allocated_shot_duration,
                transition_before="continuous",
            )
        )
    payload = json.dumps(
        {
            "render": manifest.run_id,
            "bridges": bridge_candidate_ids,
            "total": total_duration_seconds,
            "bridge_durations": bridge_duration_by_id,
            "scene_fade_seconds": scene_fade_seconds,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    run_id = f"ltx-timed-reel-{hashlib.sha256(payload).hexdigest()[:16]}"
    plan = LtxTimedReelPlan(
        run_id=run_id,
        created_at=datetime.now(UTC),
        render_manifest_path=str(render_manifest_path),
        requested_total_duration_seconds=total_duration_seconds,
        requested_bridge_duration_seconds=default_bridge_duration,
        requested_bridge_duration_overrides=overrides,
        scene_fade_seconds=scene_fade_seconds,
        items=items,
        output_duration_seconds=total_duration_seconds,
        optimization_notes=[
            "Each compatible LTX bridge receives its user-selected duration.",
            "Remaining time is distributed evenly across bridge follow-on shots.",
            "Unrelated areas use cinematic dissolves, never an invented spatial bridge.",
            f"Each unrelated scene dissolve is {scene_fade_seconds:g} seconds.",
            "Playback rates are recorded so delivery duration remains exact.",
            "Shorter remaining shot budgets intentionally make follow-on camera motion faster.",
        ],
    )
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return plan


def assemble_ltx_timed_reel(plan_path: Path, output_dir: Path) -> LtxTimedReelManifest:
    """Render an exact, bridge-aware portrait timeline from a saved pacing plan."""
    plan = LtxTimedReelPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    if not plan.items:
        raise ValueError("The timed reel plan has no timeline items.")
    missing = [item.input_path for item in plan.items if not Path(item.input_path).is_file()]
    if missing:
        raise FileNotFoundError(f"Timed reel input is missing: {', '.join(missing)}")
    require_ffmpeg()
    payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True).encode()
    run_id = f"ltx-timed-render-{hashlib.sha256(payload).hexdigest()[:16]}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "reel.mp4"
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for item in plan.items:
        command.extend(["-i", item.input_path])
    filters: list[str] = []
    for index, item in enumerate(plan.items):
        filters.extend(
            [
                f"[{index}:v]trim=duration={item.source_duration_seconds:.6f},"
                f"setpts=PTS/{item.playback_speed:.9f},split=2[fgraw{index}][bgraw{index}]",
                f"[bgraw{index}]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,boxblur=20:10[bg{index}]",
                f"[fgraw{index}]scale=1080:608:force_original_aspect_ratio=decrease[fg{index}]",
                f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2,format=yuv420p,"
                f"fps=30,settb=AVTB,setpts=PTS-STARTPTS[v{index}]",
            ]
        )
    current_label = "v0"
    current_duration = plan.items[0].delivery_duration_seconds
    for index, item in enumerate(plan.items[1:], start=1):
        output_label = f"timeline{index}"
        if item.transition_before == "cinematic_dissolve":
            fade_seconds = min(
                plan.scene_fade_seconds,
                current_duration - 0.01,
                item.delivery_duration_seconds - 0.01,
            )
            filters.append(
                f"[{current_label}][v{index}]xfade=transition=fade:duration={fade_seconds:.6f}:"
                f"offset={current_duration - fade_seconds:.6f}[{output_label}]"
            )
            current_duration += item.delivery_duration_seconds - fade_seconds
        else:
            filters.append(f"[{current_label}][v{index}]concat=n=2:v=1:a=0[{output_label}]")
            current_duration += item.delivery_duration_seconds
        current_label = output_label
    output_frames = round(plan.output_duration_seconds * 30)
    filters.append(
        f"[{current_label}]fps=30,trim=end_frame={output_frames},setpts=PTS-STARTPTS[video]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[video]",
            "-r",
            "30",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    run_ffmpeg(command)
    video = probe_video(output)
    if abs(video.duration_seconds - plan.output_duration_seconds) > 0.04:
        raise ValueError("Timed reel render duration does not match its requested plan.")
    manifest = LtxTimedReelManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        plan_path=str(plan_path),
        items=plan.items,
        output_path=str(output),
        output_sha256=_sha256(output),
        video=video,
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
