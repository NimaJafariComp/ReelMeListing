"""Orchestration for a deterministic Phase 1 listing reel."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from listing_to_reel.media.ffmpeg import build_reel_command, probe_video, require_ffmpeg, run_ffmpeg
from listing_to_reel.media.models import ReelRequest, ReelRunManifest
from listing_to_reel.media.normalize import normalize_for_vertical_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(request: ReelRequest, source_hashes: dict[str, str]) -> str:
    payload = {
        "source_hashes": source_hashes,
        "settings": request.settings.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"reel-{hashlib.sha256(encoded).hexdigest()[:16]}"


def assemble_reel(request: ReelRequest) -> ReelRunManifest:
    """Normalize images, render a vertical reel, validate it, and record lineage."""
    require_ffmpeg()
    source_hashes = {str(path): _sha256(path) for path in request.source_paths}
    run_id = _run_id(request, source_hashes)
    run_dir = request.output_dir / run_id
    normalized_dir = run_dir / "normalized"
    normalized_paths = [
        normalized_dir / f"{index:03d}.jpg"
        for index in range(1, len(request.source_paths) + 1)
    ]

    for source_path, normalized_path in zip(request.source_paths, normalized_paths, strict=True):
        normalize_for_vertical_frame(
            source_path, normalized_path, request.settings.width, request.settings.height
        )

    output_path = run_dir / "listing_reel.mp4"
    command = build_reel_command(request, normalized_paths, output_path)
    run_ffmpeg(command)
    video = probe_video(output_path)

    if video.width != request.settings.width or video.height != request.settings.height:
        raise ValueError("Encoded reel dimensions do not match the requested settings.")
    if not 10.0 <= video.duration_seconds <= 15.25:
        raise ValueError(
            f"Encoded reel duration is outside the MVP range: {video.duration_seconds:.3f}s"
        )

    manifest = ReelRunManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        source_paths=[str(path) for path in request.source_paths],
        source_sha256=source_hashes,
        normalized_paths=[str(path) for path in normalized_paths],
        settings=request.settings,
        ffmpeg_command=command,
        output_path=str(output_path),
        output_sha256=_sha256(output_path),
        video=video,
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
