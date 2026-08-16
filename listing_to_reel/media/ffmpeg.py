"""FFmpeg command construction and post-encode validation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from listing_to_reel.media.models import ReelRequest, VideoMetadata


class FFmpegUnavailableError(RuntimeError):
    """Raised when deterministic rendering cannot find the required FFmpeg tools."""


class FFmpegEncodingError(RuntimeError):
    """Raised when FFmpeg exits unsuccessfully."""


def require_ffmpeg() -> None:
    """Require both encoder and probe binaries before rendering."""
    missing = [binary for binary in ("ffmpeg", "ffprobe") if shutil.which(binary) is None]
    if missing:
        raise FFmpegUnavailableError(
            f"Missing required media tools: {', '.join(missing)}. Install FFmpeg first."
        )


def build_reel_command(
    request: ReelRequest, normalized_paths: list[Path], output_path: Path
) -> list[str]:
    """Build a single-threaded, fixed-parameter FFmpeg render command."""
    settings = request.settings
    clip_duration = request.clip_duration_seconds
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    for image_path in normalized_paths:
        command.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(settings.fps),
                "-t",
                f"{clip_duration:.6f}",
                "-i",
                str(image_path),
            ]
        )

    filter_parts: list[str] = []
    for index in range(len(normalized_paths)):
        filter_parts.append(
            f"[{index}:v]zoompan=z='min(zoom+{settings.zoom_increment_per_frame:.7f}\\,{settings.zoom_max:.3f})'"
            f":d=1:s={settings.width}x{settings.height}:fps={settings.fps},"
            f"trim=duration={clip_duration:.6f},setpts=PTS-STARTPTS[v{index}]"
        )

    previous_label = "v0"
    for index in range(1, len(normalized_paths)):
        output_label = f"x{index}"
        offset = index * (clip_duration - settings.transition_seconds)
        filter_parts.append(
            f"[{previous_label}][v{index}]xfade=transition=fade:duration="
            f"{settings.transition_seconds:.6f}:offset={offset:.6f}[{output_label}]"
        )
        previous_label = output_label

    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            f"[{previous_label}]",
            "-an",
            "-r",
            str(settings.fps),
            "-threads",
            "1",
            "-c:v",
            "libx264",
            "-preset",
            settings.preset,
            "-crf",
            str(settings.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-metadata",
            "creation_time=1970-01-01T00:00:00Z",
            str(output_path),
        ]
    )
    return command


def run_ffmpeg(command: list[str]) -> None:
    """Run FFmpeg without shell interpolation and surface its error message."""
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise FFmpegEncodingError(error.stderr.strip() or "FFmpeg reel encoding failed.") from error


def probe_video(path: Path) -> VideoMetadata:
    """Validate duration, format, and dimensions of a completed MP4."""
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(probe_command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    return VideoMetadata(
        codec_name=stream["codec_name"],
        width=stream["width"],
        height=stream["height"],
        pixel_format=stream["pix_fmt"],
        frame_rate=stream["r_frame_rate"],
        duration_seconds=float(payload["format"]["duration"]),
    )
