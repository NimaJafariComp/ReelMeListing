from pathlib import Path

from PIL import Image

from listing_to_reel.media.ffmpeg import build_reel_command
from listing_to_reel.media.models import ReelRequest, ReelSettings


def _source_images(tmp_path: Path, count: int = 5) -> list[Path]:
    paths = []
    for index in range(count):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (320, 180), color=(index, index, index)).save(path)
        paths.append(path)
    return paths


def test_reel_request_distributes_target_duration_across_sources(tmp_path: Path) -> None:
    request = ReelRequest(source_paths=_source_images(tmp_path))

    assert request.clip_duration_seconds == 2.8


def test_ffmpeg_command_has_ordered_inputs_transitions_and_stable_metadata(tmp_path: Path) -> None:
    source_paths = _source_images(tmp_path, count=3)
    request = ReelRequest(
        source_paths=source_paths, settings=ReelSettings(target_duration_seconds=10.0)
    )
    normalized_paths = [tmp_path / f"normalized-{index}.jpg" for index in range(3)]

    command = build_reel_command(request, normalized_paths, tmp_path / "listing_reel.mp4")
    filter_graph = command[command.index("-filter_complex") + 1]

    assert command.count("-loop") == 3
    assert filter_graph.count("xfade=transition=fade") == 2
    assert "offset=3.166667" in filter_graph
    assert "offset=6.333333" in filter_graph
    assert "creation_time=1970-01-01T00:00:00Z" in command
