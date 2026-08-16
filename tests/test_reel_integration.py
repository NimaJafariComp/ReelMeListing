import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from listing_to_reel.media.models import ReelRequest, ReelSettings
from listing_to_reel.media.reel import assemble_reel


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_assemble_reel_writes_a_valid_vertical_mp4_and_manifest(tmp_path: Path) -> None:
    source_paths = []
    for index, color in enumerate(((20, 40, 60), (80, 100, 120)), start=1):
        source_path = tmp_path / f"source-{index}.jpg"
        Image.new("RGB", (320, 180), color=color).save(source_path)
        source_paths.append(source_path)

    manifest = assemble_reel(
        ReelRequest(
            source_paths=source_paths,
            output_dir=tmp_path / "runs",
            settings=ReelSettings(
                width=108,
                height=192,
                fps=10,
                target_duration_seconds=10.0,
                preset="ultrafast",
            ),
        )
    )

    output_path = Path(manifest.output_path)
    manifest_path = output_path.with_name("manifest.json")
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert output_path.is_file()
    assert manifest.video.codec_name == "h264"
    assert (manifest.video.width, manifest.video.height) == (108, 192)
    assert manifest.video.duration_seconds == pytest.approx(10.0, abs=0.1)
    assert persisted_manifest["output_sha256"] == manifest.output_sha256
