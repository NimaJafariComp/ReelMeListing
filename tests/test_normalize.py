from pathlib import Path

from PIL import Image

from listing_to_reel.media.normalize import normalize_for_vertical_frame


def test_normalize_creates_requested_vertical_rgb_frame(tmp_path: Path) -> None:
    source = tmp_path / "landscape.png"
    output = tmp_path / "normalized.jpg"
    Image.new("RGBA", (400, 200), color=(12, 34, 56, 255)).save(source)

    normalize_for_vertical_frame(source, output, width=108, height=192)

    with Image.open(output) as image:
        assert image.size == (108, 192)
        assert image.mode == "RGB"
