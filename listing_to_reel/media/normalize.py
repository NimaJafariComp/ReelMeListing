"""Deterministic image normalization for vertical reel frames."""

from pathlib import Path

from PIL import Image, ImageOps


def normalize_for_vertical_frame(
    source_path: Path, output_path: Path, width: int, height: int
) -> None:
    """Apply EXIF orientation, RGB conversion, and a centered 9:16 cover crop."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source:
        normalized = ImageOps.exif_transpose(source).convert("RGB")
        frame = ImageOps.fit(
            normalized,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        frame.save(output_path, format="JPEG", quality=95, optimize=False, progressive=False)
