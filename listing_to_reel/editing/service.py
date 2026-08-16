"""Provider-independent orchestration and lineage persistence for Phase 3."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import yaml

from listing_to_reel.analysis.models import InputQualityDecision, InputQualityReport
from listing_to_reel.core.environment import collect_environment_snapshot
from listing_to_reel.editing.models import (
    EditRequest,
    EditRunManifest,
    GeneratedCandidate,
    ImageEditingFile,
)


class ImageEditor(Protocol):
    """A replaceable image-editing adapter contract."""

    def generate(
        self, request: EditRequest, candidate_dir: Path
    ) -> tuple[str, list[GeneratedCandidate]]: ...


def load_image_editor_config(path: Path):
    """Load the typed baseline image-editor configuration."""
    with path.open("r", encoding="utf-8") as config_file:
        return ImageEditingFile.model_validate(yaml.safe_load(config_file)).image_editor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_quality_report(request: EditRequest, source_hash: str) -> InputQualityReport:
    report = InputQualityReport.model_validate_json(
        request.input_quality_report_path.read_text(encoding="utf-8")
    )
    if report.source_sha256 != source_hash:
        raise ValueError("Input-quality report does not belong to the requested source image.")
    if report.decision is not InputQualityDecision.ACCEPTED:
        raise ValueError("Phase 3 requires an accepted input-quality report.")
    return report


def _run_id(request: EditRequest, source_hash: str) -> str:
    payload = {
        "source_hash": source_hash,
        "instruction": request.instruction,
        "seed": request.seed,
        "configuration": request.configuration.model_dump(mode="json"),
        "runtime_profile": request.runtime_profile.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"edit-{hashlib.sha256(encoded).hexdigest()[:16]}"


def generate_edit_candidates(request: EditRequest, editor: ImageEditor) -> EditRunManifest:
    """Generate candidates only when the source passed Phase 2 quality gates."""
    source_hash = _sha256(request.source_path)
    _validated_quality_report(request, source_hash)
    run_id = _run_id(request, source_hash)
    run_dir = request.output_dir / run_id
    resolved_revision, candidates = editor.generate(request, run_dir / "candidates")
    if len(candidates) != request.configuration.candidate_count:
        raise ValueError("Image editor returned an unexpected number of candidates.")

    manifest = EditRunManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        source_path=str(request.source_path),
        source_sha256=source_hash,
        input_quality_report_path=str(request.input_quality_report_path),
        instruction=request.instruction,
        configuration=request.configuration,
        requested_model_revision=request.configuration.model_revision,
        resolved_model_revision=resolved_revision,
        runtime_profile_name=request.runtime_profile_name,
        runtime_profile=request.runtime_profile,
        environment=collect_environment_snapshot(),
        candidates=candidates,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
