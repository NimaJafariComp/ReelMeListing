"""Deterministic image checks, ranking, and human-review artifact handling."""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

from listing_to_reel.editing.models import EditRunManifest
from listing_to_reel.evaluation.models import (
    CandidateDecision,
    CandidateEvaluation,
    CandidateMetrics,
    EvaluationConfig,
    EvaluationFile,
    EvaluationReport,
    FinalDecisionRecord,
    HumanReviewDecision,
    RunDecision,
)


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load deterministic Phase 4 thresholds."""
    with path.open("r", encoding="utf-8") as config_file:
        return EvaluationFile.model_validate(yaml.safe_load(config_file)).evaluation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gray(path: Path, maximum_dimension: int) -> np.ndarray:
    with Image.open(path) as image_file:
        rgb = np.asarray(ImageOps.exif_transpose(image_file).convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, maximum_dimension / max(height, width))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _candidate_at_source_shape(candidate: np.ndarray, source: np.ndarray) -> np.ndarray:
    return cv2.resize(candidate, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)


def _edge_f1(
    source: np.ndarray, candidate: np.ndarray, minimum_edge_pixels: int
) -> tuple[float | None, int, int]:
    source_edges = cv2.Canny(source, 50, 150)
    candidate_edges = cv2.Canny(candidate, 50, 150)
    source_count = int(np.count_nonzero(source_edges))
    candidate_count = int(np.count_nonzero(candidate_edges))
    if source_count < minimum_edge_pixels or candidate_count < minimum_edge_pixels:
        return None, source_count, candidate_count
    source_dilated = cv2.dilate(source_edges, np.ones((3, 3), np.uint8))
    candidate_dilated = cv2.dilate(candidate_edges, np.ones((3, 3), np.uint8))
    precision = float(np.mean(source_dilated[candidate_edges > 0] > 0))
    recall = float(np.mean(candidate_dilated[source_edges > 0] > 0))
    return (2 * precision * recall / max(precision + recall, 1e-9)), source_count, candidate_count


def _vertical_error(gray: np.ndarray) -> float | None:
    edges = cv2.Canny(gray, 50, 150)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=max(20, round(min(gray.shape[:2]) * 0.12)),
        maxLineGap=12,
    )
    if raw_lines is None:
        return None
    weighted_errors: list[float] = []
    for x1, y1, x2, y2 in raw_lines[:, 0]:
        error = abs(90.0 - abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if error <= 30:
            weighted_errors.extend([float(error)] * max(1, round(np.hypot(x2 - x1, y2 - y1) / 10)))
    return float(np.median(weighted_errors)) if weighted_errors else None


def _candidate_metrics(
    source: np.ndarray, candidate: np.ndarray, config: EvaluationConfig
) -> CandidateMetrics:
    candidate = _candidate_at_source_shape(candidate, source)
    edge_f1, source_edges, candidate_edges = _edge_f1(source, candidate, config.min_edge_pixels)
    source_blur = float(cv2.Laplacian(source, cv2.CV_64F).var())
    candidate_blur = float(cv2.Laplacian(candidate, cv2.CV_64F).var())
    source_vertical = _vertical_error(source)
    candidate_vertical = _vertical_error(candidate)
    vertical_delta = (
        abs(candidate_vertical - source_vertical)
        if source_vertical is not None and candidate_vertical is not None
        else None
    )
    return CandidateMetrics(
        edge_f1=edge_f1,
        source_edge_pixels=source_edges,
        candidate_edge_pixels=candidate_edges,
        blur_ratio=candidate_blur / max(source_blur, 1e-6),
        source_blur_variance=source_blur,
        candidate_blur_variance=candidate_blur,
        black_pixel_fraction=float(np.mean(candidate <= 5)),
        mean_luminance_delta=abs(float(candidate.mean()) - float(source.mean())),
        vertical_line_delta_degrees=vertical_delta,
    )


def _evaluate_metrics(
    metrics: CandidateMetrics, config: EvaluationConfig
) -> tuple[list[str], CandidateDecision, float]:
    reasons: list[str] = []
    rejected = False
    if metrics.black_pixel_fraction >= config.black_pixel_reject_fraction:
        reasons.append("artifact_black_frame_rejected")
        rejected = True
    if metrics.edge_f1 is None:
        reasons.append("structural_edges_insufficient_review")
    elif metrics.edge_f1 <= config.edge_f1_reject:
        reasons.append("structural_edge_overlap_rejected")
        rejected = True
    elif metrics.edge_f1 <= config.edge_f1_warning:
        reasons.append("structural_edge_overlap_warning")
    if metrics.blur_ratio is not None and metrics.blur_ratio <= config.blur_ratio_reject:
        reasons.append("artifact_blur_rejected")
        rejected = True
    elif metrics.blur_ratio is not None and metrics.blur_ratio <= config.blur_ratio_warning:
        reasons.append("artifact_blur_warning")
    if (
        metrics.vertical_line_delta_degrees is not None
        and metrics.vertical_line_delta_degrees >= config.vertical_line_delta_warning_degrees
    ):
        reasons.append("vertical_line_drift_review")

    edge_score = metrics.edge_f1 if metrics.edge_f1 is not None else 0.0
    blur_score = min(1.0, metrics.blur_ratio or 0.0)
    black_score = 1.0 - metrics.black_pixel_fraction
    score = max(0.0, min(1.0, 0.60 * edge_score + 0.25 * blur_score + 0.15 * black_score))
    decision = CandidateDecision.REJECTED if rejected else CandidateDecision.QUEUED_FOR_HUMAN_REVIEW
    return reasons, decision, score


def evaluate_edit_run(
    edit_run_manifest_path: Path, config: EvaluationConfig, output_dir: Path
) -> EvaluationReport:
    """Evaluate every persisted Phase 3 candidate and emit a reviewable report."""
    edit_manifest = EditRunManifest.model_validate_json(
        edit_run_manifest_path.read_text(encoding="utf-8")
    )
    source_path = Path(edit_manifest.source_path)
    if _sha256(source_path) != edit_manifest.source_sha256:
        raise ValueError("Edit manifest source hash no longer matches its source image.")
    source = _read_gray(source_path, config.analysis_max_dimension)
    evaluations: list[CandidateEvaluation] = []
    for candidate in edit_manifest.candidates:
        candidate_path = Path(candidate.artifact_path)
        candidate_hash = _sha256(candidate_path)
        if candidate_hash != candidate.sha256:
            raise ValueError(f"Candidate hash mismatch for {candidate_path}.")
        candidate_image = _read_gray(candidate_path, config.analysis_max_dimension)
        metrics = _candidate_metrics(source, candidate_image, config)
        reasons, decision, score = _evaluate_metrics(metrics, config)
        evaluations.append(
            CandidateEvaluation(
                candidate_index=candidate.candidate_index,
                candidate_path=str(candidate_path),
                candidate_sha256=candidate_hash,
                metrics=metrics,
                reason_codes=reasons,
                score=score,
                decision=decision,
            )
        )

    viable = [item for item in evaluations if item.decision is not CandidateDecision.REJECTED]
    recommended = max(viable, key=lambda item: item.score, default=None)
    run_decision = (
        RunDecision.QUEUED_FOR_HUMAN_REVIEW if recommended else RunDecision.RETRY_RECOMMENDED
    )
    reason_codes = [reason for candidate in evaluations for reason in candidate.reason_codes]
    report_id = f"quality-{edit_manifest.run_id.removeprefix('edit-')}"
    report_dir = output_dir / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report = EvaluationReport(
        report_id=report_id,
        created_at=datetime.now(UTC),
        edit_run_manifest_path=str(edit_run_manifest_path),
        edit_run_id=edit_manifest.run_id,
        source_path=edit_manifest.source_path,
        source_sha256=edit_manifest.source_sha256,
        configuration=config,
        candidates=evaluations,
        recommended_candidate_index=(recommended.candidate_index if recommended else None),
        run_decision=run_decision,
        reason_codes=sorted(set(reason_codes)),
    )
    worksheet_path = export_review_worksheet(report, report_dir / "review.csv")
    report = report.model_copy(update={"review_worksheet_path": str(worksheet_path)})
    (report_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def export_review_worksheet(report: EvaluationReport, destination: Path) -> Path:
    """Create a reviewer-facing CSV that hides machine ranking and score fields."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as worksheet:
        writer = csv.DictWriter(
            worksheet,
            fieldnames=["blind_candidate_id", "candidate_path", "decision", "reviewer", "notes"],
        )
        writer.writeheader()
        for candidate in report.candidates:
            if candidate.decision is CandidateDecision.QUEUED_FOR_HUMAN_REVIEW:
                writer.writerow(
                    {
                        "blind_candidate_id": f"C{candidate.candidate_index:02d}",
                        "candidate_path": candidate.candidate_path,
                        "decision": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )
    return destination


def import_human_review(
    evaluation_report_path: Path, worksheet_path: Path, output_dir: Path
) -> FinalDecisionRecord:
    """Resolve a Phase 4 run after a reviewer fills the exported CSV."""
    report = EvaluationReport.model_validate_json(
        evaluation_report_path.read_text(encoding="utf-8")
    )
    queued = {
        f"C{candidate.candidate_index:02d}": candidate
        for candidate in report.candidates
        if candidate.decision is CandidateDecision.QUEUED_FOR_HUMAN_REVIEW
    }
    decisions: list[HumanReviewDecision] = []
    with worksheet_path.open("r", encoding="utf-8", newline="") as worksheet:
        for row in csv.DictReader(worksheet):
            if not row.get("decision", "").strip():
                continue
            decision = HumanReviewDecision(
                blind_candidate_id=row["blind_candidate_id"].strip(),
                decision=row["decision"].strip(),
                reviewer=row["reviewer"].strip(),
                notes=row.get("notes", "").strip(),
            )
            if decision.blind_candidate_id not in queued:
                raise ValueError(
                    f"Review references an ineligible candidate: {decision.blind_candidate_id}"
                )
            decisions.append(decision)
    if not decisions:
        raise ValueError("No completed human-review decisions were supplied.")
    accepted = next(
        (item for item in decisions if item.decision is CandidateDecision.ACCEPTED_BY_HUMAN), None
    )
    selected = queued[accepted.blind_candidate_id].candidate_index if accepted else None
    record = FinalDecisionRecord(
        created_at=datetime.now(UTC),
        evaluation_report_path=str(evaluation_report_path),
        edit_run_id=report.edit_run_id,
        decision=RunDecision.ACCEPTED if accepted else RunDecision.REJECTED,
        selected_candidate_index=selected,
        reviewer=accepted.reviewer if accepted else decisions[0].reviewer,
        notes=accepted.notes if accepted else "All reviewed candidates were rejected.",
        reason_codes=([] if accepted else ["human_review_rejected_all"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final-decision.json").write_text(
        record.model_dump_json(indent=2), encoding="utf-8"
    )
    return record
