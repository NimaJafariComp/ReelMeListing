"""Command-line entry points for reproducibility checks."""

import json
from pathlib import Path

import typer

from listing_to_reel.analysis.input_quality import analyze_input_image, load_input_quality_config
from listing_to_reel.benchmarking.models import BenchmarkStage
from listing_to_reel.benchmarking.service import compare_ltx_benchmarks, record_ltx_benchmark
from listing_to_reel.core.config import load_runtime_config
from listing_to_reel.core.environment import collect_environment_snapshot
from listing_to_reel.editing.instruct_pix2pix import InstructPix2PixEditor
from listing_to_reel.editing.models import EditRequest
from listing_to_reel.editing.service import generate_edit_candidates, load_image_editor_config
from listing_to_reel.evaluation.service import (
    evaluate_edit_run,
    import_human_review,
    load_evaluation_config,
)
from listing_to_reel.finetuning.models import (
    LoRADatasetManifest,
    LoRAReadinessRequest,
    ReadinessEvidence,
)
from listing_to_reel.finetuning.service import assess_lora_readiness, load_lora_pilot_config
from listing_to_reel.media.models import ReelRequest
from listing_to_reel.media.reel import assemble_reel
from listing_to_reel.video.ltx_comfyui import (
    assemble_ltx_portrait_reel,
    assemble_ltx_timed_reel,
    evaluate_ltx_render,
    load_ltx_comfyui_config,
    plan_ltx_timed_reel,
    render_ltx_views,
)
from listing_to_reel.video.models import (
    HeroVideoRequest,
    InterpolationConfig,
    LtxBridgeCandidate,
    LtxBridgeKind,
    LtxMotionTreatment,
    LtxRenderRequest,
    LtxSourceView,
    MultiShotInput,
    MultiShotVideoRequest,
    PropertyShotRole,
)
from listing_to_reel.video.service import (
    StableVideoDiffusionGenerator,
    evaluate_hero_video,
    generate_hero_video,
    interpolate_hero_video,
    load_video_generator_config,
    plan_multishot_ltx_video,
)

app = typer.Typer(no_args_is_help=True, help="Listing-to-Reel development commands.")


lora_app = typer.Typer(no_args_is_help=True, help="Phase 8 image-editing LoRA readiness commands.")
app.add_typer(lora_app, name="lora")


@lora_app.command("assess")
def assess_lora(
    dataset_manifest: Path = typer.Option(..., "--dataset-manifest", exists=True, readable=True),
    evidence: Path = typer.Option(..., "--evidence", exists=True, readable=True),
    config: Path = typer.Option(
        Path("configs/lora_pilot.yaml"), "--config", exists=True, readable=True
    ),
    output_dir: Path = typer.Option(Path("runs/lora-readiness"), "--output-dir"),
) -> None:
    """Assess whether a licensed paired dataset justifies one frozen-base LoRA pilot."""
    dataset = LoRADatasetManifest.model_validate_json(dataset_manifest.read_text(encoding="utf-8"))
    readiness_evidence = ReadinessEvidence.model_validate_json(evidence.read_text(encoding="utf-8"))
    report = assess_lora_readiness(
        LoRAReadinessRequest(
            dataset=dataset,
            evidence=readiness_evidence,
            configuration=load_lora_pilot_config(config),
        )
    )
    run_dir = output_dir / report.report_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(report.model_dump_json(indent=2))


@app.command("config-check")
def config_check(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
    """Validate and print a resolved runtime configuration."""
    runtime_config = load_runtime_config(config)
    typer.echo(runtime_config.model_dump_json(indent=2))


@app.command("environment")
def environment() -> None:
    """Print the current reproducibility environment snapshot."""
    snapshot = collect_environment_snapshot()
    typer.echo(snapshot.model_dump_json(indent=2))


benchmark_app = typer.Typer(no_args_is_help=True, help="Phase 6 CUDA LTX benchmark commands.")
app.add_typer(benchmark_app, name="benchmark")


def _benchmark_stages(values: list[str]) -> list[BenchmarkStage]:
    stages: list[BenchmarkStage] = []
    for value in values:
        name, separator, duration_text = value.partition("=")
        if not name or not separator:
            raise typer.BadParameter("Each --stage must be name=seconds.")
        try:
            stages.append(BenchmarkStage(name=name, duration_seconds=float(duration_text)))
        except ValueError as error:
            raise typer.BadParameter("Benchmark stage seconds must be numeric.") from error
    return stages


@benchmark_app.command("record-ltx")
def benchmark_record_ltx(
    label: str = typer.Option(..., "--label", min=1),
    cohort: str = typer.Option(..., "--cohort", min=1),
    render_manifest: Path = typer.Option(..., "--render-manifest", exists=True, readable=True),
    quality_report: Path = typer.Option(..., "--quality-report", exists=True, readable=True),
    stage: list[str] = typer.Option(
        ...,
        "--stage",
        help="Measured stage as name=seconds; include generation, repeat for load/qa/assembly.",
    ),
    runtime_config: Path = typer.Option(
        Path("configs/remote_cuda.yaml"), "--runtime-config", exists=True, readable=True
    ),
    profile: str = typer.Option("remote_cuda", "--profile"),
    output_dir: Path = typer.Option(
        Path("runs/benchmarks"), help="Ignored benchmark output directory."
    ),
) -> None:
    """Record one hardware-specific LTX benchmark from measured stage timings."""
    profiles = load_runtime_config(runtime_config).runtime_profiles
    if profile not in profiles:
        raise typer.BadParameter(f"Unknown runtime profile: {profile}")
    record = record_ltx_benchmark(
        label=label,
        cohort=cohort,
        render_manifest_path=render_manifest,
        quality_report_path=quality_report,
        runtime_profile_name=profile,
        runtime_profile=profiles[profile],
        stages=_benchmark_stages(stage),
        output_dir=output_dir,
    )
    typer.echo(record.model_dump_json(indent=2))


@benchmark_app.command("compare-ltx")
def benchmark_compare_ltx(
    baseline: Path = typer.Option(..., "--baseline", exists=True, readable=True),
    candidate: Path = typer.Option(..., "--candidate", exists=True, readable=True),
    output_dir: Path = typer.Option(
        Path("runs/benchmarks"), help="Ignored comparison output directory."
    ),
) -> None:
    """Compare two same-workload LTX benchmarks and emit an acceptance decision."""
    comparison = compare_ltx_benchmarks(baseline, candidate, output_dir)
    typer.echo(comparison.model_dump_json(indent=2))


reel_app = typer.Typer(no_args_is_help=True, help="Deterministic reel assembly commands.")
app.add_typer(reel_app, name="reel")


@reel_app.command("assemble")
def assemble(
    image: list[Path] = typer.Option(
        ..., "--image", help="Ordered source image; repeat per slide."
    ),
    output_dir: Path = typer.Option(Path("runs"), help="Directory for ignored reel artifacts."),
) -> None:
    """Create a fixed 9:16 listing reel with no model inference."""
    manifest = assemble_reel(ReelRequest(source_paths=image, output_dir=output_dir))
    typer.echo(manifest.model_dump_json(indent=2))


analysis_app = typer.Typer(no_args_is_help=True, help="Deterministic input and output analysis.")
app.add_typer(analysis_app, name="analyze")


@analysis_app.command("input")
def analyze_input(
    image: list[Path] = typer.Option(
        ...,
        "--image",
        exists=True,
        readable=True,
        help="Source image; repeat for one property listing.",
    ),
    config: Path = typer.Option(
        Path("configs/input_quality.yaml"), "--config", exists=True, readable=True
    ),
    output_dir: Path = typer.Option(Path("runs/input-quality"), help="Report output directory."),
) -> None:
    """Evaluate one or more source photos before generative processing."""
    quality_config = load_input_quality_config(config)
    reports = [analyze_input_image(path, quality_config, output_dir) for path in image]
    typer.echo(json.dumps([report.model_dump(mode="json") for report in reports], indent=2))


edit_app = typer.Typer(no_args_is_help=True, help="Pretrained image-editing baseline commands.")
app.add_typer(edit_app, name="edit")


@edit_app.command("image")
def edit_image(
    image: Path = typer.Option(..., "--image", exists=True, readable=True),
    input_quality_report: Path = typer.Option(
        ..., "--input-quality-report", exists=True, readable=True
    ),
    instruction: str = typer.Option(..., "--instruction", min=10),
    runtime_config: Path = typer.Option(
        Path("configs/local_mps.yaml"), "--runtime-config", exists=True, readable=True
    ),
    profile: str = typer.Option("local_mps", "--profile"),
    config: Path = typer.Option(
        Path("configs/image_editing.yaml"), "--config", exists=True, readable=True
    ),
    seed: int = typer.Option(1, "--seed", min=0),
    output_dir: Path = typer.Option(
        Path("runs/edits"), help="Candidate artifact output directory."
    ),
) -> None:
    """Generate InstructPix2Pix candidates for an accepted source photo."""
    runtime_profiles = load_runtime_config(runtime_config).runtime_profiles
    if profile not in runtime_profiles:
        raise typer.BadParameter(f"Unknown runtime profile: {profile}")
    request = EditRequest(
        source_path=image,
        input_quality_report_path=input_quality_report,
        instruction=instruction,
        seed=seed,
        output_dir=output_dir,
        runtime_profile_name=profile,
        runtime_profile=runtime_profiles[profile],
        configuration=load_image_editor_config(config),
    )
    manifest = generate_edit_candidates(request, InstructPix2PixEditor())
    typer.echo(manifest.model_dump_json(indent=2))


evaluate_app = typer.Typer(no_args_is_help=True, help="Phase 4 candidate evaluation commands.")
app.add_typer(evaluate_app, name="evaluate")


@evaluate_app.command("images")
def evaluate_images(
    edit_run_manifest: Path = typer.Option(..., "--edit-run-manifest", exists=True, readable=True),
    config: Path = typer.Option(
        Path("configs/evaluation.yaml"), "--config", exists=True, readable=True
    ),
    output_dir: Path = typer.Option(
        Path("runs/evaluations"), help="Ignored Phase 4 report output directory."
    ),
) -> None:
    """Evaluate one complete Phase 3 edit run and export a blinded review worksheet."""
    report = evaluate_edit_run(edit_run_manifest, load_evaluation_config(config), output_dir)
    typer.echo(report.model_dump_json(indent=2))


@evaluate_app.command("import-review")
def evaluate_import_review(
    evaluation_report: Path = typer.Option(..., "--evaluation-report", exists=True, readable=True),
    worksheet: Path = typer.Option(..., "--worksheet", exists=True, readable=True),
    output_dir: Path = typer.Option(
        Path("runs/evaluations"), help="Ignored Phase 4 decision output directory."
    ),
) -> None:
    """Record a completed human decision from the exported review worksheet."""
    decision = import_human_review(evaluation_report, worksheet, output_dir)
    typer.echo(decision.model_dump_json(indent=2))


video_app = typer.Typer(no_args_is_help=True, help="Phase 5 hero-video generation and QA commands.")
app.add_typer(video_app, name="video")


@video_app.command("plan-ltx-multishot")
def plan_ltx_multishot(
    property_id: str = typer.Option(..., "--property-id", min=1),
    shot: list[str] = typer.Option(
        ...,
        "--shot",
        help="Approved view as role=path/to/final-decision.json; repeat four to six times.",
    ),
    output_dir: Path = typer.Option(
        Path("runs/video-plans"), help="Ignored LTX multi-shot plan output directory."
    ),
) -> None:
    """Plan four to six independently generated LTX shots for one property."""
    planned_inputs = []
    for value in shot:
        role, separator, decision_path = value.partition("=")
        if not separator or not decision_path:
            raise typer.BadParameter("Each --shot must be role=path/to/final-decision.json.")
        try:
            planned_inputs.append(
                MultiShotInput(role=PropertyShotRole(role), final_decision_path=Path(decision_path))
            )
        except ValueError as error:
            choices = ", ".join(item.value for item in PropertyShotRole)
            message = f"Unknown shot role '{role}'. Use one of: {choices}."
            raise typer.BadParameter(message) from error
    request = MultiShotVideoRequest(
        property_id=property_id,
        shots=planned_inputs,
        output_dir=output_dir,
    )
    manifest = plan_multishot_ltx_video(request)
    typer.echo(manifest.model_dump_json(indent=2))


@video_app.command("render-ltx")
def render_ltx(
    property_id: str = typer.Option(..., "--property-id", min=1),
    source: list[str] = typer.Option(
        ...,
        "--source",
        help=(
            "View as name=path,to_treatment; treatment is "
            "slow_lateral_gimbal_glide or gentle_dolly_in."
        ),
    ),
    bridge_candidate: list[str] = typer.Option(
        [],
        "--bridge-candidate",
        help=(
            "Explicitly selected compatible pair, optionally with its invented duration in "
            "seconds (for example patio-to-backyard=3.5). Valid range: 2–6 seconds."
        ),
    ),
    bridge_pair: list[str] = typer.Option(
        [],
        "--bridge-pair",
        help="Custom pair as id=from_view,to_view,spatial_overlap|lighting_only.",
    ),
    config: Path = typer.Option(
        Path("configs/ltx_comfyui.yaml"), "--config", exists=True, readable=True
    ),
    output_dir: Path = typer.Option(
        Path("runs/ltx-videos"), help="Ignored LTX candidate output directory."
    ),
) -> None:
    """Render native-16:9 LTX source-anchored clips through ComfyUI only."""
    source_views = []
    for value in source:
        name, separator, remainder = value.partition("=")
        path_text, treatment_separator, treatment_text = remainder.rpartition(",")
        if not separator or not treatment_separator or not name or not path_text:
            raise typer.BadParameter("Each --source must be name=path,to_treatment.")
        try:
            source_views.append(
                LtxSourceView(
                    name=name,
                    source_path=Path(path_text),
                    treatment=LtxMotionTreatment(treatment_text),
                )
            )
        except ValueError as error:
            choices = ", ".join(item.value for item in LtxMotionTreatment)
            raise typer.BadParameter(f"Unknown LTX treatment; use one of: {choices}.") from error
    bridge_candidate_ids: list[str] = []
    bridge_duration_seconds: dict[str, float] = {}
    custom_bridge_candidates: list[LtxBridgeCandidate] = []
    for value in bridge_pair:
        candidate_id, separator, pair_text = value.partition("=")
        parts = pair_text.split(",")
        if not candidate_id or not separator or len(parts) != 3:
            raise typer.BadParameter(
                "Each --bridge-pair must be id=from_view,to_view,spatial_overlap|lighting_only."
            )
        from_view, to_view, kind_text = parts
        try:
            kind = LtxBridgeKind(kind_text)
        except ValueError as error:
            raise typer.BadParameter(
                "Bridge kind must be spatial_overlap or lighting_only."
            ) from error
        custom_bridge_candidates.append(
            LtxBridgeCandidate(
                candidate_id=candidate_id,
                kind=kind,
                from_view=from_view,
                to_view=to_view,
                prompt="User-selected compatible source-view pair; human-review candidate only.",
                reason=(
                    "Reject if architecture, landscaping, perspective, or geometry changes; "
                    "use an intentional cut when overlap is not sufficient."
                ),
            )
        )
    for value in bridge_candidate:
        candidate_id, separator, duration_text = value.partition("=")
        if not candidate_id:
            raise typer.BadParameter("Each --bridge-candidate requires a candidate name.")
        if separator:
            try:
                duration_seconds = float(duration_text)
            except ValueError as error:
                raise typer.BadParameter("Bridge duration must be a number of seconds.") from error
            if not 2.0 <= duration_seconds <= 6.0:
                raise typer.BadParameter("Bridge duration must be between 2 and 6 seconds.")
            bridge_duration_seconds[candidate_id] = duration_seconds
        bridge_candidate_ids.append(candidate_id)
    manifest = render_ltx_views(
        LtxRenderRequest(
            property_id=property_id,
            source_views=source_views,
            bridge_candidate_ids=bridge_candidate_ids,
            bridge_duration_seconds=bridge_duration_seconds,
            bridge_candidates=custom_bridge_candidates,
            configuration=load_ltx_comfyui_config(config),
            output_dir=output_dir,
        )
    )
    typer.echo(manifest.model_dump_json(indent=2))


@video_app.command("qa-ltx")
def qa_ltx(
    render_manifest: Path = typer.Option(..., "--render-manifest", exists=True, readable=True),
    output_dir: Path = typer.Option(
        Path("runs/ltx-quality"), help="Ignored LTX QA report directory."
    ),
) -> None:
    """Perform structural/temporal screening and export a mandatory human-review worksheet."""
    report = evaluate_ltx_render(render_manifest, output_dir)
    typer.echo(report.model_dump_json(indent=2))


@video_app.command("assemble-ltx")
def assemble_ltx(
    render_manifest: Path = typer.Option(..., "--render-manifest", exists=True, readable=True),
    accepted_clip: list[str] = typer.Option(
        ...,
        "--accepted-clip",
        help="Human-approved clip name from the LTX render manifest; repeat in edit order.",
    ),
    output_dir: Path = typer.Option(
        Path("runs/ltx-reels"), help="Ignored portrait delivery output directory."
    ),
) -> None:
    """Create a portrait reel from explicitly human-approved landscape candidates only."""
    manifest = assemble_ltx_portrait_reel(render_manifest, accepted_clip, output_dir)
    typer.echo(manifest.model_dump_json(indent=2))


@video_app.command("plan-ltx-reel")
def plan_ltx_reel(
    render_manifest: Path = typer.Option(..., "--render-manifest", exists=True, readable=True),
    total_seconds: float = typer.Option(20.0, "--total-seconds", min=8.0, max=120.0),
    bridge_seconds: float = typer.Option(3.0, "--bridge-seconds", min=2.0, max=6.0),
    bridge_duration: list[str] = typer.Option(
        [],
        "--bridge-duration",
        help="Optional per-bridge override as candidate=seconds; valid range: 2–6 seconds.",
    ),
    scene_fade_seconds: float = typer.Option(
        0.45,
        "--scene-fade-seconds",
        min=0.2,
        max=1.0,
        help="Cinematic cross-dissolve length between unrelated property areas.",
    ),
    bridge: list[str] = typer.Option(
        ...,
        "--bridge",
        help="Human-selected compatible generated bridge candidate; repeat in reel order.",
    ),
    output_dir: Path = typer.Option(
        Path("runs/ltx-timed-reels"), help="Ignored timing-plan output directory."
    ),
) -> None:
    """Plan exact reel pacing from a desired total length and bridge duration."""
    duration_overrides: dict[str, float] = {}
    for value in bridge_duration:
        candidate_id, separator, duration_text = value.partition("=")
        if not candidate_id or not separator:
            raise typer.BadParameter("Each --bridge-duration must be candidate=seconds.")
        try:
            duration = float(duration_text)
        except ValueError as error:
            raise typer.BadParameter("Bridge duration must be a number of seconds.") from error
        if not 2.0 <= duration <= 6.0:
            raise typer.BadParameter("Bridge duration must be between 2 and 6 seconds.")
        duration_overrides[candidate_id] = duration
    unknown_overrides = set(duration_overrides) - set(bridge)
    if unknown_overrides:
        raise typer.BadParameter(
            "Every --bridge-duration must name a selected --bridge: "
            + ", ".join(sorted(unknown_overrides))
        )
    plan = plan_ltx_timed_reel(
        render_manifest,
        bridge,
        total_seconds,
        duration_overrides or bridge_seconds,
        output_dir,
        scene_fade_seconds,
    )
    typer.echo(plan.model_dump_json(indent=2))


@video_app.command("assemble-ltx-timed-reel")
def assemble_ltx_timed_reel_command(
    plan: Path = typer.Option(..., "--plan", exists=True, readable=True),
    output_dir: Path = typer.Option(
        Path("runs/ltx-timed-reels"), help="Ignored rendered timed-reel output directory."
    ),
) -> None:
    """Render a saved bridge-and-clip pacing plan into a complete portrait candidate reel."""
    manifest = assemble_ltx_timed_reel(plan, output_dir)
    typer.echo(manifest.model_dump_json(indent=2))


@video_app.command("generate")
def generate_video(
    final_decision: Path = typer.Option(..., "--final-decision", exists=True, readable=True),
    runtime_config: Path = typer.Option(
        Path("configs/remote_cuda.yaml"), "--runtime-config", exists=True, readable=True
    ),
    profile: str = typer.Option("remote_cuda", "--profile"),
    config: Path = typer.Option(
        Path("configs/video_generation.yaml"), "--config", exists=True, readable=True
    ),
    seed: int = typer.Option(1, "--seed", min=0),
    output_dir: Path = typer.Option(
        Path("runs/videos"), help="Ignored hero-video output directory."
    ),
) -> None:
    """Generate a CUDA-only, four-second hero-video candidate from an approved image."""
    profiles = load_runtime_config(runtime_config).runtime_profiles
    if profile not in profiles:
        raise typer.BadParameter(f"Unknown runtime profile: {profile}")
    request = HeroVideoRequest(
        final_decision_path=final_decision,
        seed=seed,
        output_dir=output_dir,
        runtime_profile_name=profile,
        runtime_profile=profiles[profile],
        configuration=load_video_generator_config(config),
    )
    manifest = generate_hero_video(request, StableVideoDiffusionGenerator())
    typer.echo(manifest.model_dump_json(indent=2))


@video_app.command("qa")
def video_qa(
    video_manifest: Path = typer.Option(..., "--video-manifest", exists=True, readable=True),
    output_dir: Path = typer.Option(
        Path("runs/video-quality"), help="Ignored QA report directory."
    ),
) -> None:
    """Extract temporal QA evidence and a reviewer worksheet for a hero-video candidate."""
    report = evaluate_hero_video(video_manifest, output_dir)
    typer.echo(report.model_dump_json(indent=2))


@video_app.command("interpolate")
def interpolate_video(
    video_manifest: Path = typer.Option(..., "--video-manifest", exists=True, readable=True),
    target_fps: int = typer.Option(30, "--target-fps", min=24, max=60),
    output_dir: Path = typer.Option(
        Path("runs/interpolated-videos"), help="Ignored interpolation output directory."
    ),
) -> None:
    """Create a four-second 24/30fps delivery candidate with motion interpolation."""
    manifest = interpolate_hero_video(
        video_manifest, InterpolationConfig(target_fps=target_fps), output_dir
    )
    typer.echo(manifest.model_dump_json(indent=2))
