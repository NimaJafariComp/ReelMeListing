"""Command-line entry points for reproducibility checks."""

import json
from pathlib import Path

import typer

from listing_to_reel.analysis.input_quality import analyze_input_image, load_input_quality_config
from listing_to_reel.core.config import load_runtime_config
from listing_to_reel.core.environment import collect_environment_snapshot
from listing_to_reel.editing.instruct_pix2pix import InstructPix2PixEditor
from listing_to_reel.editing.models import EditRequest
from listing_to_reel.editing.service import generate_edit_candidates, load_image_editor_config
from listing_to_reel.media.models import ReelRequest
from listing_to_reel.media.reel import assemble_reel

app = typer.Typer(no_args_is_help=True, help="Listing-to-Reel development commands.")


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
