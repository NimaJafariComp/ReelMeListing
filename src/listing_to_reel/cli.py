"""Command-line entry points for reproducibility checks."""

from pathlib import Path

import typer

from listing_to_reel.core.config import load_runtime_config
from listing_to_reel.core.environment import collect_environment_snapshot

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
