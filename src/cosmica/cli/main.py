import importlib.metadata
from typing import Annotated

import typer

from .plot import main as plot_main

app = typer.Typer()


def _version_callback(value: bool) -> None:  # noqa: FBT001
    if value:
        version = importlib.metadata.version("cosmica")
        typer.echo(f"COSMICA CLI {version}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", help="Show version information", is_eager=True, callback=_version_callback),
    ] = None,
) -> None:
    """COSMICA CLI."""


app.command("plot")(plot_main)
