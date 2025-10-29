import importlib.metadata
from typing import Annotated

import typer
from rich.console import Console

from .plot import main as plot_main

app = typer.Typer()

console_err = Console(stderr=True)


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
    """COSMICA CLI.

    Warning: The 'cosmica' CLI will be deprecated in future versions.
    """
    # Show deprecation of cosmica CLI in future version
    console_err.print(
        "[yellow]Warning: The 'cosmica' CLI will be deprecated in future versions.[/yellow]",
    )


app.command("plot")(plot_main)
