"""Package initialization module for cosmica."""

from typing import Annotated

from typing_extensions import Doc


def main() -> None:
    """Print a greeting message."""
    print("Hello from cosmica!")  # noqa: T201


def square(x: Annotated[int, Doc("The integer to square.")]) -> Annotated[int, Doc("The square of the integer.")]:
    """Return the square of the given integer."""
    return x * x
