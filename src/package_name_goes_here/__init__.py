"""Package initialization module for package-name-goes-here."""

from typing import Annotated

from typing_extensions import Doc


def main() -> None:
    """Print a greeting message."""
    print("Hello from package-name-goes-here!")  # noqa: T201


def square(x: Annotated[int, Doc("The integer to square.")]) -> Annotated[int, Doc("The square of the integer.")]:
    """Return the square of the given integer."""
    return x * x
