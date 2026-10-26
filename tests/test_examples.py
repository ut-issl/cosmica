import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"


@pytest.mark.parametrize("script", ["basic_simulation.py", "orbit_propagation.py"])
def test_example_runs(script: str) -> None:
    subprocess.run(  # noqa: S603
        [sys.executable, EXAMPLES_DIR / script],
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
