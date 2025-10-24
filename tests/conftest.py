from pathlib import Path

import pytest

from cosmica.dynamics import MultiOrbitalPlaneConstellation
from cosmica.models import Gateway

CONSTELLATION_FILE = Path(__file__).parent / "data" / "constellation-test.toml"
GATEWAY_FILE = Path(__file__).parent / "data" / "gateways-test.toml"

assert all(path.exists() for path in (CONSTELLATION_FILE, GATEWAY_FILE))


@pytest.fixture
def constellation():
    return MultiOrbitalPlaneConstellation.from_toml_file(CONSTELLATION_FILE)


@pytest.fixture
def gateways():
    return Gateway.from_toml_file(GATEWAY_FILE)
