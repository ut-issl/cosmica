from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

from cosmica.models import Gateway
from cosmica.scenario.gateway_network import DEFAULT_GATEWAYS, build_default_gateway_network, build_gateway_network


def test_build_default_gateway_network_count() -> None:
    gateways = build_default_gateway_network(n_stations=2)
    assert len(gateways) == 2
    assert all(isinstance(gateway, Gateway) for gateway in gateways)


def test_build_default_gateway_network_indexes() -> None:
    gateways = build_default_gateway_network(indexes=[1, 3])
    gateway_ids = [gateway.id for gateway in gateways]
    assert gateway_ids == [1, 3]
    for gateway in gateways:
        expected = DEFAULT_GATEWAYS[gateway.id]
        assert np.isclose(gateway.latitude, np.deg2rad(expected["lat_deg"]))
        assert np.isclose(gateway.longitude, np.deg2rad(expected["lon_deg"]))
        assert np.isclose(gateway.minimum_elevation, np.deg2rad(expected["min_el_deg"]))


def test_build_gateway_network_optional_n_terminals() -> None:
    gateway_map: Mapping[Hashable, Mapping[str, float | int]] = {
        "gw-1": {
            "lat_deg": 10.0,
            "lon_deg": 20.0,
            "min_el_deg": 5.0,
        },
    }
    gateways = build_gateway_network(gateway_map)
    assert len(gateways) == 1
    assert gateways[0].n_terminals == 1


def test_build_gateway_network_converts_degrees_to_radians() -> None:
    gateway_map: Mapping[Hashable, Mapping[str, float | int]] = {
        0: {
            "lat_deg": 45.0,
            "lon_deg": -90.0,
            "min_el_deg": 30.0,
            "n_terminals": 2,
            "altitude": 500.0,
        },
    }
    gateways = build_gateway_network(gateway_map)
    gateway = gateways[0]
    assert np.isclose(gateway.latitude, np.deg2rad(45.0))
    assert np.isclose(gateway.longitude, np.deg2rad(-90.0))
    assert np.isclose(gateway.minimum_elevation, np.deg2rad(30.0))
    assert gateway.altitude == 500.0
    assert gateway.n_terminals == 2


def test_build_gateway_network_missing_required_field() -> None:
    gateway_map: Mapping[Hashable, Mapping[str, float | int]] = {
        0: {
            "lat_deg": 45.0,
            "lon_deg": -90.0,
        },
    }
    with pytest.raises(AssertionError):
        build_gateway_network(gateway_map)


def test_build_gateway_network_invalid_n_terminals() -> None:
    gateway_map: Mapping[Hashable, Mapping[str, float | int]] = {
        0: {
            "lat_deg": 45.0,
            "lon_deg": -90.0,
            "min_el_deg": 30.0,
            "n_terminals": 0,
        },
    }
    with pytest.raises(AssertionError):
        build_gateway_network(gateway_map)
