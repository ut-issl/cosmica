from typing import cast

import numpy as np
import pytest

from cosmica.models import Constellation, Gateway, GatewayOGS, Scenario


def test_gateway_validation_rejects_invalid_latitude() -> None:
    with pytest.raises(AssertionError):
        Gateway(
            id=0,
            latitude=np.deg2rad(120.0),
            longitude=np.deg2rad(0.0),
            minimum_elevation=np.deg2rad(10.0),
        )


def test_gateway_validation_rejects_non_integer_terminals() -> None:
    with pytest.raises(AssertionError):
        Gateway(
            id=0,
            latitude=np.deg2rad(10.0),
            longitude=np.deg2rad(0.0),
            minimum_elevation=np.deg2rad(10.0),
            n_terminals=cast("int", 1.5),
        )


def _gateway_ogs(
    *,
    latitude: float = np.deg2rad(10.0),
    minimum_elevation: float = np.deg2rad(10.0),
    aperture_size: float = 1.0,
    rytov_variance: float = 0.5,
) -> GatewayOGS[int]:
    return GatewayOGS(
        id=0,
        latitude=latitude,
        longitude=np.deg2rad(0.0),
        minimum_elevation=minimum_elevation,
        aperture_size=aperture_size,
        rytov_variance=rytov_variance,
    )


def test_gateway_ogs_is_a_gateway() -> None:
    ogs = _gateway_ogs(aperture_size=0.4, rytov_variance=0.2)

    assert isinstance(ogs, Gateway)
    assert (ogs.aperture_size, ogs.rytov_variance) == (0.4, 0.2)


def test_gateway_ogs_keeps_its_own_global_id() -> None:
    assert _gateway_ogs().global_id == "GW_OGS-0"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"latitude": np.deg2rad(120.0)}, id="gateway-latitude"),
        pytest.param({"minimum_elevation": np.deg2rad(-1.0)}, id="gateway-minimum-elevation"),
        pytest.param({"aperture_size": 0.0}, id="aperture-size"),
        pytest.param({"rytov_variance": -0.1}, id="rytov-variance"),
    ],
)
def test_gateway_ogs_validation_rejects_invalid_fields(kwargs: dict[str, float]) -> None:
    with pytest.raises(AssertionError):
        _gateway_ogs(**kwargs)


def test_gateway_and_gateway_ogs_with_same_id_are_distinct_nodes() -> None:
    gateway = Gateway(id=0, latitude=np.deg2rad(10.0), longitude=np.deg2rad(0.0), minimum_elevation=np.deg2rad(10.0))
    ogs = _gateway_ogs()

    assert gateway != ogs
    assert ogs != gateway
    assert ogs == _gateway_ogs(aperture_size=0.4)
    assert len({gateway, ogs}) == 2


def test_scenario_accepts_gateway_ogs_as_gateway() -> None:
    gateway = Gateway(id=0, latitude=np.deg2rad(10.0), longitude=np.deg2rad(0.0), minimum_elevation=np.deg2rad(10.0))
    ogs = _gateway_ogs()

    scenario = Scenario(
        time=np.array([np.datetime64("2026-01-01T00:00:00")]),
        constellation=Constellation(satellites={}),
        gateways=[gateway, ogs],
    )

    assert scenario.gateways == [gateway, ogs]
