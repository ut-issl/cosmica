from collections.abc import Mapping

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest

from cosmica.dtos import DynamicsData
from cosmica.models import (
    CircularSatelliteOrbitModel,
    ConstellationSatellite,
    Gateway,
    Satellite,
    UserSatellite,
)
from cosmica.topology import ElevationBasedG2USTopologyBuilder, ManualG2USTopologyBuilder
from cosmica.utils.constants import EARTH_RADIUS

_TIME = np.datetime64("2026-01-01") + np.arange(3).astype("timedelta64[s]")
_ORBIT = CircularSatelliteOrbitModel(
    semi_major_axis=7_000_000.0,
    inclination=0.0,
    raan=0.0,
    phase_at_epoch=0.0,
    epoch=_TIME[0],
)
_ZENITH = np.array([7_000_000.0, 0.0, 0.0])
_NADIR = -_ZENITH


def _gateway(*, minimum_elevation: float = 0.0) -> Gateway[str]:
    return Gateway(id="gateway", latitude=0.0, longitude=0.0, minimum_elevation=minimum_elevation)


def _dynamics_data(positions_ecef: Mapping[Satellite, npt.NDArray[np.floating]]) -> DynamicsData:
    n_time = len(_TIME)
    zero_vectors: dict[Satellite, npt.NDArray[np.floating]] = {
        satellite: np.zeros((n_time, 3)) for satellite in positions_ecef
    }
    return DynamicsData(
        time=_TIME,
        dcm_eci2ecef=np.repeat(np.eye(3)[None, :, :], n_time, axis=0),
        satellite_position_eci=zero_vectors,
        satellite_velocity_eci=zero_vectors,
        satellite_position_ecef={
            satellite: np.broadcast_to(position, (n_time, 3)).copy() for satellite, position in positions_ecef.items()
        },
        satellite_attitude_angular_velocity_eci=zero_vectors,
        sun_direction_eci=np.zeros((n_time, 3)),
        sun_direction_ecef=np.zeros((n_time, 3)),
    )


def _edges(graph: nx.DiGraph) -> set[tuple[object, object]]:
    return set(graph.edges)


def test_elevation_based_g2us_ignores_satellites_outside_supplied_collection() -> None:
    gateway = _gateway()
    visible = UserSatellite(id="visible", orbit=_ORBIT)
    hidden = UserSatellite(id="hidden", orbit=_ORBIT)
    # Dynamics data also holds a constellation satellite that is not a user satellite.
    other = ConstellationSatellite(id="other", orbit=_ORBIT)
    dynamics_data = _dynamics_data({other: _ZENITH, visible: _ZENITH, hidden: _NADIR})

    graphs = ElevationBasedG2USTopologyBuilder().build(
        user_satellites=(visible, hidden),
        ground_nodes=(gateway,),
        dynamics_data=dynamics_data,
    )

    assert len(graphs) == len(_TIME)
    for graph in graphs:
        assert set(graph.nodes) == {gateway, visible, hidden}
        assert _edges(graph) == {(gateway, visible), (visible, gateway)}


def test_elevation_based_g2us_follows_visibility_over_time() -> None:
    gateway = _gateway()
    satellite = UserSatellite(id="satellite", orbit=_ORBIT)
    dynamics_data = _dynamics_data({satellite: _ZENITH})
    dynamics_data.satellite_position_ecef[satellite][1:] = _NADIR

    graphs = ElevationBasedG2USTopologyBuilder().build(
        user_satellites=(satellite,),
        ground_nodes=(gateway,),
        dynamics_data=dynamics_data,
    )

    assert [graph.has_edge(gateway, satellite) for graph in graphs] == [True, False, False]


@pytest.mark.parametrize(
    ("minimum_elevation_deg", "expected_linked"),
    [(40.0, True), (50.0, False)],
)
def test_elevation_based_g2us_applies_minimum_elevation(
    minimum_elevation_deg: float,
    *,
    expected_linked: bool,
) -> None:
    gateway = _gateway(minimum_elevation=np.deg2rad(minimum_elevation_deg))
    satellite = UserSatellite(id="satellite", orbit=_ORBIT)
    # 45 degrees above the horizon of a ground node on the equator at the prime meridian.
    offset = 1_000e3
    dynamics_data = _dynamics_data({satellite: np.array([EARTH_RADIUS + offset, offset, 0.0])})

    graphs = ElevationBasedG2USTopologyBuilder().build(
        user_satellites=(satellite,),
        ground_nodes=(gateway,),
        dynamics_data=dynamics_data,
    )

    for graph in graphs:
        assert graph.has_edge(gateway, satellite) is expected_linked
        assert graph.has_edge(satellite, gateway) is expected_linked


def test_manual_g2us_uses_custom_connections_at_every_time_step() -> None:
    gateway = _gateway()
    connected = UserSatellite(id="connected", orbit=_ORBIT)
    unconnected = UserSatellite(id="unconnected", orbit=_ORBIT)
    # Geometry is ignored: the connected satellite is below the horizon.
    dynamics_data = _dynamics_data({connected: _NADIR, unconnected: _ZENITH})

    graphs = ManualG2USTopologyBuilder(custom_connections={gateway: connected}).build(
        user_satellites=(connected, unconnected),
        ground_nodes=(gateway,),
        dynamics_data=dynamics_data,
    )

    assert len(graphs) == len(_TIME)
    for graph in graphs:
        assert set(graph.nodes) == {gateway, connected, unconnected}
        assert _edges(graph) == {(gateway, connected), (connected, gateway)}
