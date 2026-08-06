import networkx as nx
import numpy as np
import numpy.typing as npt

from cosmica.dtos import DynamicsData
from cosmica.models import CircularSatelliteOrbitModel, Constellation, ConstellationSatellite, Gateway
from cosmica.topology import build_max_visibility_handover_g2c_topology

type _Satellite = ConstellationSatellite[str, CircularSatelliteOrbitModel]
type _TopologyFixture = tuple[list[nx.DiGraph], tuple[Gateway[str], ...], tuple[_Satellite, ...]]


def _build_topologies(
    visibility_by_satellite: tuple[tuple[bool, ...], ...],
    *,
    n_ground_nodes: int = 1,
) -> _TopologyFixture:
    n_time = len(visibility_by_satellite[0])
    assert all(len(visibility) == n_time for visibility in visibility_by_satellite)
    time = np.datetime64("2026-01-01") + np.arange(n_time).astype("timedelta64[s]")
    orbit = CircularSatelliteOrbitModel(
        semi_major_axis=7_000_000.0,
        inclination=0.0,
        raan=0.0,
        phase_at_epoch=0.0,
        epoch=time[0],
    )
    satellites = tuple(
        ConstellationSatellite(id=f"satellite-{satellite_index}", orbit=orbit)
        for satellite_index in range(len(visibility_by_satellite))
    )
    constellation = Constellation(satellites=dict(enumerate(satellites)))
    ground_nodes = tuple(
        Gateway(id=f"ground-{ground_index}", latitude=0.0, longitude=0.0, minimum_elevation=0.0)
        for ground_index in range(n_ground_nodes)
    )

    zero_vectors: dict[_Satellite, npt.NDArray[np.floating]] = {
        satellite: np.zeros((n_time, 3)) for satellite in satellites
    }
    satellite_position_ecef: dict[_Satellite, npt.NDArray[np.floating]] = {
        satellite: np.column_stack(
            (
                np.where(visibility, 7_000_000.0, -7_000_000.0),
                np.zeros((n_time, 2)),
            ),
        )
        for satellite, visibility in zip(satellites, visibility_by_satellite, strict=True)
    }
    dynamics_data = DynamicsData(
        time=time,
        dcm_eci2ecef=np.zeros((n_time, 3, 3)),
        satellite_position_eci=zero_vectors,
        satellite_velocity_eci=zero_vectors,
        satellite_position_ecef=satellite_position_ecef,
        satellite_attitude_angular_velocity_eci=zero_vectors,
        sun_direction_eci=np.zeros((n_time, 3)),
        sun_direction_ecef=np.zeros((n_time, 3)),
    )

    graphs = build_max_visibility_handover_g2c_topology(
        constellation,
        ground_nodes=ground_nodes,
        dynamics_data=dynamics_data,
    )
    return graphs, ground_nodes, satellites


def test_max_visibility_handover_leaves_contended_ground_node_unassigned() -> None:
    graphs, ground_nodes, (invisible_satellite, visible_satellite) = _build_topologies(
        ((False, False, False), (True, True, True)),
        n_ground_nodes=2,
    )

    for graph in graphs:
        assert set(graph.neighbors(ground_nodes[0])) == {visible_satellite}
        assert set(graph.neighbors(ground_nodes[1])) == set()
        assert graph.degree[invisible_satellite] == 0


def test_max_visibility_handover_leaves_ground_node_without_visibility_unassigned() -> None:
    graphs, (ground_node,), satellites = _build_topologies(
        ((False, False, False), (False, False, False)),
    )

    for graph in graphs:
        assert set(graph.neighbors(ground_node)) == set()
        assert all(graph.degree[satellite] == 0 for satellite in satellites)


def test_max_visibility_handover_preserves_connection_until_handover() -> None:
    graphs, (ground_node,), (first_satellite, second_satellite) = _build_topologies(
        ((True, True, False), (False, True, True)),
    )

    assert [set(graph.neighbors(ground_node)) for graph in graphs] == [
        {first_satellite},
        {first_satellite},
        {second_satellite},
    ]
