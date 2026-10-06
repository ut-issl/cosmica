import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from cosmica.dtos import DynamicsData
from cosmica.models import Constellation, ConstellationSatellite, GatewayOGS
from cosmica.visualization.equirectangular import draw_snapshot
from tests.factories import make_satellite


def test_draw_snapshot_draws_gateway_ogs_and_its_links_as_gateway() -> None:
    satellite = make_satellite(1)
    ogs = GatewayOGS(id=0, latitude=np.deg2rad(10.0), longitude=np.deg2rad(20.0), minimum_elevation=0.0)
    position = np.array([[7_000_000.0, 0.0, 0.0]])
    zero = np.zeros((1, 3))
    vectors: dict[ConstellationSatellite[int], npt.NDArray[np.floating]] = {satellite: zero}
    dynamics_data = DynamicsData(
        time=np.array([np.datetime64("2026-01-01T00:00:00")]),
        dcm_eci2ecef=np.eye(3)[None, :, :],
        satellite_position_eci={satellite: position},
        satellite_velocity_eci=vectors,
        satellite_position_ecef={satellite: position},
        satellite_attitude_angular_velocity_eci=vectors,
        sun_direction_eci=zero,
        sun_direction_ecef=zero,
    )
    graph = nx.DiGraph([(satellite, ogs), (ogs, satellite)])
    ax = Figure().add_subplot()

    draw_snapshot(
        graph=graph,
        constellation=Constellation(satellites={(0, 0): satellite}),
        dynamics_data=dynamics_data[0],
        ax=ax,
    )

    gateway_markers = [
        collection
        for collection in ax.collections
        if isinstance(collection, PathCollection) and collection.get_label() == "Gateway"
    ]
    assert len(gateway_markers) == 1
    np.testing.assert_allclose(np.asarray(gateway_markers[0].get_offsets(), dtype=np.float64), np.array([[20.0, 10.0]]))
    assert len([patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]) == 1
    assert "Feeder links" in [line.get_label() for line in ax.lines]
