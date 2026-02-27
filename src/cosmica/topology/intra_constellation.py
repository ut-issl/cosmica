__all__ = [
    "ConstellationTimeSeriesTopologyBuilder",
    "ConstellationTopologyBuilder",
    "ManhattanTimeSeriesTopologyBuilder",
    "ManhattanTopologyBuilder",
    "build_manhattan_time_series_topology",
    "build_manhattan_topology",
]
from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np

from cosmica.dtos import DynamicsData
from cosmica.models import Constellation, ConstellationSatellite
from cosmica.utils.vector import normalize, unit_vector_to_azimuth_elevation

type PlaneId = int
type InPlaneIndex = int


class ConstellationTopologyBuilder[TConstellation: Constellation, TGraph: nx.Graph](ABC):
    @abstractmethod
    def build(
        self,
        *,
        constellation: TConstellation,
    ) -> TGraph: ...


class ConstellationTimeSeriesTopologyBuilder[TConstellation: Constellation, TGraph: nx.Graph](ABC):
    @abstractmethod
    def build(
        self,
        *,
        constellation: TConstellation,
        dynamics_data: DynamicsData,
    ) -> list[TGraph]: ...


class ManhattanTopologyBuilder(
    ConstellationTopologyBuilder[Constellation[tuple[PlaneId, InPlaneIndex]], nx.Graph],
):
    def __init__(
        self,
        *,
        inter_plane_offset: int = 0,
        last_to_first_plane_offset: int = 0,
    ) -> None:
        self.inter_plane_offset = inter_plane_offset
        self.last_to_first_plane_offset = last_to_first_plane_offset

    def build(
        self,
        *,
        constellation: Constellation[tuple[PlaneId, InPlaneIndex]],
    ) -> nx.Graph:
        return build_manhattan_topology(
            constellation,
            inter_plane_offset=self.inter_plane_offset,
            last_to_first_plane_offset=self.last_to_first_plane_offset,
        )


class ManhattanTimeSeriesTopologyBuilder(
    ConstellationTimeSeriesTopologyBuilder[Constellation[tuple[PlaneId, InPlaneIndex]], nx.Graph],
):
    def __init__(
        self,
        *,
        inter_plane_offset: int = 0,
        last_to_first_plane_offset: int = 0,
        max_latitude: float = np.deg2rad(90.0),
    ) -> None:
        self.inter_plane_offset = inter_plane_offset
        self.last_to_first_plane_offset = last_to_first_plane_offset
        self.max_latitude = max_latitude

    def build(
        self,
        *,
        constellation: Constellation[tuple[PlaneId, InPlaneIndex]],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        return build_manhattan_time_series_topology(
            constellation,
            dynamics_data=dynamics_data,
            inter_plane_offset=self.inter_plane_offset,
            last_to_first_plane_offset=self.last_to_first_plane_offset,
            max_latitude=self.max_latitude,
        )


# ---------------------------------------------------------------------------
# New functions using Constellation model
# ---------------------------------------------------------------------------


def _group_by_plane(
    constellation: Constellation[tuple[PlaneId, InPlaneIndex]],
) -> list[list[ConstellationSatellite]]:
    """Group satellites by plane and sort by in-plane index.

    Returns a list of planes (sorted by plane_id), where each plane is a list
    of satellite objects sorted by in_plane_index.

    The structural information comes entirely from the dict keys
    `(plane_id, in_plane_index)` — never from `satellite.id`.
    See the module docstring of `cosmica.models.constellation` for details
    on the two-ID distinction.
    """
    # Collect (plane_id, in_plane_index, satellite) triples from dict keys
    entries_dd: defaultdict[PlaneId, list[tuple[InPlaneIndex, ConstellationSatellite]]] = defaultdict(list)
    for (plane_id, in_plane_index), satellite in constellation.satellites.items():
        entries_dd[plane_id].append((in_plane_index, satellite))

    entries = dict(entries_dd)

    # Sort planes by plane_id, satellites within each plane by in_plane_index
    planes: list[list[ConstellationSatellite]] = [
        [sat for _, sat in sorted(entries[plane_id])] for plane_id in sorted(entries)
    ]

    return planes


def build_manhattan_topology(
    constellation: Constellation[tuple[PlaneId, InPlaneIndex]],
    *,
    inter_plane_offset: int = 0,
    last_to_first_plane_offset: int = 0,
) -> nx.Graph:
    """Build a Manhattan (grid) topology for a multi-plane constellation.

    The constellation must be parameterized as `Constellation[tuple[int, int]]`
    where each key is `(plane_id, in_plane_index)`. All structural information
    (plane membership, intra-plane ordering, inter-plane ordering) is derived
    from these dict keys — **not** from orbital parameters or `satellite.id`.

    All planes must have the same number of satellites.

    Args:
        constellation: Constellation with `(plane_id, in_plane_index)` keys.
        inter_plane_offset: Index offset when connecting adjacent planes.
        last_to_first_plane_offset: Additional offset for the wrap-around
            connection from the last plane back to the first.

    Returns:
        A networkx Graph with satellites as nodes and Manhattan grid edges.

    """
    planes = _group_by_plane(constellation)

    graph = nx.Graph()

    # Add all satellite objects as graph nodes.
    # We use the satellite *object* (not the structural key) as the graph node,
    # because graph nodes must be the same objects used as DynamicsData keys.
    for satellite in constellation.satellites.values():
        graph.add_node(satellite)

    # Intra-plane edges: ring connection within each plane
    for plane in planes:
        for i, satellite in enumerate(plane):
            src = satellite
            dst = plane[(i + 1) % len(plane)]
            graph.add_edge(src, dst)

    # Inter-plane edges: connect satellite at index i in plane p
    # to satellite at index (i + offset) in plane p+1
    for plane_idx in range(len(planes)):
        next_plane_idx = (plane_idx + 1) % len(planes)
        plane = planes[plane_idx]
        next_plane = planes[next_plane_idx]

        assert len(plane) == len(next_plane), (
            "Manhattan topology requires all planes to have the same number of satellites. "
            f"Plane {plane_idx} has {len(plane)}, plane {next_plane_idx} has {len(next_plane)}."
        )

        offset = inter_plane_offset
        if plane_idx == len(planes) - 1:
            offset += last_to_first_plane_offset

        for i, satellite in enumerate(plane):
            src = satellite
            dst = next_plane[(i + offset) % len(next_plane)]
            graph.add_edge(src, dst)

    return graph


def build_manhattan_time_series_topology(
    constellation: Constellation[tuple[int, int]],
    *,
    dynamics_data: DynamicsData,
    inter_plane_offset: int = 0,
    last_to_first_plane_offset: int = 0,
    max_latitude: float = np.deg2rad(90.0),
) -> list[nx.Graph]:
    """Build time-varying Manhattan topology, disabling inter-plane links near poles.

    Same as :func:`build_manhattan_topology`, but produces one graph per time
    step. Inter-plane edges are omitted at time steps where both endpoints
    exceed `max_latitude` (polar region avoidance).

    Args:
        constellation: Constellation with `(plane_id, in_plane_index)` keys.
        dynamics_data: Time-series dynamics data for satellite positions.
        inter_plane_offset: Index offset when connecting adjacent planes.
        last_to_first_plane_offset: Additional offset for the wrap-around
            connection from the last plane back to the first.
        max_latitude: Latitude threshold (radians) above which inter-plane
            links are disabled.

    Returns:
        A list of networkx Graphs, one per time step.

    """
    planes = _group_by_plane(constellation)

    def construct_graph(time_idx: int) -> nx.Graph:
        graph = nx.Graph()

        for satellite in constellation.satellites.values():
            graph.add_node(satellite)

        # Intra-plane edges: ring connection (same at every time step)
        for plane in planes:
            for i, satellite in enumerate(plane):
                src = satellite
                dst = plane[(i + 1) % len(plane)]
                graph.add_edge(src, dst)

        # Inter-plane edges: skip if both satellites are in the polar region
        for plane_idx in range(len(planes)):
            next_plane_idx = (plane_idx + 1) % len(planes)
            plane = planes[plane_idx]
            next_plane = planes[next_plane_idx]

            assert len(plane) == len(next_plane), (
                "Manhattan topology requires all planes to have the same number of satellites. "
                f"Plane {plane_idx} has {len(plane)}, plane {next_plane_idx} has {len(next_plane)}."
            )

            offset = inter_plane_offset
            if plane_idx == len(planes) - 1:
                offset += last_to_first_plane_offset

            for i, satellite in enumerate(plane):
                src = satellite
                dst = next_plane[(i + offset) % len(next_plane)]

                _, latitude_src = unit_vector_to_azimuth_elevation(
                    normalize(dynamics_data.satellite_position_ecef[src][time_idx]),
                )
                _, latitude_dst = unit_vector_to_azimuth_elevation(
                    normalize(dynamics_data.satellite_position_ecef[dst][time_idx]),
                )
                if abs(latitude_src) > max_latitude and abs(latitude_dst) > max_latitude:
                    continue

                graph.add_edge(src, dst)

        return graph

    return [construct_graph(time_idx) for time_idx in range(len(dynamics_data.time))]
