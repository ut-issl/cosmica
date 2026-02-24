__all__ = [
    "ConstellationTimeSeriesTopologyBuilder",
    "ConstellationTopologyBuilder",
    "ManhattanTimeSeriesTopologyBuilder",
    "ManhattanTopologyBuilder",
    "build_manhattan_time_series_topology",
    "build_manhattan_topology",
]
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from typing_extensions import deprecated

from cosmica.dtos import DynamicsData
from cosmica.dynamics import (
    CircularSatelliteOrbit,
    MultiOrbitalPlaneConstellation,
    SatelliteConstellation,
)
from cosmica.models import Constellation, ConstellationSatellite
from cosmica.utils.vector import normalize, unit_vector_to_azimuth_elevation


@deprecated("Use build_manhattan_topology() function instead.")
class ConstellationTopologyBuilder[TConstellation: SatelliteConstellation, TGraph: nx.Graph](ABC):
    @abstractmethod
    def build(
        self,
        *,
        constellation: TConstellation,
    ) -> TGraph: ...


@deprecated("Use build_manhattan_time_series_topology() function instead.")
class ConstellationTimeSeriesTopologyBuilder[TConstellation: SatelliteConstellation, TGraph: nx.Graph](ABC):
    @abstractmethod
    def build(
        self,
        *,
        constellation: TConstellation,
        dynamics_data: DynamicsData,
    ) -> list[TGraph]: ...


@deprecated("Use build_manhattan_topology() function instead.")
class ManhattanTopologyBuilder(
    ConstellationTopologyBuilder[MultiOrbitalPlaneConstellation[CircularSatelliteOrbit], nx.Graph],
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
        constellation: MultiOrbitalPlaneConstellation,
    ) -> nx.Graph:
        graph = nx.Graph()

        # Add nodes
        for satellite in constellation.satellites:
            graph.add_node(satellite)

        # Intra-plane edges
        # Connect satellites to the two satellites in the same plane with the closest phase angle
        for plane_id in constellation.plane_ids:
            satellites_in_plane = sorted(
                constellation.plane_id_to_satellites[plane_id],
                key=lambda satellite: constellation.satellite_orbits[satellite].phase_at_epoch,
            )
            for plane_idx, satellite in enumerate(satellites_in_plane):
                src, dst = satellite, satellites_in_plane[(plane_idx + 1) % len(satellites_in_plane)]
                graph.add_edge(src, dst)

        # Inter-plane edges
        def _get_first_satellite(plane_id: int) -> ConstellationSatellite:
            return sorted(
                constellation.plane_id_to_satellites[plane_id],
                key=lambda satellite: satellite.id.satellite_id,
            )[0]

        # Sort planes by raan
        plane_ids = sorted(
            constellation.plane_ids,
            key=lambda plane_id: constellation.satellite_orbits[_get_first_satellite(plane_id)].raan,
        )
        for plane_idx, plane_id in enumerate(plane_ids):
            next_plane_id = plane_ids[(plane_idx + 1) % len(plane_ids)]
            inter_plane_offset_ = self.inter_plane_offset
            if plane_idx == len(plane_ids) - 1:
                # Connect the last plane to the first plane
                inter_plane_offset_ += self.last_to_first_plane_offset

            satellites_in_plane = sorted(
                constellation.plane_id_to_satellites[plane_id],
                key=lambda satellite: satellite.id.satellite_id,
            )
            satellites_in_next_plane = sorted(
                constellation.plane_id_to_satellites[next_plane_id],
                key=lambda satellite: satellite.id.satellite_id,
            )
            assert len(satellites_in_plane) == len(satellites_in_next_plane), (
                "Number of satellites in each plane must be the same. "
                f"plane_id={plane_id}, next_plane_id={next_plane_id}"
            )
            for sat_idx, satellite in enumerate(satellites_in_plane):
                src = satellite
                dst = satellites_in_next_plane[(sat_idx + inter_plane_offset_) % len(satellites_in_plane)]
                graph.add_edge(src, dst)

        return graph


@deprecated("Use build_manhattan_time_series_topology() function instead.")
class ManhattanTimeSeriesTopologyBuilder(
    ConstellationTimeSeriesTopologyBuilder[MultiOrbitalPlaneConstellation[CircularSatelliteOrbit], nx.Graph],
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
        constellation: MultiOrbitalPlaneConstellation,
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        def construct_graph(time_idx: int) -> nx.Graph:
            graph = nx.Graph()

            # Add nodes
            for satellite in constellation.satellites:
                graph.add_node(satellite)

            # Intra-plane edges
            # Connect satellites to the two satellites in the same plane with the closest phase angle
            for plane_id in constellation.plane_ids:
                satellites_in_plane = sorted(
                    constellation.plane_id_to_satellites[plane_id],
                    key=lambda satellite: constellation.satellite_orbits[satellite].phase_at_epoch,
                )
                for plane_idx, satellite in enumerate(satellites_in_plane):
                    src, dst = satellite, satellites_in_plane[(plane_idx + 1) % len(satellites_in_plane)]
                    graph.add_edge(src, dst)

            # Inter-plane edges
            def _get_first_satellite(plane_id: int) -> ConstellationSatellite:
                return sorted(
                    constellation.plane_id_to_satellites[plane_id],
                    key=lambda satellite: satellite.id.satellite_id,
                )[0]

            # Sort planes by raan
            plane_ids = sorted(
                constellation.plane_ids,
                key=lambda plane_id: constellation.satellite_orbits[_get_first_satellite(plane_id)].raan,
            )
            for plane_idx, plane_id in enumerate(plane_ids):
                next_plane_id = plane_ids[(plane_idx + 1) % len(plane_ids)]
                inter_plane_offset_ = self.inter_plane_offset
                if plane_idx == len(plane_ids) - 1:
                    # Connect the last plane to the first plane
                    inter_plane_offset_ += self.last_to_first_plane_offset

                satellites_in_plane = sorted(
                    constellation.plane_id_to_satellites[plane_id],
                    key=lambda satellite: satellite.id.satellite_id,
                )
                satellites_in_next_plane = sorted(
                    constellation.plane_id_to_satellites[next_plane_id],
                    key=lambda satellite: satellite.id.satellite_id,
                )
                assert len(satellites_in_plane) == len(satellites_in_next_plane), (
                    "Number of satellites in each plane must be the same. "
                    f"plane_id={plane_id}, next_plane_id={next_plane_id}"
                )
                for sat_idx, satellite in enumerate(satellites_in_plane):
                    src = satellite
                    dst = satellites_in_next_plane[(sat_idx + inter_plane_offset_) % len(satellites_in_plane)]

                    # Orbital intersection in the polar region
                    _, latitude_src = unit_vector_to_azimuth_elevation(
                        normalize(dynamics_data.satellite_position_ecef[src][time_idx]),
                    )
                    _, latitude_dst = unit_vector_to_azimuth_elevation(
                        normalize(dynamics_data.satellite_position_ecef[dst][time_idx]),
                    )
                    if abs(latitude_src) > self.max_latitude and abs(latitude_dst) > self.max_latitude:
                        continue

                    graph.add_edge(src, dst)

            # Ensure the constructed graph is returned for each time index
            return graph

        return [construct_graph(time_idx) for time_idx in range(len(dynamics_data.time))]


# ---------------------------------------------------------------------------
# New functions using Constellation model
# ---------------------------------------------------------------------------


def _group_by_plane(
    constellation: Constellation[tuple[int, int]],
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
    entries: dict[int, list[tuple[int, ConstellationSatellite]]] = {}
    for (plane_id, in_plane_index), satellite in constellation.satellites.items():
        entries.setdefault(plane_id, []).append((in_plane_index, satellite))

    # Sort planes by plane_id, satellites within each plane by in_plane_index
    planes: list[list[ConstellationSatellite]] = [
        [sat for _, sat in sorted(entries[plane_id])] for plane_id in sorted(entries)
    ]

    return planes


def build_manhattan_topology(
    constellation: Constellation[tuple[int, int]],
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
