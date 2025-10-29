__all__ = [
    "ConstellationTimeSeriesTopologyBuilder",
    "ConstellationTopologyBuilder",
    "ManhattanTimeSeriesTopologyBuilder",
    "ManhattanTopologyBuilder",
]
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np

from cosmica.dtos import DynamicsData
from cosmica.dynamics import (
    CircularSatelliteOrbit,
    MultiOrbitalPlaneConstellation,
    SatelliteConstellation,
)
from cosmica.models import ConstellationSatellite
from cosmica.utils.vector import normalize, unit_vector_to_azimuth_elevation


class ConstellationTopologyBuilder[TConstellation: SatelliteConstellation, TGraph: nx.Graph](ABC):
    @abstractmethod
    def build(
        self,
        *,
        constellation: TConstellation,
    ) -> TGraph: ...


class ConstellationTimeSeriesTopologyBuilder[TConstellation: SatelliteConstellation, TGraph: nx.Graph](ABC):
    @abstractmethod
    def build(
        self,
        *,
        constellation: TConstellation,
        dynamics_data: DynamicsData,
    ) -> list[TGraph]: ...


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
