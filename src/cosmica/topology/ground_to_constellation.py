__all__ = [
    "ElevationBasedG2CTopologyBuilder",
    "GroundToConstellationTopologyBuilder",
    "ManualG2CTopologyBuilder",
    "build_elevation_based_g2c_topology",
    "build_manual_g2c_topology",
    "build_max_visibility_handover_g2c_topology",
]
import logging
from abc import ABC, abstractmethod
from collections.abc import Collection
from itertools import product

import networkx as nx
import numpy as np
import numpy.typing as npt

from cosmica.dtos import DynamicsData
from cosmica.models import Constellation, ConstellationSatellite, Gateway, Node, StationaryOnGroundUser
from cosmica.utils.coordinates import ecef2aer

logger = logging.getLogger(__name__)


class GroundToConstellationTopologyBuilder[TConstellation: Constellation, TNode: Node, TGraph: nx.Graph](ABC):
    @abstractmethod
    def build(
        self,
        *,
        constellation: TConstellation,
        ground_nodes: Collection[TNode],
        dynamics_data: DynamicsData,
    ) -> list[TGraph]: ...


class ElevationBasedG2CTopologyBuilder(
    GroundToConstellationTopologyBuilder[Constellation, Gateway | StationaryOnGroundUser, nx.Graph],
):
    def build(
        self,
        *,
        constellation: Constellation,
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        return build_elevation_based_g2c_topology(
            constellation,
            ground_nodes=ground_nodes,
            dynamics_data=dynamics_data,
        )


class ManualG2CTopologyBuilder(
    GroundToConstellationTopologyBuilder[Constellation, Gateway | StationaryOnGroundUser, nx.Graph],
):
    def __init__(self, custom_connections: dict[Gateway | StationaryOnGroundUser, ConstellationSatellite]) -> None:
        self.custom_connections: dict[Gateway | StationaryOnGroundUser, ConstellationSatellite] = custom_connections

    def build(
        self,
        *,
        constellation: Constellation,
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        return build_manual_g2c_topology(
            constellation,
            ground_nodes=ground_nodes,
            dynamics_data=dynamics_data,
            custom_connections=self.custom_connections,
        )


class MaxVisibilityHandOverG2CTopologyBuilder(
    GroundToConstellationTopologyBuilder[Constellation, Gateway | StationaryOnGroundUser, nx.Graph],
):
    def build(
        self,
        *,
        constellation: Constellation,
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        return build_max_visibility_handover_g2c_topology(
            constellation,
            ground_nodes=ground_nodes,
            dynamics_data=dynamics_data,
        )


# ---------------------------------------------------------------------------
# New functions using Constellation model
# ---------------------------------------------------------------------------


def build_elevation_based_g2c_topology(
    constellation: Constellation,
    *,
    ground_nodes: Collection[Gateway | StationaryOnGroundUser],
    dynamics_data: DynamicsData,
) -> list[nx.Graph]:
    """Build ground-to-constellation topology based on elevation angle.

    Connects ground nodes to constellation satellites when the satellite's
    elevation angle (as seen from the ground node) exceeds the ground node's
    minimum elevation threshold.

    This function only needs the satellite set, so it accepts `Constellation`
    with any `SatelliteId` type.

    Args:
        constellation: Constellation (any SatelliteId type).
        ground_nodes: Ground stations or users.
        dynamics_data: Time-series dynamics data.

    Returns:
        A list of networkx Graphs, one per time step.

    """
    logger.info(f"Number of time steps: {len(dynamics_data.time)}")
    ground_nodes_list = list(ground_nodes)
    satellites = list(constellation.satellites.values())

    n_satellites = len(satellites)
    visibility = np.zeros((len(ground_nodes_list), n_satellites, len(dynamics_data.time)), dtype=np.bool_)
    for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
        enumerate(ground_nodes_list),
        enumerate(satellites),
    ):
        _, elevation, _ = ecef2aer(
            x=dynamics_data.satellite_position_ecef[satellite][:, 0],
            y=dynamics_data.satellite_position_ecef[satellite][:, 1],
            z=dynamics_data.satellite_position_ecef[satellite][:, 2],
            lat0=ground_node.latitude,
            lon0=ground_node.longitude,
            h0=ground_node.altitude,
            deg=False,
        )
        visibility[ground_node_idx, sat_idx, :] = elevation >= ground_node.minimum_elevation

    def construct_graph(visibility: npt.NDArray[np.bool_]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(satellites)
        graph.add_nodes_from(ground_nodes_list)

        for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
            enumerate(ground_nodes_list),
            enumerate(satellites),
        ):
            if visibility[ground_node_idx, sat_idx]:
                graph.add_edge(ground_node, satellite)

        return graph

    return [construct_graph(visibility[:, :, time_idx]) for time_idx in range(len(dynamics_data.time))]


def build_manual_g2c_topology(
    constellation: Constellation,
    *,
    ground_nodes: Collection[Gateway | StationaryOnGroundUser],
    dynamics_data: DynamicsData,
    custom_connections: dict[Gateway | StationaryOnGroundUser, ConstellationSatellite],
) -> list[nx.Graph]:
    """Build ground-to-constellation topology with manually specified connections.

    Creates a static topology (same at every time step) based on the provided
    `custom_connections` mapping.

    Args:
        constellation: Constellation (any SatelliteId type).
        ground_nodes: Ground stations or users.
        dynamics_data: Time-series dynamics data (used only for time step count).
        custom_connections: Mapping from ground nodes to their connected satellites.

    Returns:
        A list of identical networkx Graphs, one per time step.

    """
    logger.info(f"Number of time steps: {len(dynamics_data.time)}")
    ground_nodes_list = list(ground_nodes)
    satellites = list(constellation.satellites.values())

    def construct_graph() -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(satellites)
        graph.add_nodes_from(ground_nodes_list)

        for ground_node, satellite in custom_connections.items():
            graph.add_edge(ground_node, satellite)

        return graph

    return [construct_graph() for _ in range(len(dynamics_data.time))]


def build_max_visibility_handover_g2c_topology(  # noqa: C901
    constellation: Constellation,
    *,
    ground_nodes: Collection[Gateway | StationaryOnGroundUser],
    dynamics_data: DynamicsData,
) -> list[nx.Graph]:
    """Build ground-to-constellation topology with maximum-visibility handover.

    Each ground node connects to the satellite with the longest remaining
    visibility duration. Handover occurs when the current satellite is no
    longer visible.

    Args:
        constellation: Constellation (any SatelliteId type).
        ground_nodes: Ground stations or users.
        dynamics_data: Time-series dynamics data.

    Returns:
        A list of networkx Graphs, one per time step.

    """
    logger.info(f"Number of time steps: {len(dynamics_data.time)}")
    ground_nodes_list = list(ground_nodes)
    satellites = list(constellation.satellites.values())

    n_time = len(dynamics_data.time)
    n_gateways = len(ground_nodes_list)
    n_satellites = len(satellites)
    visibility = np.zeros((n_gateways, n_satellites, n_time), dtype=np.bool_)
    for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
        enumerate(ground_nodes_list),
        enumerate(satellites),
    ):
        _, elevation, _ = ecef2aer(
            x=dynamics_data.satellite_position_ecef[satellite][:, 0],
            y=dynamics_data.satellite_position_ecef[satellite][:, 1],
            z=dynamics_data.satellite_position_ecef[satellite][:, 2],
            lat0=ground_node.latitude,
            lon0=ground_node.longitude,
            h0=ground_node.altitude,
            deg=False,
        )
        visibility[ground_node_idx, sat_idx, :] = elevation >= ground_node.minimum_elevation

    # Calculate remaining visibility time steps (backward pass)
    left_visibility_time_step = np.zeros((n_gateways, n_satellites, n_time), dtype=np.int_)
    for time_idx in reversed(range(n_time)):
        left_visibility_time_step[:, :, time_idx] = (
            np.where(
                visibility[:, :, time_idx],
                left_visibility_time_step[:, :, time_idx + 1] + 1,
                0,
            )
            if time_idx != n_time - 1
            else visibility[:, :, time_idx].astype(int)
        )

    # Select satellite for each ground node
    select_sat_idx = np.full(n_gateways, -1, dtype=np.int_)
    link_available = np.zeros((n_gateways, n_satellites, n_time), dtype=np.bool_)

    def select_max_visibility_satellite(
        ground_node_idx: int,
        time_idx: int,
    ) -> None:
        # Exclude satellites already connected to other ground nodes
        masked_left_visibility_time_step = np.copy(left_visibility_time_step[ground_node_idx, :, time_idx])
        for sat_idx in select_sat_idx:
            if sat_idx == -1:
                continue
            masked_left_visibility_time_step[sat_idx] = 0
        select_sat_idx[ground_node_idx] = np.argmax(masked_left_visibility_time_step)

    for time_idx in range(n_time):
        for ground_node_idx, _ground_node in enumerate(ground_nodes_list):
            if time_idx == 0:
                if left_visibility_time_step[ground_node_idx, :, time_idx].sum == 0:
                    select_sat_idx[ground_node_idx] = -1
                    continue
                select_max_visibility_satellite(ground_node_idx, time_idx)
            else:
                if left_visibility_time_step[ground_node_idx, :, time_idx].sum == 0:
                    select_sat_idx[ground_node_idx] = -1
                    continue
                if select_sat_idx[ground_node_idx] >= 0:
                    if left_visibility_time_step[ground_node_idx, select_sat_idx[ground_node_idx], time_idx] > 0:
                        pass
                    else:
                        select_max_visibility_satellite(ground_node_idx, time_idx)
                else:
                    select_max_visibility_satellite(ground_node_idx, time_idx)
            link_available[ground_node_idx, select_sat_idx[ground_node_idx], time_idx] = True

    def construct_graph(link_available: npt.NDArray[np.bool_]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(satellites)
        graph.add_nodes_from(ground_nodes_list)

        for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
            enumerate(ground_nodes_list),
            enumerate(satellites),
        ):
            if link_available[ground_node_idx, sat_idx]:
                graph.add_edge(ground_node, satellite)

        return graph

    return [construct_graph(link_available[:, :, time_idx]) for time_idx in range(len(dynamics_data.time))]
