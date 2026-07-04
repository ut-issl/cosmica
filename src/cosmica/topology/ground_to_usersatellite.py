__all__ = [
    "ElevationBasedG2USTopologyBuilder",
    "GroundToUserSatelliteTopologyBuilder",
    "ManualG2USTopologyBuilder",
]
import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable
from itertools import product

import networkx as nx
import numpy as np
import numpy.typing as npt

from cosmica.dtos import DynamicsData
from cosmica.models import Gateway, Node, StationaryOnGroundUser
from cosmica.models.satellite import UserSatellite
from cosmica.utils.coordinates import ecef2aer

logger = logging.getLogger(__name__)


class GroundToUserSatelliteTopologyBuilder[TUserSatellite: UserSatellite, TNode: Node, TGraph: nx.Graph](ABC):
    @abstractmethod
    def build(
        self,
        *,
        user_satellites: Collection[TUserSatellite],
        ground_nodes: Collection[TNode],
        dynamics_data: DynamicsData,
    ) -> list[TGraph]: ...


class ElevationBasedG2USTopologyBuilder(
    GroundToUserSatelliteTopologyBuilder[UserSatellite, Gateway | StationaryOnGroundUser, nx.DiGraph],
):
    def build(
        self,
        *,
        user_satellites: Collection[UserSatellite],
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.DiGraph]:
        logger.info(f"Number of time steps: {len(dynamics_data.time)}")
        ground_nodes = list(ground_nodes)

        n_satellites = len(dynamics_data.satellite_position_eci)
        visibility = np.zeros((len(ground_nodes), n_satellites, len(dynamics_data.time)), dtype=np.bool_)
        for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
            enumerate(ground_nodes),
            enumerate(user_satellites),
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

        def construct_graph(visibility: npt.NDArray[np.bool_]) -> nx.DiGraph:
            graph: nx.Graph[Node[Hashable]] = nx.Graph()
            graph.add_nodes_from(user_satellites)
            graph.add_nodes_from(ground_nodes)

            for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
                enumerate(ground_nodes),
                enumerate(user_satellites),
            ):
                if visibility[ground_node_idx, sat_idx]:
                    graph.add_edge(ground_node, satellite)

            # Each physical link is bidirectional: represent it as two directed edges
            return graph.to_directed()

        return [construct_graph(visibility[:, :, time_idx]) for time_idx in range(len(dynamics_data.time))]


class ManualG2USTopologyBuilder(
    GroundToUserSatelliteTopologyBuilder[UserSatellite, Gateway | StationaryOnGroundUser, nx.DiGraph],
):
    def __init__(self, custom_connections: dict[Gateway | StationaryOnGroundUser, UserSatellite]) -> None:
        self.custom_connections: dict[Gateway | StationaryOnGroundUser, UserSatellite] = custom_connections

    def build(
        self,
        *,
        user_satellites: Collection[UserSatellite],
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.DiGraph]:
        logger.info(f"Number of time steps: {len(dynamics_data.time)}")
        ground_nodes = list(ground_nodes)

        def construct_graph() -> nx.DiGraph:
            graph: nx.Graph[Node[Hashable]] = nx.Graph()
            graph.add_nodes_from(user_satellites)
            graph.add_nodes_from(ground_nodes)

            for ground_node, satellite in self.custom_connections.items():
                graph.add_edge(ground_node, satellite)

            # Each physical link is bidirectional: represent it as two directed edges
            return graph.to_directed()

        # dynamics_data.time の長さに応じたグラフを返す
        return [construct_graph() for _ in range(len(dynamics_data.time))]
