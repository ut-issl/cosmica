__all__ = [
    "ElevationBasedG2CTopologyBuilder",
    "GroundToConstellationTopologyBuilder",
    "ManualG2CTopologyBuilder",
]
import logging
from abc import ABC, abstractmethod
from collections.abc import Collection
from itertools import product

import networkx as nx
import numpy as np
import numpy.typing as npt

from cosmica.dtos import DynamicsData
from cosmica.dynamics import SatelliteConstellation
from cosmica.models import ConstellationSatellite, Gateway, Node, StationaryOnGroundUser
from cosmica.utils.coordinates import ecef2aer

logger = logging.getLogger(__name__)


class GroundToConstellationTopologyBuilder[TConstellation: SatelliteConstellation, TNode: Node, TGraph: nx.Graph](ABC):
    @abstractmethod
    def build(
        self,
        *,
        constellation: TConstellation,
        ground_nodes: Collection[TNode],
        dynamics_data: DynamicsData,
    ) -> list[TGraph]: ...


class ElevationBasedG2CTopologyBuilder(
    GroundToConstellationTopologyBuilder[SatelliteConstellation, Gateway | StationaryOnGroundUser, nx.Graph],
):
    def build(
        self,
        *,
        constellation: SatelliteConstellation,
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        logger.info(f"Number of time steps: {len(dynamics_data.time)}")
        ground_nodes = list(ground_nodes)

        n_satellites = len(dynamics_data.satellite_position_eci)
        visibility = np.zeros((len(ground_nodes), n_satellites, len(dynamics_data.time)), dtype=np.bool_)
        for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
            enumerate(ground_nodes),
            enumerate(constellation.satellites),
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
            graph.add_nodes_from(constellation.satellites)
            graph.add_nodes_from(ground_nodes)

            for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
                enumerate(ground_nodes),
                enumerate(constellation.satellites),
            ):
                if visibility[ground_node_idx, sat_idx]:
                    graph.add_edge(ground_node, satellite)

            return graph

        return [construct_graph(visibility[:, :, time_idx]) for time_idx in range(len(dynamics_data.time))]


class ManualG2CTopologyBuilder(
    GroundToConstellationTopologyBuilder[SatelliteConstellation, Gateway | StationaryOnGroundUser, nx.Graph],
):
    def __init__(self, custom_connections: dict[Gateway | StationaryOnGroundUser, ConstellationSatellite]) -> None:
        self.custom_connections: dict[Gateway | StationaryOnGroundUser, ConstellationSatellite] = custom_connections

    def build(
        self,
        *,
        constellation: SatelliteConstellation,
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        logger.info(f"Number of time steps: {len(dynamics_data.time)}")
        ground_nodes = list(ground_nodes)

        def construct_graph() -> nx.Graph:
            graph = nx.Graph()
            graph.add_nodes_from(constellation.satellites)
            graph.add_nodes_from(ground_nodes)

            for ground_node, satellite in self.custom_connections.items():
                graph.add_edge(ground_node, satellite)

            return graph

        # dynamics_data.time の長さに応じたグラフを返す
        return [construct_graph() for _ in range(len(dynamics_data.time))]


# class MaxElevationHandOverG2CTopologyBuilder(
#     GroundToConstellationTopologyBuilder[SatelliteConstellation, Gateway | StationaryOnGroundUser, nx.Graph],
# ):
#     def build(
#         self,
#         *,
#         constellation: SatelliteConstellation,
#         ground_nodes: Collection[Gateway | StationaryOnGroundUser],
#         dynamics_data: DynamicsData,
#     ) -> list[nx.Graph]:
#         logger.info(f"Number of time steps: {len(dynamics_data.time)}")
#         ground_nodes = list(ground_nodes)

#         n_satellites = len(dynamics_data.satellite_position_eci)
#         visibility = np.zeros((len(ground_nodes), n_satellites, len(dynamics_data.time)), dtype=np.bool_)
#         elevation_angles = np.zeros((len(ground_nodes), n_satellites, len(dynamics_data.time)))
#         for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
#             enumerate(ground_nodes),
#             enumerate(constellation.satellites),
#         ):
#             _, elevation, _ = ecef2aer(
#                 x=dynamics_data.satellite_position_ecef[satellite][:, 0],
#                 y=dynamics_data.satellite_position_ecef[satellite][:, 1],
#                 z=dynamics_data.satellite_position_ecef[satellite][:, 2],
#                 lat0=ground_node.latitude,
#                 lon0=ground_node.longitude,
#                 h0=ground_node.altitude,
#                 deg=False,
#             )
#             elevation_angles[ground_node_idx, sat_idx, :] = elevation
#             visibility[ground_node_idx, sat_idx, :] = elevation >= ground_node.minimum_elevation

#         def construct_graph(visibility: npt.NDArray[np.bool_]) -> nx.Graph:
#             graph = nx.Graph()
#             graph.add_nodes_from(constellation.satellites)
#             graph.add_nodes_from(ground_nodes)

#             for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
#                 enumerate(ground_nodes),
#                 enumerate(constellation.satellites),
#             ):
#                 if visibility[ground_node_idx, sat_idx]:
#                     graph.add_edge(ground_node, satellite)

#             return graph

#         return [construct_graph(visibility[:, :, time_idx]) for time_idx in range(len(dynamics_data.time))]


class MaxVisibilityHandOverG2CTopologyBuilder(
    GroundToConstellationTopologyBuilder[SatelliteConstellation, Gateway | StationaryOnGroundUser, nx.Graph],
):
    def build(  # noqa: C901
        self,
        *,
        constellation: SatelliteConstellation,
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        logger.info(f"Number of time steps: {len(dynamics_data.time)}")
        ground_nodes = list(ground_nodes)

        n_time = len(dynamics_data.time)
        n_gateways = len(ground_nodes)
        n_satellites = len(dynamics_data.satellite_position_eci)
        visibility = np.zeros((n_gateways, n_satellites, n_time), dtype=np.bool_)
        for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
            enumerate(ground_nodes),
            enumerate(constellation.satellites),
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

        # 残接続時間を計算
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

        # 地上局に対する通信する衛星の選択
        select_sat_idx = np.full(
            n_gateways,
            -1,
            dtype=np.int_,
        )  # -1: 通信しない TODO: -1でなくNanしたいが, int型ではエラーが出る
        # connection_accumulation_time = np.zeros(n_gateways, dtype=np.float64)
        link_available = np.zeros((n_gateways, n_satellites, n_time), dtype=np.bool_)

        def select_max_visibility_satellite(
            ground_node_idx: int,
            time_idx: int,
        ) -> None:
            # 他の地上局と通信している衛星は除く
            masked_left_visibility_time_step = np.copy(left_visibility_time_step[ground_node_idx, :, time_idx])
            for sat_idx in select_sat_idx:
                if sat_idx == -1:
                    continue
                masked_left_visibility_time_step[sat_idx] = 0
            # 残り可視時間が最大の衛星を選択
            select_sat_idx[ground_node_idx] = np.argmax(masked_left_visibility_time_step)

        for time_idx in range(n_time):
            for ground_node_idx, _ground_node in enumerate(ground_nodes):
                if time_idx == 0:
                    # すべての衛星に対して可視でない
                    if left_visibility_time_step[ground_node_idx, :, time_idx].sum == 0:
                        select_sat_idx[ground_node_idx] = -1
                        continue
                    # 残り可視時間が最大の衛星を選択
                    select_max_visibility_satellite(ground_node_idx, time_idx)
                else:
                    # すべての衛星に対して可視でない
                    if left_visibility_time_step[ground_node_idx, :, time_idx].sum == 0:
                        select_sat_idx[ground_node_idx] = -1
                        continue
                    # 前のタイムステップで接続している衛星がいる
                    if select_sat_idx[ground_node_idx] >= 0:
                        # 現在のタイムステップも可視 → 同じ衛星に接続
                        if left_visibility_time_step[ground_node_idx, select_sat_idx[ground_node_idx], time_idx] > 0:
                            # # 累積連続可視時間を更新
                            # connection_accumulation_time[ground_node_idx] += (
                            #     dynamics_data.time[time_idx] - dynamics_data.time[time_idx - 1]
                            # ) / np.timedelta64(1, "s")
                            pass
                        else:
                            # 残り可視時間が最大の衛星を選択
                            select_max_visibility_satellite(ground_node_idx, time_idx)
                    # 前のタイムステップで接続している衛星がいない
                    else:
                        # 残り可視時間が最大の衛星を選択
                        select_max_visibility_satellite(ground_node_idx, time_idx)
                link_available[ground_node_idx, select_sat_idx[ground_node_idx], time_idx] = True

        def construct_graph(link_available: npt.NDArray[np.bool_]) -> nx.Graph:
            graph = nx.Graph()
            graph.add_nodes_from(constellation.satellites)
            graph.add_nodes_from(ground_nodes)

            for (ground_node_idx, ground_node), (sat_idx, satellite) in product(
                enumerate(ground_nodes),
                enumerate(constellation.satellites),
            ):
                if link_available[ground_node_idx, sat_idx]:
                    graph.add_edge(ground_node, satellite)

            return graph

        return [construct_graph(link_available[:, :, time_idx]) for time_idx in range(len(dynamics_data.time))]
