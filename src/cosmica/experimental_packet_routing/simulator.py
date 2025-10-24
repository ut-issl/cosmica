import copy
import logging
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx import Graph
from tqdm import tqdm
from typing_extensions import Doc

from cosmica.experimental_packet_routing.dtos import (
    PacketRoutingResult,
    PacketRoutingSetting,
)
from cosmica.models.demand import (
    ConstantCommunicationDemand,
    Demand,
    TemporaryCommunicationDemand,
)
from cosmica.models.node import Node, NodeGID

from .calc_forwarding_table_from_graph import (
    initialize_forwarding_table_list_from_space_time_graph,
)
from .case_definitions import BackupCaseType
from .comm_data import CommDataDemand, CommDataLSA
from .lsa_case_definition import LsaCaseType
from .node_knowledge import NodeKnowledge
from .space_time_graph import SpaceTimeGraph
from .use_node_knowledge import calc_next_node_from_node_knowledge
from .utility import get_edge_data, has_edge_bidirectional, remove_edge_safe

logger = logging.getLogger(__name__)


# TODO(Takashima): 汎用的なsimulation classの作成
class PacketCommunicationSimulator:
    """時系列のパケットレベルの通信シミュレーションを実施するクラス."""

    # 前提
    # - グラフ上のNodeはシミュレーション期間で変化しない (attributeは変化しても良い)
    # - edgeがdelayの属性を持つ

    def __init__(
        self,
        time: Annotated[
            npt.NDArray[np.datetime64],
            Doc("Array of datetime64 representing the time points"),
        ],
        all_graphs_with_comm_performance: Annotated[
            list[Graph],
            Doc("List of Graph objects representing communication performance over time"),
        ],
        nodes_dict: Annotated[dict[NodeGID, Node], Doc("Dictionary mapping NodeGID to Node objects")],
        demands: Annotated[list[Demand], Doc("List of demands")],
        *,
        backup_case: Annotated[
            BackupCaseType,
            Doc("Backup case scenario identifier, default is 'no-backup'"),
        ] = "no-backup",
        hop_limit: Annotated[int, Doc("Hop limit for backup routing, default is 1")] = 1,
        packet_size: Annotated[int, Doc("Size of the packet for communication [bit], default is 10,000")] = int(1e4),
        space_time_graph: Annotated[
            SpaceTimeGraph,
            Doc("SpaceTimeGraph object"),
        ],
    ) -> None:
        assert len(time) == len(all_graphs_with_comm_performance)

        self.time: npt.NDArray[np.datetime64] = time
        self.all_graphs_with_comm_performance: list[Graph] = all_graphs_with_comm_performance
        self.nodes_dict: dict[NodeGID, Node] = nodes_dict
        self.demands: list[Demand] = demands

        self.backup_case: BackupCaseType = backup_case
        self.hop_limit: int = hop_limit
        self.packet_size: int = packet_size

        self.node_knowledge_known_by_each_node: dict[Node, NodeKnowledge] = {}
        self.space_time_graph = space_time_graph
        for node in tqdm(self.nodes_dict.values(), desc="Initializing node knowledge"):
            self.node_knowledge_known_by_each_node[node] = NodeKnowledge(
                target_node=node,
                space_time_graph=copy.deepcopy(self.space_time_graph),
                forwarding_table_time_list=initialize_forwarding_table_list_from_space_time_graph(
                    space_time_graph=self.space_time_graph,
                    src_node=node,
                    weight="weight",
                    backup_case=self.backup_case,
                    hops_limit=self.hop_limit,
                ),
            )

    def create_packet_routing_setting(self) -> PacketRoutingSetting:
        """Generate and return a PacketRoutingSetting object."""
        return PacketRoutingSetting(
            time=self.time,
            nodes_dict=self.nodes_dict,
            demands=self.demands,
            backup_case=self.backup_case,
            hop_limit=self.hop_limit,
            packet_size=self.packet_size,
        )

    def run(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        rng: Annotated[
            np.random.Generator | None,
            Doc("NumPy random number generator. If None, use default."),
        ] = None,
        edge_remove_schedule: Annotated[
            list[tuple[np.datetime64, tuple[Node, Node]]] | None,
            Doc("List of edge removal schedule"),
        ] = None,
        failure_detection_time: Annotated[np.timedelta64 | None, Doc("Time taken for failure detection")] = None,
        enable_random_routing_when_edge_failure: Annotated[
            bool,
            Doc("Flag to enable random routing when edge failure"),
        ] = False,
        prevent_loop: Annotated[bool, Doc("Flag to prevent loop in routing")] = False,
        with_lsa: Annotated[bool, Doc("Flag to enable LSA data")] = True,
        lsa_case: Annotated[
            LsaCaseType,
            Doc("Case scenario identifier for LSA data, default is 'nominal'"),
        ] = "from-source-to-all",
    ) -> PacketRoutingResult:
        ## デフォルト値の設定 ========================================
        rng = rng if rng is not None else np.random.default_rng()
        edge_remove_schedule = edge_remove_schedule if edge_remove_schedule is not None else []
        failure_detection_time = (
            failure_detection_time if failure_detection_time is not None else np.timedelta64(10, "ms")
        )

        ## シミュレーション情報をログ出力 ========================================
        logger.info(f"Starting packet communication simulation with {len(self.time)} time steps")
        logger.info(f"Number of nodes: {len(self.nodes_dict)}")
        logger.info(f"Number of demands: {len(self.demands)}")
        if edge_remove_schedule:
            logger.info(f"Edge removal schedule: {len(edge_remove_schedule)} events")
            for timing, edge in edge_remove_schedule:
                logger.info(f"  - {edge[0].id} <-> {edge[1].id} at {timing}")

        ## データ格納用のリスト ========================================
        all_graphs_after_simulation: list[Graph] = []  # シミュレーション後のグラフ履歴
        comm_data_demand_list: list[CommDataDemand] = []
        comm_data_lsa_list: list[CommDataLSA] = []

        ## シミュレーションの実行 ========================================
        for time_idx, current_time in tqdm(
            enumerate(self.time),
            desc="Running packet simulation",
            total=len(self.time),
        ):
            if time_idx < len(self.time) - 1:
                time_step: np.timedelta64 = self.time[time_idx + 1] - self.time[time_idx]
                time_step_s: float = float(time_step / np.timedelta64(1, "s"))

            ## タイムステップにおけるグラフの生成 ========================================
            _graph: Graph = copy.deepcopy(self.all_graphs_with_comm_performance[time_idx])

            # Initialize the edge attributes
            nx.set_edge_attributes(
                _graph,
                {edge: {"bandwidth_usage_for_demand_data": 0} for edge in _graph.edges},
            )
            nx.set_edge_attributes(
                _graph,
                {edge: {"bandwidth_usage_for_lsa_data": 0} for edge in _graph.edges},
            )

            ## Edgeの切断 ========================================
            for edge_remove_timing, edge_to_remove in edge_remove_schedule:
                if current_time >= edge_remove_timing:
                    logger.info(
                        f"Edge {edge_to_remove[0].id} <-> {edge_to_remove[1].id} disconnected at {current_time}",
                    )
                    remove_edge_safe(_graph, *edge_to_remove)
                    # Edge上にあるデータはパケットロスと判断
                    packet_loss_count_demand = 0
                    for comm_data_demand in comm_data_demand_list:
                        if isinstance(comm_data_demand.current_position, tuple) and set(
                            comm_data_demand.current_position,
                        ) == set(edge_to_remove):
                            comm_data_demand.packet_loss = True
                            comm_data_demand.delay = np.inf
                            packet_loss_count_demand += 1
                    packet_loss_count_lsa = 0
                    for comm_data_lsa in comm_data_lsa_list:
                        if isinstance(comm_data_lsa.current_position, tuple) and set(
                            comm_data_lsa.current_position,
                        ) == set(edge_to_remove):
                            comm_data_lsa.packet_loss = True
                            comm_data_lsa.delay = np.inf
                            packet_loss_count_lsa += 1
                    if packet_loss_count_demand > 0 or packet_loss_count_lsa > 0:
                        logger.info(
                            f"Packet loss due to edge disconnection: "
                            f"{packet_loss_count_demand} demand packets, {packet_loss_count_lsa} LSA packets",
                        )

            ## Edgeの復旧 ========================================
            # TODO(Takashima): 未実装

            ## 通信データの生成 ========================================
            # LSAデータの生成
            if with_lsa:
                for edge_remove_timing, edge_to_remove in edge_remove_schedule:
                    failure_detection_timing: np.datetime64 = edge_remove_timing + failure_detection_time
                    if (time_idx == 0 and failure_detection_timing <= self.time[time_idx]) or (
                        time_idx > 0 and self.time[time_idx - 1] < failure_detection_timing <= self.time[time_idx]
                    ):
                        # 切断Edgeに接続している node 情報を更新
                        for detect_node in edge_to_remove:
                            # 自身の情報を更新
                            self.node_knowledge_known_by_each_node[detect_node].update_node_knowledge_based_on_lsa(
                                comm_data_lsa=CommDataLSA(
                                    data_size=self.packet_size,
                                    packet_size=self.packet_size,
                                    packet_num=1,
                                    dst_node=detect_node,
                                    next_node=detect_node,
                                    current_position=detect_node,
                                    path=[detect_node],
                                    generated_time=current_time,
                                    time=current_time,
                                    time_from_generated=0.0,
                                    time_remaining_for_current_position=0.0,
                                    failure_position=edge_to_remove,
                                ),
                                current_time=current_time,
                                weight="weight",
                                backup_case=self.backup_case,
                                hops_limit=self.hop_limit,
                            )
                        # LSAデータの生成
                        if lsa_case == "from-source-to-all":
                            #  切断Edgeに接続している node から 他のすべての node に対して LSA データを生成
                            for detect_node in edge_to_remove:
                                for dst_node in self.nodes_dict.values():
                                    if dst_node == detect_node:
                                        # dst_node = detect_node の場合はスキップ
                                        continue
                                    next_node = calc_next_node_from_node_knowledge(
                                        node_knowledge=self.node_knowledge_known_by_each_node[detect_node],
                                        dst_node=dst_node,
                                        path=[detect_node],
                                        current_time=current_time,
                                        enable_random_routing_when_edge_failure=enable_random_routing_when_edge_failure,
                                        prevent_loop=prevent_loop,
                                    )
                                    if next_node is None:
                                        continue
                                    comm_data_lsa = CommDataLSA(
                                        data_size=self.packet_size,
                                        packet_size=self.packet_size,
                                        packet_num=1,
                                        dst_node=dst_node,
                                        next_node=next_node,
                                        current_position=detect_node,
                                        path=[detect_node],
                                        generated_time=current_time,
                                        time=current_time,
                                        time_from_generated=0.0,
                                        time_remaining_for_current_position=0.0,
                                        failure_position=edge_to_remove,
                                    )
                                    comm_data_lsa_list.append(comm_data_lsa)
                        elif lsa_case == "adjacent":
                            # 切断Edgeに接続している node から 隣接 node に対して LSA データを生成
                            for detect_node in edge_to_remove:
                                for dst_node in list(_graph.neighbors(detect_node)):
                                    next_node = calc_next_node_from_node_knowledge(
                                        node_knowledge=self.node_knowledge_known_by_each_node[detect_node],
                                        dst_node=dst_node,
                                        path=[detect_node],
                                        current_time=current_time,
                                        enable_random_routing_when_edge_failure=enable_random_routing_when_edge_failure,
                                        prevent_loop=prevent_loop,
                                    )
                                    if next_node is None:
                                        continue
                                    comm_data_lsa = CommDataLSA(
                                        data_size=self.packet_size,
                                        packet_size=self.packet_size,
                                        packet_num=1,
                                        dst_node=dst_node,
                                        next_node=next_node,
                                        current_position=detect_node,
                                        path=[detect_node],
                                        generated_time=current_time,
                                        time=current_time,
                                        time_from_generated=0.0,
                                        time_remaining_for_current_position=0.0,
                                        failure_position=edge_to_remove,
                                    )
                                    comm_data_lsa_list.append(comm_data_lsa)

            # Demandデータの生成
            # 1パケットずつ扱うとシミュレーション時間がかかるので、宛先が同じ & 生成時刻が同じパケットをcomm_dataとしてまとめている  # noqa: E501
            for demand in self.demands:
                if isinstance(demand, ConstantCommunicationDemand) or (
                    isinstance(demand, TemporaryCommunicationDemand) and demand.is_active(current_time)
                ):
                    if demand.distribution == "uniform":
                        packet_num = int(demand.transmission_rate * time_step_s / self.packet_size)
                    elif demand.distribution == "poisson":
                        packet_num = int(rng.poisson(lam=demand.transmission_rate * time_step_s / self.packet_size))

                    next_node = calc_next_node_from_node_knowledge(
                        node_knowledge=self.node_knowledge_known_by_each_node[self.nodes_dict[demand.source]],
                        dst_node=self.nodes_dict[demand.destination],
                        path=[self.nodes_dict[demand.source]],
                        current_time=current_time,
                        enable_random_routing_when_edge_failure=enable_random_routing_when_edge_failure,
                        prevent_loop=prevent_loop,
                    )
                    if next_node is None:
                        continue
                    comm_data_demand = CommDataDemand(
                        demand_id=demand.id,
                        data_size=packet_num * self.packet_size,
                        packet_size=self.packet_size,
                        packet_num=packet_num,
                        dst_node=self.nodes_dict[demand.destination],
                        next_node=next_node,
                        current_position=self.nodes_dict[demand.source],
                        path=[self.nodes_dict[demand.source]],
                        generated_time=current_time,
                        time=current_time,
                        time_from_generated=0.0,
                        time_remaining_for_current_position=0.0,
                    )
                    comm_data_demand_list.append(comm_data_demand)

            ## 通信データの伝播 ========================================
            # TODO(Takashima): comm_data_lsa_listやcomm_data_demand_listについて、届いたデータやパケットロスのデータを削除しない場合、forのループ回数が増大していくので、処理を検討  # noqa: E501

            # LSAデータの伝播
            if with_lsa:
                for comm_data_lsa in comm_data_lsa_list:
                    if comm_data_lsa.packet_loss or comm_data_lsa.reach_dst:
                        continue
                    if comm_data_lsa.generated_time > current_time + time_step:
                        continue

                    # 共通処理
                    if comm_data_lsa.time > current_time:
                        comm_data_lsa.time_remaining_for_current_position -= float(
                            (current_time + time_step - comm_data_lsa.time) / np.timedelta64(1, "s"),
                        )
                        comm_data_lsa.time = current_time + time_step
                    else:
                        comm_data_lsa.time_remaining_for_current_position -= time_step_s
                        comm_data_lsa.time += time_step

                    # Node -> Edge、Edge -> Nodeなど境界を超える時のデータの処理
                    while comm_data_lsa.time_remaining_for_current_position < 0:
                        # Node -> Edge
                        if not isinstance(comm_data_lsa.current_position, tuple):
                            next_edge: tuple[Node, Node] = (
                                comm_data_lsa.current_position,
                                comm_data_lsa.next_node,
                            )

                            # 転送先のEdgeが存在しない時 -> パケットロスと判断
                            if not has_edge_bidirectional(_graph, *next_edge):
                                comm_data_lsa.packet_loss = True
                                comm_data_lsa.delay = np.inf
                                break

                            # Congestionの考慮 -> LSAデータについては、占有bandwidthの考慮は行わない

                            # 現在位置の更新
                            comm_data_lsa.current_position = next_edge
                            edge_data = get_edge_data(_graph, *next_edge)
                            if edge_data is not None:
                                edge_data["bandwidth_usage_for_lsa_data"] += comm_data_lsa.data_size / time_step_s

                                # 現在位置に留まる残時間の更新
                                comm_data_lsa.time_remaining_for_current_position += edge_data["delay"]
                                comm_data_lsa.time_from_generated += edge_data["delay"]

                        # Edge -> Node
                        elif isinstance(comm_data_lsa.current_position, tuple):
                            # 現在位置の更新
                            comm_data_lsa.current_position = comm_data_lsa.next_node
                            comm_data_lsa.path.append(comm_data_lsa.current_position)

                            # 現在の位置が目的地の場合
                            if comm_data_lsa.current_position == comm_data_lsa.dst_node:
                                comm_data_lsa.reach_dst = True
                                comm_data_lsa.delay = comm_data_lsa.time_from_generated
                                logger.info(
                                    f"LSA data received at {comm_data_lsa.dst_node.id} "
                                    f"(delay: {comm_data_lsa.delay:.6f}s, failure: {comm_data_lsa.failure_position})",
                                )
                                if lsa_case == "from-source-to-all":
                                    self.node_knowledge_known_by_each_node[
                                        comm_data_lsa.current_position
                                    ].update_node_knowledge_based_on_lsa(
                                        comm_data_lsa,
                                        current_time,
                                        weight="weight",
                                        backup_case=self.backup_case,
                                        hops_limit=self.hop_limit,
                                    )
                                elif lsa_case == "adjacent":
                                    already_registered = False
                                    for failure_assumed_edge in self.node_knowledge_known_by_each_node[
                                        comm_data_lsa.current_position
                                    ].failure_assumed_edge_list:
                                        if isinstance(comm_data_lsa.failure_position, tuple) and (
                                            set(failure_assumed_edge) == set(comm_data_lsa.failure_position)
                                        ):
                                            already_registered = True

                                    # 同じリンク切断が既に登録されている場合は何もしない
                                    if already_registered:
                                        logger.info(
                                            f"LSA data received at {comm_data_lsa.dst_node.id} "
                                            f"(already registered failure: {comm_data_lsa.failure_position})",
                                        )
                                    # 登録されていないリンク切断情報の場合のみ、情報をアップデートし、隣接ノードに伝搬
                                    else:
                                        logger.info(
                                            f"LSA data received at {comm_data_lsa.dst_node.id} "
                                            f"(new failure info: {comm_data_lsa.failure_position})",
                                        )
                                        self.node_knowledge_known_by_each_node[
                                            comm_data_lsa.current_position
                                        ].update_node_knowledge_based_on_lsa(
                                            comm_data_lsa,
                                            current_time,
                                            weight="weight",
                                            backup_case=self.backup_case,
                                            hops_limit=self.hop_limit,
                                        )

                                        # 隣接nodeにLSAデータを伝搬
                                        graph_known_by_node = self.node_knowledge_known_by_each_node[
                                            comm_data_lsa.current_position
                                        ].space_time_graph.get_space_time_graph_at_time(current_time)
                                        next_destination_list = list(
                                            graph_known_by_node.neighbors(comm_data_lsa.current_position),
                                        )

                                        for dst_node in next_destination_list:
                                            next_node = calc_next_node_from_node_knowledge(
                                                node_knowledge=self.node_knowledge_known_by_each_node[
                                                    comm_data_lsa.current_position
                                                ],
                                                dst_node=dst_node,
                                                path=[comm_data_lsa.current_position],
                                                current_time=current_time,
                                                enable_random_routing_when_edge_failure=enable_random_routing_when_edge_failure,
                                                prevent_loop=prevent_loop,
                                            )
                                            if next_node is None:
                                                continue

                                            generated_time = (
                                                comm_data_lsa.generated_time
                                                + np.timedelta64(
                                                    int(comm_data_lsa.delay * 1e3),
                                                    "ms",
                                                )  # TODO(): 現状だとms単位で生成され、より細かい単位に対応できない
                                                + np.timedelta64(
                                                    10,
                                                    "ms",
                                                )  # TODO(): 処理遅延についてパラメータで設定できるようにする
                                            )
                                            comm_data_lsa_new = CommDataLSA(
                                                data_size=self.packet_size,
                                                packet_size=self.packet_size,
                                                packet_num=1,
                                                dst_node=dst_node,
                                                next_node=next_node,
                                                current_position=comm_data_lsa.current_position,
                                                path=[comm_data_lsa.current_position],
                                                generated_time=generated_time,
                                                time=generated_time,
                                                time_from_generated=0,
                                                time_remaining_for_current_position=0,
                                                failure_position=comm_data_lsa.failure_position,
                                            )
                                            comm_data_lsa_list.append(comm_data_lsa_new)
                                else:
                                    pass

                                break  # 目的地に到達した場合は、次の通信データに移る

                            # 現在の位置が目的地でない場合
                            # 現在位置に留まる残時間の更新
                            # TODO(Takashima): 実際にはBufferに入れたりするのでその処理を検討
                            comm_data_lsa.time_remaining_for_current_position += _graph.nodes[
                                comm_data_lsa.current_position
                            ]["delay"]
                            comm_data_lsa.time_from_generated += _graph.nodes[comm_data_lsa.current_position]["delay"]

                            # ルーティングテーブルの参照
                            next_node = calc_next_node_from_node_knowledge(
                                node_knowledge=self.node_knowledge_known_by_each_node[comm_data_lsa.current_position],
                                dst_node=comm_data_lsa.dst_node,
                                path=comm_data_lsa.path,
                                current_time=current_time,
                                enable_random_routing_when_edge_failure=enable_random_routing_when_edge_failure,
                                prevent_loop=prevent_loop,
                            )
                            if next_node is None:
                                comm_data_lsa.packet_loss = True
                                comm_data_lsa.delay = np.inf
                                break
                            comm_data_lsa.next_node = next_node

            # Demandデータの伝播
            # TODO(Takashima): 現状は先に生成されたデータの伝播が優先されるので, 優先順位について検討
            for comm_data_demand in comm_data_demand_list:
                if comm_data_demand.packet_loss or comm_data_demand.reach_dst:
                    continue
                if comm_data_demand.generated_time > current_time + time_step:
                    continue

                # 共通処理
                if comm_data_demand.time > current_time:
                    comm_data_demand.time_remaining_for_current_position -= float(
                        (current_time + time_step - comm_data_demand.time) / np.timedelta64(1, "s"),
                    )
                    comm_data_demand.time = current_time + time_step
                else:
                    comm_data_demand.time_remaining_for_current_position -= time_step_s
                    comm_data_demand.time += time_step

                # Node -> Edge、Edge -> Nodeなど境界を超える時のデータの処理
                while comm_data_demand.time_remaining_for_current_position < 0:
                    # Node -> Edge
                    if not isinstance(comm_data_demand.current_position, tuple):
                        next_edge = (
                            comm_data_demand.current_position,
                            comm_data_demand.next_node,
                        )

                        # 転送先のEdgeが存在しない時 -> パケットロスと判断
                        if not has_edge_bidirectional(_graph, *next_edge):
                            comm_data_demand.packet_loss = True
                            comm_data_demand.delay = np.inf
                            break

                        # Congestionの考慮
                        edge_data = get_edge_data(_graph, *next_edge)
                        if edge_data is None:
                            comm_data_demand.packet_loss = True
                            comm_data_demand.delay = np.inf
                            break

                        if (
                            edge_data["bandwidth_usage_for_demand_data"] + comm_data_demand.data_size / time_step_s
                            > edge_data["link_capacity"]
                        ):
                            comm_data_demand.packet_loss = True
                            comm_data_demand.delay = np.inf
                            break

                        # 現在位置の更新
                        comm_data_demand.current_position = next_edge
                        edge_data["bandwidth_usage_for_demand_data"] += comm_data_demand.data_size / time_step_s

                        # 現在位置に留まる残時間の更新
                        comm_data_demand.time_remaining_for_current_position += edge_data["delay"]
                        comm_data_demand.time_from_generated += edge_data["delay"]

                    # Edge -> Node
                    elif isinstance(comm_data_demand.current_position, tuple):
                        # 現在位置の更新
                        comm_data_demand.current_position = comm_data_demand.next_node
                        comm_data_demand.path.append(comm_data_demand.current_position)

                        # 現在の位置が目的地の場合
                        if comm_data_demand.current_position == comm_data_demand.dst_node:
                            comm_data_demand.reach_dst = True
                            comm_data_demand.delay = comm_data_demand.time_from_generated
                            break  # 目的地に到達した場合は、次の通信データに移る

                        # 現在の位置が目的地でない場合
                        # 現在位置に留まる残時間の更新
                        # TODO(Takashima): 実際にはBufferに入れたりするのでその処理を検討
                        # -> 手計算では, forwarding rate が10Gbpsの場合 Queuing Delayは10-100μsになりそうで小さい
                        comm_data_demand.time_remaining_for_current_position += _graph.nodes[
                            comm_data_demand.current_position
                        ]["delay"]
                        comm_data_demand.time_from_generated += _graph.nodes[comm_data_demand.current_position]["delay"]

                        # ルーティングテーブルの参照
                        next_node = calc_next_node_from_node_knowledge(
                            node_knowledge=self.node_knowledge_known_by_each_node[comm_data_demand.current_position],
                            dst_node=comm_data_demand.dst_node,
                            path=comm_data_demand.path,
                            current_time=current_time,
                            enable_random_routing_when_edge_failure=enable_random_routing_when_edge_failure,
                            prevent_loop=prevent_loop,
                        )
                        if next_node is None:
                            comm_data_demand.packet_loss = True
                            comm_data_demand.delay = np.inf
                            break
                        comm_data_demand.next_node = next_node

            ## Save the graph to the list of graphs after simulation for each time step ================================
            all_graphs_after_simulation.append(_graph)

        ## Save simulation results ========================================
        # シミュレーション結果の統計をログ出力
        total_demand_packets = len(comm_data_demand_list)
        successful_demand_packets = sum(1 for comm in comm_data_demand_list if comm.reach_dst)
        packet_loss_demand_packets = sum(1 for comm in comm_data_demand_list if comm.packet_loss)

        total_lsa_packets = len(comm_data_lsa_list)
        successful_lsa_packets = sum(1 for comm in comm_data_lsa_list if comm.reach_dst)
        packet_loss_lsa_packets = sum(1 for comm in comm_data_lsa_list if comm.packet_loss)

        logger.info("Simulation completed")
        logger.info(
            f"Demand packets - Total: {total_demand_packets}, "
            f"Successful: {successful_demand_packets}, "
            f"Lost: {packet_loss_demand_packets}",
        )
        logger.info(
            f"LSA packets - Total: {total_lsa_packets}, "
            f"Successful: {successful_lsa_packets}, "
            f"Lost: {packet_loss_lsa_packets}",
        )

        if successful_demand_packets > 0:
            avg_delay_demand = np.mean([comm.delay for comm in comm_data_demand_list if comm.reach_dst])
            logger.info(f"Average delay for successful demand packets: {avg_delay_demand:.6f}s")

        return PacketRoutingResult(
            all_graphs_after_simulation=all_graphs_after_simulation,
            comm_data_demand_list=comm_data_demand_list,
            comm_data_lsa_list=comm_data_lsa_list,
            node_knowledge_known_by_each_node=self.node_knowledge_known_by_each_node,
        )
