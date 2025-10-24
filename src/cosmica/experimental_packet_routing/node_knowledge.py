__all__ = [
    "NodeKnowledge",
]

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from cosmica.models.node import Node

from .calc_forwarding_table_from_graph import (
    initialize_forwarding_table_list_from_space_time_graph,
)
from .case_definitions import BackupCaseType
from .comm_data import CommDataLSA
from .forwarding_table_time_list import ForwardingTableTimeList
from .space_time_graph import SpaceTimeGraph


@dataclass(kw_only=True, slots=True)
class NodeKnowledge:
    target_node: Node
    space_time_graph: SpaceTimeGraph
    forwarding_table_time_list: ForwardingTableTimeList
    failure_assumed_edge_list: list[tuple[Node, Node]] = field(default_factory=list)
    routing_recalculation_period: list[list[np.datetime64]] = field(default_factory=list)  # [start, end] のリストで構成
    receive_lsa_time_list: list[np.datetime64] | None = None
    routing_recalculation_interval: np.timedelta64 = field(default_factory=lambda: np.timedelta64(2, "s"))

    # TODO (Takashima): 各ノードのアクションなので、NodeKnowledgeではなく別の場所での処理に変更する
    def update_node_knowledge_based_on_lsa(
        self,
        comm_data_lsa: CommDataLSA,
        current_time: np.datetime64,
        weight: str | Callable = "weight",
        *,
        backup_case: BackupCaseType = "no-backup",
        hops_limit: int = 1,
    ) -> None:
        """Receive LSA and edit the network information."""
        if isinstance(comm_data_lsa.failure_position, tuple):
            # Edit the failure assumed edge
            self.failure_assumed_edge_list.append(comm_data_lsa.failure_position)

            # Edit the routing recalculation period
            # TODO(Takashima): routing_recalculation_periodについてNode内の処理負荷の影響を考慮する
            self.routing_recalculation_period.append(
                [current_time, current_time + self.routing_recalculation_interval],
            )

            # TODO(Takashima): 持っているSpaceTimeGraphやForwardingTableTimeList自体が変更されるので, 複製するかどうか検討  # noqa: E501
            # 新しい SpaceTimeGraph の作成
            self.space_time_graph.update_space_time_graph_for_failure_edge(
                comm_data_lsa.failure_position,
                current_time + self.routing_recalculation_interval,
            )
            # 新しい ForwardingTableTimeList の作成
            # TODO(takashima): 新しい SpaceTimeGraph が作成されるたびに ForwardingTableTimeList を作成するのは効率が悪いので, 修正が必要  # noqa: E501
            # TODO(Takashima): caseやhops_limitについて参照するのはコードがややこしくなるので, どこかで設定するようにする  # noqa: E501
            self.forwarding_table_time_list = initialize_forwarding_table_list_from_space_time_graph(
                self.space_time_graph,
                self.target_node,
                weight=weight,
                backup_case=backup_case,
                hops_limit=hops_limit,
            )
            return

        else:
            # TODO(Takashima): failure_position が Node の場合の処理を検討
            return
