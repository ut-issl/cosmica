import warnings
from typing import TYPE_CHECKING

import numpy as np

from cosmica.models.node import Node

from .node_knowledge import NodeKnowledge

if TYPE_CHECKING:
    from networkx import Graph

    from cosmica.experimental_packet_routing.forwarding_table import ForwardingTableInformation

    from .forwarding_table import ForwardingTable


def calc_next_node_from_node_knowledge(  # noqa: C901, PLR0912, PLR0915
    node_knowledge: NodeKnowledge,
    dst_node: Node,
    path: list[Node],
    current_time: np.datetime64,
    *,
    enable_random_routing_when_edge_failure: bool = False,
    prevent_loop: bool = False,
) -> Node | None:
    """Calculate the next node from the network information."""
    rng: np.random.Generator = np.random.default_rng()

    # Step 1: Find the index of the closest time snapshot before or at current_time
    time_index = max(
        i for i, time in enumerate(node_knowledge.forwarding_table_time_list.time_for_snapshots) if time <= current_time
    )

    # Step 2: Select the forwarding table
    do_backup: bool = False

    # Use the nominal forwarding table for the identified time snapshot
    forwarding_table: ForwardingTable = (
        node_knowledge.forwarding_table_time_list.nominal_forwarding_table_for_snapshots[time_index]
    )

    # TODO(Takashima): 3つ以上のリンク切断バックアップの処理を検討する

    # Backup forwarding tables (both single and dual) are used when the routing recalculation period is active
    if node_knowledge.forwarding_table_time_list.backup_forwarding_tables_for_snapshots:
        backup_forwarding_tables: dict[frozenset[frozenset[Node]], ForwardingTable] = (
            node_knowledge.forwarding_table_time_list.backup_forwarding_tables_for_snapshots[time_index]
        )
        if any(start <= current_time <= end for start, end in node_knowledge.routing_recalculation_period):
            # Dual link backup forwarding tables
            dual_backup_forwarding_table_candidates: dict[frozenset[frozenset[Node]], ForwardingTable] = {}

            for failure_assumed_edges, backup_table in backup_forwarding_tables.items():
                if len(failure_assumed_edges) == 2:  # Dual backup case
                    # frozensetを使用しているので、集合演算で直接比較可能
                    failure_edge_frozensets = {frozenset([u, v]) for u, v in node_knowledge.failure_assumed_edge_list}
                    if failure_assumed_edges.issubset(failure_edge_frozensets):
                        dual_backup_forwarding_table_candidates[failure_assumed_edges] = backup_table

            # TODO(Takashima): 候補が複数ある場合の処理については、もう少し検討する
            if len(dual_backup_forwarding_table_candidates) == 0:
                pass
            elif len(dual_backup_forwarding_table_candidates) == 1:
                forwarding_table = next(iter(dual_backup_forwarding_table_candidates.values()))
                do_backup = True
            elif len(dual_backup_forwarding_table_candidates) > 1:
                pass

        # Single link backup forwarding tables are used when the routing recalculation period is active
        if any(start <= current_time <= end for start, end in node_knowledge.routing_recalculation_period):
            # TODO(Takashima): 複数の同時リンク切断への対応方法は色々あるので検討する
            single_backup_forwarding_table_candidates: dict[frozenset[frozenset[Node]], ForwardingTable] = {}
            # TODO(Takashima): エッジを探すときに、frozensetを使用しているため順序を気にする必要はない
            for edge_key, backup_table in backup_forwarding_tables.items():
                if len(edge_key) == 1:  # Single edge failure
                    edge_set = next(iter(edge_key))  # Get the single edge frozenset
                    # frozensetなので順序を気にせずに直接比較可能
                    failure_edge_frozensets = {frozenset([u, v]) for u, v in node_knowledge.failure_assumed_edge_list}
                    if edge_set in failure_edge_frozensets:
                        single_backup_forwarding_table_candidates[edge_key] = backup_table

            # TODO(Takashima): 候補が複数ある場合の処理については、もう少し検討する
            if len(single_backup_forwarding_table_candidates) == 0:
                pass
            elif len(single_backup_forwarding_table_candidates) == 1:
                forwarding_table = next(iter(single_backup_forwarding_table_candidates.values()))
                do_backup = True
            elif len(single_backup_forwarding_table_candidates) > 1:
                # # すべてのバックアップフォワーディングテーブルの次宛先が同じ場合は、バックアップフォワーディングテーブルのいずれかを選択する  # noqa: E501
                # if all(
                #     forwarding_table_candidate.find_entry(dst_node).next_node
                #     == next(iter(forwarding_table_candidates.values())).find_entry(dst_node).next_node
                #     for forwarding_table_candidate in forwarding_table_candidates.values()
                # ):
                #     forwarding_table = next(iter(forwarding_table_candidates.values()))
                #     do_backup = True

                # # (u,v)が地上局-コンステ間のエッジの場合は、そのエントリを選択する
                # for (u, v), forwarding_table_candidate in forwarding_table_candidates.items():
                #     if (isinstance(u, Gateway) and isinstance(v, ConstellationSatellite)) or (
                #         isinstance(v, Gateway) and isinstance(u, ConstellationSatellite)
                #     ):
                #         forwarding_table = forwarding_table_candidate
                #         do_backup = True
                #         break
                pass

        # ループを防ぐために、直前に通ったエッジのバックアップテーブルを保持している場合は、そのエッジのバックアップフォワーディングテーブルを使用する  # noqa: E501
        # ただしバックアップフォワーディングテーブルを使用する場合は、そのバックアップフォワーディングテーブルに従う
        if prevent_loop and (not do_backup) and (len(path) >= 2):
            # frozensetを使用しているので順序を気にせずに直接検索可能
            edge_key = frozenset([frozenset([path[-2], path[-1]])])

            if edge_key in backup_forwarding_tables:
                forwarding_table = backup_forwarding_tables[edge_key]

    # Step 3: Find the forwarding entry for the destination node
    forwarding_entry: ForwardingTableInformation | None = forwarding_table.find_entry(dst_node)

    if forwarding_entry:
        # Step 4: Return the next node for the destination
        # If random routing is enabled, return a random next node from the list of next nodes
        next_node: Node | None = forwarding_entry.next_node
    else:
        # If no forwarding entry is found, raise an exception or return random next node
        msg: str = f"No forwarding entry found for destination node {dst_node}"
        warnings.warn(msg, stacklevel=2)
        next_node = None

    # リンク切断時にランダムにnext nodeを選択する
    # (エッジが切断されていることを知っているが, フォワーディングテーブルには反映されていない場合)
    if enable_random_routing_when_edge_failure:
        graph: Graph = node_knowledge.space_time_graph.graph_for_snapshots[time_index]
        src_node: Node = node_knowledge.target_node
        neighbor_nodes = list(graph.neighbors(src_node))
        if next_node is None or (
            {(src_node, next_node), (next_node, src_node)} & set(node_knowledge.failure_assumed_edge_list)
        ):
            next_node = rng.choice(neighbor_nodes)

    # next_nodeが過去通ってきたパスに含まれる場合に、パスに含まれない隣接ノードをランダムに選択する
    # 隣接の全てのノードが過去通ってきたパスに含まれる場合は、ランダムに選択する
    if prevent_loop and (not do_backup) and (next_node in path):
        graph = node_knowledge.space_time_graph.graph_for_snapshots[time_index]
        src_node = node_knowledge.target_node
        neighbor_nodes = list(graph.neighbors(src_node))
        if all(node in path for node in neighbor_nodes):
            next_node = rng.choice(neighbor_nodes)
        else:
            remain_neighbor_nodes: list = [node for node in neighbor_nodes if node not in path]
            next_node = rng.choice(remain_neighbor_nodes)

    return next_node
