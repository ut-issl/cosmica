import logging
from collections.abc import Callable
from itertools import combinations

import networkx as nx

from cosmica.models import ConstellationSatellite, Gateway
from cosmica.models.node import Node

from .case_definitions import BackupCaseType
from .forwarding_table import ForwardingTable
from .forwarding_table_time_list import ForwardingTableTimeList
from .space_time_graph import SpaceTimeGraph
from .utility import has_edge_bidirectional

logger = logging.getLogger(__name__)


def initialize_forwarding_table_list_from_space_time_graph(  # noqa: C901
    space_time_graph: SpaceTimeGraph,
    src_node: Node,
    *,
    weight: str | Callable = "weight",
    backup_case: BackupCaseType = "no-backup",
    hops_limit: int = 1,
) -> ForwardingTableTimeList:
    """Calculate the forwarding table for a given space-time graph."""
    if backup_case == "no-backup":
        nominal_forwarding_table_for_snapshots: list[ForwardingTable] = []

        for graph in space_time_graph.graph_for_snapshots:
            nominal_forwarding_table: ForwardingTable = _calc_nominal_forwarding_table_from_graph_optimized(
                graph,
                src_node,
                weight=weight,
            )
            nominal_forwarding_table_for_snapshots.append(nominal_forwarding_table)
        return ForwardingTableTimeList(
            time_for_snapshots=space_time_graph.time_for_snapshots,
            nominal_forwarding_table_for_snapshots=nominal_forwarding_table_for_snapshots,
        )

    elif backup_case == "backup-feeder-links":
        nominal_forwarding_table_for_snapshots = []
        backup_forwarding_tables_for_snapshots: list[dict[frozenset[frozenset[Node]], ForwardingTable]] = []

        for graph in space_time_graph.graph_for_snapshots:
            nominal_forwarding_table = _calc_nominal_forwarding_table_from_graph_optimized(
                graph,
                src_node,
                weight=weight,
            )
            nominal_forwarding_table_for_snapshots.append(nominal_forwarding_table)
            failure_assumed_edges: set = _find_edges_from_all_gateway_to_constellation(graph)
            backup_forwarding_tables: dict[frozenset[frozenset[Node]], ForwardingTable] = (
                _calc_multi_failure_backup_forwarding_table_from_graph(
                    graph,
                    src_node,
                    failure_assumed_edges,
                    combination_range=1,
                    weight=weight,
                )
            )
            backup_forwarding_tables_for_snapshots.append(backup_forwarding_tables)
        return ForwardingTableTimeList(
            time_for_snapshots=space_time_graph.time_for_snapshots,
            nominal_forwarding_table_for_snapshots=nominal_forwarding_table_for_snapshots,
            backup_forwarding_tables_for_snapshots=backup_forwarding_tables_for_snapshots,
        )
    elif backup_case == "backup-n-hops-links":
        nominal_forwarding_table_for_snapshots = []
        backup_forwarding_tables_for_snapshots = []

        for graph in space_time_graph.graph_for_snapshots:
            nominal_forwarding_table = _calc_nominal_forwarding_table_from_graph_optimized(
                graph,
                src_node,
                weight=weight,
            )
            nominal_forwarding_table_for_snapshots.append(nominal_forwarding_table)
            failure_assumed_edges = _find_edges_within_hops_from_source(
                graph,
                src_node,
                hops_limit=hops_limit,
            )
            backup_forwarding_tables = _calc_multi_failure_backup_forwarding_table_from_graph(
                graph,
                src_node,
                failure_assumed_edges,
                combination_range=1,
                weight=weight,
            )
            backup_forwarding_tables_for_snapshots.append(backup_forwarding_tables)
        return ForwardingTableTimeList(
            time_for_snapshots=space_time_graph.time_for_snapshots,
            nominal_forwarding_table_for_snapshots=nominal_forwarding_table_for_snapshots,
            backup_forwarding_tables_for_snapshots=backup_forwarding_tables_for_snapshots,
        )
    elif backup_case == "backup-n-hops-links-and-feeder-links":
        nominal_forwarding_table_for_snapshots = []
        backup_forwarding_tables_for_snapshots = []

        for graph in space_time_graph.graph_for_snapshots:
            nominal_forwarding_table = _calc_nominal_forwarding_table_from_graph_optimized(
                graph,
                src_node,
                weight=weight,
            )
            nominal_forwarding_table_for_snapshots.append(nominal_forwarding_table)
            failure_assumed_edges = _find_edges_within_hops_from_source(
                graph,
                src_node,
                hops_limit=hops_limit,
            ) | _find_edges_from_all_gateway_to_constellation(graph)
            backup_forwarding_tables = _calc_multi_failure_backup_forwarding_table_from_graph(
                graph,
                src_node,
                failure_assumed_edges,
                combination_range=1,
                weight=weight,
            )
            backup_forwarding_tables_for_snapshots.append(backup_forwarding_tables)
        return ForwardingTableTimeList(
            time_for_snapshots=space_time_graph.time_for_snapshots,
            nominal_forwarding_table_for_snapshots=nominal_forwarding_table_for_snapshots,
            backup_forwarding_tables_for_snapshots=backup_forwarding_tables_for_snapshots,
        )
    elif backup_case == "dual-backup-n-hops-links-and-feeder-links":
        nominal_forwarding_table_for_snapshots = []
        backup_forwarding_tables_for_snapshots = []

        for graph in space_time_graph.graph_for_snapshots:
            nominal_forwarding_table = _calc_nominal_forwarding_table_from_graph_optimized(
                graph,
                src_node,
                weight=weight,
            )
            nominal_forwarding_table_for_snapshots.append(nominal_forwarding_table)
            failure_assumed_edges = _find_edges_within_hops_from_source(
                graph,
                src_node,
                hops_limit=hops_limit,
            ) | _find_edges_from_all_gateway_to_constellation(graph)

            backup_forwarding_tables = _calc_multi_failure_backup_forwarding_table_from_graph(
                graph,
                src_node,
                failure_assumed_edges,
                combination_range=2,
                weight=weight,
            )

            # single backupとdual backupを統合
            backup_forwarding_tables_for_snapshots.append(backup_forwarding_tables)
        return ForwardingTableTimeList(
            time_for_snapshots=space_time_graph.time_for_snapshots,
            nominal_forwarding_table_for_snapshots=nominal_forwarding_table_for_snapshots,
            backup_forwarding_tables_for_snapshots=backup_forwarding_tables_for_snapshots,
        )
    # TODO(Takashima): 必要に応じて他のケースを追加
    else:
        logger.warning(f"Backup case '{backup_case}' not recognized. Returning None.")
        return None


def _calc_nominal_forwarding_table_from_graph(
    graph: nx.Graph,
    src_node: Node,
    *,
    weight: str | Callable = "weight",
) -> ForwardingTable:
    """Calculate the forwarding table for a given graph."""
    shortest_paths = dict(nx.single_source_dijkstra_path(graph, src_node, weight=weight))
    del shortest_paths[src_node]

    forwarding_table = ForwardingTable()
    for key, value in shortest_paths.items():
        forwarding_table.update_entry(destination=key, next_node=value[1])
    return forwarding_table


def _calc_nominal_forwarding_table_from_graph_optimized(
    graph: nx.Graph,
    src_node: Node,
    *,
    weight: str | Callable = "weight",
) -> ForwardingTable:
    """Optimized version of nominal forwarding table calculation."""
    try:
        # Use single_source_dijkstra for both path and distance in one call
        _distances, paths = nx.single_source_dijkstra(graph, src_node, weight=weight)
        paths.pop(src_node, None)

        forwarding_table = ForwardingTable()
        for destination, path in paths.items():
            if len(path) > 1:  # Ensure path has at least one hop
                forwarding_table.update_entry(destination=destination, next_node=path[1])
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        forwarding_table = ForwardingTable()

    return forwarding_table


def _calc_multi_failure_backup_forwarding_table_from_graph(
    graph: nx.Graph,
    src_node: Node,
    failure_assumed_edges: set[tuple[Node, Node]],
    combination_range: int,
    *,
    weight: str | Callable = "weight",
) -> dict[frozenset[frozenset[Node]], ForwardingTable]:
    """Calculate the forwarding table considering multiple simultaneous link failures.

    Args:
        graph: The network graph
        src_node: Source node for calculating paths
        failure_assumed_edges: Set of edges that can potentially fail
        combination_range: Maximum number of simultaneous failures to consider.
                         For example, if combination_range=3, it will generate scenarios
                         for 1, 2, and 3 simultaneous edge failures.
        weight: Weight function for shortest path calculation

    Returns:
        Dictionary mapping failure scenarios to forwarding tables

    """
    forwarding_table_dict: dict = {}

    # Pre-calculate all shortest paths for reference
    all_shortest_paths = dict(nx.single_source_dijkstra_path(graph, src_node, weight=weight))
    all_shortest_paths.pop(src_node, None)

    # Generate all combinations of edges from 1 to combination_range
    for failure_count in range(1, combination_range + 1):
        for edge_combination in combinations(failure_assumed_edges, failure_count):
            # Use graph view instead of deep copy for better performance
            modified_graph = _create_graph_without_edges(graph, edge_combination)

            try:
                shortest_paths = dict(nx.single_source_dijkstra_path(modified_graph, src_node, weight=weight))
                shortest_paths.pop(src_node, None)

                forwarding_table = ForwardingTable()
                for key, value in shortest_paths.items():
                    if len(value) > 1:  # Ensure path has at least one hop
                        forwarding_table.update_entry(destination=key, next_node=value[1])

                # Convert tuple of tuples to frozenset of frozensets for orderless representation
                edge_key = frozenset(frozenset([edge[0], edge[1]]) for edge in edge_combination)
                forwarding_table_dict[edge_key] = forwarding_table
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # If there's no path to the source after failures, create empty forwarding table
                edge_key = frozenset(frozenset([edge[0], edge[1]]) for edge in edge_combination)
                forwarding_table_dict[edge_key] = ForwardingTable()

    return forwarding_table_dict


def _batch_process_failure_scenarios(
    graph: nx.Graph,
    src_node: Node,
    failure_scenarios: set[tuple],
    *,
    weight: str | Callable = "weight",
) -> dict[frozenset[frozenset[Node]], ForwardingTable]:
    """Process multiple failure scenarios efficiently using batch processing."""
    forwarding_table_dict = {}

    for edge_combination in failure_scenarios:
        # Use the optimized graph view method
        modified_graph = _create_graph_without_edges(graph, edge_combination)

        try:
            # Use the optimized forwarding table calculation
            forwarding_table = _calc_nominal_forwarding_table_from_graph_optimized(
                modified_graph,
                src_node,
                weight=weight,
            )
            # Convert tuple of tuples to frozenset of frozensets for orderless representation
            edge_key = frozenset(frozenset([edge[0], edge[1]]) for edge in edge_combination)
            forwarding_table_dict[edge_key] = forwarding_table
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            edge_key = frozenset(frozenset([edge[0], edge[1]]) for edge in edge_combination)
            forwarding_table_dict[edge_key] = ForwardingTable()

    return forwarding_table_dict


def _create_graph_without_edges(graph: nx.Graph, edges_to_remove: tuple) -> nx.Graph:
    """Create a graph view without specified edges."""

    # Create a subgraph view that excludes the specified edges
    def edge_filter(u: Node, v: Node) -> bool:
        return (u, v) not in edges_to_remove and (v, u) not in edges_to_remove

    return nx.subgraph_view(graph, filter_edge=edge_filter)


def _find_edges_from_nearest_gateway_to_constellation(
    graph: nx.Graph,
    src_node: Node,
    *,
    weight: str | Callable = "weight",
) -> set:
    """Find the nearest gateway to the source node and returns the associated edges."""
    gateways: set[Gateway] = {node for node in graph.nodes if isinstance(node, Gateway)}
    constellation_satellites: set[ConstellationSatellite] = {
        node for node in graph.nodes if isinstance(node, ConstellationSatellite)
    }

    nearest_gateways: set = set()
    min_distance = float("inf")

    for gateway in gateways:
        try:
            distance = nx.dijkstra_path_length(graph, src_node, gateway, weight)
            if distance < min_distance:
                min_distance = distance
                nearest_gateways = {gateway}
            elif distance == min_distance:
                nearest_gateways.add(gateway)
        except nx.NetworkXNoPath:
            continue

    return {
        (gw, sat)
        for gw in nearest_gateways
        for sat in constellation_satellites
        if has_edge_bidirectional(graph, gw, sat)
    }


def _find_edges_from_all_gateway_to_constellation(
    graph: nx.Graph,
) -> set:
    """Find all edges between gateways and constellation satellites."""
    edges_between_gateway_and_constellation = set()

    for u, v in graph.edges():
        # 片側がGateway、片側がConstellationSatelliteかチェック
        if (isinstance(u, Gateway) and isinstance(v, ConstellationSatellite)) or (
            isinstance(u, ConstellationSatellite) and isinstance(v, Gateway)
        ):
            edges_between_gateway_and_constellation.add((u, v))

    return edges_between_gateway_and_constellation


def _find_edges_within_hops_from_source(
    graph: nx.Graph,
    src_node: Node,
    *,
    hops_limit: int = 1,
) -> set:
    """Find all edges within a specified number of hops from the source node."""
    # 幅優先探索 (BFS) により指定されたホップ数以内のエッジを収集
    edges_within_n_hops: set = set()
    visited_nodes: set[Node] = set()
    queue: list[tuple[Node, int]] = [(src_node, 0)]
    visited_nodes.add(src_node)

    while queue:
        current_node, hop_count = queue.pop(0)
        if hop_count < hops_limit:
            for neighbor in graph.neighbors(current_node):
                edge = (current_node, neighbor)
                reverse_edge = (neighbor, current_node)

                # より効率的なエッジの重複チェック
                if edge not in edges_within_n_hops and reverse_edge not in edges_within_n_hops:
                    edges_within_n_hops.add(edge)

                # 訪問済みノードのチェックを追加して冗長な処理を回避
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, hop_count + 1))

    return edges_within_n_hops
