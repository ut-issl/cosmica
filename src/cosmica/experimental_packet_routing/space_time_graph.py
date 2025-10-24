__all__ = [
    "SpaceTimeGraph",
]

import copy
from dataclasses import dataclass
from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx import Graph
from tqdm import tqdm

from cosmica.models.node import Node

from .utility import has_edge_bidirectional, remove_edge_safe


@dataclass(kw_only=True, slots=True)
class SpaceTimeGraph:
    """Base model for a space time graph."""

    time_for_snapshots: list[np.datetime64]
    graph_for_snapshots: list[Graph]

    def __init__(
        self,
        time_for_snapshots: list[np.datetime64],
        graph_for_snapshots: list[Graph],
    ) -> None:
        self.time_for_snapshots = time_for_snapshots
        self.graph_for_snapshots = graph_for_snapshots

    @classmethod
    def make_space_time_graph_from_graph(  # noqa: C901, PLR0912
        cls,
        time: npt.NDArray[np.datetime64],
        graphs: list[Graph],
        *,
        check_interval_time: np.timedelta64 | None = None,
    ) -> Self:
        """Update time and graph lists if the graph shape changes."""
        # ──────────────────────────────────────────────────────────────────────
        if check_interval_time is None:
            # Old behaviour (unchanged) ───────────────────────────────────────
            time_for_snapshots: list[np.datetime64] = [time[0]]
            graph_for_snapshots: list[Graph] = [copy.deepcopy(graphs[0])]
            for _time, _graph in tqdm(zip(time[1:], graphs[1:], strict=False), total=len(time) - 1):
                if not nx.is_isomorphic(_graph, graph_for_snapshots[-1]):
                    time_for_snapshots.append(_time)
                    graph_for_snapshots.append(copy.deepcopy(_graph))
            return cls(
                time_for_snapshots=time_for_snapshots,
                graph_for_snapshots=graph_for_snapshots,
            )

        # ──────────────────────────────────────────────────────────────────────
        # New t-second aggregation logic
        # ──────────────────────────────────────────────────────────────────────
        start, stop = time[0], time[-1]
        # Interval boundaries: t, 2t, …, ≤ stop
        boundaries: npt.NDArray[np.datetime64] = np.arange(
            start,
            stop + check_interval_time,
            check_interval_time,
        )

        time_for_snapshots = []
        graph_for_snapshots = []

        for boundary in tqdm(boundaries, total=len(boundaries)):
            # ── indices for past / future windows ────────────────────────────
            past_mask = (time > boundary - check_interval_time) & (time <= boundary)
            future_mask = (time > boundary) & (time < boundary + check_interval_time)

            if not past_mask.any() and not future_mask.any():
                continue  # no information

            past_idx = np.where(past_mask)[0]
            future_idx = np.where(future_mask)[0]

            # ── graph defined exactly at (or just before) boundary ───────────
            idx_boundary = max(0, int(np.searchsorted(time, boundary, side="right")) - 1)
            graph_at_boundary = copy.deepcopy(graphs[idx_boundary])

            # ── ① add every edge that ever appeared in the past window ───────
            edges_past = set().union(*(graphs[i].edges() for i in past_idx)) if past_idx.size else set()
            for u, v in edges_past:
                if not has_edge_bidirectional(graph_at_boundary, u, v):
                    last_i = max(i for i in past_idx if has_edge_bidirectional(graphs[i], u, v))
                    # Find the actual edge direction that exists
                    if graphs[last_i].has_edge(u, v):
                        graph_at_boundary.add_edge(u, v, **graphs[last_i].edges[u, v])
                    else:
                        graph_at_boundary.add_edge(u, v, **graphs[last_i].edges[v, u])

            # ── ② remove edges that will disappear in the coming interval ────
            if future_idx.size:
                stable_future = set.intersection(*(set(graphs[i].edges()) for i in future_idx))
                for u, v in list(graph_at_boundary.edges()):
                    if (u, v) not in stable_future and (v, u) not in stable_future:
                        remove_edge_safe(graph_at_boundary, u, v)

            # ── ③ overwrite attributes with those at the mid-point (n+½)t ───
            half = check_interval_time // np.int64(2)
            mid_time = boundary + half  # (n+½)t
            mid_i = min(len(time) - 1, int(np.searchsorted(time, mid_time, side="left")))
            graph_at_midpoint = graphs[mid_i]

            for node, data in graph_at_midpoint.nodes(data=True):
                if graph_at_boundary.has_node(node):
                    graph_at_boundary.nodes[node].update(data)

            for u, v, data in graph_at_midpoint.edges(data=True):
                if has_edge_bidirectional(graph_at_boundary, u, v):
                    # Update the edge that actually exists in the boundary graph
                    if graph_at_boundary.has_edge(u, v):
                        graph_at_boundary.edges[u, v].update(data)
                    else:
                        graph_at_boundary.edges[v, u].update(data)

            # ── ④ recompute weights (edge delay + max node delay) ───────────
            for u, v, d in graph_at_boundary.edges(data=True):
                edge_delay = d.get("delay", 0)
                node_delay = max(graph_at_boundary.nodes[u].get("delay", 0), graph_at_boundary.nodes[v].get("delay", 0))
                graph_at_boundary.edges[u, v]["weight"] = edge_delay + node_delay
            for n in graph_at_boundary.nodes:
                graph_at_boundary.nodes[n]["weight"] = graph_at_boundary.nodes[n].get("delay", 0)

            # ── ⑤ store snapshot, skipping duplicates ───────────────────────
            if (not graph_for_snapshots) or (not nx.is_isomorphic(graph_at_boundary, graph_for_snapshots[-1])):
                time_for_snapshots.append(boundary)
                graph_for_snapshots.append(graph_at_boundary)

        return cls(
            time_for_snapshots=time_for_snapshots,
            graph_for_snapshots=graph_for_snapshots,
        )

    def update_space_time_graph_for_failure_edge(
        self,
        failure_edge: tuple[Node, Node],
        update_time: np.datetime64,
    ) -> Self:
        """Update the space time graph for the failure edge."""
        # Find the index of the closest time snapshot before or at graph_update_time
        closest_snapshot_index: int = max(i for i, time in enumerate(self.time_for_snapshots) if time <= update_time)

        # If the closest snapshot is the last snapshot, add a new snapshot
        if closest_snapshot_index == len(self.time_for_snapshots) - 1:
            graph: Graph = copy.deepcopy(self.graph_for_snapshots[closest_snapshot_index])
            if remove_edge_safe(graph, *failure_edge):
                self.time_for_snapshots.append(update_time)
                self.graph_for_snapshots.append(graph)

        else:
            for i in range(closest_snapshot_index + 1, len(self.time_for_snapshots)):
                graph = self.graph_for_snapshots[i]
                remove_edge_safe(graph, *failure_edge)

            graph = copy.deepcopy(self.graph_for_snapshots[closest_snapshot_index])
            if remove_edge_safe(graph, *failure_edge):
                self.time_for_snapshots.insert(closest_snapshot_index + 1, update_time)
                self.graph_for_snapshots.insert(closest_snapshot_index + 1, graph)

        return self

    def get_space_time_graph_at_time(self, time: np.datetime64) -> Graph:
        """Get the space time graph at the specified time."""
        closest_snapshot_index: int = max(i for i, _time in enumerate(self.time_for_snapshots) if _time <= time)
        return self.graph_for_snapshots[closest_snapshot_index]
