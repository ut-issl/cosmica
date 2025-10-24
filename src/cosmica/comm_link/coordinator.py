__all__ = [
    "CommLinkCalculationCoordinator",
]
import logging
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import product

import numpy as np
from tqdm import tqdm

from cosmica.dtos import DynamicsData
from cosmica.models import Node

from .base import CommLinkCalculator, CommLinkPerformance

type EdgeType = tuple[type[Node], type[Node]]

logger = logging.getLogger(__name__)


class CommLinkCalculationCoordinator:
    def __init__(
        self,
        *,
        calculator_assignment: Mapping[EdgeType, CommLinkCalculator],
        directed: bool = False,
    ) -> None:
        self.calculator_assignment = dict(calculator_assignment)
        self.directed = directed

        if not self.directed:
            assert len(set(self.calculator_assignment)) == len(
                self.calculator_assignment,
            ), "All edge types must be unique."

    def _calc(
        self,
        edges: Collection[tuple[Node, Node]],
        *,
        dynamics_data: DynamicsData,
    ) -> dict[tuple[Node, Node], CommLinkPerformance]:
        if any(isinstance(calculator, CommLinkCalculator) for calculator in self.calculator_assignment.values()):
            msg = "Only memoryless calculators are supported. Use calc_time_series instead."
            raise ValueError(msg)
        # Categorize edges based on node types
        edge_group_dd: defaultdict[EdgeType, list[tuple[Node, Node]]] = defaultdict(list)
        for src, dst in edges:
            edge_group_dd[type(src), type(dst)].append((src, dst))
        edge_group = dict(edge_group_dd)

        link_performance: dict[tuple[Node, Node], CommLinkPerformance] = {}
        for edge_type, edges_of_type in edge_group.items():
            if edge_type in self.calculator_assignment:
                calculator = self.calculator_assignment[edge_type]
                link_performance.update(calculator.calc(edges_of_type, dynamics_data=dynamics_data))
            elif not self.directed and edge_type[::-1] in self.calculator_assignment:
                calculator = self.calculator_assignment[(edge_type[1], edge_type[0])]
                edges_inverted = [(dst, src) for src, dst in edges_of_type]
                result_inverted = calculator.calc(edges_inverted, dynamics_data=dynamics_data)
                link_performance.update({(src, dst): result_inverted[(dst, src)] for src, dst in edges_of_type})
            else:
                msg = f"No calculator found for edge type {edge_type}"
                raise ValueError(msg)

        return link_performance

    def calc(
        self,
        edges_time_series: Sequence[Collection[tuple[Node, Node]]],
        *,
        dynamics_data: DynamicsData,
        rng: np.random.Generator,
    ) -> list[dict[tuple[Node, Node], CommLinkPerformance]]:
        assert len(dynamics_data.time) == len(edges_time_series)

        if self.directed:
            msg = "calc_time_series is only supported for undirected calculators."
            raise NotImplementedError(msg)
        return self._calc_undirected(edges_time_series, dynamics_data=dynamics_data, rng=rng)

    def _calc_undirected(
        self,
        edges_time_series: Sequence[Collection[tuple[Node, Node]]],
        *,
        dynamics_data: DynamicsData,
        rng: np.random.Generator,
    ) -> list[dict[tuple[Node, Node], CommLinkPerformance]]:
        supported_edge_types = set(self.calculator_assignment)

        def sort_edge_end_nodes(edge: tuple[Node, Node]) -> tuple[Node, Node]:
            assert len(edge) == 2, "Edge must have exactly two nodes."
            if (type(edge[0]), type(edge[1])) in supported_edge_types:
                return edge
            elif (type(edge[1]), type(edge[0])) in supported_edge_types:
                return edge[::-1]
            else:
                msg = f"No calculator found for edge type {(type(edge[0]), type(edge[1]))}"
                raise ValueError(msg)

        # "Sort" the end nodes so the order matches that of the calculator arguments
        edges_time_series_sorted = [{sort_edge_end_nodes(edge) for edge in edges} for edges in edges_time_series]

        # Categorize edges based on node types
        all_edge_types: set[EdgeType] = {
            (type(src), type(dst)) for edges in edges_time_series_sorted for src, dst in edges
        }
        edges_time_series_by_type_dd: defaultdict[EdgeType, list[set[tuple[Node, Node]]]] = defaultdict(list)
        for edges, edge_type in product(edges_time_series_sorted, all_edge_types):
            edges_per_type = {
                (src, dst) for src, dst in edges if isinstance(src, edge_type[0]) and isinstance(dst, edge_type[1])
            }
            edges_time_series_by_type_dd[edge_type].append(edges_per_type)
        edges_time_series_by_type = dict(edges_time_series_by_type_dd)

        performance_time_series_by_type: dict[EdgeType, list[dict[tuple[Node, Node], CommLinkPerformance]]] = {}
        for edge_type, edges_time_series_of_type in edges_time_series_by_type.items():
            logger.info(
                f"Calculating performance for edge type {edge_type} with {len(edges_time_series_of_type)} time steps.",
            )
            calculator = self.calculator_assignment[edge_type]
            performance_time_series_by_type[edge_type] = calculator.calc(
                edges_time_series_of_type,
                dynamics_data=dynamics_data,
                rng=rng,
            )

        # Merge the results
        performance_time_series = []
        for time_index in tqdm(range(len(dynamics_data.time)), desc="Merging performance results"):
            performance = {}
            for perf_of_type in performance_time_series_by_type.values():
                performance.update(perf_of_type[time_index])
            performance_time_series.append(performance)

        return performance_time_series
