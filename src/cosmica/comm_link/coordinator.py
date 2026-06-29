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
    """Coordinate comm link calculators over directed topology graphs.

    Each directed edge (src, dst) is dispatched as-is to the calculator registered for the
    exact node-type pair (type(src), type(dst)); a calculator handles exactly one link
    direction. Since directed topology graphs represent every physical link as two directed
    edges, a calculator must be registered for BOTH orientations of each link type — e.g.
    (Satellite, Gateway) for the downlink and (Gateway, Satellite) for the uplink.
    """

    def __init__(
        self,
        *,
        calculator_assignment: Mapping[EdgeType, CommLinkCalculator],
    ) -> None:
        self.calculator_assignment = dict(calculator_assignment)

    def calc(
        self,
        edges_time_series: Sequence[Collection[tuple[Node, Node]]],
        *,
        dynamics_data: DynamicsData,
        rng: np.random.Generator,
    ) -> list[dict[tuple[Node, Node], CommLinkPerformance]]:
        assert len(dynamics_data.time) == len(edges_time_series)

        # Categorize edges by (source type, destination type), keeping edge direction as-is
        all_edge_types: set[EdgeType] = {(type(src), type(dst)) for edges in edges_time_series for src, dst in edges}
        unsupported_edge_types = all_edge_types - set(self.calculator_assignment)
        if unsupported_edge_types:
            msg = (
                f"No calculator registered for edge types: "
                f"{sorted((t1.__name__, t2.__name__) for t1, t2 in unsupported_edge_types)}. "
                "Note that both directions of a physical link need a registered calculator "
                "(e.g. (Satellite, Gateway) for the downlink and (Gateway, Satellite) for the uplink)."
            )
            raise ValueError(msg)

        edges_time_series_by_type_dd: defaultdict[EdgeType, list[set[tuple[Node, Node]]]] = defaultdict(list)
        for edges, edge_type in product(edges_time_series, all_edge_types):
            # Match on the exact node types (consistent with the registration check above) so that
            # each edge belongs to exactly one edge type even if both a class and its base class
            # are registered; isinstance matching would dispatch such an edge to both calculators.
            edges_per_type = {(src, dst) for src, dst in edges if (type(src), type(dst)) == edge_type}
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
            performance: dict[tuple[Node, Node], CommLinkPerformance] = {}
            for perf_of_type in performance_time_series_by_type.values():
                performance.update(perf_of_type[time_index])
            performance_time_series.append(performance)

        return performance_time_series
