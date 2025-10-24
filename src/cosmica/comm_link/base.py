__all__ = [
    "CommLinkPerformance",
    "MemorylessCommLinkCalculator",
    "MemorylessCommLinkCalculatorWrapper",
]
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from typing import TypedDict

import numpy as np

from cosmica.dtos import DynamicsData
from cosmica.models import Node


# Use TypedDict instead of dataclass so it can be treated nicely by nx.set_edge_attributes
# Ref: https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.set_edge_attributes.html
class CommLinkPerformance(TypedDict):
    link_capacity: float  # [bps]
    delay: float  # [s]
    link_available: bool


class MemorylessCommLinkCalculator[T: Node, U: Node](ABC):
    @abstractmethod
    def calc(
        self,
        edges: Collection[tuple[T, U]],
        *,
        dynamics_data: DynamicsData,
        rng: np.random.Generator,
    ) -> dict[tuple[T, U], CommLinkPerformance]:
        """Calculate communication link performance for each edge in a network."""
        raise NotImplementedError


class CommLinkCalculator[T: Node, U: Node](ABC):
    @abstractmethod
    def calc(
        self,
        edges_time_series: Sequence[Collection[tuple[T, U]]],
        *,
        dynamics_data: DynamicsData,
        rng: np.random.Generator,
    ) -> list[dict[tuple[T, U], CommLinkPerformance]]:
        """Calculate communication link performance for each edge in a network."""
        raise NotImplementedError


class MemorylessCommLinkCalculatorWrapper[T: Node, U: Node](CommLinkCalculator[T, U]):
    """Convert a memoryless calculator to a time series calculator by calling the memoryless one at each time step."""

    def __init__(
        self,
        memoryless_calculator: MemorylessCommLinkCalculator[T, U],
        /,
    ) -> None:
        self.memoryless_calculator = memoryless_calculator

    def calc(
        self,
        edges_time_series: Sequence[Collection[tuple[T, U]]],
        *,
        dynamics_data: DynamicsData,
        rng: np.random.Generator,
    ) -> list[dict[tuple[T, U], CommLinkPerformance]]:
        assert len(edges_time_series) == len(
            dynamics_data.time,
        ), "edges_time_series must have the same length as dynamics_data.time"
        return [
            self.memoryless_calculator.calc(edges=edges, dynamics_data=dynamics_data[time_index], rng=rng)
            for time_index, edges in enumerate(edges_time_series)
        ]
