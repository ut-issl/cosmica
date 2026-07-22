__all__ = [
    "InternetToGatewayCommLinkCalculator",
]
from collections.abc import Collection
from typing import Any

import numpy as np

from cosmica.dtos import DynamicsData
from cosmica.models import Gateway, Internet, Satellite

from .base import CommLinkPerformance, MemorylessCommLinkCalculator


class InternetToGatewayCommLinkCalculator(MemorylessCommLinkCalculator[Internet[Any], Gateway[Any]]):
    """Calculate the internet -> gateway communication link performance."""

    def __init__(
        self,
        *,
        link_capacity: float,
        delay: float,
    ) -> None:
        self.link_capacity = link_capacity
        self.delay = delay

    def calc(
        self,
        edges: Collection[tuple[Internet[Any], Gateway[Any]]],
        *,
        dynamics_data: DynamicsData[Satellite[Any]],  # noqa: ARG002 For interface compatibility
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[tuple[Internet[Any], Gateway[Any]], CommLinkPerformance]:
        return {
            (internet, gateway): CommLinkPerformance(
                link_capacity=self.link_capacity,
                delay=self.delay,
                link_available=True,
            )
            for internet, gateway in edges
        }
