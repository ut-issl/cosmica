__all__ = [
    "GatewayToInternetCommLinkCalculator",
]
from collections.abc import Collection
from typing import Any

import numpy as np

from cosmica.dtos import DynamicsData
from cosmica.models import Gateway, Internet, Satellite

from .base import CommLinkPerformance, MemorylessCommLinkCalculator


class GatewayToInternetCommLinkCalculator(MemorylessCommLinkCalculator[Gateway[Any], Internet[Any]]):
    """Calculate the gateway -> internet communication link performance."""

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
        edges: Collection[tuple[Gateway[Any], Internet[Any]]],
        *,
        dynamics_data: DynamicsData[Satellite[Any]],  # noqa: ARG002 For interface compatibility
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[tuple[Gateway[Any], Internet[Any]], CommLinkPerformance]:
        return {
            (gateway, internet): CommLinkPerformance(
                link_capacity=self.link_capacity,
                delay=self.delay,
                link_available=True,
            )
            for gateway, internet in edges
        }
