__all__ = [
    "GatewayToInternetCommLinkCalculator",
]
from collections.abc import Collection

import numpy as np

from cosmica.dtos import DynamicsData
from cosmica.models import Gateway, Internet

from .base import CommLinkPerformance, MemorylessCommLinkCalculator


class GatewayToInternetCommLinkCalculator(MemorylessCommLinkCalculator[Gateway, Internet]):
    """Calculates the communication link between a gateway and an internet."""

    def __init__(
        self,
        *,
        gateway_to_internet_link_capacity: float,
        gateway_to_internet_delay: float,
    ) -> None:
        self.gateway_to_internet_link_capacity = gateway_to_internet_link_capacity
        self.gateway_to_internet_delay = gateway_to_internet_delay

    def calc(
        self,
        edges: Collection[tuple[Gateway, Internet]],
        *,
        dynamics_data: DynamicsData,  # noqa: ARG002 For interface compatibility
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[tuple[Gateway, Internet], CommLinkPerformance]:
        return {
            (gateway, internet): CommLinkPerformance(
                link_capacity=self.gateway_to_internet_link_capacity,
                delay=self.gateway_to_internet_delay,
                link_available=True,
            )
            for gateway, internet in edges
        }
