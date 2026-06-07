__all__ = [
    "GatewayToInternetCommLinkCalculator",
    "InternetToGatewayCommLinkCalculator",
]
from collections.abc import Collection

import numpy as np

from cosmica.dtos import DynamicsData
from cosmica.models import Gateway, Internet

from .base import CommLinkPerformance, MemorylessCommLinkCalculator


class GatewayToInternetCommLinkCalculator(MemorylessCommLinkCalculator[Gateway, Internet]):
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
        edges: Collection[tuple[Gateway, Internet]],
        *,
        dynamics_data: DynamicsData,  # noqa: ARG002 For interface compatibility
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[tuple[Gateway, Internet], CommLinkPerformance]:
        return {
            (gateway, internet): CommLinkPerformance(
                link_capacity=self.link_capacity,
                delay=self.delay,
                link_available=True,
            )
            for gateway, internet in edges
        }


class InternetToGatewayCommLinkCalculator(MemorylessCommLinkCalculator[Internet, Gateway]):
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
        edges: Collection[tuple[Internet, Gateway]],
        *,
        dynamics_data: DynamicsData,  # noqa: ARG002 For interface compatibility
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[tuple[Internet, Gateway], CommLinkPerformance]:
        return {
            (internet, gateway): CommLinkPerformance(
                link_capacity=self.internet_to_gateway_link_capacity,
                delay=self.internet_to_gateway_delay,
                link_available=True,
            )
            for internet, gateway in edges
        }
