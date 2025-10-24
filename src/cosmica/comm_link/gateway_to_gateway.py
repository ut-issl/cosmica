__all__ = [
    "GatewayToGatewayCommLinkCalculator",
]

from collections.abc import Collection

import numpy as np

from cosmica.dtos import DynamicsData
from cosmica.models import Gateway
from cosmica.utils.constants import EARTH_RADIUS, SPEED_OF_LIGHT

from .base import CommLinkPerformance, MemorylessCommLinkCalculator


class GatewayToGatewayCommLinkCalculator(MemorylessCommLinkCalculator[Gateway, Gateway]):
    """Calculate gateway-to-gateway communication link performance using great circle distance."""

    def __init__(
        self,
        *,
        gateway_to_gateway_bandwidth: float,
        refractive_index: float = 1.5,
    ) -> None:
        """Initialize the gateway-to-gateway communication link calculator."""
        self.gateway_to_gateway_bandwidth = gateway_to_gateway_bandwidth
        self.refractive_index = refractive_index

    def calc(
        self,
        edges: Collection[tuple[Gateway, Gateway]],
        *,
        dynamics_data: DynamicsData,  # noqa: ARG002 For interface compatibility
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[tuple[Gateway, Gateway], CommLinkPerformance]:
        """Calculate communication link performance for gateway-to-gateway edges."""
        return {
            edge: self._calc_gateway_to_gateway(
                gateway1=edge[0],
                gateway2=edge[1],
            )
            for edge in edges
        }

    def _calc_gateway_to_gateway(
        self,
        gateway1: Gateway,
        gateway2: Gateway,
    ) -> CommLinkPerformance:
        """Calculate communication link performance between two gateways."""
        # Calculate great circle distance between gateways
        distance = self._calc_great_circle_distance(
            lat1=gateway1.latitude,
            lon1=gateway1.longitude,
            lat2=gateway2.latitude,
            lon2=gateway2.longitude,
        )

        # Calculate propagation delay with refractive index
        propagation_speed = SPEED_OF_LIGHT / self.refractive_index
        delay = distance / propagation_speed

        # Gateway-to-gateway links are always available
        link_available = True

        return CommLinkPerformance(
            link_capacity=self.gateway_to_gateway_bandwidth if link_available else 0.0,
            delay=float(delay),
            link_available=link_available,
        )

    @staticmethod
    def _calc_great_circle_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate the great circle distance between two points on Earth's surface using the haversine formula."""
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return EARTH_RADIUS * c
