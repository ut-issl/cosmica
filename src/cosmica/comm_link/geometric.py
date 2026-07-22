__all__ = [
    "GeometricCommLinkCalculator",
]
import warnings
from collections.abc import Collection
from itertools import chain
from typing import Any

import numpy as np
import numpy.typing as npt

from cosmica.dtos import DynamicsData
from cosmica.models import ConstellationSatellite, Gateway, Satellite
from cosmica.utils.constants import SPEED_OF_LIGHT
from cosmica.utils.coordinates import ecef2aer

from .base import CommLinkPerformance, MemorylessCommLinkCalculator

# TODO(Nomura): Separate different calculators into different classes
# so that the user can choose which calculator to use for each edge type.


type _NodeType = ConstellationSatellite[Any, Any] | Gateway[Any]
type _EdgeType = tuple[_NodeType, _NodeType]


class GeometricCommLinkCalculator(MemorylessCommLinkCalculator[_NodeType, _NodeType]):
    """Calculate geometric communication link performance for each directed edge in a network.

    Each input edge (src, dst) is the directed link src -> dst and gets its own entry.
    """

    def __init__(
        self,
        *,
        inter_satellite_link_capacity: float,
        satellite_to_gateway_link_capacity: float,
        max_inter_satellite_distance: float = float("inf"),
        lowest_altitude: float = 0.0,
        max_relative_angular_velocity: float = float("inf"),
    ) -> None:
        warnings.warn(
            "GeometricCommLinkCalculator is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.max_inter_satellite_distance = max_inter_satellite_distance
        self.inter_satellite_link_capacity = inter_satellite_link_capacity
        self.satellite_to_gateway_link_capacity = satellite_to_gateway_link_capacity
        self.lowest_altitude = lowest_altitude
        self.max_relative_angular_velocity = max_relative_angular_velocity

    def calc(
        self,
        edges: Collection[_EdgeType],
        *,
        dynamics_data: DynamicsData[Satellite[Any]],
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[_EdgeType, CommLinkPerformance]:
        """Calculate geometric communication link performance for each directed edge in a network."""
        performance: dict[_EdgeType, CommLinkPerformance] = {}
        for node1, node2 in edges:
            if isinstance(node1, ConstellationSatellite) and isinstance(node2, ConstellationSatellite):
                performance[(node1, node2)] = self._calc_satellite_to_satellite(
                    positions_eci=(
                        dynamics_data.satellite_position_eci[node1],
                        dynamics_data.satellite_position_eci[node2],
                    ),
                    velocities_eci=(
                        dynamics_data.satellite_velocity_eci[node1],
                        dynamics_data.satellite_velocity_eci[node2],
                    ),
                    attitude_angular_velocities_eci=(
                        dynamics_data.satellite_attitude_angular_velocity_eci[node1],
                        dynamics_data.satellite_attitude_angular_velocity_eci[node2],
                    ),
                )
            elif isinstance(node1, ConstellationSatellite) and isinstance(node2, Gateway):
                performance[(node1, node2)] = self._calc_satellite_to_gateway(
                    dynamics_data.satellite_position_ecef[node1],
                    node2,
                )
            elif isinstance(node1, Gateway) and isinstance(node2, ConstellationSatellite):
                performance[(node1, node2)] = self._calc_satellite_to_gateway(
                    dynamics_data.satellite_position_ecef[node2],
                    node1,
                )
            else:
                msg = f"Invalid edge: {node1} -> {node2}"
                raise TypeError(msg)
        return performance

    def _calc_satellite_to_satellite(
        self,
        *,
        positions_eci: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
        velocities_eci: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
        attitude_angular_velocities_eci: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    ) -> CommLinkPerformance:
        for vec in chain(positions_eci, velocities_eci):
            assert vec.shape == (3,)

        distance = np.linalg.norm(positions_eci[1] - positions_eci[0])

        # Calculate relative angular velocity due to translational motion
        # The calculated angular velocity is that of the second satellite relative to the first satellite.
        relative_position_eci = positions_eci[1] - positions_eci[0]
        relative_velocity_eci = velocities_eci[1] - velocities_eci[0]
        relative_angular_velocity_translational_eci = (
            np.cross(relative_position_eci, relative_velocity_eci) / distance**2
        )

        relative_angular_velocities = (
            relative_angular_velocity_translational_eci - attitude_angular_velocities_eci[0],
            # Note: The relative angular velocity should be negated to get the angular velocity of the first satellite
            # relative to the second satellite.
            -relative_angular_velocity_translational_eci - attitude_angular_velocities_eci[1],
        )

        link_available = distance < self.max_inter_satellite_distance and all(
            np.linalg.norm(relative_angular_velocity) < self.max_relative_angular_velocity
            for relative_angular_velocity in relative_angular_velocities
        )

        return CommLinkPerformance(
            link_capacity=self.inter_satellite_link_capacity if link_available else 0.0,
            delay=float(distance / SPEED_OF_LIGHT),
            link_available=bool(link_available),
        )

    def _calc_satellite_to_gateway(
        self,
        satellite_position_ecef: npt.NDArray[np.floating],
        gateway: Gateway[Any],
    ) -> CommLinkPerformance:
        assert satellite_position_ecef.shape == (3,)
        _, elevation, srange = ecef2aer(
            x=satellite_position_ecef[0],
            y=satellite_position_ecef[1],
            z=satellite_position_ecef[2],
            lat0=gateway.latitude,
            lon0=gateway.longitude,
            h0=gateway.altitude,
            deg=False,
        )
        link_available = bool(elevation >= gateway.minimum_elevation)
        return CommLinkPerformance(
            link_capacity=self.satellite_to_gateway_link_capacity if link_available else 0.0,
            delay=float(srange / SPEED_OF_LIGHT),
            link_available=link_available,
        )
