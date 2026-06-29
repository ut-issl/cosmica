__all__ = ["SatToSatBinaryCommLinkCalculatorWithRateCalc"]

from collections.abc import Collection
from itertools import chain
from typing import Annotated

import numpy as np
import numpy.typing as npt
from typing_extensions import Doc

from cosmica.dtos import DynamicsData
from cosmica.models import Satellite
from cosmica.utils.constants import BOLTZ_CONST, SPEED_OF_LIGHT
from cosmica.utils.vector import angle_between, is_satellite_in_eclipse

from .base import CommLinkPerformance, MemorylessCommLinkCalculator


class SatToSatBinaryCommLinkCalculatorWithRateCalc(MemorylessCommLinkCalculator[Satellite, Satellite]):
    """Calculate satellite-to-satellite communication link performance for each directed edge.

    The link performance is calculated as a binary value, i.e., 1 if the link is available and 0 otherwise.

    Each input edge (src, dst) is the directed link src -> dst and gets its own entry.
    Both directions of a physical link are handled by whatever registrations the user
    sets up (a single registration for homogeneous satellite types, or one per
    orientation for heterogeneous types such as user satellite <-> constellation).
    """

    def __init__(
        self,
        *,
        max_inter_satellite_distance: float = float("inf"),
        lowest_altitude: float = 0.0,
        max_relative_angular_velocity: float = float("inf"),
        sun_exclusion_angle: float = 0.0,
        lct_power: float = 1.0,
        available_link_capacity: float = 1e9,
        lna_gain: float,
        noise_figure: float = 4,
    ) -> None:
        self.max_inter_satellite_distance = max_inter_satellite_distance
        self.lowest_altitude = lowest_altitude
        self.max_relative_angular_velocity = max_relative_angular_velocity
        self.sun_exclusion_angle = sun_exclusion_angle
        self.lct_p0 = lct_power
        self.link_capacity = available_link_capacity
        self.lna_gain = lna_gain
        self.noise_figure = noise_figure

    def calc(
        self,
        edges: Collection[tuple[Satellite, Satellite]],
        *,
        dynamics_data: DynamicsData,
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[tuple[Satellite, Satellite], CommLinkPerformance]:
        return {
            edge: self._calc_satellite_to_satellite(
                positions_eci=(
                    dynamics_data.satellite_position_eci[edge[0]],
                    dynamics_data.satellite_position_eci[edge[1]],
                ),
                velocities_eci=(
                    dynamics_data.satellite_velocity_eci[edge[0]],
                    dynamics_data.satellite_velocity_eci[edge[1]],
                ),
                attitude_angular_velocities_eci=(
                    dynamics_data.satellite_attitude_angular_velocity_eci[edge[0]],
                    dynamics_data.satellite_attitude_angular_velocity_eci[edge[1]],
                ),
                sun_direction_eci=dynamics_data.sun_direction_eci,
            )
            for edge in edges
        }

    def _calc_capacity_from_distance(
        self,
        distance: Annotated[
            float,
            Doc("Distance between satellites"),
        ],
    ) -> float:
        power = self.lct_p0 / distance**2  # can be replaced later by gaussian beam loss
        t_sys = 150  # K
        noise_factor = 10 ** (self.noise_figure / 10)
        noise = t_sys * self.link_capacity * BOLTZ_CONST * noise_factor
        gain = 10 ** (self.lna_gain / 10)  # convert from dB to W
        snr = gain * power / noise
        return self.link_capacity * np.log2(1 + snr)

    def _calc_satellite_to_satellite(
        self,
        *,
        positions_eci: Annotated[
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
            Doc("Position vectors in ECI frame. Shape: (3,)"),
        ],
        velocities_eci: Annotated[
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
            Doc("Velocity vectors in ECI frame. Shape: (3,)"),
        ],
        attitude_angular_velocities_eci: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
        sun_direction_eci: Annotated[
            npt.NDArray[np.floating],
            Doc("Sun direction vector in ECI frame. Shape: (3,)"),
        ],
    ) -> CommLinkPerformance:
        """Calculate binary communication link performance between two satellites."""
        for vec in chain(positions_eci, velocities_eci):
            assert vec.shape == (3,), f"Position and velocity vectors must be 3-dimensional, but got shape {vec.shape}"

        distance = float(np.linalg.norm(positions_eci[1] - positions_eci[0]))

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

        # Check if either satellite is in eclipse - if so, ignore sun exclusion angle for that satellite
        satellite_b_in_eclipse = is_satellite_in_eclipse(positions_eci[1], sun_direction_eci)

        # Calculate sun exclusion angle constraints for each direction
        # If satellite is in eclipse, skip sun exclusion angle check for that direction
        sun_exclusion_satisfied = True

        if not satellite_b_in_eclipse:
            # Check sun exclusion angle from satellite B's perspective (B looking towards A)
            edge_sun_angle_b_to_a = angle_between(-relative_position_eci, sun_direction_eci)
            if edge_sun_angle_b_to_a < self.sun_exclusion_angle:
                sun_exclusion_satisfied = False

        link_available = bool(
            distance < self.max_inter_satellite_distance
            and all(
                float(np.linalg.norm(relative_angular_velocity)) < self.max_relative_angular_velocity
                for relative_angular_velocity in relative_angular_velocities
            )
            and sun_exclusion_satisfied,
        )

        return CommLinkPerformance(
            link_capacity=self._calc_capacity_from_distance(distance) if link_available else 0.0,
            delay=float(distance / SPEED_OF_LIGHT),
            link_available=link_available,
        )
