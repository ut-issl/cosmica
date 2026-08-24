__all__ = [
    "OTC2OTCBinaryCommLinkCalculator",
    "SatToSatBinaryCommLinkCalculator",
    "SatToSatBinaryMemoryCommLinkCalculator",
]

import logging
from collections.abc import Collection, Sequence
from itertools import chain
from typing import Annotated

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from typing_extensions import Doc

from cosmica.dtos import DynamicsData
from cosmica.models import OpticalCommunicationTerminal, Satellite, SatelliteTerminal
from cosmica.utils.constants import SPEED_OF_LIGHT
from cosmica.utils.vector import (
    angle_between,
    is_line_segment_clear_of_earth,
    is_satellite_in_eclipse,
    normalize,
    unit_vector_to_azimuth_elevation,
)

from .base import CommLinkCalculator, CommLinkPerformance, MemorylessCommLinkCalculator

logger = logging.getLogger(__name__)


class SatToSatBinaryCommLinkCalculator(MemorylessCommLinkCalculator[Satellite, Satellite]):
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
        link_capacity: float,
        max_inter_satellite_distance: float = float("inf"),
        lowest_altitude: float = 0.0,
        max_relative_angular_velocity: float = float("inf"),
        sun_exclusion_angle: float = 0.0,
    ) -> None:
        self.link_capacity = link_capacity
        self.max_inter_satellite_distance = max_inter_satellite_distance
        self.lowest_altitude = lowest_altitude
        self.max_relative_angular_velocity = max_relative_angular_velocity
        self.sun_exclusion_angle = sun_exclusion_angle

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
            and is_line_segment_clear_of_earth(
                positions_eci[0],
                positions_eci[1],
                lowest_altitude=self.lowest_altitude,
            )
            and all(
                float(np.linalg.norm(relative_angular_velocity)) < self.max_relative_angular_velocity
                for relative_angular_velocity in relative_angular_velocities
            )
            and sun_exclusion_satisfied,
        )

        return CommLinkPerformance(
            link_capacity=self.link_capacity if link_available else 0.0,
            delay=float(distance / SPEED_OF_LIGHT),
            link_available=link_available,
        )


class SatToSatBinaryMemoryCommLinkCalculator(CommLinkCalculator[Satellite, Satellite]):
    """Calculate satellite-to-satellite communication link performance for each directed edge in a network.

    The link performance is calculated as a binary value, i.e., 1 if the link is available and 0 otherwise.

    Link acquisition delay is tracked independently per directed edge (src, dst): when a directed
    edge appears or its underlying memoryless availability drops, only that direction goes through
    (re)acquisition. Since availability can be direction-asymmetric (e.g. the sun exclusion angle is
    checked at the receiver only), the two directions of a physical link may be in different
    acquisition states.
    """

    def __init__(
        self,
        *,
        memoryless_calculator: MemorylessCommLinkCalculator[Satellite, Satellite],
        link_acquisition_time: float = 60.0,
        skip_link_acquisition_at_simulation_start: bool = True,
    ) -> None:
        self.memoryless_calculator = memoryless_calculator
        self.link_acquisition_time = link_acquisition_time
        self.skip_link_acquisition_at_simulation_start = skip_link_acquisition_at_simulation_start

    def calc(
        self,
        edges_time_series: Sequence[Collection[tuple[Satellite, Satellite]]],
        *,
        dynamics_data: DynamicsData,
        rng: np.random.Generator,
    ) -> list[dict[tuple[Satellite, Satellite], CommLinkPerformance]]:
        assert len(edges_time_series) == len(dynamics_data.time)

        comm_link_time_series: list[dict[tuple[Satellite, Satellite], CommLinkPerformance]] = []

        # ― per-directed-edge state ―
        # Link acquisition is tracked independently for each directed edge (src, dst).
        link_acquisition_start_time: dict[tuple[Satellite, Satellite], np.datetime64] = {}
        prev_edges: frozenset[tuple[Satellite, Satellite]] = frozenset()

        for time_index, edges_snapshot in enumerate(edges_time_series):
            current_time: np.datetime64 = dynamics_data.time[time_index]

            comm_link = self.memoryless_calculator.calc(
                edges=edges_snapshot,
                dynamics_data=dynamics_data[time_index],
                rng=rng,
            )

            edges_snapshot_set = frozenset(edges_snapshot)

            # ── update “first-seen” bookkeeping ──────────────────── ★
            new_edges = edges_snapshot_set - prev_edges
            for edge in new_edges:
                if self.skip_link_acquisition_at_simulation_start and time_index == 0:
                    link_acquisition_start_time[edge] = current_time - np.timedelta64(
                        int(self.link_acquisition_time),
                        "s",
                    )
                else:
                    link_acquisition_start_time[edge] = current_time

            disappeared_edges = prev_edges - edges_snapshot_set
            for edge in disappeared_edges:  # フェードアウトしたら状態を消去
                link_acquisition_start_time.pop(edge, None)
            prev_edges = edges_snapshot_set
            # ───────────────────────────────────────────────────────

            for edge in edges_snapshot_set:
                if comm_link[edge]["link_available"] is False:
                    link_acquisition_start_time[edge] = current_time

                # --- link acquisition delay ------------ ★
                within_link_acquisition = (
                    float(
                        (current_time - link_acquisition_start_time[edge]) / np.timedelta64(1, "s"),
                    )
                    < self.link_acquisition_time
                )

                if within_link_acquisition:
                    comm_link[edge] = CommLinkPerformance(
                        link_capacity=0.0,
                        delay=np.inf,
                        link_available=False,
                    )
                # ----------------------------------------------------

            comm_link_time_series.append(comm_link)

        return comm_link_time_series


class OTC2OTCBinaryCommLinkCalculator(CommLinkCalculator[SatelliteTerminal, SatelliteTerminal]):
    """Calculate satellite-to-satellite communication link performance for each terminal in a network.

    The link performance is calculated as a binary value, i.e., 1 if the link is available and 0 otherwise.
    """

    def __init__(
        self,
        *,
        link_capacity: float,
        max_inter_satellite_distance: float = float("inf"),
        lowest_altitude: float = 0.0,
        max_relative_angular_velocity: float = float("inf"),
        sun_exclusion_angle: float = 0.0,
    ) -> None:
        self.link_capacity = link_capacity
        self.max_inter_satellite_distance = max_inter_satellite_distance
        self.lowest_altitude = lowest_altitude
        self.max_relative_angular_velocity = max_relative_angular_velocity
        self.sun_exclusion_angle = sun_exclusion_angle

    def calc(
        self,
        edges_time_series: Sequence[Collection[tuple[SatelliteTerminal, SatelliteTerminal]]],
        *,
        dynamics_data: DynamicsData,
        rng: Generator,  # noqa: ARG002
    ) -> list[dict[tuple[SatelliteTerminal, SatelliteTerminal], CommLinkPerformance]]:
        if len(edges_time_series) != len(dynamics_data.time):
            msg = "edges_time_series must have the same length as dynamics_data.time"
            raise ValueError(msg)
        if not edges_time_series:
            return []

        time_deltas = np.diff(dynamics_data.time) / np.timedelta64(1, "s")
        if not np.all(np.isfinite(time_deltas)) or np.any(time_deltas <= 0.0):
            msg = "dynamics_data.time must be strictly increasing"
            raise ValueError(msg)

        terminal_memo: dict[tuple[SatelliteTerminal, SatelliteTerminal], list[tuple[float, float]]] = {}
        comm_link_time_series = []

        for time_index, edges_snapshot in enumerate(edges_time_series):
            snapshot_dynamics = dynamics_data[time_index]
            time_delta = 1.0 if time_index == 0 else float(time_deltas[time_index - 1])
            edges_performance = {}
            current_terminal_memo: dict[
                tuple[SatelliteTerminal, SatelliteTerminal],
                list[tuple[float, float]],
            ] = {}

            for edge in edges_snapshot:
                previous_terminal_directions = terminal_memo.get(edge, [(0.0, 0.0), (0.0, 0.0)])
                comm_link_performance, terminal_directions = self._calc_satellite_to_satellite(
                    positions_eci=(
                        snapshot_dynamics.satellite_position_eci[edge[0]],
                        snapshot_dynamics.satellite_position_eci[edge[1]],
                    ),
                    velocities_eci=(
                        snapshot_dynamics.satellite_velocity_eci[edge[0]],
                        snapshot_dynamics.satellite_velocity_eci[edge[1]],
                    ),
                    attitude_angular_velocities_eci=(
                        snapshot_dynamics.satellite_attitude_angular_velocity_eci[edge[0]],
                        snapshot_dynamics.satellite_attitude_angular_velocity_eci[edge[1]],
                    ),
                    sun_direction_eci=snapshot_dynamics.sun_direction_eci,
                    terminals=(
                        edge[0].terminal,
                        edge[1].terminal,
                    ),
                    previous_terminal_directions=previous_terminal_directions,
                    time_delta=time_delta,
                )

                edges_performance[edge] = comm_link_performance
                current_terminal_memo[edge] = terminal_directions

            comm_link_time_series.append(edges_performance)
            terminal_memo = current_terminal_memo

        return comm_link_time_series

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
        terminals: Annotated[
            tuple[OpticalCommunicationTerminal, OpticalCommunicationTerminal],
            Doc("Optical Communication Terminals of pair of satellites"),
        ],
        previous_terminal_directions: Annotated[
            list[tuple[float, float]],
            Doc("Azimuth and elevation values for each terminal in the previous time step"),
        ],
        time_delta: Annotated[
            float,
            Doc("Float representing time difference to previous configuration in the time series"),
        ],
    ) -> tuple[CommLinkPerformance, list[tuple[float, float]]]:
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
        satellite_a_in_eclipse = is_satellite_in_eclipse(positions_eci[0], sun_direction_eci)
        satellite_b_in_eclipse = is_satellite_in_eclipse(positions_eci[1], sun_direction_eci)

        # Calculate sun exclusion angle constraints for each direction
        # If satellite is in eclipse, skip sun exclusion angle check for that direction
        sun_exclusion_satisfied = True

        if not satellite_a_in_eclipse:
            # Check sun exclusion angle from satellite A's perspective (A looking towards B)
            edge_sun_angle_a_to_b = angle_between(relative_position_eci, sun_direction_eci)
            if edge_sun_angle_a_to_b < self.sun_exclusion_angle:
                sun_exclusion_satisfied = False

        if not satellite_b_in_eclipse:
            # Check sun exclusion angle from satellite B's perspective (B looking towards A)
            edge_sun_angle_b_to_a = angle_between(-relative_position_eci, sun_direction_eci)
            if edge_sun_angle_b_to_a < self.sun_exclusion_angle:
                sun_exclusion_satisfied = False

        terminal_directions = self._calc_terminal_directions(relative_position_eci)
        terminal_angular_velocity = [
            [(terminal[0] - previous_terminal[0]) / time_delta, (terminal[1] - previous_terminal[1]) / time_delta]
            for terminal, previous_terminal in zip(terminal_directions, previous_terminal_directions, strict=False)
        ]
        link_available = bool(
            distance < self.max_inter_satellite_distance
            and is_line_segment_clear_of_earth(
                positions_eci[0],
                positions_eci[1],
                lowest_altitude=self.lowest_altitude,
            )
            and all(
                float(np.linalg.norm(relative_angular_velocity)) < self.max_relative_angular_velocity
                for relative_angular_velocity in relative_angular_velocities
            )
            and sun_exclusion_satisfied
            and all(
                # For terminal direction checks, a more careful implementation is required
                # using the appropriate coordinate frame transformations
                # terminal_directions[i][0] < terminal.azimuth_max
                # and terminal_directions[i][0] > terminal.azimuth_min
                # and terminal_directions[i][1] > terminal.elevation_min
                # and terminal_directions[i][1] < terminal.elevation_max
                terminal_angular_velocity[i][0] < terminal.angular_velocity_max
                and terminal_angular_velocity[i][1] < terminal.angular_velocity_max
                for i, terminal in enumerate(terminals)
            ),
        )

        return (
            CommLinkPerformance(
                link_capacity=self.link_capacity if link_available else 0.0,
                delay=float(distance / SPEED_OF_LIGHT),
                link_available=link_available,
            ),
            terminal_directions,
        )

    def _calc_terminal_directions(self, relative_position: npt.NDArray) -> list[tuple[float, float]]:
        unitary_direction = normalize(relative_position)
        terminal_a = unit_vector_to_azimuth_elevation(unitary_direction)
        terminal_b = unit_vector_to_azimuth_elevation(
            np.concatenate((unitary_direction[:-1], [-unitary_direction[-1]])),
        )
        # Note: Here we only inverted the z component of the unit vector, as it is assumed the terminals are located
        # at opposite faces
        return [terminal_a, terminal_b]
