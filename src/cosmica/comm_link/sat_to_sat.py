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
from cosmica.utils.vector import angle_between, is_line_segment_clear_of_earth, is_satellite_in_eclipse

from .base import CommLinkCalculator, CommLinkPerformance, MemorylessCommLinkCalculator
from .terminal_geometry import (
    TerminalPointing,
    calc_terminal_angular_rate,
    calc_terminal_pointing,
    is_pointing_rate_within_limit,
    is_pointing_within_field_of_regard,
)

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
    """Calculate time-series performance for links between body-mounted optical terminals.

    For each endpoint, the ECI line of sight is transformed first by the snapshot's
    ECI-to-body attitude from :class:`~cosmica.dtos.DynamicsData`, then by the
    terminal's fixed body-to-terminal mounting transform:
    ``u_terminal = C_body2terminal @ C_eci2body @ u_eci``. The two endpoints use
    opposite ECI line-of-sight vectors and their own transforms. In terminal
    coordinates, ``+x`` has zero azimuth and elevation, azimuth rotates toward
    ``+y``, and elevation rotates toward ``+z``.

    A link is available only when its distance, Earth clearance, relative angular
    velocity, two-ended Sun exclusion, both terminals' fields of regard, and both
    terminals' axis slew rates satisfy their configured limits. Slew is estimated
    only across consecutive snapshots containing the same directed edge. Therefore,
    the first appearance after simulation start or an absence skips the slew-rate
    constraint while still applying every instantaneous constraint.
    """

    def __init__(
        self,
        *,
        link_capacity: Annotated[float, Doc("Available-link capacity in bits per second.")],
        max_inter_satellite_distance: Annotated[
            float,
            Doc("Exclusive maximum terminal separation in meters."),
        ] = float("inf"),
        lowest_altitude: Annotated[
            float,
            Doc("Minimum line-of-sight altitude above Earth's surface in meters, inclusive."),
        ] = 0.0,
        max_relative_angular_velocity: Annotated[
            float,
            Doc("Exclusive maximum line-of-sight angular velocity at either satellite in radians per second."),
        ] = float("inf"),
        sun_exclusion_angle: Annotated[
            float,
            Doc("Minimum angle from either illuminated terminal's line of sight to the Sun, in radians."),
        ] = 0.0,
    ) -> None:
        self.link_capacity = link_capacity
        self.max_inter_satellite_distance = max_inter_satellite_distance
        self.lowest_altitude = lowest_altitude
        self.max_relative_angular_velocity = max_relative_angular_velocity
        self.sun_exclusion_angle = sun_exclusion_angle

    def calc(
        self,
        edges_time_series: Annotated[
            Sequence[Collection[tuple[SatelliteTerminal, SatelliteTerminal]]],
            Doc("Directed terminal edges for each simulation snapshot."),
        ],
        *,
        dynamics_data: Annotated[
            DynamicsData,
            Doc("Time-series dynamics including ECI-to-body attitudes for every terminal endpoint."),
        ],
        rng: Annotated[Generator, Doc("Random generator reserved for calculator-interface compatibility.")],  # noqa: ARG002
    ) -> list[dict[tuple[SatelliteTerminal, SatelliteTerminal], CommLinkPerformance]]:
        """Calculate terminal-link performance for every time-series snapshot."""
        if len(edges_time_series) != len(dynamics_data.time):
            msg = "edges_time_series must have the same length as dynamics_data.time"
            raise ValueError(msg)
        if not edges_time_series:
            return []

        time_deltas = np.diff(dynamics_data.time) / np.timedelta64(1, "s")
        if not np.all(np.isfinite(time_deltas)) or np.any(time_deltas <= 0.0):
            msg = "dynamics_data.time must be strictly increasing"
            raise ValueError(msg)

        terminal_memo: dict[
            tuple[SatelliteTerminal, SatelliteTerminal],
            tuple[TerminalPointing, TerminalPointing],
        ] = {}
        comm_link_time_series = []

        for time_index, edges_snapshot in enumerate(edges_time_series):
            snapshot_dynamics = dynamics_data[time_index]
            time_delta = 1.0 if time_index == 0 else float(time_deltas[time_index - 1])
            edges_performance = {}
            current_terminal_memo: dict[
                tuple[SatelliteTerminal, SatelliteTerminal],
                tuple[TerminalPointing, TerminalPointing],
            ] = {}

            for edge in edges_snapshot:
                missing_attitudes = tuple(
                    endpoint
                    for endpoint in edge
                    if endpoint not in snapshot_dynamics.satellite_attitude_dcm_eci2body
                )
                if missing_attitudes:
                    msg = (
                        "satellite_attitude_dcm_eci2body must contain an ECI-to-body "
                        f"attitude for every terminal endpoint; missing {missing_attitudes!r}"
                    )
                    raise ValueError(msg)

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
                    attitude_dcms_eci2body=(
                        snapshot_dynamics.satellite_attitude_dcm_eci2body[edge[0]],
                        snapshot_dynamics.satellite_attitude_dcm_eci2body[edge[1]],
                    ),
                    sun_direction_eci=snapshot_dynamics.sun_direction_eci,
                    terminals=(
                        edge[0].terminal,
                        edge[1].terminal,
                    ),
                    previous_terminal_directions=terminal_memo.get(edge),
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
        attitude_angular_velocities_eci: Annotated[
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
            Doc("Satellite body angular-velocity vectors in ECI coordinates. Shape per endpoint: (3,)."),
        ],
        attitude_dcms_eci2body: Annotated[
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
            Doc("Direction-cosine matrices transforming ECI components to body components. Shape: (3, 3)."),
        ],
        sun_direction_eci: Annotated[
            npt.NDArray[np.floating],
            Doc("Sun direction vector in ECI frame. Shape: (3,)"),
        ],
        terminals: Annotated[
            tuple[OpticalCommunicationTerminal, OpticalCommunicationTerminal],
            Doc("Optical Communication Terminals of pair of satellites"),
        ],
        previous_terminal_directions: Annotated[
            tuple[TerminalPointing, TerminalPointing] | None,
            Doc("Previous endpoint pointings, or None when the directed edge was not present in the prior snapshot."),
        ],
        time_delta: Annotated[
            float,
            Doc("Elapsed time from the previous simulation snapshot in seconds."),
        ],
    ) -> tuple[CommLinkPerformance, tuple[TerminalPointing, TerminalPointing]]:
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

        terminal_directions = (
            calc_terminal_pointing(
                relative_position_eci,
                dcm_eci2body=attitude_dcms_eci2body[0],
                terminal=terminals[0],
            ),
            calc_terminal_pointing(
                -relative_position_eci,
                dcm_eci2body=attitude_dcms_eci2body[1],
                terminal=terminals[1],
            ),
        )
        field_of_regard_satisfied = all(
            is_pointing_within_field_of_regard(direction, terminal)
            for direction, terminal in zip(terminal_directions, terminals, strict=True)
        )
        slew_rate_satisfied = previous_terminal_directions is None or all(
            is_pointing_rate_within_limit(
                calc_terminal_angular_rate(direction, previous_direction, time_delta=time_delta),
                terminal,
            )
            for direction, previous_direction, terminal in zip(
                terminal_directions,
                previous_terminal_directions,
                terminals,
                strict=True,
            )
        )
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
            and field_of_regard_satisfied
            and slew_rate_satisfied,
        )

        return (
            CommLinkPerformance(
                link_capacity=self.link_capacity if link_available else 0.0,
                delay=float(distance / SPEED_OF_LIGHT),
                link_available=link_available,
            ),
            terminal_directions,
        )
