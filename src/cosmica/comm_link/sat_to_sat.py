__all__ = [
    "OTC2OTCBinaryCommLinkCalculator",
    "SatToSatBinaryCommLinkCalculator",
    "SatToSatBinaryMemoryCommLinkCalculator",
]

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Annotated

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from typing_extensions import Doc

from cosmica.dtos import DirectedCommunicationLink, DynamicsData
from cosmica.models import OpticalCommunicationTerminal, Satellite
from cosmica.utils.constants import SPEED_OF_LIGHT
from cosmica.utils.vector import angle_between, is_line_segment_clear_of_earth, is_satellite_in_eclipse

from .base import AssignedCommLinkCalculator, CommLinkCalculator, CommLinkPerformance, MemorylessCommLinkCalculator
from .terminal_geometry import (
    TerminalPointing,
    calc_terminal_angular_rate,
    calc_terminal_pointing,
    is_pointing_rate_within_limit,
    is_pointing_within_field_of_regard,
)

logger = logging.getLogger(__name__)

type OpticalSatelliteLink[
    SourceSatellite: Satellite,
    SourceTerminal: OpticalCommunicationTerminal,
    DestinationSatellite: Satellite,
    DestinationTerminal: OpticalCommunicationTerminal,
] = DirectedCommunicationLink[
    SourceSatellite,
    SourceTerminal,
    DestinationSatellite,
    DestinationTerminal,
]


type _TerminalPointings = tuple[TerminalPointing, TerminalPointing]


@dataclass(frozen=True, slots=True)
class _TerminalState:
    observed_at: np.datetime64
    pointings: _TerminalPointings


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


class OTC2OTCBinaryCommLinkCalculator(
    AssignedCommLinkCalculator[
        OpticalSatelliteLink[Satellite, OpticalCommunicationTerminal, Satellite, OpticalCommunicationTerminal]
    ],
):
    """Calculate performance for optical links with explicitly assigned satellite terminals.

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

    def calc[
        SourceSatellite: Satellite,
        SourceTerminal: OpticalCommunicationTerminal,
        DestinationSatellite: Satellite,
        DestinationTerminal: OpticalCommunicationTerminal,
    ](
        self,
        links_time_series: Sequence[
            Collection[OpticalSatelliteLink[SourceSatellite, SourceTerminal, DestinationSatellite, DestinationTerminal]]
        ],
        *,
        dynamics_data: DynamicsData,
        rng: Generator,  # noqa: ARG002
    ) -> list[
        dict[
            OpticalSatelliteLink[SourceSatellite, SourceTerminal, DestinationSatellite, DestinationTerminal],
            CommLinkPerformance,
        ]
    ]:
        previous_terminal_state_by_link: dict[
            OpticalSatelliteLink[SourceSatellite, SourceTerminal, DestinationSatellite, DestinationTerminal],
            _TerminalState,
        ] = {}
        comm_link_time_series = []

        for links_snapshot, dynamics_snapshot in zip(links_time_series, dynamics_data, strict=True):
            links_performance_snapshot = {}
            current_terminal_state_by_link: dict[
                OpticalSatelliteLink[SourceSatellite, SourceTerminal, DestinationSatellite, DestinationTerminal],
                _TerminalState,
            ] = {}
            current_time = dynamics_snapshot.time

            for link in links_snapshot:
                source_satellite = link.source.node
                destination_satellite = link.destination.node
                missing_attitudes = tuple(
                    satellite
                    for satellite in (source_satellite, destination_satellite)
                    if satellite not in dynamics_snapshot.satellite_attitude_dcm_eci2body
                )
                if missing_attitudes:
                    msg = (
                        "satellite_attitude_dcm_eci2body must contain an ECI-to-body "
                        f"attitude for every link endpoint; missing {missing_attitudes!r}"
                    )
                    raise ValueError(msg)

                previous_terminal_state = previous_terminal_state_by_link.get(link)
                previous_terminal_observation = (
                    None
                    if previous_terminal_state is None
                    else (
                        previous_terminal_state.pointings,
                        float((current_time - previous_terminal_state.observed_at) / np.timedelta64(1, "s")),
                    )
                )

                comm_link_performance, terminal_pointings = self._calc_satellite_to_satellite(
                    positions_eci=(
                        dynamics_snapshot.satellite_position_eci[source_satellite],
                        dynamics_snapshot.satellite_position_eci[destination_satellite],
                    ),
                    velocities_eci=(
                        dynamics_snapshot.satellite_velocity_eci[source_satellite],
                        dynamics_snapshot.satellite_velocity_eci[destination_satellite],
                    ),
                    attitude_angular_velocities_eci=(
                        dynamics_snapshot.satellite_attitude_angular_velocity_eci[source_satellite],
                        dynamics_snapshot.satellite_attitude_angular_velocity_eci[destination_satellite],
                    ),
                    attitude_dcms_eci2body=(
                        dynamics_snapshot.satellite_attitude_dcm_eci2body[source_satellite],
                        dynamics_snapshot.satellite_attitude_dcm_eci2body[destination_satellite],
                    ),
                    sun_direction_eci=dynamics_snapshot.sun_direction_eci,
                    terminals=(
                        link.source.terminal,
                        link.destination.terminal,
                    ),
                    previous_terminal_observation=previous_terminal_observation,
                )

                links_performance_snapshot[link] = comm_link_performance
                current_terminal_state_by_link[link] = _TerminalState(
                    observed_at=current_time,
                    pointings=terminal_pointings,
                )

            comm_link_time_series.append(links_performance_snapshot)
            previous_terminal_state_by_link = current_terminal_state_by_link

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
        previous_terminal_observation: Annotated[
            tuple[_TerminalPointings, float] | None,
            Doc("Previous terminal pointings and elapsed time in seconds, or None for a first observation"),
        ],
    ) -> tuple[CommLinkPerformance, _TerminalPointings]:
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

        terminal_pointings = (
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
            is_pointing_within_field_of_regard(pointing, terminal)
            for pointing, terminal in zip(terminal_pointings, terminals, strict=True)
        )
        match previous_terminal_observation:
            case None:
                slew_rate_satisfied = True
            case (previous_terminal_pointings, elapsed_time_seconds):
                slew_rate_satisfied = all(
                    is_pointing_rate_within_limit(
                        calc_terminal_angular_rate(
                            pointing,
                            previous_pointing,
                            time_delta=elapsed_time_seconds,
                        ),
                        terminal,
                    )
                    for pointing, previous_pointing, terminal in zip(
                        terminal_pointings,
                        previous_terminal_pointings,
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
            terminal_pointings,
        )
