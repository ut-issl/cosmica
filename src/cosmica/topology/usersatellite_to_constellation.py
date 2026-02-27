__all__ = [
    "MaxConnectionTimeUS2CTopologyBuilder",
    "UserSatelliteToConstellationTopologyBuilder",
    "build_max_connection_time_us2c_topology",
]
import logging
from abc import ABC, abstractmethod
from collections.abc import Collection
from itertools import product

import networkx as nx
import numpy as np
import numpy.typing as npt

from cosmica.dtos import DynamicsData
from cosmica.models import Constellation, UserSatellite
from cosmica.utils.vector import angle_between

logger = logging.getLogger(__name__)


class UserSatelliteToConstellationTopologyBuilder[
    TConstellation: Constellation,
    TUserSatellite: UserSatellite,
    TGraph: nx.Graph,
](ABC):
    @abstractmethod
    def build(
        self,
        *,
        constellation: TConstellation,
        user_satellites: Collection[TUserSatellite],
        dynamics_data: DynamicsData,
    ) -> list[TGraph]:
        """Build time-varying topology connecting user satellites to constellation."""
        ...


class MaxConnectionTimeUS2CTopologyBuilder(
    UserSatelliteToConstellationTopologyBuilder[Constellation, UserSatellite, nx.Graph],
):
    """Topology builder connecting user satellites to constellation with longest connection time.

    This builder calculates the visibility duration between user satellites and constellation
    satellites, then selects the constellation satellite that provides the longest continuous
    connection time for each user satellite. This minimizes handovers and provides stable
    connections.

    """

    def __init__(
        self,
        max_distance: float = float("inf"),
        max_relative_angular_velocity: float = float("inf"),
        sun_exclusion_angle: float = 0.0,
    ) -> None:
        self.max_distance = max_distance
        self.max_relative_angular_velocity = max_relative_angular_velocity
        self.sun_exclusion_angle = sun_exclusion_angle

    def build(
        self,
        *,
        constellation: Constellation,
        user_satellites: Collection[UserSatellite],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        return build_max_connection_time_us2c_topology(
            constellation=constellation,
            user_satellites=user_satellites,
            dynamics_data=dynamics_data,
            max_distance=self.max_distance,
            max_relative_angular_velocity=self.max_relative_angular_velocity,
            sun_exclusion_angle=self.sun_exclusion_angle,
        )


# ---------------------------------------------------------------------------
# New functions using Constellation model
# ---------------------------------------------------------------------------


def build_max_connection_time_us2c_topology(  # noqa: C901, PLR0912, PLR0915
    constellation: Constellation,
    *,
    user_satellites: Collection[UserSatellite],
    dynamics_data: DynamicsData,
    max_distance: float = float("inf"),
    max_relative_angular_velocity: float = float("inf"),
    sun_exclusion_angle: float = 0.0,
) -> list[nx.Graph]:
    """Build user-satellite-to-constellation topology with longest connection time.

    Selects the constellation satellite that provides the longest continuous
    connection time for each user satellite, minimizing handovers.

    This function only needs the satellite set, so it accepts `Constellation`
    with any `SatelliteId` type.

    Args:
        constellation: Constellation (any SatelliteId type).
        user_satellites: User satellites to connect.
        dynamics_data: Time-series dynamics data.
        max_distance: Maximum distance constraint.
        max_relative_angular_velocity: Maximum relative angular velocity constraint.
        sun_exclusion_angle: Sun exclusion angle constraint (radians).

    Returns:
        A list of networkx Graphs, one per time step.

    """
    logger.info(f"Building user-to-constellation topology for {len(user_satellites)} user satellites")
    logger.info(f"Number of time steps: {len(dynamics_data.time)}")

    user_satellites_list = list(user_satellites)
    constellation_satellites = list(constellation.satellites.values())

    n_time = len(dynamics_data.time)
    n_users = len(user_satellites_list)
    n_constellation = len(constellation_satellites)

    # Calculate visibility based on distance, sun exclusion angle, and relative angular velocity constraints
    visibility = np.zeros((n_users, n_constellation, n_time), dtype=np.bool_)

    for (user_idx, user_sat), (const_idx, const_sat) in product(
        enumerate(user_satellites_list),
        enumerate(constellation_satellites),
    ):
        user_pos_eci = dynamics_data.satellite_position_eci[user_sat]
        const_pos_eci = dynamics_data.satellite_position_eci[const_sat]
        user_vel_eci = dynamics_data.satellite_velocity_eci[user_sat]
        const_vel_eci = dynamics_data.satellite_velocity_eci[const_sat]
        user_att_ang_vel_eci = dynamics_data.satellite_attitude_angular_velocity_eci[user_sat]
        const_att_ang_vel_eci = dynamics_data.satellite_attitude_angular_velocity_eci[const_sat]

        relative_pos_eci = const_pos_eci - user_pos_eci
        relative_vel_eci = const_vel_eci - user_vel_eci

        distances = np.linalg.norm(relative_pos_eci, axis=1)

        relative_angular_velocities_list = []
        for t in range(n_time):
            distance = distances[t]
            if distance > 0:
                relative_angular_velocity_translational = (
                    np.cross(relative_pos_eci[t], relative_vel_eci[t]) / distance**2
                )
                rel_ang_vel_user = relative_angular_velocity_translational - user_att_ang_vel_eci[t]
                rel_ang_vel_const = -relative_angular_velocity_translational - const_att_ang_vel_eci[t]
                max_rel_ang_vel = np.maximum(
                    np.linalg.norm(rel_ang_vel_user),
                    np.linalg.norm(rel_ang_vel_const),
                )
                relative_angular_velocities_list.append(float(max_rel_ang_vel))
            else:
                relative_angular_velocities_list.append(float("inf"))

        relative_angular_velocities = np.array(relative_angular_velocities_list)

        sun_angles = np.array(
            [angle_between(relative_pos_eci[t], dynamics_data.sun_direction_eci[t]) for t in range(n_time)],
        )

        distance_ok = distances <= max_distance
        sun_ok = (sun_angles >= sun_exclusion_angle) & (sun_angles <= (np.pi - sun_exclusion_angle))
        angular_velocity_ok = relative_angular_velocities <= max_relative_angular_velocity

        visibility[user_idx, const_idx, :] = distance_ok & sun_ok & angular_velocity_ok

    # Calculate remaining connection time (backward pass)
    remaining_connection_time = np.zeros((n_users, n_constellation, n_time), dtype=np.int_)
    for time_idx in reversed(range(n_time)):
        if time_idx == n_time - 1:
            remaining_connection_time[:, :, time_idx] = visibility[:, :, time_idx].astype(int)
        else:
            remaining_connection_time[:, :, time_idx] = np.where(
                visibility[:, :, time_idx],
                remaining_connection_time[:, :, time_idx + 1] + 1,
                0,
            )

    # Select constellation satellite for each user satellite at each time step
    selected_constellation_idx = np.full(n_users, -1, dtype=np.int_)
    link_available = np.zeros((n_users, n_constellation, n_time), dtype=np.bool_)

    def select_max_connection_satellite(user_idx: int, time_idx: int) -> None:
        user_remaining_times = remaining_connection_time[user_idx, :, time_idx].copy()
        for other_user_idx in range(n_users):
            if other_user_idx != user_idx and selected_constellation_idx[other_user_idx] >= 0:
                user_remaining_times[selected_constellation_idx[other_user_idx]] = 0
        if user_remaining_times.max() > 0:
            selected_constellation_idx[user_idx] = int(np.argmax(user_remaining_times))
        else:
            selected_constellation_idx[user_idx] = -1

    for time_idx in range(n_time):
        selected_constellation_idx.fill(-1)

        for user_idx in range(n_users):
            if time_idx == 0:
                select_max_connection_satellite(user_idx, time_idx)
            else:
                prev_selected = np.where(link_available[user_idx, :, time_idx - 1])[0]
                if len(prev_selected) > 0:
                    prev_sat_idx = prev_selected[0]
                    if visibility[user_idx, prev_sat_idx, time_idx]:
                        selected_constellation_idx[user_idx] = prev_sat_idx
                    else:
                        select_max_connection_satellite(user_idx, time_idx)
                else:
                    select_max_connection_satellite(user_idx, time_idx)

            if selected_constellation_idx[user_idx] >= 0:
                link_available[user_idx, selected_constellation_idx[user_idx], time_idx] = True

    def construct_graph(link_available_at_time: npt.NDArray[np.bool_]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(user_satellites_list)
        graph.add_nodes_from(constellation_satellites)

        for user_idx, user_sat in enumerate(user_satellites_list):
            for const_idx, const_sat in enumerate(constellation_satellites):
                if link_available_at_time[user_idx, const_idx]:
                    graph.add_edge(user_sat, const_sat)

        return graph

    return [construct_graph(link_available[:, :, time_idx]) for time_idx in range(n_time)]
