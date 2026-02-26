__all__ = [
    "HybridUS2CG2CTopologyBuilder",
    "build_hybrid_us2c_g2c_topology",
]

import logging
from collections.abc import Collection
from itertools import product

import networkx as nx
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from cosmica.dtos import DynamicsData
from cosmica.models import Constellation, ConstellationSatellite, Gateway, StationaryOnGroundUser, UserSatellite
from cosmica.utils.coordinates import ecef2aer, geodetic2ecef
from cosmica.utils.vector import angle_between

logger = logging.getLogger(__name__)


class HybridUS2CG2CTopologyBuilder:
    """Hybrid topology builder for user satellites and ground stations to constellation.

    This builder combines the functionality of MaxConnectionTimeUS2CTopologyBuilder and
    MaxVisibilityHandOverG2CTopologyBuilder. It prioritizes user satellite connections
    and then optimizes ground station connections based on maximum visibility duration.

    For user satellites, constraints include distance, sun exclusion angle, and relative angular velocity.
    For ground stations, constraints include elevation angle and sun exclusion angle.
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
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        """Build hybrid topology connecting user satellites and ground stations to constellation."""
        return build_hybrid_us2c_g2c_topology(
            constellation=constellation,
            user_satellites=user_satellites,
            ground_nodes=ground_nodes,
            dynamics_data=dynamics_data,
            max_distance=self.max_distance,
            max_relative_angular_velocity=self.max_relative_angular_velocity,
            sun_exclusion_angle=self.sun_exclusion_angle,
        )


def _handle_user_satellite_connection(
    user_idx: int,
    user_satellites: list[UserSatellite],
    constellation_satellites: list[ConstellationSatellite],
    user_visibility: npt.NDArray[np.bool_],
    user_remaining_connection_time: npt.NDArray[np.int_],
    time_idx: int,
    user_connections: dict[int, int],
    assigned_satellites: set[int],
    graph: nx.Graph,
) -> None:
    """Handle connection logic for a single user satellite."""
    user_sat = user_satellites[user_idx]

    # Check if user satellite has an existing connection
    if user_idx in user_connections:
        current_sat_idx = user_connections[user_idx]

        # Check if current connection is still visible
        if user_visibility[user_idx, current_sat_idx, time_idx]:
            # Maintain existing connection
            const_sat = constellation_satellites[current_sat_idx]
            graph.add_edge(user_sat, const_sat)
            assigned_satellites.add(current_sat_idx)
            logger.debug(
                f"Time {time_idx}: User satellite {user_idx} "
                f"maintaining connection to constellation satellite {current_sat_idx}",
            )
            return

        # Connection lost, remove from tracking
        del user_connections[user_idx]
        logger.debug(
            f"Time {time_idx}: User satellite {user_idx} lost connection to constellation satellite {current_sat_idx}",
        )

    # Find new connection (only if no existing connection or connection lost)
    user_remaining_times = user_remaining_connection_time[user_idx, :, time_idx].copy()
    for assigned_sat_idx in assigned_satellites:
        user_remaining_times[assigned_sat_idx] = 0

    if user_remaining_times.max() > 0:
        selected_sat_idx = int(np.argmax(user_remaining_times))
        const_sat = constellation_satellites[selected_sat_idx]

        # Add connection if visible
        if user_visibility[user_idx, selected_sat_idx, time_idx]:
            graph.add_edge(user_sat, const_sat)
            assigned_satellites.add(selected_sat_idx)
            user_connections[user_idx] = selected_sat_idx
            logger.debug(
                f"Time {time_idx}: User satellite {user_idx} "
                f"connected to new constellation satellite {selected_sat_idx}",
            )


def _handle_ground_node_connection(
    ground_idx: int,
    ground_nodes: list[Gateway | StationaryOnGroundUser],
    constellation_satellites: list[ConstellationSatellite],
    ground_visibility: npt.NDArray[np.bool_],
    ground_remaining_visibility_time: npt.NDArray[np.int_],
    time_idx: int,
    ground_connections: dict[int, int],
    assigned_satellites: set[int],
    graph: nx.Graph,
) -> None:
    """Handle connection logic for a single ground node."""
    ground_node = ground_nodes[ground_idx]

    # Check if ground node has an existing connection
    if ground_idx in ground_connections:
        current_sat_idx = ground_connections[ground_idx]

        # Check if current connection is still visible
        if ground_visibility[ground_idx, current_sat_idx, time_idx]:
            # Maintain existing connection
            const_sat = constellation_satellites[current_sat_idx]
            graph.add_edge(ground_node, const_sat)
            assigned_satellites.add(current_sat_idx)
            logger.debug(
                f"Time {time_idx}: Ground node {ground_idx} "
                f"maintaining connection to constellation satellite {current_sat_idx}",
            )
            return

        # Connection lost, remove from tracking
        del ground_connections[ground_idx]
        logger.debug(
            f"Time {time_idx}: Ground node {ground_idx} lost connection to constellation satellite {current_sat_idx}",
        )

    # Find new connection (only if no existing connection or connection lost)
    ground_remaining_times = ground_remaining_visibility_time[ground_idx, :, time_idx].copy()
    for assigned_sat_idx in assigned_satellites:
        ground_remaining_times[assigned_sat_idx] = 0

    if ground_remaining_times.max() > 0:
        selected_sat_idx = int(np.argmax(ground_remaining_times))
        const_sat = constellation_satellites[selected_sat_idx]

        # Add connection if visible
        if ground_visibility[ground_idx, selected_sat_idx, time_idx]:
            graph.add_edge(ground_node, const_sat)
            assigned_satellites.add(selected_sat_idx)
            ground_connections[ground_idx] = selected_sat_idx
            logger.debug(
                f"Time {time_idx}: Ground node {ground_idx} "
                f"connected to new constellation satellite {selected_sat_idx}",
            )


# ---------------------------------------------------------------------------
# New functions using Constellation model
# ---------------------------------------------------------------------------


def _calculate_user_satellite_visibility(
    user_satellites: list[UserSatellite],
    constellation_satellites: list[ConstellationSatellite],
    dynamics_data: DynamicsData,
    *,
    max_distance: float,
    max_relative_angular_velocity: float,
    sun_exclusion_angle: float,
) -> npt.NDArray[np.bool_]:
    """Calculate visibility between user satellites and constellation satellites."""
    n_time = len(dynamics_data.time)
    n_users = len(user_satellites)
    n_constellation = len(constellation_satellites)

    visibility = np.zeros((n_users, n_constellation, n_time), dtype=np.bool_)
    for (user_idx, user_sat), (const_idx, const_sat) in tqdm(
        product(enumerate(user_satellites), enumerate(constellation_satellites)),
        total=n_users * n_constellation,
        desc="Calculating user satellite visibility",
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

        relative_angular_velocities = []
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
                relative_angular_velocities.append(float(max_rel_ang_vel))
            else:
                relative_angular_velocities.append(float("inf"))

        relative_angular_velocities_array = np.array(relative_angular_velocities)
        sun_angles = np.array(
            [angle_between(relative_pos_eci[t], dynamics_data.sun_direction_eci[t]) for t in range(n_time)],
        )

        distance_ok = distances <= max_distance
        sun_ok = (sun_angles >= sun_exclusion_angle) & (sun_angles <= (np.pi - sun_exclusion_angle))
        angular_velocity_ok = relative_angular_velocities_array <= max_relative_angular_velocity
        visibility[user_idx, const_idx, :] = distance_ok & sun_ok & angular_velocity_ok

    return visibility


def _calculate_ground_visibility(
    ground_nodes: list[Gateway | StationaryOnGroundUser],
    constellation_satellites: list[ConstellationSatellite],
    dynamics_data: DynamicsData,
    *,
    sun_exclusion_angle: float,
) -> npt.NDArray[np.bool_]:
    """Calculate visibility between ground nodes and constellation satellites."""
    n_time = len(dynamics_data.time)
    n_ground = len(ground_nodes)
    n_constellation = len(constellation_satellites)

    visibility = np.zeros((n_ground, n_constellation, n_time), dtype=np.bool_)
    for (ground_idx, ground_node), (sat_idx, satellite) in tqdm(
        product(enumerate(ground_nodes), enumerate(constellation_satellites)),
        total=n_ground * n_constellation,
        desc="Calculating ground node visibility",
    ):
        _, elevation, _ = ecef2aer(
            x=dynamics_data.satellite_position_ecef[satellite][:, 0],
            y=dynamics_data.satellite_position_ecef[satellite][:, 1],
            z=dynamics_data.satellite_position_ecef[satellite][:, 2],
            lat0=ground_node.latitude,
            lon0=ground_node.longitude,
            h0=ground_node.altitude,
            deg=False,
        )
        elevation_ok = elevation >= ground_node.minimum_elevation

        ground_x_ecef, ground_y_ecef, ground_z_ecef = geodetic2ecef(
            lat=ground_node.latitude,
            lon=ground_node.longitude,
            alt=ground_node.altitude,
            deg=False,
        )
        ground_pos_ecef = np.array([ground_x_ecef, ground_y_ecef, ground_z_ecef])
        relative_pos_ecef = dynamics_data.satellite_position_ecef[satellite] - ground_pos_ecef

        sun_angles = np.array(
            [angle_between(relative_pos_ecef[t], dynamics_data.sun_direction_eci[t]) for t in range(n_time)],
        )
        sun_ok = sun_angles >= sun_exclusion_angle

        visibility[ground_idx, sat_idx, :] = elevation_ok & sun_ok

    return visibility


def _calculate_remaining_connection_time(
    visibility: npt.NDArray[np.bool_],
) -> npt.NDArray[np.int_]:
    """Calculate remaining connection time for each node-satellite pair.

    Traverses time backwards so that ``remaining[i, j, t]`` gives the number
    of consecutive visible time steps from *t* onward.
    """
    n_nodes, n_satellites, n_time = visibility.shape
    remaining = np.zeros((n_nodes, n_satellites, n_time), dtype=np.int_)
    for t in reversed(range(n_time)):
        if t == n_time - 1:
            remaining[:, :, t] = visibility[:, :, t].astype(int)
        else:
            remaining[:, :, t] = np.where(visibility[:, :, t], remaining[:, :, t + 1] + 1, 0)
    return remaining


def _assign_connection(
    node_idx: int,
    node: UserSatellite | Gateway | StationaryOnGroundUser,
    constellation_satellites: list[ConstellationSatellite],
    visibility: npt.NDArray[np.bool_],
    remaining: npt.NDArray[np.int_],
    time_idx: int,
    connections: dict[int, int],
    assigned_satellites: set[int],
    graph: nx.Graph,
) -> None:
    """Try to keep an existing link or greedily pick the one with the longest remaining visibility."""
    # Try to maintain existing connection
    if node_idx in connections:
        current_sat_idx = connections[node_idx]
        if visibility[node_idx, current_sat_idx, time_idx]:
            graph.add_edge(node, constellation_satellites[current_sat_idx])
            assigned_satellites.add(current_sat_idx)
            return
        del connections[node_idx]

    # Find new connection with maximum remaining visibility
    remaining_times = remaining[node_idx, :, time_idx].copy()
    for assigned_sat_idx in assigned_satellites:
        remaining_times[assigned_sat_idx] = 0

    if remaining_times.max() > 0:
        selected = int(np.argmax(remaining_times))
        if visibility[node_idx, selected, time_idx]:
            graph.add_edge(node, constellation_satellites[selected])
            assigned_satellites.add(selected)
            connections[node_idx] = selected


def _build_topology_graphs(
    *,
    n_time: int,
    constellation_satellites: list[ConstellationSatellite],
    user_satellites: list[UserSatellite],
    ground_nodes: list[Gateway | StationaryOnGroundUser],
    user_visibility: npt.NDArray[np.bool_],
    ground_visibility: npt.NDArray[np.bool_],
    user_remaining: npt.NDArray[np.int_],
    ground_remaining: npt.NDArray[np.int_],
) -> list[nx.Graph]:
    """Build one graph per time step using greedy max-remaining-time assignment."""
    user_connections: dict[int, int] = {}
    ground_connections: dict[int, int] = {}

    graphs = []
    for time_idx in tqdm(range(n_time), desc="Building topology graphs"):
        assigned_satellites: set[int] = set()
        graph = nx.Graph()
        graph.add_nodes_from(constellation_satellites)
        graph.add_nodes_from(user_satellites)
        graph.add_nodes_from(ground_nodes)

        # Phase 1: Handle user satellites (priority)
        for user_idx, user_sat in enumerate(user_satellites):
            _assign_connection(
                user_idx,
                user_sat,
                constellation_satellites,
                user_visibility,
                user_remaining,
                time_idx,
                user_connections,
                assigned_satellites,
                graph,
            )

        # Phase 2: Handle ground nodes
        for ground_idx, ground_node in enumerate(ground_nodes):
            _assign_connection(
                ground_idx,
                ground_node,
                constellation_satellites,
                ground_visibility,
                ground_remaining,
                time_idx,
                ground_connections,
                assigned_satellites,
                graph,
            )

        graphs.append(graph)

    return graphs


def build_hybrid_us2c_g2c_topology(
    constellation: Constellation,
    *,
    user_satellites: Collection[UserSatellite],
    ground_nodes: Collection[Gateway | StationaryOnGroundUser],
    dynamics_data: DynamicsData,
    max_distance: float = float("inf"),
    max_relative_angular_velocity: float = float("inf"),
    sun_exclusion_angle: float = 0.0,
) -> list[nx.Graph]:
    """Build hybrid topology connecting user satellites and ground stations to constellation.

    Combines user-satellite and ground-station connectivity. User satellite
    connections are prioritized, then ground station connections are optimized
    based on maximum visibility duration.

    This function only needs the satellite set, so it accepts `Constellation`
    with any `SatelliteId` type.

    Args:
        constellation: Constellation (any SatelliteId type).
        user_satellites: User satellites to connect.
        ground_nodes: Ground stations or users.
        dynamics_data: Time-series dynamics data.
        max_distance: Maximum distance constraint for user satellite links.
        max_relative_angular_velocity: Maximum relative angular velocity constraint.
        sun_exclusion_angle: Sun exclusion angle constraint (radians).

    Returns:
        A list of networkx Graphs, one per time step.

    """
    logger.info(
        f"Building hybrid topology for {len(user_satellites)} user satellites and {len(ground_nodes)} ground nodes",
    )
    logger.info(f"Number of time steps: {len(dynamics_data.time)}")

    user_satellites_list = list(user_satellites)
    ground_nodes_list = list(ground_nodes)
    constellation_satellites = list(constellation.satellites.values())

    n_time = len(dynamics_data.time)

    # Calculate user satellite visibility
    logger.info("Calculating user satellite visibility...")
    user_visibility = _calculate_user_satellite_visibility(
        user_satellites_list,
        constellation_satellites,
        dynamics_data,
        max_distance=max_distance,
        max_relative_angular_velocity=max_relative_angular_velocity,
        sun_exclusion_angle=sun_exclusion_angle,
    )

    # Calculate ground node visibility
    logger.info("Calculating ground node visibility...")
    ground_visibility = _calculate_ground_visibility(
        ground_nodes_list,
        constellation_satellites,
        dynamics_data,
        sun_exclusion_angle=sun_exclusion_angle,
    )

    # Calculate remaining connection times (backward pass)
    user_remaining = _calculate_remaining_connection_time(user_visibility)
    ground_remaining = _calculate_remaining_connection_time(ground_visibility)

    # Build topology for each time step
    logger.info("Building topology graphs for each time step...")
    return _build_topology_graphs(
        n_time=n_time,
        constellation_satellites=constellation_satellites,
        user_satellites=user_satellites_list,
        ground_nodes=ground_nodes_list,
        user_visibility=user_visibility,
        ground_visibility=ground_visibility,
        user_remaining=user_remaining,
        ground_remaining=ground_remaining,
    )
