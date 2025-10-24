__all__ = [
    "HybridUS2CG2CTopologyBuilder",
]

import logging
from collections.abc import Collection
from itertools import product

import networkx as nx
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from cosmica.dtos import DynamicsData
from cosmica.dynamics import SatelliteConstellation
from cosmica.models import ConstellationSatellite, Gateway, StationaryOnGroundUser, UserSatellite
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
        constellation: SatelliteConstellation,
        user_satellites: Collection[UserSatellite],
        ground_nodes: Collection[Gateway | StationaryOnGroundUser],
        dynamics_data: DynamicsData,
    ) -> list[nx.Graph]:
        """Build hybrid topology connecting user satellites and ground stations to constellation."""
        logger.info(
            f"Building hybrid topology for {len(user_satellites)} user satellites and {len(ground_nodes)} ground nodes",
        )
        logger.info(f"Number of time steps: {len(dynamics_data.time)}")

        user_satellites = list(user_satellites)
        ground_nodes = list(ground_nodes)
        constellation_satellites = list(constellation.satellites)

        n_time = len(dynamics_data.time)

        # Calculate user satellite visibility
        logger.info("Calculating user satellite visibility...")
        user_visibility = self._calculate_user_satellite_visibility(
            user_satellites,
            constellation_satellites,
            dynamics_data,
        )

        # Calculate ground node visibility
        logger.info("Calculating ground node visibility...")
        ground_visibility = self._calculate_ground_visibility(
            ground_nodes,
            constellation_satellites,
            dynamics_data,
        )

        # Calculate remaining connection time for user satellites
        user_remaining_connection_time = self._calculate_remaining_connection_time(user_visibility)

        # Calculate remaining visibility time for ground nodes
        ground_remaining_visibility_time = self._calculate_remaining_connection_time(ground_visibility)

        # Initialize connection tracking
        user_connections: dict[int, int] = {}  # user_idx -> constellation_satellite_idx
        ground_connections: dict[int, int] = {}  # ground_idx -> constellation_satellite_idx

        # Build topology for each time step
        logger.info("Building topology graphs for each time step...")
        graphs = []
        for time_idx in tqdm(range(n_time), desc="Building topology graphs"):
            graph = self._build_graph_at_time(
                time_idx,
                constellation_satellites,
                user_satellites,
                ground_nodes,
                user_visibility,
                ground_visibility,
                user_remaining_connection_time,
                ground_remaining_visibility_time,
                user_connections,
                ground_connections,
            )
            graphs.append(graph)

        return graphs

    def _calculate_user_satellite_visibility(
        self,
        user_satellites: list[UserSatellite],
        constellation_satellites: list[ConstellationSatellite],
        dynamics_data: DynamicsData,
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
            # Get positions and velocities
            user_pos_eci = dynamics_data.satellite_position_eci[user_sat]
            const_pos_eci = dynamics_data.satellite_position_eci[const_sat]
            user_vel_eci = dynamics_data.satellite_velocity_eci[user_sat]
            const_vel_eci = dynamics_data.satellite_velocity_eci[const_sat]
            user_att_ang_vel_eci = dynamics_data.satellite_attitude_angular_velocity_eci[user_sat]
            const_att_ang_vel_eci = dynamics_data.satellite_attitude_angular_velocity_eci[const_sat]

            # Calculate relative position and velocity vectors
            relative_pos_eci = const_pos_eci - user_pos_eci
            relative_vel_eci = const_vel_eci - user_vel_eci

            # Calculate distances
            distances = np.linalg.norm(relative_pos_eci, axis=1)

            # Calculate relative angular velocities
            relative_angular_velocities = []
            for t in range(n_time):
                distance = distances[t]
                if distance > 0:
                    # Calculate relative angular velocity due to translational motion
                    relative_angular_velocity_translational = (
                        np.cross(relative_pos_eci[t], relative_vel_eci[t]) / distance**2
                    )

                    # Calculate relative angular velocities considering attitude angular velocities
                    rel_ang_vel_user = relative_angular_velocity_translational - user_att_ang_vel_eci[t]
                    rel_ang_vel_const = -relative_angular_velocity_translational - const_att_ang_vel_eci[t]

                    # Take maximum of the two relative angular velocities
                    max_rel_ang_vel = np.maximum(
                        np.linalg.norm(rel_ang_vel_user),
                        np.linalg.norm(rel_ang_vel_const),
                    )
                    relative_angular_velocities.append(float(max_rel_ang_vel))
                else:
                    relative_angular_velocities.append(float("inf"))

            relative_angular_velocities_array = np.array(relative_angular_velocities)

            # Calculate sun angles
            sun_angles = np.array(
                [angle_between(relative_pos_eci[t], dynamics_data.sun_direction_eci[t]) for t in range(n_time)],
            )

            # Check all constraints
            distance_ok = distances <= self.max_distance
            sun_ok = (sun_angles >= self.sun_exclusion_angle) & (sun_angles <= (np.pi - self.sun_exclusion_angle))
            angular_velocity_ok = relative_angular_velocities_array <= self.max_relative_angular_velocity

            # Combined visibility
            visibility[user_idx, const_idx, :] = distance_ok & sun_ok & angular_velocity_ok

        return visibility

    def _calculate_ground_visibility(
        self,
        ground_nodes: list[Gateway | StationaryOnGroundUser],
        constellation_satellites: list[ConstellationSatellite],
        dynamics_data: DynamicsData,
    ) -> npt.NDArray[np.bool_]:
        """Calculate visibility between ground nodes and constellation satellites.

        Considers both elevation angle and sun exclusion angle constraints.
        """
        n_time = len(dynamics_data.time)
        n_ground = len(ground_nodes)
        n_constellation = len(constellation_satellites)

        visibility = np.zeros((n_ground, n_constellation, n_time), dtype=np.bool_)

        for (ground_idx, ground_node), (sat_idx, satellite) in tqdm(
            product(enumerate(ground_nodes), enumerate(constellation_satellites)),
            total=n_ground * n_constellation,
            desc="Calculating ground node visibility",
        ):
            # Calculate elevation angle
            _, elevation, _ = ecef2aer(
                x=dynamics_data.satellite_position_ecef[satellite][:, 0],
                y=dynamics_data.satellite_position_ecef[satellite][:, 1],
                z=dynamics_data.satellite_position_ecef[satellite][:, 2],
                lat0=ground_node.latitude,
                lon0=ground_node.longitude,
                h0=ground_node.altitude,
                deg=False,
            )
            # Check elevation constraint
            elevation_ok = elevation >= ground_node.minimum_elevation

            # Calculate ground node position in ECEF
            ground_x_ecef, ground_y_ecef, ground_z_ecef = geodetic2ecef(
                lat=ground_node.latitude,
                lon=ground_node.longitude,
                alt=ground_node.altitude,
                deg=False,
            )
            ground_pos_ecef = np.array([ground_x_ecef, ground_y_ecef, ground_z_ecef])

            # Calculate relative position vector from ground node to satellite
            relative_pos_ecef = dynamics_data.satellite_position_ecef[satellite] - ground_pos_ecef

            # Calculate sun angles for each time step
            sun_angles = np.array(
                [angle_between(relative_pos_ecef[t], dynamics_data.sun_direction_eci[t]) for t in range(n_time)],
            )

            # Check sun exclusion angle constraint (avoid pointing too close to sun)
            sun_ok = sun_angles >= self.sun_exclusion_angle

            # Combined visibility considering both elevation and sun exclusion
            visibility[ground_idx, sat_idx, :] = elevation_ok & sun_ok

        return visibility

    def _calculate_remaining_connection_time(
        self,
        visibility: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.int_]:
        """Calculate remaining connection time for each node-satellite pair."""
        n_nodes, n_satellites, n_time = visibility.shape
        remaining_time = np.zeros((n_nodes, n_satellites, n_time), dtype=np.int_)

        # Traverse time backwards to calculate remaining visibility duration
        for time_idx in reversed(range(n_time)):
            if time_idx == n_time - 1:
                # Last time step: 1 if visible, 0 otherwise
                remaining_time[:, :, time_idx] = visibility[:, :, time_idx].astype(int)
            else:
                # If visible at current time, add 1 to future remaining time
                remaining_time[:, :, time_idx] = np.where(
                    visibility[:, :, time_idx],
                    remaining_time[:, :, time_idx + 1] + 1,
                    0,
                )

        return remaining_time

    def _build_graph_at_time(
        self,
        time_idx: int,
        constellation_satellites: list[ConstellationSatellite],
        user_satellites: list[UserSatellite],
        ground_nodes: list[Gateway | StationaryOnGroundUser],
        user_visibility: npt.NDArray[np.bool_],
        ground_visibility: npt.NDArray[np.bool_],
        user_remaining_connection_time: npt.NDArray[np.int_],
        ground_remaining_visibility_time: npt.NDArray[np.int_],
        user_connections: dict[int, int],
        ground_connections: dict[int, int],
    ) -> nx.Graph:
        """Build graph for a specific time step."""
        n_users = len(user_satellites)
        n_ground = len(ground_nodes)

        # Track which constellation satellites are already assigned
        assigned_satellites: set[int] = set()

        # Create graph
        graph = nx.Graph()
        graph.add_nodes_from(constellation_satellites)
        graph.add_nodes_from(user_satellites)
        graph.add_nodes_from(ground_nodes)

        # Phase 1: Handle user satellites (priority)
        for user_idx in range(n_users):
            self._handle_user_satellite_connection(
                user_idx,
                user_satellites,
                constellation_satellites,
                user_visibility,
                user_remaining_connection_time,
                time_idx,
                user_connections,
                assigned_satellites,
                graph,
            )

        # Phase 2: Handle ground nodes
        for ground_idx in range(n_ground):
            self._handle_ground_node_connection(
                ground_idx,
                ground_nodes,
                constellation_satellites,
                ground_visibility,
                ground_remaining_visibility_time,
                time_idx,
                ground_connections,
                assigned_satellites,
                graph,
            )

        logger.debug(
            f"Time {time_idx}: {len(user_connections)} user satellites "
            f"and {len(ground_connections)} ground nodes connected",
        )

        return graph

    def _handle_user_satellite_connection(
        self,
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
                f"Time {time_idx}: User satellite {user_idx} "
                f"lost connection to constellation satellite {current_sat_idx}",
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
        self,
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
                f"Time {time_idx}: Ground node {ground_idx} "
                f"lost connection to constellation satellite {current_sat_idx}",
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
