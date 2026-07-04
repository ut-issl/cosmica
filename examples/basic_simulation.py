"""Minimal COSMICA simulation script.

Run from the repository root with:

    uv run examples/basic_simulation.py

Add ``--plot`` to show a simple 2D equirectangular snapshot plot:

    uv run examples/basic_simulation.py --plot
"""

import argparse
from collections.abc import Sequence
from itertools import pairwise

import networkx as nx
import numpy as np

from cosmica.comm_link import (
    CommLinkCalculationCoordinator,
    MemorylessCommLinkCalculatorWrapper,
    SatToGatewayBinaryCommLinkCalculator,
    SatToSatBinaryCommLinkCalculatorWithRateCalc,
)
from cosmica.dtos import DynamicsData
from cosmica.dynamics import CircularSatelliteOrbitPropagator, get_sun_direction_eci
from cosmica.models import (
    CircularSatelliteOrbitModel,
    Constellation,
    ConstellationSatellite,
    Gateway,
    Node,
    UserSatellite,
    build_walker_delta_constellation,
)
from cosmica.topology import (
    build_elevation_based_g2c_topology,
    build_manhattan_topology,
    build_max_connection_time_us2c_topology,
)
from cosmica.utils.constants import EARTH_RADIUS
from cosmica.utils.coordinates import calc_dcm_eci2ecef
from cosmica.utils.vector import rowwise_innerdot
from cosmica.visualization.equirectangular import draw_countries, draw_lat_lon_grid, draw_snapshot


def main() -> None:
    """Set up a tiny relay scenario and print one topology snapshot."""
    parser = argparse.ArgumentParser(description="Run a minimal COSMICA simulation.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a simple 2D equirectangular network plot at the selected snapshot.",
    )
    args = parser.parse_args()

    epoch = np.datetime64("2026-01-01T00:00:00")
    time = epoch + np.arange(0, np.timedelta64(2, "h"), np.timedelta64(5, "m"))

    constellation = build_walker_delta_constellation(
        semi_major_axis=EARTH_RADIUS + 1_200e3,
        inclination=np.deg2rad(55.0),
        n_total_sats=6,
        n_geometry_planes=3,
        phasing_factor=1,
        epoch=epoch,
    )
    user_satellite = UserSatellite(
        id="EO-1",
        orbit=CircularSatelliteOrbitModel(
            semi_major_axis=EARTH_RADIUS + 550e3,
            inclination=np.deg2rad(97.6),
            raan=np.deg2rad(270.0),
            phase_at_epoch=np.deg2rad(40.0),
            epoch=epoch,
        ),
    )
    gateway = Gateway(
        id="Tokyo",
        latitude=np.deg2rad(35.68),
        longitude=np.deg2rad(139.76),
        minimum_elevation=np.deg2rad(10.0),
        altitude=0.0,
    )

    # Propagate all satellites and collect the data topology builders need.
    all_satellites = set(constellation.satellites.values()) | {user_satellite}
    dcm_eci2ecef = calc_dcm_eci2ecef(time)
    sun_direction_eci = get_sun_direction_eci(time)
    sun_direction_ecef = np.einsum("tij,tj->ti", dcm_eci2ecef, sun_direction_eci)
    orbit_states = {
        satellite: CircularSatelliteOrbitPropagator(satellite.orbit).propagate(time) for satellite in all_satellites
    }
    dynamics_data = DynamicsData(
        time=time,
        dcm_eci2ecef=dcm_eci2ecef,
        sun_direction_eci=sun_direction_eci,
        sun_direction_ecef=sun_direction_ecef,
        satellite_position_eci={satellite: state.position_eci for satellite, state in orbit_states.items()},
        satellite_velocity_eci={satellite: state.velocity_eci for satellite, state in orbit_states.items()},
        satellite_position_ecef={
            satellite: state.calc_position_ecef(dcm_eci2ecef) for satellite, state in orbit_states.items()
        },
        satellite_attitude_angular_velocity_eci={
            satellite: np.cross(state.position_eci, state.velocity_eci)
            / rowwise_innerdot(state.position_eci, state.position_eci, keepdims=True)
            for satellite, state in orbit_states.items()
        },
    )

    # Build the links available at each time step.
    gateway_to_constellation = build_elevation_based_g2c_topology(
        constellation,
        ground_nodes=[gateway],
        dynamics_data=dynamics_data,
    )
    user_to_constellation = build_max_connection_time_us2c_topology(
        constellation,
        user_satellites=[user_satellite],
        dynamics_data=dynamics_data,
    )
    inter_satellite_links = build_manhattan_topology(constellation)

    # Merge link types into a full network snapshot for each time step.
    network_snapshots = [
        nx.compose_all([g2c, us2c, inter_satellite_links])
        for g2c, us2c in zip(gateway_to_constellation, user_to_constellation, strict=True)
    ]

    # Calculate communication performance for each active link.
    satellite_to_gateway_calculator = MemorylessCommLinkCalculatorWrapper(
        SatToGatewayBinaryCommLinkCalculator(link_capacity=1e9),
    )
    satellite_to_satellite_calculator = MemorylessCommLinkCalculatorWrapper(
        SatToSatBinaryCommLinkCalculatorWithRateCalc(
            lna_gain=40.0,
        ),
    )
    performance_time_series = CommLinkCalculationCoordinator(
        calculator_assignment={
            (ConstellationSatellite, Gateway): satellite_to_gateway_calculator,
            (ConstellationSatellite, ConstellationSatellite): satellite_to_satellite_calculator,
            (UserSatellite, ConstellationSatellite): satellite_to_satellite_calculator,
        },
    ).calc(
        [set(snapshot.edges) for snapshot in network_snapshots],
        dynamics_data=dynamics_data,
        rng=np.random.default_rng(seed=0),
    )

    # Inspect the snapshot with the most active links.
    snapshot_index = max(range(len(network_snapshots)), key=lambda index: network_snapshots[index].number_of_edges())
    snapshot = network_snapshots[snapshot_index]

    print(f"Simulated {len(time)} time steps from {time[0]} to {time[-1]}.")  # noqa: T201
    print(f"Constellation satellites: {len(constellation.satellites)}")  # noqa: T201
    snapshot_performance = performance_time_series[snapshot_index]
    available_links = [performance for performance in snapshot_performance.values() if performance["link_available"]]

    print(f"Snapshot time: {dynamics_data.time[snapshot_index]}")  # noqa: T201
    print(f"Active nodes: {snapshot.number_of_nodes()}, active links: {snapshot.number_of_edges()}")  # noqa: T201
    print(f"Available communication links: {len(available_links)}")  # noqa: T201

    route_edges = []
    if nx.has_path(snapshot, gateway, user_satellite):
        route = nx.shortest_path(snapshot, gateway, user_satellite)
        route_edges = list(pairwise(route))
        route_performance = [
            snapshot_performance[(src, dst)] if (src, dst) in snapshot_performance else snapshot_performance[(dst, src)]
            for src, dst in route_edges
        ]
        route_capacity_gbps = min(performance["link_capacity"] for performance in route_performance) / 1e9
        route_delay_ms = sum(performance["delay"] for performance in route_performance) * 1e3

        path = " -> ".join(str(node) for node in route)
        print(f"Gateway-to-user route: {path}")  # noqa: T201
        print("Route edge performance:")  # noqa: T201
        for (src, dst), performance in zip(route_edges, route_performance, strict=True):
            capacity_gbps = performance["link_capacity"] / 1e9
            delay_ms = performance["delay"] * 1e3
            print(f"  {src} -> {dst}: {capacity_gbps:.2f} Gbps, {delay_ms:.2f} ms")  # noqa: T201
        print(f"Route bottleneck capacity: {route_capacity_gbps:.2f} Gbps")  # noqa: T201
        print(f"Route propagation delay: {route_delay_ms:.2f} ms")  # noqa: T201
    else:
        print("No gateway-to-user route in this snapshot.")  # noqa: T201

    if args.plot:
        _draw_equirectangular_snapshot(
            snapshot=snapshot,
            constellation=constellation,
            dynamics_data=dynamics_data,
            snapshot_index=snapshot_index,
            route_edges=route_edges,
        )


def _draw_equirectangular_snapshot(
    *,
    snapshot: nx.Graph,
    constellation: Constellation[tuple[int, int]],
    dynamics_data: DynamicsData,
    snapshot_index: int,
    route_edges: Sequence[tuple[Node, Node]],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    draw_lat_lon_grid(ax=ax)
    draw_countries(ax=ax)
    draw_snapshot(
        graph=snapshot,
        constellation=constellation,
        dynamics_data=dynamics_data[snapshot_index],
        ax=ax,
        with_labels=True,
        focus_edges_list=[set(route_edges)] if route_edges else None,
        focus_edges_label_list=["Gateway-to-user route"] if route_edges else None,
    )
    ax.set_title(f"COSMICA network snapshot at {dynamics_data.time[snapshot_index]}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
