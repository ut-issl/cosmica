import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import matplotlib.animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import typer
from matplotlib.axes import Axes
from rich.console import Console
from tqdm import tqdm

from cosmica.comm_link import (
    CommLinkCalculationCoordinator,
    MemorylessCommLinkCalculatorWrapper,
    SatToGatewayBinaryCommLinkCalculator,
    SatToSatBinaryCommLinkCalculator,
)
from cosmica.dtos import DynamicsData
from cosmica.dynamics import MultiOrbitalPlaneConstellation, get_sun_direction_eci
from cosmica.models import (
    ConstellationSatellite,
    Gateway,
)
from cosmica.topology import ElevationBasedG2CTopologyBuilder
from cosmica.topology.intra_constellation import ManhattanTopologyBuilder
from cosmica.utils.coordinates import calc_dcm_eci2ecef
from cosmica.utils.vector import rowwise_innerdot
from cosmica.visualization.equirectangular import (
    draw_countries,
    draw_lat_lon_grid,
    draw_snapshot,
    draw_urban_areas,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
logger = logging.getLogger(__name__)
console = Console()
err_console = Console(stderr=True)


def main(  # noqa: C901, PLR0912, PLR0915
    constellation_path: Annotated[
        Path | None,
        typer.Option(
            "--constellation",
            help="Path to the constellation TOML file. If not provided, no satellite will be shown",
        ),
    ] = None,
    gateways_path: Annotated[
        Path | None,
        typer.Option("--gateways", help="Path to the gateways TOML file. If not provided, no gateway will be shown"),
    ] = None,
    start_time_arg: Annotated[datetime | None, typer.Option("--time", help="Start time of the plot")] = None,
    duration_arg: Annotated[
        str | None,
        typer.Option(
            "--duration",
            help="Duration of the plot. It will be parsed using `pandas.to_timedelta` (e.g., '1h', '30m'). "
            " If not provided, only one frame will be plotted.",
        ),
    ] = None,
    time_step_arg: Annotated[
        str,
        typer.Option(
            "--time-step",
            help="Time step of the plot. It will be parsed using `pandas.to_timedelta` (e.g., '1m')",
        ),
    ] = "1m",
    check_links: Annotated[  # noqa: FBT002
        bool,
        typer.Option(..., help="Check link availability and show only the available link edges"),
    ] = False,
    output: Annotated[Path | None, typer.Option(..., help="Path to the output file")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Seed for the random number generator")] = None,
    inter_plane_offset: Annotated[
        int,
        typer.Option(
            "--inter-plane-offset",
            help="Offset between the planes of the constellation",
            show_default=True,
        ),
    ] = 0,
    last_to_first_plane_offset: Annotated[
        int,
        typer.Option(
            "--last-to-first-plane-offset",
            help="Offset between the last and the first planes of the constellation",
            show_default=True,
        ),
    ] = 0,
    max_inter_satellite_distance_km: Annotated[
        float,
        typer.Option(
            "--max-inter-satellite-distance",
            help="Maximum distance between two satellites for communication (km). Used for link availability check.",
            show_default=True,
        ),
    ] = 6_000,
    lowest_altitude_km: Annotated[
        float,
        typer.Option(
            "--lowest-altitude",
            help="Lowest altitude for communication (km). Used for link availability check.",
            show_default=True,
        ),
    ] = 80,
    max_relative_angular_velocity_dps: Annotated[
        float,
        typer.Option(
            "--max-relative-angular-velocity",
            help="Maximum relative angular velocity between two satellites for communication (deg/s). "
            "Used for link availability check.",
            show_default=True,
        ),
    ] = 1,
    sun_exclusion_angle_deg: Annotated[
        float,
        typer.Option(
            "--sun-exclusion-angle",
            help="Sun exclusion angle (deg). Used for link availability check.",
            show_default=True,
        ),
    ] = 5,
) -> NoReturn:
    # Sanitize the inputs
    constellation = MultiOrbitalPlaneConstellation.from_toml_file(constellation_path) if constellation_path else None
    gateways: list[Gateway[Any]] | None = Gateway.from_toml_file(gateways_path) if gateways_path else None

    rng = np.random.default_rng(seed)

    # Construct the time array
    time: npt.NDArray[np.datetime64]
    if duration_arg is None:
        time = np.array([np.datetime64(start_time_arg)]) if start_time_arg else np.array([np.datetime64("now")])
    else:
        start_time = np.datetime64(start_time_arg) if start_time_arg else np.datetime64("now")
        try:
            duration = np.timedelta64(pd.to_timedelta(duration_arg))
        except ValueError:
            msg = "It should be parsable by `pandas.to_timedelta`."
            raise typer.BadParameter(
                msg,
                param_hint="--duration",
            ) from None
        try:
            time_step = np.timedelta64(pd.to_timedelta(time_step_arg))
        except ValueError:
            msg = "It should be parsable by `pandas.to_timedelta`."
            raise typer.BadParameter(
                msg,
                param_hint="--time-step",
            ) from None
        time = np.arange(start_time, start_time + duration, time_step)

    dcm_eci2ecef = calc_dcm_eci2ecef(time)
    sun_dir_eci = get_sun_direction_eci(time)
    sun_dir_ecef = np.stack(
        [
            dcm_eci2ecef_t @ sun_dir_eci_t  # type: ignore[operator]
            for dcm_eci2ecef_t, sun_dir_eci_t in zip(dcm_eci2ecef, sun_dir_eci, strict=True)
        ],
    )
    if constellation is not None:
        graph_c = ManhattanTopologyBuilder(
            inter_plane_offset=inter_plane_offset,
            last_to_first_plane_offset=last_to_first_plane_offset,
        ).build(constellation=constellation)

        # Calculate the dynamics data
        result = constellation.propagate(time)
        position_ecef = {satellite: result[satellite].calc_position_ecef(dcm_eci2ecef) for satellite in result}
        # attitude angular velocity is assumed to be the same as the orbital angular velocity
        satellite_attitude_angular_velocity = {
            satellite: np.cross(state.position_eci, state.velocity_eci)
            / rowwise_innerdot(state.position_eci, state.position_eci, keepdims=True)
            for satellite, state in result.items()
        }

        dynamics_data = DynamicsData(
            time=time,
            dcm_eci2ecef=dcm_eci2ecef,
            satellite_position_eci={satellite: state.position_eci for satellite, state in result.items()},
            satellite_velocity_eci={satellite: state.velocity_eci for satellite, state in result.items()},
            satellite_position_ecef=position_ecef,
            satellite_attitude_angular_velocity_eci=satellite_attitude_angular_velocity,
            sun_direction_eci=sun_dir_eci,
            sun_direction_ecef=sun_dir_ecef,
        )

        graphs_g2c = ElevationBasedG2CTopologyBuilder().build(
            constellation=constellation,
            ground_nodes=gateways if gateways is not None else [],
            dynamics_data=dynamics_data,
        )
        graphs = [nx.compose(graph_c, graph_g2c) for graph_g2c in graphs_g2c]
    else:
        if gateways is None:
            msg = "Either `constellation` or `gateways` must be provided."
            raise typer.BadParameter(msg)
        graph_g = nx.Graph()
        graph_g.add_nodes_from(gateways)
        graphs = [graph_g for _ in time]
        dynamics_data = DynamicsData(
            time=time,
            dcm_eci2ecef=dcm_eci2ecef,
            satellite_position_eci={},
            satellite_velocity_eci={},
            satellite_position_ecef={},
            satellite_attitude_angular_velocity_eci={},
            sun_direction_eci=sun_dir_eci,
            sun_direction_ecef=sun_dir_ecef,
        )

    if check_links:
        s2s_calculator = SatToSatBinaryCommLinkCalculator(
            inter_satellite_link_capacity=1e9,  # 1 Gbps
            max_inter_satellite_distance=max_inter_satellite_distance_km * 1e3,
            lowest_altitude=lowest_altitude_km * 1e3,
            max_relative_angular_velocity=np.deg2rad(max_relative_angular_velocity_dps),
            sun_exclusion_angle=np.deg2rad(sun_exclusion_angle_deg),
        )
        s2g_calculator = SatToGatewayBinaryCommLinkCalculator(
            satellite_to_gateway_link_capacity=1e9,  # 1 Gbps
        )
        coordinator = CommLinkCalculationCoordinator(
            calculator_assignment={
                (ConstellationSatellite, ConstellationSatellite): MemorylessCommLinkCalculatorWrapper(s2s_calculator),
                (ConstellationSatellite, Gateway): MemorylessCommLinkCalculatorWrapper(s2g_calculator),
            },
        )

        link_performance = coordinator.calc(
            [graph.edges for graph in graphs],
            dynamics_data=dynamics_data,
            rng=rng,
        )
        for graph, performance in zip(graphs, link_performance, strict=True):
            nx.set_edge_attributes(graph, performance)
            unavailable_edges = [edge for edge, perf in performance.items() if not perf["link_available"]]
            graph.remove_edges_from(unavailable_edges)

    if duration_arg is None:
        # Plot only one frame
        # The axis is assigned to a variable to prevent it from being garbage collected (just in case)
        _ax = draw_graph(graph=graphs[0], dynamics_data=dynamics_data[0], output_file=output)
        if output is None:
            plt.show()
    else:
        # The animation should be assigned to a variable to prevent it from being garbage collected
        _ani = animate_graphs(graphs=graphs, dynamics_data=dynamics_data)
        if output is None:
            plt.show()

    raise typer.Exit


def draw_graph(
    graph: nx.Graph,
    dynamics_data: DynamicsData,
    output_file: Path | None = None,
    *,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _fig, ax = plt.subplots(figsize=(15, 8))

    ax.set_title(f"Time: {np.datetime_as_string(dynamics_data.time, unit='s')}")

    draw_lat_lon_grid(ax=ax)
    draw_countries(ax=ax)
    draw_urban_areas(ax=ax)
    draw_snapshot(graph=graph, dynamics_data=dynamics_data, ax=ax)

    # Place the legend on the South Pacific Ocean
    ax.legend(loc="lower left")

    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")

    return ax


def animate_graphs(
    graphs: Sequence[nx.Graph],
    dynamics_data: DynamicsData,
    output_file: Path | None = None,
    *,
    writer: str | None = None,
) -> matplotlib.animation.FuncAnimation:
    ## Plot
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(15, 8))

    def animate(time_index: int) -> None:
        ax.clear()
        draw_graph(graph=graphs[time_index], dynamics_data=dynamics_data[time_index], ax=ax)

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(dynamics_data.time), interval=100)  # type: ignore[arg-type]
    if output_file is not None:
        writer = writer or "imagemagick"
        progress_bar = tqdm(total=len(dynamics_data.time))
        ani.save(
            str(output_file),
            writer=writer,
            fps=10,
            dpi=100,
            progress_callback=lambda _i, _n: progress_bar.update(1),
        )
        progress_bar.close()

    return ani
