__all__ = [
    "draw_countries",
    "draw_coverage_area",
    "draw_lat_lon_grid",
    "draw_snapshot",
    "draw_snapshot_movie",
    "draw_urban_areas",
]
import copy
import importlib.resources
import logging
from dataclasses import dataclass, fields
from functools import cache
from typing import Annotated, Any

import geopandas as gpd
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.typing import ColorType
from pymap3d.ecef import ecef2geodetic
from typing_extensions import Doc

from cosmica.dtos import DynamicsData
from cosmica.models import (
    ConstellationSatellite,
    Gateway,
    Internet,
    Node,
    StationaryOnGroundUser,
    UserSatellite,
)
from cosmica.utils.coordinates import ecef2aer

from .utils import preserve_tick_params

logger = logging.getLogger(__name__)


@cache
def _load_shapefile_from_assets_dir(filename: str) -> gpd.GeoDataFrame:
    rel_path = f"assets/{filename}"
    resource = importlib.resources.files("cosmica.visualization").joinpath(rel_path)
    with importlib.resources.as_file(resource) as f:
        logger.debug(f"Loading shapefile from {f}")
        return gpd.read_file(f)


def draw_countries(
    *,
    ax: Axes,
    zorder: int = 0,
    ocean_color: ColorType = "#caecfc",
    countries_color: ColorType = "#fdfbe6",
) -> Axes:
    ax.set_facecolor(ocean_color)
    df_countries = _load_shapefile_from_assets_dir("ne_50m_admin_0_countries_lakes.shp")
    df_countries.plot(ax=ax, color=countries_color, edgecolor="grey", linewidth=0.5, zorder=zorder)
    ax.grid(color="black", alpha=0.2)
    return ax


def draw_urban_areas(*, ax: Axes, zorder: int = 1) -> Axes:
    df_urban_areas = _load_shapefile_from_assets_dir("ne_50m_urban_areas.shp")
    df_urban_areas.plot(ax=ax, color="red", zorder=zorder)
    return ax


def draw_lat_lon_grid(*, ax: Axes) -> Axes:
    ax.set_ylabel("Latitude")
    ax.set_yticks(np.arange(-90, 90 + 1, 30))
    ax.set_yticklabels([f"{ytick}°" for ytick in np.arange(-90, 90 + 1, 30)])
    ax.set_xlabel("Longitude")
    ax.set_xticks(np.arange(-180, 180 + 1, 30))
    ax.set_xticklabels([f"{xtick}°" for xtick in np.arange(-180, 180 + 1, 30)])

    ax.set_ylim(-90, 90)
    ax.set_xlim(-180, 180)

    ax.set_aspect("equal")

    return ax


@cache
def _dummy_class_creator(class_: type) -> type:
    # This function creates a new dummy class based on the original class
    assert issubclass(class_, Node), f"{class_} is not a subclass of Node"

    # Create a new class dynamically
    class DummyClass[T](class_):
        id: T

        # Copying the `from_real` method
        @classmethod
        def from_real(cls, real_obj: Node) -> "DummyClass":
            # Create a new instance with attributes copied from the real object
            assert isinstance(real_obj, class_), f"{real_obj} is not an instance of {class_}"
            class_field_names = [fld.name for fld in fields(class_)]
            return cls(
                id=real_obj.id,
                **{attr: getattr(real_obj, attr) for attr in class_field_names if attr != "id"},
            )

    # Apply the dataclass decorator with specific parameters
    return dataclass(frozen=True, kw_only=True, slots=True)(DummyClass)


# This is necessary because nx.draw_networkx_edges may not display the legend correctly
# See: https://github.com/networkx/networkx/issues/7054
def _add_legend_to_edges(handler: Any, label: str, ax: Axes) -> None:
    if isinstance(handler, list) and len(handler) > 0 and isinstance(handler[0], FancyArrowPatch):
        fap = handler[0]
        lin1 = mlines.Line2D(
            [],
            [],
            color=fap.get_edgecolor(),
            linestyle=fap.get_linestyle(),
            label=label,
        )
        ax.add_artist(lin1)


def draw_snapshot(  # noqa: C901, PLR0915
    *,
    graph: nx.Graph,
    dynamics_data: DynamicsData[Any],
    ax: Axes,
    with_labels: bool = False,
    focus_edges_list: list[set[tuple[Node, Node]]] | None = None,
    focus_edges_label_list: list[str] | None = None,
) -> Axes:
    # None のときのデフォルト値設定
    if focus_edges_list is None:
        focus_edges_list = []
    if focus_edges_label_list is None:
        # focus_edges と同じ数だけデフォルトのラベルを作成する
        focus_edges_label_list = [f"Focus edges {i}" for i in range(len(focus_edges_list))]

    if len(focus_edges_list) != len(focus_edges_label_list):
        msg = "focus_edges と focus_edges_label の要素数が一致していません。"
        raise ValueError(msg)

    constellation_satellites_to_draw = {node for node in graph.nodes if isinstance(node, ConstellationSatellite)}
    pos_constellation = {
        satellite: np.degrees(
            np.asarray(ecef2geodetic(*dynamics_data.satellite_position_ecef[satellite], deg=False))[(1, 0),],
        )
        for satellite in constellation_satellites_to_draw
    }

    user_satellites_to_draw = {node for node in graph.nodes if isinstance(node, UserSatellite)}
    pos_user_satellites = {
        satellite: np.degrees(
            np.asarray(ecef2geodetic(*dynamics_data.satellite_position_ecef[satellite], deg=False))[(1, 0),],
        )
        for satellite in user_satellites_to_draw
    }

    gateways = {node for node in graph.nodes if isinstance(node, Gateway)}
    pos_gateways = {gateway: np.degrees(np.array([gateway.longitude, gateway.latitude])) for gateway in gateways}

    on_ground_users = {node for node in graph.nodes if isinstance(node, StationaryOnGroundUser)}
    pos_ogu = {
        on_ground_user: np.degrees(np.array([on_ground_user.longitude, on_ground_user.latitude]))
        for on_ground_user in on_ground_users
    }

    internets = {node for node in graph.nodes if isinstance(node, Internet)}
    pos_internets = {internet: [np.nan, np.nan] for internet in internets}

    pos = pos_constellation | pos_user_satellites | pos_gateways | pos_ogu | pos_internets
    nodes_to_draw: set[Node] = constellation_satellites_to_draw | user_satellites_to_draw | gateways | on_ground_users

    with preserve_tick_params(ax):
        # Draw nodes
        # Draw constellation satellites
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            ax=ax,
            nodelist=set(pos_constellation),
            node_size=100,
            node_color="tab:blue",
            node_shape="s",
            alpha=0.7,
            label="Constellation satellite",
        )
        # Draw user satellites
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            ax=ax,
            nodelist=set(pos_user_satellites),
            node_size=120,
            node_color="tab:purple",
            node_shape="D",
            alpha=0.7,
            label="User satellite",
        )
        # Draw gateways
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=gateways,
            node_shape="^",
            node_color="tab:orange",
            node_size=150,
            alpha=0.7,
            label="Gateway",
            ax=ax,
        )
        # Draw on-ground users
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=on_ground_users,
            node_shape="o",
            node_color="tab:green",
            node_size=150,
            alpha=0.7,
            label="On-ground user",
            ax=ax,
        )
        if with_labels:
            # Draw labels of constellation
            nx.draw_networkx_labels(
                graph,
                pos,
                labels={node: str(node).split("-")[-1] for node in set(pos_constellation)},
                font_size=5,
                font_color="black",
                font_family="sans-serif",
                font_weight="normal",
                alpha=0.8,
                ax=ax,
            )
            # Draw labels of user satellites
            nx.draw_networkx_labels(
                graph,
                pos,
                labels={node: str(node).split("-")[-1] for node in set(pos_user_satellites)},
                font_size=5,
                font_color="purple",
                font_family="sans-serif",
                font_weight="normal",
                alpha=0.8,
                ax=ax,
            )
            # Draw labels of gateways
            nx.draw_networkx_labels(
                graph,
                pos,
                labels={node: str(node).split("-")[-1] for node in gateways},
                font_size=7,
                font_color="blue",
                font_family="sans-serif",
                font_weight="normal",
                alpha=0.8,
                ax=ax,
            )

        # Draw edges
        graph_pos_corrected = copy.deepcopy(graph)

        focus_edges_corrected_list = [focus_edges.copy() for focus_edges in focus_edges_list]

        # Re-draw the edges with the correct direction around the globe
        for u, v in graph.edges():
            if not (u in nodes_to_draw and v in nodes_to_draw):
                continue
            if abs(pos[u][0] - pos[v][0]) > 180:
                graph_pos_corrected.remove_edge(u, v)
                u_to_east = pos[u][0] < pos[v][0]

                dummy_u = _dummy_class_creator(type(u)).from_real(u)  # type: ignore[attr-defined,arg-type]
                pos[dummy_u] = pos[u].copy()
                pos[dummy_u][0] = pos[dummy_u][0] + 360 if u_to_east else pos[dummy_u][0] - 360

                dummy_v = _dummy_class_creator(type(v)).from_real(v)  # type: ignore[attr-defined,arg-type]
                pos[dummy_v] = pos[v].copy()
                pos[dummy_v][0] = pos[dummy_v][0] + 360 if not u_to_east else pos[dummy_v][0] - 360

                graph_pos_corrected.add_edge(u, dummy_v)
                graph_pos_corrected.add_edge(dummy_u, v)

                for focus_edges, focus_edges_corrected in zip(
                    focus_edges_list,
                    focus_edges_corrected_list,
                    strict=False,
                ):
                    for edge in focus_edges:
                        if (u, v) == edge or (u, v) == edge[::-1]:
                            focus_edges_corrected.remove(edge)
                            focus_edges_corrected.add((u, dummy_v))
                            focus_edges_corrected.add((dummy_u, v))

        intra_plane_edges = {
            (u, v)
            for u, v in graph_pos_corrected.edges()
            if isinstance(u, ConstellationSatellite)
            and isinstance(v, ConstellationSatellite)
            and u.id.plane_id == v.id.plane_id
        }
        inter_plane_edges = {
            (u, v)
            for u, v in graph_pos_corrected.edges()
            if isinstance(u, ConstellationSatellite)
            and isinstance(v, ConstellationSatellite)
            and u.id.plane_id != v.id.plane_id
        }
        constellation_gateway_edges = {
            (u, v)
            for u, v in graph_pos_corrected.edges()
            if (isinstance(u, ConstellationSatellite) and isinstance(v, Gateway))
            or (isinstance(u, Gateway) and isinstance(v, ConstellationSatellite))
        }
        constellation_usersatellite_edges = {
            (u, v)
            for u, v in graph_pos_corrected.edges()
            if (isinstance(u, ConstellationSatellite) and isinstance(v, UserSatellite))
            or (isinstance(u, UserSatellite) and isinstance(v, ConstellationSatellite))
        }
        inter_gateways_edges = {
            (u, v) for u, v in graph_pos_corrected.edges() if isinstance(u, Gateway) and isinstance(v, Gateway)
        }
        other_edges_to_draw = (
            {(u, v) for u, v in graph_pos_corrected.edges() if u in nodes_to_draw and v in nodes_to_draw}
            - intra_plane_edges
            - inter_plane_edges
            - constellation_gateway_edges
            - constellation_usersatellite_edges
            - inter_gateways_edges
        )

        # Draw intra-plane edges
        h1 = nx.draw_networkx_edges(
            graph_pos_corrected,
            pos,
            edgelist=intra_plane_edges,
            width=1,
            edge_color="tab:purple",
            ax=ax,
            arrows=True,
            alpha=0.7,
            arrowstyle="<|-|>",
            label="Intra-plane links",
        )
        _add_legend_to_edges(h1, "Intra-plane links", ax=ax)

        # Draw inter-plane edges
        h2 = nx.draw_networkx_edges(
            graph_pos_corrected,
            pos,
            edgelist=inter_plane_edges,
            width=1,
            edge_color="tab:green",
            ax=ax,
            arrows=True,
            alpha=0.7,
            arrowstyle="<|-|>",
            label="Inter-plane links",
        )
        _add_legend_to_edges(h2, "Inter-plane links", ax=ax)

        # Draw constellation-gateway edges
        h3 = nx.draw_networkx_edges(
            graph_pos_corrected,
            pos,
            edgelist=constellation_gateway_edges,
            width=1,
            edge_color="tab:brown",
            ax=ax,
            arrows=True,
            alpha=0.7,
            arrowstyle="<|-|>",
            label="Feeder links",
        )
        _add_legend_to_edges(h3, "Feeder links", ax=ax)

        # Draw constellation-usersatellite edges
        h4 = nx.draw_networkx_edges(
            graph_pos_corrected,
            pos,
            edgelist=constellation_usersatellite_edges,
            width=1,
            edge_color="tab:cyan",
            ax=ax,
            arrows=True,
            alpha=0.7,
            arrowstyle="<|-|>",
            label="Service links",
        )
        _add_legend_to_edges(h4, "Service links", ax=ax)

        # Draw inter-gateways edges
        h5 = nx.draw_networkx_edges(
            graph_pos_corrected,
            pos,
            edgelist=inter_gateways_edges,
            width=1,
            edge_color="tab:olive",
            ax=ax,
            arrows=True,
            alpha=0.7,
            arrowstyle="<|-|>",
            label="Gateway links",
        )
        _add_legend_to_edges(h5, "Gateway links", ax=ax)

        # Draw other edges
        h6 = nx.draw_networkx_edges(
            graph_pos_corrected,
            pos,
            edgelist=other_edges_to_draw,
            width=1,
            edge_color="tab:gray",
            ax=ax,
            arrows=True,
            alpha=0.7,
            arrowstyle="<|-|>",
            label="Other links",
        )
        _add_legend_to_edges(h6, "Other links", ax=ax)

        for focus_edges_corrected, focus_edges_label in zip(
            focus_edges_corrected_list,
            focus_edges_label_list,
            strict=False,
        ):
            if not focus_edges_corrected:
                continue  # 空集合ならスキップ
            h7 = nx.draw_networkx_edges(
                graph_pos_corrected,
                pos,
                edgelist=focus_edges_corrected,
                width=3,
                edge_color="tab:red",
                ax=ax,
                arrows=True,
                alpha=0.7,
                arrowstyle="-",
                label=focus_edges_label,
            )
            _add_legend_to_edges(h7, focus_edges_label, ax=ax)

    return ax


def draw_coverage_area[T](
    *,
    dynamics_data: Annotated[
        DynamicsData[ConstellationSatellite[T]],
        Doc("Dynamics data with no time dimension."),
    ],
    ax: Annotated[Axes, Doc("Matplotlib axes.")],
    min_elevation: Annotated[float, Doc("Minimum elevation angle in radians.")],
    face_color: Annotated[
        str,
        Doc("Color of the coverage area. Used as the `colors` parameter of `contourf` function."),
    ] = "red",
    face_alpha: Annotated[
        float,
        Doc("Transparency of the coverage area. Used as the `alpha` parameter of `contourf` function."),
    ] = 0.25,
    draw_edges: Annotated[bool, Doc("Whether to draw the edges of the coverage area")] = True,
    edge_color: Annotated[
        str,
        Doc("Color of the coverage area edge. Used as the `colors` parameter of `contour` function."),
    ] = "red",
    edge_alpha: Annotated[
        float,
        Doc("Transparency of the coverage area edge. Used as the `alpha` parameter of `contour` function."),
    ] = 1.0,
    edge_linewidth: Annotated[
        float | None,
        Doc("Width of the coverage area edge. Used as the `linewidths` parameter of `contour` function."),
    ] = None,
    edge_linestyle: Annotated[
        str | None,
        Doc("Style of the coverage area edge. Used as the `linestyles` parameter of `contour` function."),
    ] = None,
) -> Annotated[Axes, Doc("Matplotlib axes.")]:
    """Draw the coverage area of the satellites in the constellation.

    Note that this function is not optimized for performance, and may take a few minutes to complete.
    """
    latitudes = np.radians(np.linspace(-90, 90, 180))
    longitudes = np.radians(np.linspace(-180, 180, 360))
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)

    elevation_angles = {}
    for sat, pos_ecef in dynamics_data.satellite_position_ecef.items():
        elevation_angles[sat] = np.zeros_like(lat_grid)
        for i in range(len(latitudes)):
            for j in range(len(longitudes)):
                _azimuth, elevation, _srange = ecef2aer(
                    x=pos_ecef[0],
                    y=pos_ecef[1],
                    z=pos_ecef[2],
                    lat0=latitudes[i],
                    lon0=longitudes[j],
                    h0=0,
                    deg=False,
                )
                elevation_angles[sat][j, i] = elevation

    for elevation in elevation_angles.values():
        cs = ax.contourf(
            np.degrees(lon_grid),
            np.degrees(lat_grid),
            np.degrees(elevation),
            levels=[np.degrees(min_elevation), np.inf],
            colors=face_color,
            alpha=face_alpha,
        )

        if draw_edges:
            ax.contour(
                cs,
                levels=cs.levels,
                colors=edge_color,
                alpha=edge_alpha,
                linewidths=edge_linewidth,
                linestyles=edge_linestyle,
            )

    return ax


def draw_snapshot_movie(
    *,
    graph: list[nx.Graph],
    time_array: npt.NDArray,
    dynamics_data: DynamicsData,
    time_index_for_plot: npt.NDArray,
    fig: Figure,
    ax: Axes,
    interval_ms: int = 100,
) -> FuncAnimation:
    """Draw a snapshot of the network graph for a movie."""

    def update(frame: int):  # noqa: ANN202
        ax.clear()

        time_index = time_index_for_plot[frame]

        title = f"Time: {np.datetime_as_string(time_array[time_index], unit='ms').split('T')[1]}"
        ax.set_title(title)

        draw_lat_lon_grid(ax=ax)
        draw_countries(ax=ax)
        draw_snapshot(
            graph=graph[time_index],
            dynamics_data=dynamics_data[time_index],
            ax=ax,
            with_labels=False,
        )
        ax.legend(loc="lower left")

    return FuncAnimation(fig, update, frames=len(time_index_for_plot), interval=interval_ms)
