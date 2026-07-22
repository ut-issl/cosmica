__all__ = [
    "SatToGatewayBinaryCommLinkCalculator",
    "SatToGatewayBinaryCommLinkCalculatorWithScintillation",
    "SatToGatewayBinaryMemoryCommLinkCalculator",
    "SatToGatewayStochasticBinaryCommLinkCalculator",
]

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Doc

from cosmica.comm_link.uncertainty import CloudStates
from cosmica.dtos import DynamicsData
from cosmica.models import Gateway, GatewayOGS, Node, Satellite
from cosmica.utils.constants import BOLTZ_CONST, SPEED_OF_LIGHT
from cosmica.utils.coordinates import ecef2aer, geodetic2ecef
from cosmica.utils.gauss_beam import calc_gauss_rx_power
from cosmica.utils.vector import angle_between

from .base import CommLinkCalculator, CommLinkPerformance, MemorylessCommLinkCalculator
from .uncertainty import ApertureAveragedLogNormalScintillationModel, AtmosphericScintillationModel


class SatToGatewayBinaryCommLinkCalculator(
    MemorylessCommLinkCalculator[Satellite[Any], Gateway[Any]],
):
    """Calculate satellite-to-gateway (downlink) communication link performance.

    The link performance is calculated as a binary value, i.e., 1 if the link is available and 0 otherwise.

    Handles the satellite -> gateway direction only; use `GatewayToSatBinaryCommLinkCalculator`
    for the gateway -> satellite (uplink) direction.
    """

    def __init__(
        self,
        *,
        link_capacity: Annotated[float, Doc("Capacity of the satellite -> gateway (downlink) in bps.")],
        sun_exclusion_angle: float = 0.0,
    ) -> None:
        self.link_capacity = link_capacity
        self.sun_exclusion_angle = sun_exclusion_angle

    def calc(
        self,
        edges: Collection[tuple[Satellite[Any], Gateway[Any]]],
        *,
        dynamics_data: DynamicsData[Satellite[Any]],
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[tuple[Satellite[Any], Gateway[Any]], CommLinkPerformance]:
        return {
            edge: self._calc_satellite_to_gateway(
                satellite_position_ecef=dynamics_data.satellite_position_ecef[edge[0]],
                gateway=edge[1],
                sun_direction_ecef=dynamics_data.sun_direction_ecef,
            )
            for edge in edges
        }

    def _calc_satellite_to_gateway(
        self,
        satellite_position_ecef: npt.NDArray[np.floating],
        gateway: Gateway[Any],
        sun_direction_ecef: npt.NDArray[np.floating],
    ) -> CommLinkPerformance:
        assert satellite_position_ecef.shape == (3,)
        assert sun_direction_ecef.shape == (3,)
        _, elevation, srange = ecef2aer(
            x=satellite_position_ecef[0],
            y=satellite_position_ecef[1],
            z=satellite_position_ecef[2],
            lat0=gateway.latitude,
            lon0=gateway.longitude,
            h0=gateway.altitude,
            deg=False,
        )
        x0, y0, z0 = geodetic2ecef(
            lat=gateway.latitude,
            lon=gateway.longitude,
            alt=gateway.altitude,
            deg=False,
        )
        gateway_ecef = np.array([x0, y0, z0])
        gateway_to_satellite_ecef = satellite_position_ecef - gateway_ecef
        edge_sun_angle = angle_between(gateway_to_satellite_ecef, sun_direction_ecef)
        link_available = bool(elevation >= gateway.minimum_elevation and edge_sun_angle >= self.sun_exclusion_angle)
        return CommLinkPerformance(
            link_capacity=self.link_capacity if link_available else 0.0,
            delay=float(srange / SPEED_OF_LIGHT),
            link_available=link_available,
        )


class SatToGatewayBinaryMemoryCommLinkCalculator(
    CommLinkCalculator[Satellite[Any], Gateway[Any]],
):
    """Calculate satellite-to-gateway (downlink) communication link performance with link acquisition delay.

    The link performance is calculated as a binary value, i.e., 1 if the link is available and 0 otherwise.

    Link acquisition delay is tracked independently per directed edge (satellite, gateway): when an edge
    appears or its underlying memoryless availability drops, the link goes through (re)acquisition and is
    unavailable for `link_acquisition_time` seconds.

    Handles the satellite -> gateway direction only; use `GatewayToSatBinaryMemoryCommLinkCalculator`
    for the gateway -> satellite (uplink) direction.
    """

    def __init__(
        self,
        *,
        memoryless_calculator: MemorylessCommLinkCalculator[Satellite[Any], Gateway[Any]],
        link_acquisition_time: float = 60.0,
        skip_link_acquisition_at_simulation_start: bool = True,
    ) -> None:
        self.memoryless_calculator = memoryless_calculator
        self.link_acquisition_time = link_acquisition_time
        self.skip_link_acquisition_at_simulation_start = skip_link_acquisition_at_simulation_start

    def calc(
        self,
        edges_time_series: Sequence[Collection[tuple[Satellite[Any], Gateway[Any]]]],
        *,
        dynamics_data: DynamicsData[Satellite[Any]],
        rng: np.random.Generator,
    ) -> list[dict[tuple[Satellite[Any], Gateway[Any]], CommLinkPerformance]]:
        assert len(edges_time_series) == len(dynamics_data.time)

        comm_link_time_series: list[dict[tuple[Satellite[Any], Gateway[Any]], CommLinkPerformance]] = []

        # ― per-directed-edge state ―
        # Link acquisition is tracked independently for each directed edge (satellite, gateway).
        link_acquisition_start_time: dict[tuple[Satellite[Any], Gateway[Any]], np.datetime64] = {}
        prev_edges: frozenset[tuple[Satellite[Any], Gateway[Any]]] = frozenset()

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
            for edge in disappeared_edges:  # remove edges that disappeared
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


class SatToGatewayStochasticBinaryCommLinkCalculator(
    CommLinkCalculator[Satellite[Any], Gateway[Any]],
):
    def __init__(
        self,
        *,
        memoryless_calculator: MemorylessCommLinkCalculator[Satellite[Any], Gateway[Any]],
        stochastic_model_factory: Callable[[Node[Any], Node[Any]], CloudStates[np.bool_]],
    ) -> None:
        self.memoryless_calculator = memoryless_calculator
        self.stochastic_model_factory = stochastic_model_factory

    def calc(
        self,
        edges_time_series: Sequence[Collection[tuple[Satellite[Any], Gateway[Any]]]],
        *,
        dynamics_data: DynamicsData[Satellite[Any]],
        rng: np.random.Generator,
    ) -> list[dict[tuple[Satellite[Any], Gateway[Any]], CommLinkPerformance]]:
        assert len(edges_time_series) == len(dynamics_data.time)
        all_edges = {edge for edges_snapshot in edges_time_series for edge in edges_snapshot}
        edge_interrupted = {
            edge: self.stochastic_model_factory(edge[0], edge[1]).simulate(time=dynamics_data.time, rng=rng)
            for edge in all_edges
        }

        comm_link_performance = []
        for time_index, edges_snapshot in enumerate(edges_time_series):
            comm_link = self.memoryless_calculator.calc(
                edges=edges_snapshot,
                dynamics_data=dynamics_data[time_index],
                rng=rng,
            )
            for edge in comm_link:
                if edge_interrupted[edge][time_index]:
                    comm_link[edge] = CommLinkPerformance(link_capacity=0.0, delay=0.0, link_available=False)
            comm_link_performance.append(comm_link)

        return comm_link_performance


class SatToGatewayBinaryCommLinkCalculatorWithScintillation(
    MemorylessCommLinkCalculator[Satellite[Any], GatewayOGS[Any]],
):
    """Calculate satellite-to-gateway communication link performance for each edge in a network with turbulence.

    The link performance is calculated as a binary value, i.e., 1 if the link is available and 0 otherwise.
    """

    def __init__(
        self,
        *,
        satellite_to_gateway_link_capacity: float,
        link_capacity: float,
        noise_figure: float,
        lna_gain: float,
        lct_p0: float,
        turbulence_model_factory: Callable[
            [GatewayOGS[Any]],
            AtmosphericScintillationModel,
        ] = lambda gateway: ApertureAveragedLogNormalScintillationModel(
            default_rytov_variance=gateway.rytov_variance,
            wavelength=1550 * 1e-9,
            aperture_diameter=gateway.aperture_size,
        ),
    ) -> None:
        self.satellite_to_gateway_link_capacity = satellite_to_gateway_link_capacity
        self.link_capacity = link_capacity
        self.noise_figure = noise_figure
        self.lna_gain = lna_gain
        self.lct_p0 = lct_p0
        self.turbulence_model_factory = turbulence_model_factory

    def calc(
        self,
        edges: Collection[tuple[Satellite[Any], GatewayOGS[Any]]],
        *,
        dynamics_data: DynamicsData[Satellite[Any]],
        rng: np.random.Generator,
    ) -> dict[tuple[Satellite[Any], GatewayOGS[Any]], CommLinkPerformance]:
        gateway_turbulence_map = self._map_turbulence2gateway(edges)
        return {
            edge: self._calc_satellite_to_gateway(
                satellite_position_ecef=dynamics_data.satellite_position_ecef[edge[0]],
                sun_direction_ecef=dynamics_data.sun_direction_ecef,
                sun_exclusion_angle=0.0,
                rng=rng,
                gateway=edge[1],
                gateway_turbulence_map=gateway_turbulence_map,
            )
            for edge in edges
        }

    def _map_turbulence2gateway(
        self,
        edges: Collection[tuple[Satellite[Any], GatewayOGS[Any]]],
    ) -> dict[GatewayOGS[Any], AtmosphericScintillationModel]:
        gateway_turbulence_map: dict[GatewayOGS[Any], AtmosphericScintillationModel] = {}
        for edge in edges:
            if gateway_turbulence_map.get(edge[1]) is None:
                gateway_turbulence_map[edge[1]] = self.turbulence_model_factory(edge[1])
        return gateway_turbulence_map

    def _calc_capacity(
        self,
        distance: Annotated[
            float,
            Doc("Distance between satellite and OGS"),
        ],
        gateway: Annotated[
            GatewayOGS[Any],
            Doc("Gateway instance in the satellite-gateway link"),
        ],
        turbulence_map: Annotated[
            Mapping[GatewayOGS[Any], AtmosphericScintillationModel],
            Doc("Dictionary mapping each OGS to its respective turbulence model."),
        ],
        rng: np.random.Generator,
        rytov_variance: Annotated[
            float | None,
            Doc("Rytov variance value"),
        ] = None,
    ) -> float:
        scintillation_factor = turbulence_map[gateway].sample(
            rng=rng,
            link_distance=distance,
            rytov_variance=rytov_variance,
        )
        # Downlink Capacity Calculation
        power = calc_gauss_rx_power(self.lct_p0, gateway.aperture_size, distance)
        t_sys = 300  # K
        noise_factor = 10 ** (self.noise_figure / 10)
        noise = t_sys * self.link_capacity * BOLTZ_CONST * noise_factor
        gain = 10 ** (self.lna_gain / 10)  # convert from dB to W
        snr = gain * power * scintillation_factor / (noise)
        return self.link_capacity * np.log2(1 + snr)

    def _calc_satellite_to_gateway(
        self,
        satellite_position_ecef: npt.NDArray[np.floating],
        gateway: GatewayOGS[Any],
        gateway_turbulence_map: Mapping[GatewayOGS[Any], AtmosphericScintillationModel],
        sun_direction_ecef: npt.NDArray[np.floating],
        rng: np.random.Generator,
        sun_exclusion_angle: float = 0.0,
    ) -> CommLinkPerformance:
        assert satellite_position_ecef.shape == (3,)
        assert sun_direction_ecef.shape == (3,)
        _, elevation, srange = ecef2aer(
            x=satellite_position_ecef[0],
            y=satellite_position_ecef[1],
            z=satellite_position_ecef[2],
            lat0=gateway.latitude,
            lon0=gateway.longitude,
            h0=gateway.altitude,
            deg=False,
        )
        x0, y0, z0 = geodetic2ecef(
            lat=gateway.latitude,
            lon=gateway.longitude,
            alt=gateway.altitude,
            deg=False,
        )
        gateway_ecef = np.array([x0, y0, z0])
        gateway_to_satellite_ecef = satellite_position_ecef - gateway_ecef
        distance = np.linalg.norm(gateway_to_satellite_ecef)
        edge_sun_angle = angle_between(gateway_to_satellite_ecef, sun_direction_ecef)
        link_available = bool(elevation >= gateway.minimum_elevation and edge_sun_angle >= sun_exclusion_angle)
        return CommLinkPerformance(
            link_capacity=self._calc_capacity(float(distance), gateway, gateway_turbulence_map, rng)
            if link_available
            else 0.0,
            delay=float(srange / SPEED_OF_LIGHT),
            link_available=link_available,
        )
