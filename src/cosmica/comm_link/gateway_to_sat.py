__all__ = [
    "GatewayToSatBinaryCommLinkCalculator",
    "GatewayToSatBinaryMemoryCommLinkCalculator",
]

from collections.abc import Collection, Sequence
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Doc

from cosmica.dtos import DynamicsData
from cosmica.models import Gateway, Satellite
from cosmica.utils.constants import SPEED_OF_LIGHT
from cosmica.utils.coordinates import ecef2aer, geodetic2ecef
from cosmica.utils.vector import angle_between

from .base import CommLinkCalculator, CommLinkPerformance, MemorylessCommLinkCalculator


class GatewayToSatBinaryCommLinkCalculator(
    MemorylessCommLinkCalculator[Gateway[Any], Satellite[Any]],
):
    """Calculate gateway-to-satellite (uplink) communication link performance.

    The link performance is calculated as a binary value, i.e., 1 if the link is available and 0 otherwise.

    Handles the gateway -> satellite direction only; use `SatToGatewayBinaryCommLinkCalculator`
    for the satellite -> gateway (downlink) direction.
    """

    def __init__(
        self,
        *,
        link_capacity: Annotated[float, Doc("Capacity of the gateway -> satellite (uplink) in bps.")],
        sun_exclusion_angle: float = 0.0,
    ) -> None:
        self.link_capacity = link_capacity
        self.sun_exclusion_angle = sun_exclusion_angle

    def calc(
        self,
        edges: Collection[tuple[Gateway[Any], Satellite[Any]]],
        *,
        dynamics_data: DynamicsData[Satellite[Any]],
        rng: np.random.Generator,  # noqa: ARG002 For interface compatibility
    ) -> dict[tuple[Gateway[Any], Satellite[Any]], CommLinkPerformance]:
        return {
            edge: self._calc_gateway_to_satellite(
                gateway=edge[0],
                satellite_position_ecef=dynamics_data.satellite_position_ecef[edge[1]],
                sun_direction_ecef=dynamics_data.sun_direction_ecef,
            )
            for edge in edges
        }

    def _calc_gateway_to_satellite(
        self,
        gateway: Gateway[Any],
        satellite_position_ecef: npt.NDArray[np.floating],
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


class GatewayToSatBinaryMemoryCommLinkCalculator(
    CommLinkCalculator[Gateway[Any], Satellite[Any]],
):
    """Calculate gateway-to-satellite (uplink) communication link performance with link acquisition delay.

    The link performance is calculated as a binary value, i.e., 1 if the link is available and 0 otherwise.

    Link acquisition delay is tracked independently per directed edge (gateway, satellite): when an edge
    appears or its underlying memoryless availability drops, the link goes through (re)acquisition and is
    unavailable for `link_acquisition_time` seconds.

    Handles the gateway -> satellite direction only; use `SatToGatewayBinaryMemoryCommLinkCalculator`
    for the satellite -> gateway (downlink) direction.
    """

    def __init__(
        self,
        *,
        memoryless_calculator: MemorylessCommLinkCalculator[Gateway[Any], Satellite[Any]],
        link_acquisition_time: float = 60.0,
        skip_link_acquisition_at_simulation_start: bool = True,
    ) -> None:
        self.memoryless_calculator = memoryless_calculator
        self.link_acquisition_time = link_acquisition_time
        self.skip_link_acquisition_at_simulation_start = skip_link_acquisition_at_simulation_start

    def calc(
        self,
        edges_time_series: Sequence[Collection[tuple[Gateway[Any], Satellite[Any]]]],
        *,
        dynamics_data: DynamicsData[Satellite[Any]],
        rng: np.random.Generator,
    ) -> list[dict[tuple[Gateway[Any], Satellite[Any]], CommLinkPerformance]]:
        assert len(edges_time_series) == len(dynamics_data.time)

        comm_link_time_series: list[dict[tuple[Gateway[Any], Satellite[Any]], CommLinkPerformance]] = []

        # ― per-directed-edge state ―
        # Link acquisition is tracked independently for each directed edge (gateway, satellite).
        link_acquisition_start_time: dict[tuple[Gateway[Any], Satellite[Any]], np.datetime64] = {}
        prev_edges: frozenset[tuple[Gateway[Any], Satellite[Any]]] = frozenset()

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
