"""Tests for gateway-site cloud outages and independent per-edge failures of gateway links."""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from cosmica.comm_link import (
    BinaryCloudModel,
    CloudStates,
    CommLinkPerformance,
    EdgeFailureModel,
    GatewayCloudSeries,
    GatewayToSatBinaryCommLinkCalculator,
    GatewayToSatStochasticBinaryCommLinkCalculator,
    SatToGatewayBinaryCommLinkCalculator,
    SatToGatewayEdgeFailureCommLinkCalculator,
    SatToGatewayStochasticBinaryCommLinkCalculator,
)
from cosmica.dtos import DynamicsData
from cosmica.models import ConstellationSatellite, Gateway, Satellite
from tests.factories import TEST_EPOCH, make_satellite

N_TIME = 4
TIME = TEST_EPOCH + np.arange(N_TIME).astype("timedelta64[s]")


def _make_gateway(gateway_id: int) -> Gateway[int]:
    return Gateway(id=gateway_id, latitude=0.0, longitude=0.0, minimum_elevation=0.0)


def _make_overhead_dynamics_data(satellites: list[ConstellationSatellite[int]]) -> DynamicsData:
    """Put every satellite directly above (0, 0) so that all gateway links are geometrically available."""
    overhead: dict[Satellite, npt.NDArray[np.floating]] = {
        satellite: np.tile([7_000_000.0, 0.0, 0.0], (N_TIME, 1)) for satellite in satellites
    }
    zero: dict[Satellite, npt.NDArray[np.floating]] = {satellite: np.zeros((N_TIME, 3)) for satellite in satellites}
    sun_direction = np.tile([0.0, 0.0, 1.0], (N_TIME, 1))
    return DynamicsData(
        time=TIME,
        dcm_eci2ecef=np.repeat(np.eye(3)[None, :, :], N_TIME, axis=0),
        satellite_position_eci=overhead,
        satellite_velocity_eci=zero,
        satellite_position_ecef=overhead,
        satellite_attitude_angular_velocity_eci=zero,
        sun_direction_eci=sun_direction,
        sun_direction_ecef=sun_direction,
    )


def _available[T](performance: Sequence[Mapping[T, CommLinkPerformance]], edge: T) -> list[bool]:
    return [snapshot[edge]["link_available"] for snapshot in performance]


@pytest.fixture
def satellites() -> list[ConstellationSatellite[int]]:
    return [make_satellite(1), make_satellite(2)]


@pytest.fixture
def gateways() -> list[Gateway[int]]:
    return [_make_gateway(1), _make_gateway(2)]


@pytest.fixture
def gateway_cloud(gateways: list[Gateway[int]]) -> GatewayCloudSeries:
    return GatewayCloudSeries(
        time=TIME,
        cloudy={
            gateways[0]: np.array([False, True, True, False]),
            gateways[1]: np.array([True, False, False, False]),
        },
    )


def test_downlinks_to_one_gateway_share_cloud_state(
    satellites: list[ConstellationSatellite[int]],
    gateways: list[Gateway[int]],
    gateway_cloud: GatewayCloudSeries,
) -> None:
    calculator = SatToGatewayStochasticBinaryCommLinkCalculator(
        memoryless_calculator=SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9),
        gateway_cloud=gateway_cloud,
    )
    edges = [(satellite, gateway) for satellite in satellites for gateway in gateways]

    performance = calculator.calc(
        [edges] * N_TIME,
        dynamics_data=_make_overhead_dynamics_data(satellites),
        rng=np.random.default_rng(0),
    )

    for satellite in satellites:
        assert _available(performance, (satellite, gateways[0])) == [True, False, False, True]
        assert _available(performance, (satellite, gateways[1])) == [False, True, True, True]


def test_uplinks_share_cloud_state_with_downlinks(
    satellites: list[ConstellationSatellite[int]],
    gateways: list[Gateway[int]],
    gateway_cloud: GatewayCloudSeries,
) -> None:
    dynamics_data = _make_overhead_dynamics_data(satellites)
    downlink = SatToGatewayStochasticBinaryCommLinkCalculator(
        memoryless_calculator=SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9),
        gateway_cloud=gateway_cloud,
    )
    uplink = GatewayToSatStochasticBinaryCommLinkCalculator(
        memoryless_calculator=GatewayToSatBinaryCommLinkCalculator(link_capacity=1e9),
        gateway_cloud=gateway_cloud,
    )
    down_edges = [(satellite, gateway) for satellite in satellites for gateway in gateways]
    up_edges = [(gateway, satellite) for satellite, gateway in down_edges]

    rng = np.random.default_rng(0)
    down = downlink.calc([down_edges] * N_TIME, dynamics_data=dynamics_data, rng=rng)
    up = uplink.calc([up_edges] * N_TIME, dynamics_data=dynamics_data, rng=rng)

    for satellite, gateway in down_edges:
        assert _available(up, (gateway, satellite)) == _available(down, (satellite, gateway))
        assert _available(up, (gateway, satellite)) == list(~gateway_cloud.cloudy[gateway])


def test_cloud_calculator_rejects_mismatched_time(
    satellites: list[ConstellationSatellite[int]],
    gateways: list[Gateway[int]],
) -> None:
    gateway_cloud = GatewayCloudSeries(time=TIME + np.timedelta64(1, "s"), cloudy={gateways[0]: np.zeros(N_TIME, bool)})
    calculator = SatToGatewayStochasticBinaryCommLinkCalculator(
        memoryless_calculator=SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9),
        gateway_cloud=gateway_cloud,
    )

    with pytest.raises(ValueError, match="Time array"):
        calculator.calc(
            [[(satellites[0], gateways[0])]] * N_TIME,
            dynamics_data=_make_overhead_dynamics_data(satellites),
            rng=np.random.default_rng(0),
        )


def test_cloud_calculator_rejects_gateway_without_series(
    satellites: list[ConstellationSatellite[int]],
    gateways: list[Gateway[int]],
) -> None:
    calculator = GatewayToSatStochasticBinaryCommLinkCalculator(
        memoryless_calculator=GatewayToSatBinaryCommLinkCalculator(link_capacity=1e9),
        gateway_cloud=GatewayCloudSeries(time=TIME, cloudy={gateways[0]: np.zeros(N_TIME, bool)}),
    )

    with pytest.raises(ValueError, match="No cloud series"):
        calculator.calc(
            [[(gateways[1], satellites[0])]] * N_TIME,
            dynamics_data=_make_overhead_dynamics_data(satellites),
            rng=np.random.default_rng(0),
        )


def test_gateway_cloud_series_rejects_misaligned_series(gateways: list[Gateway[int]]) -> None:
    with pytest.raises(ValueError, match="shape"):
        GatewayCloudSeries(time=TIME, cloudy={gateways[0]: np.zeros(N_TIME + 1, bool)})


class _FixedCloud(CloudStates[np.bool_]):
    def __init__(self, series: npt.NDArray[np.bool_]) -> None:
        self.series = series

    def simulate(self, time: npt.NDArray[np.datetime64], rng: np.random.Generator) -> npt.NDArray[np.bool_]:  # noqa: ARG002
        return self.series


def test_simulate_creates_one_series_per_gateway(gateways: list[Gateway[int]]) -> None:
    requested: list[Gateway] = []

    def factory(gateway: Gateway) -> CloudStates[np.bool_]:
        requested.append(gateway)
        return _FixedCloud(np.full(N_TIME, gateway.id == 1))

    gateway_cloud = GatewayCloudSeries.simulate(
        [gateways[1], gateways[0], gateways[1]],
        time=TIME,
        cloud_model_factory=factory,
        rng=np.random.default_rng(0),
    )

    assert requested == gateways
    assert gateway_cloud.cloudy.keys() == set(gateways)
    assert gateway_cloud.cloudy[gateways[0]].all()
    assert not gateway_cloud.cloudy[gateways[1]].any()


def test_simulate_is_independent_of_gateway_order(gateways: list[Gateway[int]]) -> None:
    time = TEST_EPOCH + np.arange(50).astype("timedelta64[s]")

    def simulate(order: list[Gateway[int]]) -> GatewayCloudSeries:
        return GatewayCloudSeries.simulate(
            order,
            time=time,
            cloud_model_factory=lambda _: BinaryCloudModel(),
            rng=np.random.default_rng(0),
        )

    forward, backward = simulate(gateways), simulate(gateways[::-1])

    for gateway in gateways:
        np.testing.assert_array_equal(forward.cloudy[gateway], backward.cloudy[gateway])


class _FixedFailure(EdgeFailureModel[np.bool_]):
    def __init__(self, series: npt.NDArray[np.bool_]) -> None:
        self.series = series

    def simulate(self, time: npt.NDArray[np.datetime64], rng: np.random.Generator) -> npt.NDArray[np.bool_]:  # noqa: ARG002
        return self.series


def test_edge_failures_are_independent_per_edge(
    satellites: list[ConstellationSatellite[int]],
    gateways: list[Gateway[int]],
) -> None:
    gateway = gateways[0]
    failures: dict[tuple[Satellite, Gateway], npt.NDArray[np.bool_]] = {
        (satellites[0], gateway): np.array([True, False, False, False]),
        (satellites[1], gateway): np.array([False, False, True, True]),
    }
    requested: list[tuple[Satellite, Gateway]] = []

    def factory(satellite: Satellite, gateway: Gateway) -> EdgeFailureModel[np.bool_]:
        requested.append((satellite, gateway))
        return _FixedFailure(failures[satellite, gateway])

    calculator = SatToGatewayEdgeFailureCommLinkCalculator(
        memoryless_calculator=SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9),
        edge_failure_model_factory=factory,
    )

    performance = calculator.calc(
        [list(failures)] * N_TIME,
        dynamics_data=_make_overhead_dynamics_data(satellites),
        rng=np.random.default_rng(0),
    )

    assert sorted(requested, key=lambda edge: edge[0].id) == list(failures)
    for edge, failed in failures.items():
        assert _available(performance, edge) == list(~failed)
