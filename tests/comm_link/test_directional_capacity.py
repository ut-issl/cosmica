"""Tests for per-direction (asymmetric) link capacities of comm link calculators.

Each calculator handles exactly one link direction; the coordinator dispatches each
directed edge to the calculator registered for the exact (source type, destination type).
"""

from collections.abc import Callable

import numpy as np
import pytest

from cosmica.comm_link import (
    CommLinkCalculationCoordinator,
    GatewayToSatBinaryCommLinkCalculator,
    MemorylessCommLinkCalculator,
    MemorylessCommLinkCalculatorWrapper,
    SatToGatewayBinaryCommLinkCalculator,
    SatToSatBinaryCommLinkCalculator,
    SatToSatBinaryCommLinkCalculatorWithRateCalc,
)
from cosmica.comm_link.geometric import GeometricCommLinkCalculator
from cosmica.dtos import DynamicsData
from cosmica.models import (
    CircularSatelliteOrbitModel,
    ConstellationSatellite,
    Gateway,
    Satellite,
)
from cosmica.utils.constants import EARTH_RADIUS

EPOCH = np.datetime64("2026-01-01T00:00:00")


def _make_satellite(sat_id: int, phase_deg: float = 0.0) -> ConstellationSatellite[int]:
    return ConstellationSatellite(
        id=sat_id,
        orbit=CircularSatelliteOrbitModel(
            semi_major_axis=EARTH_RADIUS + 1000e3,
            inclination=0.0,
            raan=0.0,
            phase_at_epoch=np.deg2rad(phase_deg),
            epoch=EPOCH,
        ),
    )


def _make_snapshot_dynamics_data(
    position_eci: dict[Satellite, np.ndarray],
    position_ecef: dict[Satellite, np.ndarray] | None = None,
    sun_direction: np.ndarray | None = None,
) -> DynamicsData:
    """Build a single-snapshot (no time dimension) DynamicsData for memoryless calculators."""
    zero = np.zeros(3)
    sun_direction = sun_direction if sun_direction is not None else np.array([0.0, 0.0, 1.0])
    return DynamicsData(
        time=np.array(EPOCH),
        dcm_eci2ecef=np.eye(3),
        satellite_position_eci=position_eci,
        satellite_velocity_eci=dict.fromkeys(position_eci, zero),
        satellite_position_ecef=position_ecef if position_ecef is not None else dict(position_eci),
        satellite_attitude_angular_velocity_eci=dict.fromkeys(position_eci, zero),
        sun_direction_eci=sun_direction,
        sun_direction_ecef=sun_direction,
    )


@pytest.fixture
def satellite() -> ConstellationSatellite[int]:
    return _make_satellite(1)


@pytest.fixture
def gateway() -> Gateway[int]:
    return Gateway(
        id=1,
        latitude=0.0,
        longitude=0.0,
        minimum_elevation=np.deg2rad(5),
    )


class TestSatGatewayDirectionalCalculators:
    def test_downlink_calculator_emits_only_sat_to_gateway_direction(
        self,
        satellite: ConstellationSatellite[int],
        gateway: Gateway[int],
    ) -> None:
        # Satellite directly above the gateway (gateway at lat=0, lon=0 -> ECEF x-axis)
        dynamics_data = _make_snapshot_dynamics_data(
            position_eci={satellite: np.array([EARTH_RADIUS + 1000e3, 0.0, 0.0])},
        )
        calculator = SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9)

        performance = calculator.calc(
            edges=[(satellite, gateway)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert set(performance) == {(satellite, gateway)}
        assert performance[(satellite, gateway)]["link_available"] is True
        assert performance[(satellite, gateway)]["link_capacity"] == 10e9

    def test_uplink_calculator_emits_only_gateway_to_sat_direction(
        self,
        satellite: ConstellationSatellite[int],
        gateway: Gateway[int],
    ) -> None:
        dynamics_data = _make_snapshot_dynamics_data(
            position_eci={satellite: np.array([EARTH_RADIUS + 1000e3, 0.0, 0.0])},
        )
        calculator = GatewayToSatBinaryCommLinkCalculator(link_capacity=2e9)

        performance = calculator.calc(
            edges=[(gateway, satellite)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert set(performance) == {(gateway, satellite)}
        assert performance[(gateway, satellite)]["link_available"] is True
        assert performance[(gateway, satellite)]["link_capacity"] == 2e9

    def test_uplink_and_downlink_share_the_same_geometry(
        self,
        satellite: ConstellationSatellite[int],
        gateway: Gateway[int],
    ) -> None:
        dynamics_data = _make_snapshot_dynamics_data(
            position_eci={satellite: np.array([EARTH_RADIUS + 1000e3, 0.0, 0.0])},
        )
        downlink = SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9).calc(
            edges=[(satellite, gateway)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )
        uplink = GatewayToSatBinaryCommLinkCalculator(link_capacity=2e9).calc(
            edges=[(gateway, satellite)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert downlink[(satellite, gateway)]["delay"] == uplink[(gateway, satellite)]["delay"]
        assert downlink[(satellite, gateway)]["link_available"] == uplink[(gateway, satellite)]["link_available"]

    def test_unavailable_link_has_zero_capacity(
        self,
        satellite: ConstellationSatellite[int],
        gateway: Gateway[int],
    ) -> None:
        # Satellite on the opposite side of the Earth -> below the horizon
        dynamics_data = _make_snapshot_dynamics_data(
            position_eci={satellite: np.array([-(EARTH_RADIUS + 1000e3), 0.0, 0.0])},
        )

        downlink = SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9).calc(
            edges=[(satellite, gateway)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )
        uplink = GatewayToSatBinaryCommLinkCalculator(link_capacity=2e9).calc(
            edges=[(gateway, satellite)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert downlink[(satellite, gateway)]["link_available"] is False
        assert downlink[(satellite, gateway)]["link_capacity"] == 0.0
        assert uplink[(gateway, satellite)]["link_available"] is False
        assert uplink[(gateway, satellite)]["link_capacity"] == 0.0


class TestSatToSatDirectionalDispatch:
    @pytest.fixture
    def sat_pair(self) -> tuple[ConstellationSatellite[int], ConstellationSatellite[int]]:
        return _make_satellite(1), _make_satellite(2, phase_deg=10.0)

    @pytest.fixture
    def dynamics_data(
        self,
        sat_pair: tuple[ConstellationSatellite[int], ConstellationSatellite[int]],
    ) -> DynamicsData:
        sat_a, sat_b = sat_pair
        return _make_snapshot_dynamics_data(
            position_eci={
                sat_a: np.array([EARTH_RADIUS + 1000e3, 0.0, 0.0]),
                sat_b: np.array([EARTH_RADIUS + 1000e3, 1000e3, 0.0]),
            },
        )

    def test_output_contains_exactly_the_input_directed_edges(
        self,
        sat_pair: tuple[ConstellationSatellite[int], ConstellationSatellite[int]],
        dynamics_data: DynamicsData,
    ) -> None:
        sat_a, sat_b = sat_pair
        calculator = SatToSatBinaryCommLinkCalculator(link_capacity=10e9)

        performance_single = calculator.calc(
            edges=[(sat_a, sat_b)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )
        performance_both = calculator.calc(
            edges=[(sat_a, sat_b), (sat_b, sat_a)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert set(performance_single) == {(sat_a, sat_b)}
        assert set(performance_both) == {(sat_a, sat_b), (sat_b, sat_a)}

    def test_both_directions_get_consistent_performance(
        self,
        sat_pair: tuple[ConstellationSatellite[int], ConstellationSatellite[int]],
        dynamics_data: DynamicsData,
    ) -> None:
        sat_a, sat_b = sat_pair
        calculator = SatToSatBinaryCommLinkCalculator(link_capacity=10e9)

        performance = calculator.calc(
            edges=[(sat_a, sat_b), (sat_b, sat_a)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert performance[(sat_a, sat_b)]["link_capacity"] == 10e9
        assert performance[(sat_b, sat_a)]["link_capacity"] == 10e9
        # Geometry (delay / availability) is direction-independent
        assert performance[(sat_a, sat_b)]["delay"] == performance[(sat_b, sat_a)]["delay"]
        assert performance[(sat_a, sat_b)]["link_available"] == performance[(sat_b, sat_a)]["link_available"]


class TestSatToSatDirectionalSunExclusion:
    """Receiver-side sun exclusion behavior shared by both sat-to-sat calculator variants."""

    @pytest.fixture
    def sat_pair(self) -> tuple[ConstellationSatellite[int], ConstellationSatellite[int]]:
        return _make_satellite(1), _make_satellite(2, phase_deg=10.0)

    @pytest.fixture(params=["binary", "rate_calc"])
    def make_calculator(
        self,
        request: pytest.FixtureRequest,
    ) -> Callable[[float], MemorylessCommLinkCalculator[Satellite, Satellite]]:
        """Build either sat-to-sat calculator variant with the given sun exclusion angle."""

        def factory(sun_exclusion_angle: float = 0.0) -> MemorylessCommLinkCalculator[Satellite, Satellite]:
            if request.param == "binary":
                return SatToSatBinaryCommLinkCalculator(
                    link_capacity=10e9,
                    sun_exclusion_angle=sun_exclusion_angle,
                )
            return SatToSatBinaryCommLinkCalculatorWithRateCalc(
                inter_satellite_link_capacity=10e9,
                lna_gain=30.0,
                sun_exclusion_angle=sun_exclusion_angle,
            )

        return factory

    @pytest.fixture
    def dynamics_data(
        self,
        sat_pair: tuple[ConstellationSatellite[int], ConstellationSatellite[int]],
    ) -> DynamicsData:
        sat_a, sat_b = sat_pair
        # Sun along -y: satellite B (receiver of the a -> b edge) looks towards A along -y,
        # i.e. straight into the sun, while A looking towards B (+y) faces away from it.
        # Neither satellite is in eclipse for this geometry.
        return _make_snapshot_dynamics_data(
            position_eci={
                sat_a: np.array([EARTH_RADIUS + 1000e3, 0.0, 0.0]),
                sat_b: np.array([EARTH_RADIUS + 1000e3, 1000e3, 0.0]),
            },
            sun_direction=np.array([0.0, -1.0, 0.0]),
        )

    def test_output_contains_exactly_the_input_directed_edges(
        self,
        sat_pair: tuple[ConstellationSatellite[int], ConstellationSatellite[int]],
        dynamics_data: DynamicsData,
        make_calculator: Callable[[float], MemorylessCommLinkCalculator[Satellite, Satellite]],
    ) -> None:
        sat_a, sat_b = sat_pair
        calculator = make_calculator(0.0)

        performance = calculator.calc(
            edges=[(sat_a, sat_b), (sat_b, sat_a)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert set(performance) == {(sat_a, sat_b), (sat_b, sat_a)}

    def test_sun_exclusion_is_checked_at_the_receiver_only(
        self,
        sat_pair: tuple[ConstellationSatellite[int], ConstellationSatellite[int]],
        dynamics_data: DynamicsData,
        make_calculator: Callable[[float], MemorylessCommLinkCalculator[Satellite, Satellite]],
    ) -> None:
        sat_a, sat_b = sat_pair
        calculator = make_calculator(np.deg2rad(10))

        performance = calculator.calc(
            edges=[(sat_a, sat_b), (sat_b, sat_a)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        # Receiver B looks into the sun -> a -> b is blocked; receiver A does not -> b -> a is up.
        assert performance[(sat_a, sat_b)]["link_available"] is False
        assert performance[(sat_a, sat_b)]["link_capacity"] == 0.0
        assert performance[(sat_b, sat_a)]["link_available"] is True
        assert performance[(sat_b, sat_a)]["link_capacity"] > 0.0
        # Geometry (delay) is direction-independent
        assert performance[(sat_a, sat_b)]["delay"] == performance[(sat_b, sat_a)]["delay"]

    def test_receiver_in_eclipse_is_exempt_from_sun_exclusion(
        self,
        sat_pair: tuple[ConstellationSatellite[int], ConstellationSatellite[int]],
        make_calculator: Callable[[float], MemorylessCommLinkCalculator[Satellite, Satellite]],
    ) -> None:
        sat_a, sat_b = sat_pair
        # Both satellites on the anti-sun side, inside Earth's shadow. Receiver B (of the
        # a -> b edge) looks towards A, i.e. towards the sun azimuth, but the Earth blocks
        # the sun, so the sun exclusion check is skipped and the link stays available.
        dynamics_data = _make_snapshot_dynamics_data(
            position_eci={
                sat_a: np.array([-(EARTH_RADIUS + 1500e3), 0.0, 0.0]),
                sat_b: np.array([-(EARTH_RADIUS + 2500e3), 0.0, 0.0]),
            },
            sun_direction=np.array([1.0, 0.0, 0.0]),
        )
        calculator = make_calculator(np.deg2rad(10))

        performance = calculator.calc(
            edges=[(sat_a, sat_b)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert performance[(sat_a, sat_b)]["link_available"] is True
        assert performance[(sat_a, sat_b)]["link_capacity"] > 0.0

    def test_both_directions_available_without_sun_exclusion(
        self,
        sat_pair: tuple[ConstellationSatellite[int], ConstellationSatellite[int]],
        dynamics_data: DynamicsData,
        make_calculator: Callable[[float], MemorylessCommLinkCalculator[Satellite, Satellite]],
    ) -> None:
        sat_a, sat_b = sat_pair
        calculator = make_calculator(0.0)

        performance = calculator.calc(
            edges=[(sat_a, sat_b), (sat_b, sat_a)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert performance[(sat_a, sat_b)]["link_available"] is True
        assert performance[(sat_b, sat_a)]["link_available"] is True
        assert performance[(sat_a, sat_b)]["link_capacity"] > 0.0
        assert performance[(sat_a, sat_b)]["link_capacity"] == performance[(sat_b, sat_a)]["link_capacity"]


class TestGeometricCalculatorDirectedEdges:
    def test_calc_emits_exactly_the_input_directed_edges(
        self,
        gateway: Gateway[int],
    ) -> None:
        sat_a = _make_satellite(1)
        sat_b = _make_satellite(2, phase_deg=10.0)
        dynamics_data = _make_snapshot_dynamics_data(
            position_eci={
                sat_a: np.array([EARTH_RADIUS + 1000e3, 0.0, 0.0]),
                sat_b: np.array([EARTH_RADIUS + 1000e3, 1000e3, 0.0]),
            },
        )
        with pytest.warns(DeprecationWarning, match="GeometricCommLinkCalculator is deprecated"):
            calculator = GeometricCommLinkCalculator(
                inter_satellite_link_capacity=10e9,
                satellite_to_gateway_link_capacity=5e9,
            )

        performance = calculator.calc(
            edges=[(sat_a, sat_b), (sat_b, sat_a), (sat_a, gateway), (gateway, sat_a)],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert set(performance) == {(sat_a, sat_b), (sat_b, sat_a), (sat_a, gateway), (gateway, sat_a)}
        # Both directions of a physical link share the same geometry
        assert performance[(sat_a, sat_b)]["delay"] == performance[(sat_b, sat_a)]["delay"]
        assert performance[(sat_a, gateway)]["delay"] == performance[(gateway, sat_a)]["delay"]
        assert performance[(sat_a, gateway)]["link_available"] is True
        assert performance[(sat_a, gateway)]["link_capacity"] == 5e9


class TestCoordinatorDirectionalDispatch:
    @pytest.fixture
    def dynamics_data(self, satellite: ConstellationSatellite[int]) -> DynamicsData:
        position = np.array([[EARTH_RADIUS + 1000e3, 0.0, 0.0]])  # shape (1, 3): one time step
        zero = np.zeros((1, 3))
        return DynamicsData(
            time=np.array([EPOCH]),
            dcm_eci2ecef=np.eye(3)[None, :, :],
            satellite_position_eci={satellite: position},
            satellite_velocity_eci={satellite: zero},
            satellite_position_ecef={satellite: position},
            satellite_attitude_angular_velocity_eci={satellite: zero},
            sun_direction_eci=np.array([[0.0, 0.0, 1.0]]),
            sun_direction_ecef=np.array([[0.0, 0.0, 1.0]]),
        )

    def test_each_direction_is_dispatched_to_its_own_calculator(
        self,
        satellite: ConstellationSatellite[int],
        gateway: Gateway[int],
        dynamics_data: DynamicsData,
    ) -> None:
        coordinator = CommLinkCalculationCoordinator(
            calculator_assignment={
                (ConstellationSatellite, Gateway): MemorylessCommLinkCalculatorWrapper(
                    SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9),
                ),
                (Gateway, ConstellationSatellite): MemorylessCommLinkCalculatorWrapper(
                    GatewayToSatBinaryCommLinkCalculator(link_capacity=2e9),
                ),
            },
        )

        # Directed topology graphs yield both directed edges of the physical link
        performance_time_series = coordinator.calc(
            edges_time_series=[{(satellite, gateway), (gateway, satellite)}],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert len(performance_time_series) == 1
        performance = performance_time_series[0]
        assert performance[(satellite, gateway)]["link_capacity"] == 10e9
        assert performance[(gateway, satellite)]["link_capacity"] == 2e9

    def test_edge_is_dispatched_only_to_the_exact_type_calculator(
        self,
        satellite: ConstellationSatellite[int],
        gateway: Gateway[int],
        dynamics_data: DynamicsData,
    ) -> None:
        coordinator = CommLinkCalculationCoordinator(
            calculator_assignment={
                (ConstellationSatellite, Gateway): MemorylessCommLinkCalculatorWrapper(
                    SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9),
                ),
                # Also registering the base class must not double-dispatch ConstellationSatellite
                # edges (which would silently overwrite the result of the exact-type calculator).
                (Satellite, Gateway): MemorylessCommLinkCalculatorWrapper(
                    SatToGatewayBinaryCommLinkCalculator(link_capacity=5e9),
                ),
            },
        )

        performance_time_series = coordinator.calc(
            edges_time_series=[{(satellite, gateway)}],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )

        assert performance_time_series[0][(satellite, gateway)]["link_capacity"] == 10e9

    def test_missing_direction_registration_raises(
        self,
        satellite: ConstellationSatellite[int],
        gateway: Gateway[int],
        dynamics_data: DynamicsData,
    ) -> None:
        coordinator = CommLinkCalculationCoordinator(
            calculator_assignment={
                # Only the downlink direction is registered
                (ConstellationSatellite, Gateway): MemorylessCommLinkCalculatorWrapper(
                    SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9),
                ),
            },
        )

        with pytest.raises(ValueError, match="No calculator registered"):
            coordinator.calc(
                edges_time_series=[{(satellite, gateway), (gateway, satellite)}],
                dynamics_data=dynamics_data,
                rng=np.random.default_rng(0),
            )
