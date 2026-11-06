from dataclasses import dataclass, field

import numpy as np
import pytest
from pymap3d import aer2ecef

from cosmica.comm_link import (
    AtmosphericScintillationModel,
    SatToGatewayBinaryCommLinkCalculator,
    SatToGatewayBinaryCommLinkCalculatorWithScintillation,
)
from cosmica.dtos import DynamicsData
from cosmica.models import GatewayOGS
from cosmica.utils.constants import EARTH_RADIUS
from tests.factories import make_satellite


def _make_calculator(*, sun_exclusion_angle: float) -> SatToGatewayBinaryCommLinkCalculatorWithScintillation:
    return SatToGatewayBinaryCommLinkCalculatorWithScintillation(
        satellite_to_gateway_link_capacity=10e9,
        link_capacity=1e9,
        noise_figure=3.0,
        lna_gain=30.0,
        lct_p0=1.0,
        sun_exclusion_angle=sun_exclusion_angle,
    )


@pytest.mark.parametrize(
    ("sun_direction", "exclusion_angle", "expected_available"),
    [
        ([1.0, 0.0, 0.0], 0.0, True),
        ([1.0, 0.0, 0.0], 0.1, False),
        ([0.0, 1.0, 0.0], np.pi / 2, True),
        ([0.0, 1.0, 0.0], np.pi / 2 + 0.1, False),
        ([-1.0, 0.0, 0.0], np.pi, True),
    ],
)
def test_scintillation_downlink_sun_exclusion_matches_plain_downlink(
    sun_direction: list[float],
    exclusion_angle: float,
    *,
    expected_available: bool,
):
    """Match plain-downlink Sun-angle boundaries while retaining propagation delay."""
    satellite = make_satellite(1)
    gateway = GatewayOGS(id=1, latitude=0.0, longitude=0.0, minimum_elevation=0.0, ground_cn2=0.0, rms_wind_speed=0.0)
    edge = (satellite, gateway)
    position = {satellite: np.array([EARTH_RADIUS + 1e6, 0.0, 0.0])}
    sun = np.array(sun_direction)
    dynamics = DynamicsData(
        time=np.array(np.datetime64("2026-01-01")),
        dcm_eci2ecef=np.eye(3),
        satellite_position_eci=position,
        satellite_velocity_eci={satellite: np.zeros(3)},
        satellite_position_ecef=position,
        satellite_attitude_angular_velocity_eci={satellite: np.zeros(3)},
        sun_direction_eci=sun,
        sun_direction_ecef=sun,
    )
    result = _make_calculator(sun_exclusion_angle=exclusion_angle).calc(
        [edge],
        dynamics_data=dynamics,
        rng=np.random.default_rng(0),
    )[edge]
    plain = SatToGatewayBinaryCommLinkCalculator(link_capacity=1e9, sun_exclusion_angle=exclusion_angle).calc(
        [edge],
        dynamics_data=dynamics,
        rng=np.random.default_rng(0),
    )[edge]

    assert result["link_available"] is expected_available
    assert result["link_available"] == plain["link_available"]
    assert result["delay"] == plain["delay"]
    if expected_available:
        assert result["link_capacity"] > 0.0
    else:
        assert result["link_capacity"] == 0.0


@pytest.mark.parametrize("angle", [-0.1, np.pi + 0.1, np.nan, np.inf, -np.inf])
def test_scintillation_downlink_rejects_invalid_sun_exclusion(angle: float):
    with pytest.raises(ValueError, match="sun_exclusion_angle must be finite and in \\[0, pi\\] radians"):
        _make_calculator(sun_exclusion_angle=angle)


def test_scintillation_downlink_defaults_to_no_sun_exclusion():
    calculator = SatToGatewayBinaryCommLinkCalculatorWithScintillation(
        satellite_to_gateway_link_capacity=10e9,
        link_capacity=1e9,
        noise_figure=3.0,
        lna_gain=30.0,
        lct_p0=1.0,
    )
    assert calculator.sun_exclusion_angle == 0.0


@dataclass
class _RecordingScintillationModel(AtmosphericScintillationModel):
    zenith_angles: list[float] = field(default_factory=list)

    def sample(self, rng: np.random.Generator, zenith_angle: float) -> float:  # noqa: ARG002
        self.zenith_angles.append(zenith_angle)
        return 1.0


@pytest.mark.parametrize("elevation_deg", [30.0, 90.0])
def test_scintillation_downlink_passes_link_zenith_angle_to_model(elevation_deg: float):
    model = _RecordingScintillationModel()
    calculator = SatToGatewayBinaryCommLinkCalculatorWithScintillation(
        satellite_to_gateway_link_capacity=10e9,
        link_capacity=1e9,
        noise_figure=3.0,
        lna_gain=30.0,
        lct_p0=1.0,
        turbulence_model_factory=lambda _: model,
    )
    satellite = make_satellite(1)
    gateway = GatewayOGS(id=1, latitude=0.0, longitude=0.0, minimum_elevation=0.0)
    position = {satellite: np.array(aer2ecef(0.0, elevation_deg, 1e6, 0.0, 0.0, 0.0, deg=True))}
    sun = np.array([0.0, 0.0, 1.0])
    dynamics = DynamicsData(
        time=np.array(np.datetime64("2026-01-01")),
        dcm_eci2ecef=np.eye(3),
        satellite_position_eci=position,
        satellite_velocity_eci={satellite: np.zeros(3)},
        satellite_position_ecef=position,
        satellite_attitude_angular_velocity_eci={satellite: np.zeros(3)},
        sun_direction_eci=sun,
        sun_direction_ecef=sun,
    )

    calculator.calc([(satellite, gateway)], dynamics_data=dynamics, rng=np.random.default_rng(0))

    assert model.zenith_angles == [pytest.approx(np.deg2rad(90.0 - elevation_deg), abs=1e-9)]
