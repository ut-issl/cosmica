from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from cosmica.comm_link import CommLinkPerformance, OTC2OTCBinaryCommLinkCalculator
from cosmica.dtos import CommunicationLinkEndpoint, DirectedCommunicationLink, DynamicsData
from cosmica.models import ConstellationSatellite, DirectionCosineMatrix, OpticalCommunicationTerminal
from cosmica.utils.constants import EARTH_RADIUS
from tests.factories import make_satellite, make_terminal

EPOCH = np.datetime64("2026-01-01T00:00:00")
_IDENTITY_DCM: DirectionCosineMatrix = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_ROTATE_Z_NEGATIVE_90_DCM: DirectionCosineMatrix = (
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


@dataclass(frozen=True, slots=True)
class _Endpoint:
    satellite: ConstellationSatellite[int]
    terminal: OpticalCommunicationTerminal[int]


type _TestLink = DirectedCommunicationLink[
    ConstellationSatellite[int],
    OpticalCommunicationTerminal[int],
    ConstellationSatellite[int],
    OpticalCommunicationTerminal[int],
]


def _make_endpoint(
    satellite_id: int,
    *,
    angular_velocity_max: float = float("inf"),
    azimuth_min: float = -np.pi,
    azimuth_max: float = np.pi,
    elevation_min: float = -np.pi / 2,
    elevation_max: float = np.pi / 2,
    dcm_body2terminal: DirectionCosineMatrix = _IDENTITY_DCM,
) -> _Endpoint:
    terminal = make_terminal(
        satellite_id,
        azimuth_min=azimuth_min,
        azimuth_max=azimuth_max,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        angular_velocity_max=angular_velocity_max,
        dcm_body2terminal=dcm_body2terminal,
    )
    satellite = make_satellite(satellite_id, terminals=(terminal,))
    return _Endpoint(satellite=satellite, terminal=terminal)


def _make_link(source: _Endpoint, destination: _Endpoint) -> _TestLink:
    return DirectedCommunicationLink(
        source=CommunicationLinkEndpoint(node=source.satellite, terminal=source.terminal),
        destination=CommunicationLinkEndpoint(node=destination.satellite, terminal=destination.terminal),
    )


def _repeat_dcm(dcm: DirectionCosineMatrix, sample_count: int) -> npt.NDArray[np.floating]:
    return np.repeat(np.asarray(dcm)[None, :, :], sample_count, axis=0)


def _make_dynamics_data(
    endpoint_a: _Endpoint,
    endpoint_b: _Endpoint,
    *,
    time: npt.NDArray[np.datetime64],
    line_of_sight_azimuths: Sequence[float] | npt.NDArray[np.floating] | None = None,
    dcm_eci2body_a: DirectionCosineMatrix = _IDENTITY_DCM,
    dcm_eci2body_b: DirectionCosineMatrix = _IDENTITY_DCM,
    include_endpoint_b_attitude: bool = True,
    sun_direction_eci: npt.NDArray[np.floating] | None = None,
) -> DynamicsData[ConstellationSatellite[int]]:
    sample_count = len(time)
    azimuths = np.asarray(
        line_of_sight_azimuths if line_of_sight_azimuths is not None else [np.pi / 2] * sample_count,
    )
    position_a = np.repeat(np.array([[EARTH_RADIUS + 1_000e3, 0.0, 0.0]]), sample_count, axis=0)
    line_of_sight = 100e3 * np.column_stack((np.cos(azimuths), np.sin(azimuths), np.zeros(sample_count)))
    position_b = position_a + line_of_sight
    zero = np.zeros((sample_count, 3))
    identity = _repeat_dcm(_IDENTITY_DCM, sample_count)
    satellite_a = endpoint_a.satellite
    satellite_b = endpoint_b.satellite
    positions = {satellite_a: position_a, satellite_b: position_b}
    attitudes = {satellite_a: _repeat_dcm(dcm_eci2body_a, sample_count)}
    if include_endpoint_b_attitude:
        attitudes[satellite_b] = _repeat_dcm(dcm_eci2body_b, sample_count)
    sun_direction = (
        np.repeat(np.array([[0.0, 0.0, 1.0]]), sample_count, axis=0)
        if sun_direction_eci is None
        else np.repeat(sun_direction_eci[None, :], sample_count, axis=0)
    )
    return DynamicsData(
        time=time,
        dcm_eci2ecef=identity,
        satellite_position_eci=positions,
        satellite_velocity_eci={satellite_a: zero, satellite_b: zero},
        satellite_position_ecef=positions,
        satellite_attitude_angular_velocity_eci={satellite_a: zero, satellite_b: zero},
        sun_direction_eci=sun_direction,
        sun_direction_ecef=sun_direction,
        satellite_attitude_dcm_eci2body=attitudes,
    )


def _calculate(
    calculator: OTC2OTCBinaryCommLinkCalculator,
    link: _TestLink,
    dynamics_data: DynamicsData[ConstellationSatellite[int]],
    *,
    links_time_series: Sequence[Collection[_TestLink]] | None = None,
) -> list[dict[_TestLink, CommLinkPerformance]]:
    links = links_time_series if links_time_series is not None else [{link} for _ in dynamics_data.time]
    return calculator.calc(links_time_series=links, dynamics_data=dynamics_data, rng=np.random.default_rng(0))


def test_calc_processes_each_time_step_as_a_dynamics_snapshot() -> None:
    endpoint_a = _make_endpoint(1)
    endpoint_b = _make_endpoint(2)
    link = _make_link(endpoint_a, endpoint_b)
    time = EPOCH + np.arange(2).astype("timedelta64[s]")
    dynamics_data = _make_dynamics_data(endpoint_a, endpoint_b, time=time)

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert len(performance) == 2
    assert performance[0][link]["link_available"] is True
    assert performance[1][link]["link_available"] is True


def test_calc_rejects_link_and_dynamics_series_length_mismatch() -> None:
    endpoint_a = _make_endpoint(1)
    endpoint_b = _make_endpoint(2)
    link = _make_link(endpoint_a, endpoint_b)
    time = EPOCH + np.arange(2).astype("timedelta64[s]")
    dynamics_data = _make_dynamics_data(endpoint_a, endpoint_b, time=time)

    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is longer"):
        _calculate(
            OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
            link,
            dynamics_data,
            links_time_series=[{link}],
        )


def test_calc_requires_body_attitudes_for_both_link_endpoints() -> None:
    endpoint_a = _make_endpoint(1)
    endpoint_b = _make_endpoint(2)
    link = _make_link(endpoint_a, endpoint_b)
    dynamics_data = _make_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=np.array([EPOCH]),
        include_endpoint_b_attitude=False,
    )

    with pytest.raises(ValueError, match="attitude for every link endpoint"):
        _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)


@pytest.mark.parametrize(
    ("dcm_body2terminal", "dcm_eci2body"),
    [
        (_ROTATE_Z_NEGATIVE_90_DCM, _IDENTITY_DCM),
        (_IDENTITY_DCM, _ROTATE_Z_NEGATIVE_90_DCM),
    ],
    ids=["terminal-mounting", "body-attitude"],
)
def test_calc_transforms_pointing_through_body_and_terminal_frames(
    dcm_body2terminal: DirectionCosineMatrix,
    dcm_eci2body: DirectionCosineMatrix,
) -> None:
    endpoint_a = _make_endpoint(
        1,
        azimuth_min=np.deg2rad(-1.0),
        azimuth_max=np.deg2rad(1.0),
        dcm_body2terminal=dcm_body2terminal,
    )
    endpoint_b = _make_endpoint(2)
    link = _make_link(endpoint_a, endpoint_b)
    dynamics_data = _make_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=np.array([EPOCH]),
        dcm_eci2body_a=dcm_eci2body,
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[0][link]["link_available"] is True


@pytest.mark.parametrize("restricted_endpoint", ["source", "destination"])
def test_calc_enforces_field_of_regard_at_both_endpoints(restricted_endpoint: str) -> None:
    restricted_bounds = {"azimuth_min": np.deg2rad(-10.0), "azimuth_max": np.deg2rad(10.0)}
    endpoint_a = _make_endpoint(1, **(restricted_bounds if restricted_endpoint == "source" else {}))
    endpoint_b = _make_endpoint(2, **(restricted_bounds if restricted_endpoint == "destination" else {}))
    link = _make_link(endpoint_a, endpoint_b)
    dynamics_data = _make_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=np.array([EPOCH]),
        line_of_sight_azimuths=[np.deg2rad(30.0)],
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[0][link]["link_available"] is False


def test_calc_unwraps_azimuth_rate_across_pi_boundary() -> None:
    max_rate = np.deg2rad(3.0)
    endpoint_a = _make_endpoint(1, angular_velocity_max=max_rate)
    endpoint_b = _make_endpoint(2, angular_velocity_max=max_rate)
    link = _make_link(endpoint_a, endpoint_b)
    dynamics_data = _make_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=EPOCH + np.arange(2).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([179.0, -179.0]),
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[1][link]["link_available"] is True


def test_calc_rejects_large_negative_terminal_rate() -> None:
    max_rate = np.deg2rad(5.0)
    endpoint_a = _make_endpoint(1, angular_velocity_max=max_rate)
    endpoint_b = _make_endpoint(2, angular_velocity_max=max_rate)
    link = _make_link(endpoint_a, endpoint_b)
    dynamics_data = _make_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=EPOCH + np.arange(2).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([10.0, 0.0]),
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[1][link]["link_available"] is False


def test_calc_converts_time_delta_to_seconds_for_terminal_rate() -> None:
    max_rate = np.deg2rad(5.0)
    endpoint_a = _make_endpoint(1, angular_velocity_max=max_rate)
    endpoint_b = _make_endpoint(2, angular_velocity_max=max_rate)
    link = _make_link(endpoint_a, endpoint_b)
    dynamics_data = _make_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=EPOCH + np.array([0, 2]).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([0.0, 8.0]),
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[1][link]["link_available"] is True


def test_calc_skips_slew_check_when_link_reappears_after_absence() -> None:
    endpoint_a = _make_endpoint(1, angular_velocity_max=0.0)
    endpoint_b = _make_endpoint(2, angular_velocity_max=0.0)
    link = _make_link(endpoint_a, endpoint_b)
    dynamics_data = _make_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=EPOCH + np.arange(3).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([0.0, 90.0, 179.0]),
    )

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
        link,
        dynamics_data,
        links_time_series=[{link}, set(), {link}],
    )

    assert performance[0][link]["link_available"] is True
    assert performance[2][link]["link_available"] is True


def test_calc_applies_sun_exclusion_at_both_terminals() -> None:
    endpoint_a = _make_endpoint(1)
    endpoint_b = _make_endpoint(2)
    link = _make_link(endpoint_a, endpoint_b)
    dynamics_data = _make_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=np.array([EPOCH]),
        sun_direction_eci=np.array([0.0, 1.0, 0.0]),
    )

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9, sun_exclusion_angle=np.deg2rad(10.0)),
        link,
        dynamics_data,
    )

    assert performance[0][link]["link_available"] is False
