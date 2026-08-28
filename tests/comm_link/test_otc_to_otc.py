from collections.abc import Collection, Sequence

import numpy as np
import pytest

from cosmica.comm_link import CommLinkPerformance, OTC2OTCBinaryCommLinkCalculator
from cosmica.dtos import DynamicsData
from cosmica.models import ConstellationSatellite, DirectionCosineMatrix
from tests.factories import (
    TEST_EPOCH,
    OpticalSatelliteLink,
    make_link_from_endpoints,
    make_optical_link_dynamics_data,
    make_optical_satellite_endpoint,
)

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


def _calculate(
    calculator: OTC2OTCBinaryCommLinkCalculator,
    link: OpticalSatelliteLink,
    dynamics_data: DynamicsData[ConstellationSatellite[int]],
    *,
    links_time_series: Sequence[Collection[OpticalSatelliteLink]] | None = None,
) -> list[dict[OpticalSatelliteLink, CommLinkPerformance]]:
    links = links_time_series if links_time_series is not None else [{link} for _ in dynamics_data.time]
    return calculator.calc(links_time_series=links, dynamics_data=dynamics_data, rng=np.random.default_rng(0))


def test_calc_processes_each_time_step_as_a_dynamics_snapshot() -> None:
    endpoint_a = make_optical_satellite_endpoint(1)
    endpoint_b = make_optical_satellite_endpoint(2)
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    time = TEST_EPOCH + np.arange(2).astype("timedelta64[s]")
    dynamics_data = make_optical_link_dynamics_data(endpoint_a, endpoint_b, time=time)

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert len(performance) == 2
    assert performance[0][link]["link_available"] is True
    assert performance[1][link]["link_available"] is True


def test_calc_rejects_link_and_dynamics_series_length_mismatch() -> None:
    endpoint_a = make_optical_satellite_endpoint(1)
    endpoint_b = make_optical_satellite_endpoint(2)
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    time = TEST_EPOCH + np.arange(2).astype("timedelta64[s]")
    dynamics_data = make_optical_link_dynamics_data(endpoint_a, endpoint_b, time=time)

    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is longer"):
        _calculate(
            OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
            link,
            dynamics_data,
            links_time_series=[{link}],
        )


def test_calc_requires_body_attitudes_for_both_link_endpoints() -> None:
    endpoint_a = make_optical_satellite_endpoint(1)
    endpoint_b = make_optical_satellite_endpoint(2)
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    dynamics_data = make_optical_link_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=np.array([TEST_EPOCH]),
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
    endpoint_a = make_optical_satellite_endpoint(
        1,
        azimuth_min=np.deg2rad(-1.0),
        azimuth_max=np.deg2rad(1.0),
        dcm_body2terminal=dcm_body2terminal,
    )
    endpoint_b = make_optical_satellite_endpoint(2)
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    dynamics_data = make_optical_link_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=np.array([TEST_EPOCH]),
        dcm_eci2body_a=dcm_eci2body,
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[0][link]["link_available"] is True


@pytest.mark.parametrize("restricted_endpoint", ["source", "destination"])
def test_calc_enforces_field_of_regard_at_both_endpoints(restricted_endpoint: str) -> None:
    restricted_bounds = {"azimuth_min": np.deg2rad(-10.0), "azimuth_max": np.deg2rad(10.0)}
    endpoint_a = make_optical_satellite_endpoint(1, **(restricted_bounds if restricted_endpoint == "source" else {}))
    endpoint_b = make_optical_satellite_endpoint(
        2,
        **(restricted_bounds if restricted_endpoint == "destination" else {}),
    )
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    dynamics_data = make_optical_link_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=np.array([TEST_EPOCH]),
        line_of_sight_azimuths=[np.deg2rad(30.0)],
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[0][link]["link_available"] is False


def test_calc_unwraps_azimuth_rate_across_pi_boundary() -> None:
    max_rate = np.deg2rad(3.0)
    endpoint_a = make_optical_satellite_endpoint(1, angular_velocity_max=max_rate)
    endpoint_b = make_optical_satellite_endpoint(2, angular_velocity_max=max_rate)
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    dynamics_data = make_optical_link_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=TEST_EPOCH + np.arange(2).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([179.0, -179.0]),
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[1][link]["link_available"] is True


def test_calc_rejects_large_negative_terminal_rate() -> None:
    max_rate = np.deg2rad(5.0)
    endpoint_a = make_optical_satellite_endpoint(1, angular_velocity_max=max_rate)
    endpoint_b = make_optical_satellite_endpoint(2, angular_velocity_max=max_rate)
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    dynamics_data = make_optical_link_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=TEST_EPOCH + np.arange(2).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([10.0, 0.0]),
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[1][link]["link_available"] is False


def test_calc_converts_time_delta_to_seconds_for_terminal_rate() -> None:
    max_rate = np.deg2rad(5.0)
    endpoint_a = make_optical_satellite_endpoint(1, angular_velocity_max=max_rate)
    endpoint_b = make_optical_satellite_endpoint(2, angular_velocity_max=max_rate)
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    dynamics_data = make_optical_link_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=TEST_EPOCH + np.array([0, 2]).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([0.0, 8.0]),
    )

    performance = _calculate(OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9), link, dynamics_data)

    assert performance[1][link]["link_available"] is True


def test_calc_skips_slew_check_when_link_reappears_after_absence() -> None:
    endpoint_a = make_optical_satellite_endpoint(1, angular_velocity_max=0.0)
    endpoint_b = make_optical_satellite_endpoint(2, angular_velocity_max=0.0)
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    dynamics_data = make_optical_link_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=TEST_EPOCH + np.arange(3).astype("timedelta64[s]"),
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
    endpoint_a = make_optical_satellite_endpoint(1)
    endpoint_b = make_optical_satellite_endpoint(2)
    link = make_link_from_endpoints(endpoint_a, endpoint_b)
    dynamics_data = make_optical_link_dynamics_data(
        endpoint_a,
        endpoint_b,
        time=np.array([TEST_EPOCH]),
        sun_direction_eci=np.array([0.0, 1.0, 0.0]),
    )

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9, sun_exclusion_angle=np.deg2rad(10.0)),
        link,
        dynamics_data,
    )

    assert performance[0][link]["link_available"] is False
