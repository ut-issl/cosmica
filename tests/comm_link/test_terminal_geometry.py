import numpy as np

from cosmica.comm_link.terminal_geometry import (
    TerminalPointing,
    calc_terminal_angular_rate,
    calc_terminal_pointing,
    is_pointing_rate_within_limit,
    is_pointing_within_field_of_regard,
    shortest_angular_difference,
)
from tests.factories import make_terminal

_ROTATE_Z_NEGATIVE_90_DCM = (
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


def test_calc_terminal_pointing_applies_body_attitude_and_terminal_mounting() -> None:
    terminal = make_terminal(1, dcm_body2terminal=_ROTATE_Z_NEGATIVE_90_DCM)
    dcm_eci2body = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
    )

    pointing = calc_terminal_pointing(
        np.array([0.0, 1.0, 0.0]),
        dcm_eci2body=dcm_eci2body,
        terminal=terminal,
    )

    assert np.isclose(pointing.azimuth, -np.pi / 2)
    assert np.isclose(pointing.elevation, 0.0)


def test_shortest_angular_difference_unwraps_pi_boundary() -> None:
    difference = shortest_angular_difference(np.deg2rad(-179.0), np.deg2rad(179.0))

    assert np.isclose(difference, np.deg2rad(2.0))


def test_calc_terminal_angular_rate_uses_absolute_axis_rates() -> None:
    rate = calc_terminal_angular_rate(
        TerminalPointing(azimuth=np.deg2rad(-10.0), elevation=np.deg2rad(-20.0)),
        TerminalPointing(azimuth=0.0, elevation=0.0),
        time_delta=2.0,
    )

    assert np.isclose(rate.azimuth, np.deg2rad(5.0))
    assert np.isclose(rate.elevation, np.deg2rad(10.0))


def test_terminal_constraints_use_inclusive_bounds() -> None:
    terminal = make_terminal(
        1,
        angular_velocity_max=np.deg2rad(5.0),
        azimuth_min=np.deg2rad(-10.0),
        azimuth_max=np.deg2rad(10.0),
        dcm_body2terminal=_ROTATE_Z_NEGATIVE_90_DCM,
    )
    pointing = TerminalPointing(azimuth=np.deg2rad(10.0), elevation=0.0)
    rate = calc_terminal_angular_rate(
        pointing,
        TerminalPointing(azimuth=np.deg2rad(5.0), elevation=0.0),
        time_delta=1.0,
    )

    assert is_pointing_within_field_of_regard(pointing, terminal)
    assert is_pointing_rate_within_limit(rate, terminal)
