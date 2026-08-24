from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from cosmica.comm_link import CommLinkPerformance, OTC2OTCBinaryCommLinkCalculator
from cosmica.dtos import DynamicsData
from cosmica.models import DirectionCosineMatrix, SatelliteTerminal
from cosmica.utils.constants import EARTH_RADIUS

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


def _make_terminal(
    satellite_id: int,
    *,
    angular_velocity_max: float = float("inf"),
    azimuth_min: float = -np.pi,
    azimuth_max: float = np.pi,
    elevation_min: float = -np.pi / 2,
    elevation_max: float = np.pi / 2,
    dcm_body2terminal: DirectionCosineMatrix = _IDENTITY_DCM,
) -> SatelliteTerminal[int]:
    return SatelliteTerminal(
        id=satellite_id,
        terminal_id=satellite_id,
        azimuth_min=azimuth_min,
        azimuth_max=azimuth_max,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        angular_velocity_max=angular_velocity_max,
        dcm_body2terminal=dcm_body2terminal,
    )


def _repeat_dcm(dcm: DirectionCosineMatrix, n_time: int) -> npt.NDArray[np.floating]:
    return np.repeat(np.asarray(dcm)[None, :, :], n_time, axis=0)


def _make_dynamics_data(
    terminal_a: SatelliteTerminal[int],
    terminal_b: SatelliteTerminal[int],
    *,
    time: npt.NDArray[np.datetime64],
    line_of_sight_azimuths: Sequence[float] | npt.NDArray[np.floating] | None = None,
    dcm_eci2body_a: DirectionCosineMatrix = _IDENTITY_DCM,
    dcm_eci2body_b: DirectionCosineMatrix = _IDENTITY_DCM,
    include_terminal_b_attitude: bool = True,
    sun_direction_eci: npt.NDArray[np.floating] | None = None,
) -> DynamicsData[SatelliteTerminal[int]]:
    n_time = len(time)
    azimuths = np.asarray(line_of_sight_azimuths if line_of_sight_azimuths is not None else [np.pi / 2] * n_time)
    position_a = np.repeat(np.array([[EARTH_RADIUS + 1_000e3, 0.0, 0.0]]), n_time, axis=0)
    line_of_sight = 100e3 * np.column_stack((np.cos(azimuths), np.sin(azimuths), np.zeros(n_time)))
    position_b = position_a + line_of_sight
    zero = np.zeros((n_time, 3))
    identity = _repeat_dcm(_IDENTITY_DCM, n_time)
    positions = {terminal_a: position_a, terminal_b: position_b}
    attitudes = {terminal_a: _repeat_dcm(dcm_eci2body_a, n_time)}
    if include_terminal_b_attitude:
        attitudes[terminal_b] = _repeat_dcm(dcm_eci2body_b, n_time)
    sun_direction = (
        np.repeat(np.array([[0.0, 0.0, 1.0]]), n_time, axis=0)
        if sun_direction_eci is None
        else np.repeat(sun_direction_eci[None, :], n_time, axis=0)
    )
    return DynamicsData(
        time=time,
        dcm_eci2ecef=identity,
        satellite_position_eci=positions,
        satellite_velocity_eci={terminal_a: zero, terminal_b: zero},
        satellite_position_ecef=positions,
        satellite_attitude_angular_velocity_eci={terminal_a: zero, terminal_b: zero},
        sun_direction_eci=sun_direction,
        sun_direction_ecef=sun_direction,
        satellite_attitude_dcm_eci2body=attitudes,
    )


def _calculate(
    calculator: OTC2OTCBinaryCommLinkCalculator,
    terminal_a: SatelliteTerminal[int],
    terminal_b: SatelliteTerminal[int],
    dynamics_data: DynamicsData[SatelliteTerminal[int]],
    *,
    edges_time_series: Sequence[set[tuple[SatelliteTerminal[int], SatelliteTerminal[int]]]] | None = None,
) -> list[dict[tuple[SatelliteTerminal, SatelliteTerminal], CommLinkPerformance]]:
    edge = (terminal_a, terminal_b)
    edges = edges_time_series if edges_time_series is not None else [{edge} for _ in dynamics_data.time]
    return calculator.calc(
        edges_time_series=edges,
        dynamics_data=dynamics_data,
        rng=np.random.default_rng(0),
    )


def test_calc_processes_each_time_step_as_a_dynamics_snapshot() -> None:
    terminal_a = _make_terminal(1)
    terminal_b = _make_terminal(2)
    time = EPOCH + np.arange(2).astype("timedelta64[s]")
    dynamics_data = _make_dynamics_data(terminal_a, terminal_b, time=time)
    edge = (terminal_a, terminal_b)

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
        terminal_a,
        terminal_b,
        dynamics_data,
    )

    assert len(performance) == 2
    assert performance[0][edge]["link_available"] is True
    assert performance[1][edge]["link_available"] is True


def test_calc_rejects_edge_and_dynamics_series_length_mismatch() -> None:
    terminal_a = _make_terminal(1)
    terminal_b = _make_terminal(2)
    time = EPOCH + np.arange(2).astype("timedelta64[s]")
    dynamics_data = _make_dynamics_data(terminal_a, terminal_b, time=time)
    calculator = OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9)

    with pytest.raises(ValueError, match="same length"):
        calculator.calc(
            edges_time_series=[{(terminal_a, terminal_b)}],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )


def test_calc_rejects_non_increasing_time() -> None:
    terminal_a = _make_terminal(1)
    terminal_b = _make_terminal(2)
    dynamics_data = _make_dynamics_data(terminal_a, terminal_b, time=np.array([EPOCH, EPOCH]))

    with pytest.raises(ValueError, match="strictly increasing"):
        _calculate(
            OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
            terminal_a,
            terminal_b,
            dynamics_data,
            edges_time_series=[set(), set()],
        )


def test_calc_requires_body_attitudes_for_both_terminal_endpoints() -> None:
    terminal_a = _make_terminal(1)
    terminal_b = _make_terminal(2)
    dynamics_data = _make_dynamics_data(
        terminal_a,
        terminal_b,
        time=np.array([EPOCH]),
        include_terminal_b_attitude=False,
    )

    with pytest.raises(ValueError, match="attitude for every terminal endpoint"):
        _calculate(
            OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
            terminal_a,
            terminal_b,
            dynamics_data,
        )


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
    terminal_a = _make_terminal(
        1,
        azimuth_min=np.deg2rad(-1.0),
        azimuth_max=np.deg2rad(1.0),
        dcm_body2terminal=dcm_body2terminal,
    )
    terminal_b = _make_terminal(2)
    dynamics_data = _make_dynamics_data(
        terminal_a,
        terminal_b,
        time=np.array([EPOCH]),
        dcm_eci2body_a=dcm_eci2body,
    )

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
        terminal_a,
        terminal_b,
        dynamics_data,
    )

    assert performance[0][(terminal_a, terminal_b)]["link_available"] is True


@pytest.mark.parametrize("restricted_endpoint", ["source", "destination"])
def test_calc_enforces_field_of_regard_at_both_endpoints(restricted_endpoint: str) -> None:
    restricted_bounds = {"azimuth_min": np.deg2rad(-10.0), "azimuth_max": np.deg2rad(10.0)}
    terminal_a = _make_terminal(1, **(restricted_bounds if restricted_endpoint == "source" else {}))
    terminal_b = _make_terminal(2, **(restricted_bounds if restricted_endpoint == "destination" else {}))
    dynamics_data = _make_dynamics_data(
        terminal_a,
        terminal_b,
        time=np.array([EPOCH]),
        line_of_sight_azimuths=[np.deg2rad(30.0)],
    )

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
        terminal_a,
        terminal_b,
        dynamics_data,
    )

    assert performance[0][(terminal_a, terminal_b)]["link_available"] is False


def test_calc_unwraps_azimuth_rate_across_pi_boundary() -> None:
    max_rate = np.deg2rad(3.0)
    terminal_a = _make_terminal(1, angular_velocity_max=max_rate)
    terminal_b = _make_terminal(2, angular_velocity_max=max_rate)
    dynamics_data = _make_dynamics_data(
        terminal_a,
        terminal_b,
        time=EPOCH + np.arange(2).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([179.0, -179.0]),
    )

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
        terminal_a,
        terminal_b,
        dynamics_data,
    )

    assert performance[1][(terminal_a, terminal_b)]["link_available"] is True


def test_calc_rejects_large_negative_terminal_rate() -> None:
    max_rate = np.deg2rad(5.0)
    terminal_a = _make_terminal(1, angular_velocity_max=max_rate)
    terminal_b = _make_terminal(2, angular_velocity_max=max_rate)
    dynamics_data = _make_dynamics_data(
        terminal_a,
        terminal_b,
        time=EPOCH + np.arange(2).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([10.0, 0.0]),
    )

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
        terminal_a,
        terminal_b,
        dynamics_data,
    )

    assert performance[1][(terminal_a, terminal_b)]["link_available"] is False


def test_calc_converts_time_delta_to_seconds_for_terminal_rate() -> None:
    max_rate = np.deg2rad(5.0)
    terminal_a = _make_terminal(1, angular_velocity_max=max_rate)
    terminal_b = _make_terminal(2, angular_velocity_max=max_rate)
    dynamics_data = _make_dynamics_data(
        terminal_a,
        terminal_b,
        time=EPOCH + np.array([0, 2]).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([0.0, 8.0]),
    )

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
        terminal_a,
        terminal_b,
        dynamics_data,
    )

    assert performance[1][(terminal_a, terminal_b)]["link_available"] is True


def test_calc_skips_slew_check_when_edge_reappears_after_absence() -> None:
    terminal_a = _make_terminal(1, angular_velocity_max=0.0)
    terminal_b = _make_terminal(2, angular_velocity_max=0.0)
    dynamics_data = _make_dynamics_data(
        terminal_a,
        terminal_b,
        time=EPOCH + np.arange(3).astype("timedelta64[s]"),
        line_of_sight_azimuths=np.deg2rad([0.0, 90.0, 179.0]),
    )
    edge = (terminal_a, terminal_b)

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9),
        terminal_a,
        terminal_b,
        dynamics_data,
        edges_time_series=[{edge}, set(), {edge}],
    )

    assert performance[0][edge]["link_available"] is True
    assert performance[2][edge]["link_available"] is True


def test_calc_applies_sun_exclusion_at_both_terminals() -> None:
    terminal_a = _make_terminal(1)
    terminal_b = _make_terminal(2)
    dynamics_data = _make_dynamics_data(
        terminal_a,
        terminal_b,
        time=np.array([EPOCH]),
        sun_direction_eci=np.array([0.0, 1.0, 0.0]),
    )

    performance = _calculate(
        OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9, sun_exclusion_angle=np.deg2rad(10.0)),
        terminal_a,
        terminal_b,
        dynamics_data,
    )

    assert performance[0][(terminal_a, terminal_b)]["link_available"] is False
