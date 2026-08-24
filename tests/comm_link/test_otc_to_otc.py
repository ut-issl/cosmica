import numpy as np
import pytest

from cosmica.comm_link import OTC2OTCBinaryCommLinkCalculator
from cosmica.dtos import DynamicsData
from cosmica.models import SatelliteTerminal
from cosmica.utils.constants import EARTH_RADIUS

EPOCH = np.datetime64("2026-01-01T00:00:00")


def _make_terminal(satellite_id: int, *, angular_velocity_max: float = float("inf")) -> SatelliteTerminal[int]:
    return SatelliteTerminal(
        id=satellite_id,
        terminal_id=satellite_id,
        azimuth_min=-np.pi,
        azimuth_max=np.pi,
        elevation_min=-np.pi / 2,
        elevation_max=np.pi / 2,
        angular_velocity_max=angular_velocity_max,
    )


def _make_dynamics_data(
    terminal_a: SatelliteTerminal[int],
    terminal_b: SatelliteTerminal[int],
    *,
    time: np.ndarray,
) -> DynamicsData[SatelliteTerminal[int]]:
    n_time = len(time)
    position_a = np.repeat(np.array([[EARTH_RADIUS + 1_000e3, 0.0, 0.0]]), n_time, axis=0)
    position_b = np.repeat(np.array([[EARTH_RADIUS + 1_000e3, 1_000e3, 0.0]]), n_time, axis=0)
    zero = np.zeros((n_time, 3))
    identity = np.repeat(np.eye(3)[None, :, :], n_time, axis=0)
    positions = {terminal_a: position_a, terminal_b: position_b}
    return DynamicsData(
        time=time,
        dcm_eci2ecef=identity,
        satellite_position_eci=positions,
        satellite_velocity_eci={terminal_a: zero, terminal_b: zero},
        satellite_position_ecef=positions,
        satellite_attitude_angular_velocity_eci={terminal_a: zero, terminal_b: zero},
        sun_direction_eci=np.repeat(np.array([[0.0, 0.0, 1.0]]), n_time, axis=0),
        sun_direction_ecef=np.repeat(np.array([[0.0, 0.0, 1.0]]), n_time, axis=0),
        satellite_attitude_dcm_eci2body={terminal_a: identity, terminal_b: identity},
    )


def test_calc_processes_each_time_step_as_a_dynamics_snapshot() -> None:
    terminal_a = _make_terminal(1)
    terminal_b = _make_terminal(2)
    time = EPOCH + np.arange(2).astype("timedelta64[s]")
    dynamics_data = _make_dynamics_data(terminal_a, terminal_b, time=time)
    edge = (terminal_a, terminal_b)
    calculator = OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9)

    performance = calculator.calc(
        edges_time_series=[{edge}, {edge}],
        dynamics_data=dynamics_data,
        rng=np.random.default_rng(0),
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
    calculator = OTC2OTCBinaryCommLinkCalculator(link_capacity=10e9)

    with pytest.raises(ValueError, match="strictly increasing"):
        calculator.calc(
            edges_time_series=[set(), set()],
            dynamics_data=dynamics_data,
            rng=np.random.default_rng(0),
        )
