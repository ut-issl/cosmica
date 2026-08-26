import numpy as np
import pytest

from cosmica.dtos import DynamicsData

EPOCH = np.datetime64("2026-01-01T00:00:00")


def _make_dynamics_data(time: np.ndarray) -> DynamicsData:
    sample_count = len(time)
    sun_direction = np.zeros((sample_count, 3))
    return DynamicsData(
        time=time,
        dcm_eci2ecef=np.repeat(np.eye(3)[None, :, :], sample_count, axis=0),
        satellite_position_eci={},
        satellite_velocity_eci={},
        satellite_position_ecef={},
        satellite_attitude_angular_velocity_eci={},
        sun_direction_eci=sun_direction,
        sun_direction_ecef=sun_direction,
    )


@pytest.mark.parametrize(
    "time",
    [
        np.array([EPOCH, EPOCH]),
        np.array([EPOCH + np.timedelta64(1, "s"), EPOCH]),
    ],
    ids=["duplicate", "decreasing"],
)
def test_dynamics_data_rejects_non_increasing_time(time: np.ndarray) -> None:
    with pytest.raises(ValueError, match="time must be strictly increasing"):
        _make_dynamics_data(time)


def test_dynamics_data_iteration_yields_time_snapshots() -> None:
    time = np.array([EPOCH, EPOCH + np.timedelta64(1, "s")])

    snapshots = list(_make_dynamics_data(time))

    assert [snapshot.time for snapshot in snapshots] == list(time)
    assert all(snapshot.data_shape == () for snapshot in snapshots)
