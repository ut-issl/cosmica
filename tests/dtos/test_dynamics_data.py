import numpy as np
import pytest

from cosmica.dtos import DynamicsData
from tests.factories import make_satellite

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


def _make_attitude_dynamics_data(attitude_dcm: np.ndarray) -> DynamicsData:
    satellite = make_satellite(1)
    sample_count = len(attitude_dcm)
    vectors = np.zeros((sample_count, 3))
    return DynamicsData(
        time=EPOCH + np.arange(sample_count).astype("timedelta64[s]"),
        dcm_eci2ecef=np.repeat(np.eye(3)[None, :, :], sample_count, axis=0),
        satellite_position_eci={satellite: vectors},
        satellite_velocity_eci={satellite: vectors},
        satellite_position_ecef={satellite: vectors},
        satellite_attitude_angular_velocity_eci={satellite: vectors},
        sun_direction_eci=vectors,
        sun_direction_ecef=vectors,
        satellite_attitude_dcm_eci2body={satellite: attitude_dcm},
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


def test_dynamics_data_slices_satellite_body_attitude() -> None:
    attitude_dcm = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    dynamics_data = _make_attitude_dynamics_data(attitude_dcm)
    satellite = next(iter(dynamics_data.satellite_attitude_dcm_eci2body))

    snapshot = dynamics_data[1]

    np.testing.assert_array_equal(snapshot.satellite_attitude_dcm_eci2body[satellite], np.eye(3))


def test_dynamics_data_rejects_non_rotation_body_attitude() -> None:
    attitude_dcm = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    attitude_dcm[1, 2, 2] = -1.0

    with pytest.raises(ValueError, match="orthonormal, right-handed"):
        _make_attitude_dynamics_data(attitude_dcm)
