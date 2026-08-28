import numpy as np
import pytest

from tests.factories import TEST_EPOCH, make_attitude_dynamics_data, make_dynamics_data


@pytest.mark.parametrize(
    "time",
    [
        np.array([TEST_EPOCH, TEST_EPOCH]),
        np.array([TEST_EPOCH + np.timedelta64(1, "s"), TEST_EPOCH]),
    ],
    ids=["duplicate", "decreasing"],
)
def test_dynamics_data_rejects_non_increasing_time(time: np.ndarray) -> None:
    with pytest.raises(ValueError, match="time must be strictly increasing"):
        make_dynamics_data(time)


def test_dynamics_data_iteration_yields_time_snapshots() -> None:
    time = np.array([TEST_EPOCH, TEST_EPOCH + np.timedelta64(1, "s")])

    snapshots = list(make_dynamics_data(time))

    assert [snapshot.time for snapshot in snapshots] == list(time)
    assert all(snapshot.data_shape == () for snapshot in snapshots)


def test_dynamics_data_slices_satellite_body_attitude() -> None:
    attitude_dcm = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    dynamics_data = make_attitude_dynamics_data(attitude_dcm)
    satellite = next(iter(dynamics_data.satellite_attitude_dcm_eci2body))

    snapshot = dynamics_data[1]

    np.testing.assert_array_equal(snapshot.satellite_attitude_dcm_eci2body[satellite], np.eye(3))


def test_dynamics_data_rejects_non_rotation_body_attitude() -> None:
    attitude_dcm = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    attitude_dcm[1, 2, 2] = -1.0

    with pytest.raises(ValueError, match="orthonormal, right-handed"):
        make_attitude_dynamics_data(attitude_dcm)
