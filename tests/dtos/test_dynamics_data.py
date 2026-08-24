import numpy as np
import pytest

from cosmica.dtos import DynamicsData
from cosmica.models import SatelliteTerminal


def _make_satellite_terminal() -> SatelliteTerminal[int]:
    return SatelliteTerminal(
        id=1,
        terminal_id=1,
        azimuth_min=-np.pi,
        azimuth_max=np.pi,
        elevation_min=-np.pi / 2,
        elevation_max=np.pi / 2,
        angular_velocity_max=1.0,
    )


def _make_dynamics_data(attitude_dcm: np.ndarray) -> DynamicsData[SatelliteTerminal[int]]:
    satellite = _make_satellite_terminal()
    n_time = attitude_dcm.shape[0]
    vectors = np.zeros((n_time, 3))
    return DynamicsData(
        time=np.datetime64("2026-01-01") + np.arange(n_time).astype("timedelta64[s]"),
        dcm_eci2ecef=np.repeat(np.eye(3)[None, :, :], n_time, axis=0),
        satellite_position_eci={satellite: vectors},
        satellite_velocity_eci={satellite: vectors},
        satellite_position_ecef={satellite: vectors},
        satellite_attitude_angular_velocity_eci={satellite: vectors},
        sun_direction_eci=vectors,
        sun_direction_ecef=vectors,
        satellite_attitude_dcm_eci2body={satellite: attitude_dcm},
    )


def test_dynamics_data_slices_satellite_body_attitude() -> None:
    attitude_dcm = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    dynamics_data = _make_dynamics_data(attitude_dcm)
    satellite = next(iter(dynamics_data.satellite_attitude_dcm_eci2body))

    snapshot = dynamics_data[1]

    np.testing.assert_array_equal(snapshot.satellite_attitude_dcm_eci2body[satellite], np.eye(3))


def test_dynamics_data_rejects_non_rotation_body_attitude() -> None:
    attitude_dcm = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    attitude_dcm[1, 2, 2] = -1.0

    with pytest.raises(ValueError, match="orthonormal, right-handed"):
        _make_dynamics_data(attitude_dcm)
