import numpy as np
import numpy.typing as npt
import pytest

from cosmica.dynamics import CircularSatelliteOrbit, SatelliteOrbitState
from cosmica.utils.constants import EARTH_MU


def test_satellite_orbit_state():
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([4.0, 5.0, 6.0])
    state = SatelliteOrbitState(position_eci=position, velocity_eci=velocity)
    assert np.all(state.position_eci == position)
    assert np.all(state.velocity_eci == velocity)


def test_circular_satellite_orbit_mean_motion():
    epoch = np.datetime64("2021-01-01T00:00:00")
    orbit = CircularSatelliteOrbit(semi_major_axis=7000, inclination=0, raan=0, phase_at_epoch=0, epoch=epoch)
    expected_mean_motion = np.sqrt(EARTH_MU / orbit.semi_major_axis**3)
    assert orbit.mean_motion == pytest.approx(expected_mean_motion, rel=1e-9)


def test_circular_satellite_orbit_propagate():
    epoch = np.datetime64("2021-01-01T00:00:00")
    semi_major_axis = 7000
    inclination = np.radians(28.5)
    raan = np.radians(45)
    phase_at_epoch = np.radians(0)
    orbit = CircularSatelliteOrbit(
        semi_major_axis=semi_major_axis,
        inclination=inclination,
        raan=raan,
        phase_at_epoch=phase_at_epoch,
        epoch=epoch,
    )

    # Propagate for 0 seconds, the result should be the same as the initial position and velocity
    time = np.array([epoch])
    result = orbit.propagate(time)
    assert isinstance(result, SatelliteOrbitState)

    position_eci = result.position_eci
    velocity_eci = result.velocity_eci

    initial_position_eci = orbit.dcm_orbit_to_eci @ np.array([semi_major_axis, 0, 0])
    initial_velocity_eci = orbit.dcm_orbit_to_eci @ np.array([0, semi_major_axis * orbit.mean_motion, 0])

    assert np.allclose(position_eci, initial_position_eci, rtol=1e-9)
    assert np.allclose(velocity_eci, initial_velocity_eci, rtol=1e-9)


@pytest.mark.parametrize(
    ("inclination", "raan", "expected_dcm"),
    [
        # Equatorial orbit with RAAN = 0. +X_o -> +X_i, +Y_o -> +Y_i, +Z_o -> +Z_i
        (0, 0, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        # Equatorial orbit with RAAN = pi/2. +X_o -> +Y_i, +Y_o -> -X_i, +Z_o -> +Z_i
        (0, np.pi / 2, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
        # Polar orbit with inclination = pi/2, RAAN = 0. +X_o -> +X_i, +Y_o -> +Z_i, +Z_o -> -Y_i
        (np.pi / 2, 0, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
        # Polar orbit with inclination = pi/2, RAAN = pi/2. +X_o -> +Y_i, +Y_o -> +Z_i, +Z_o -> -X_i
        (np.pi / 2, np.pi / 2, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])),
    ],
)
def test_circular_satellite_orbit_dcm_orbit_to_eci(
    inclination: float,
    raan: float,
    expected_dcm: npt.NDArray[np.float64],
):
    epoch = np.datetime64("2021-01-01T00:00:00")
    orbit = CircularSatelliteOrbit(
        semi_major_axis=7000,
        inclination=inclination,
        raan=raan,
        phase_at_epoch=0,
        epoch=epoch,
    )
    assert np.allclose(orbit.dcm_orbit_to_eci, expected_dcm, rtol=1e-9, atol=1e-9)
