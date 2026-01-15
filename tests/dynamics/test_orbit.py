import numpy as np
import numpy.typing as npt
import pytest

from cosmica.dynamics import (
    CircularSatelliteOrbitPropagator,
    EllipticalSatelliteOrbitPropagator,
    SatelliteOrbitState,
)
from cosmica.models import CircularSatelliteOrbitModel, EllipticalSatelliteOrbitModel, GravityModel
from cosmica.utils.constants import EARTH_MU, EARTH_RADIUS


class TestSatelliteOrbitState:
    """Test the SatelliteOrbitState dataclass."""

    def test_init_valid(self):
        """Test basic initialization with valid data."""
        pos = np.array([[1000.0, 2000.0, 3000.0]])
        vel = np.array([[100.0, 200.0, 300.0]])
        state = SatelliteOrbitState(position_eci=pos, velocity_eci=vel)
        assert np.array_equal(state.position_eci, pos)
        assert np.array_equal(state.velocity_eci, vel)

    def test_shape_validation(self):
        """Test that shapes are validated correctly."""
        # Valid shapes
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        vel = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        state = SatelliteOrbitState(position_eci=pos, velocity_eci=vel)
        assert state.position_eci.shape == (2, 3)
        assert state.velocity_eci.shape == (2, 3)

    def test_shape_mismatch_raises(self):
        """Test that mismatched shapes raise an error."""
        pos = np.array([[1.0, 2.0, 3.0]])
        vel = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        with pytest.raises(AssertionError):
            SatelliteOrbitState(position_eci=pos, velocity_eci=vel)

    def test_invalid_last_dimension_raises(self):
        """Test that non-3D vectors raise an error."""
        pos = np.array([[1.0, 2.0]])  # Only 2 dimensions
        vel = np.array([[0.1, 0.2]])
        with pytest.raises(AssertionError):
            SatelliteOrbitState(position_eci=pos, velocity_eci=vel)

    def test_getitem_single_index(self):
        """Test indexing with a single index."""
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        vel = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        state = SatelliteOrbitState(position_eci=pos, velocity_eci=vel)

        state_0 = state[0]
        assert np.array_equal(state_0.position_eci, pos[0])
        assert np.array_equal(state_0.velocity_eci, vel[0])

    def test_getitem_slice(self):
        """Test slicing."""
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        vel = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        state = SatelliteOrbitState(position_eci=pos, velocity_eci=vel)

        state_slice = state[0:2]
        assert np.array_equal(state_slice.position_eci, pos[0:2])
        assert np.array_equal(state_slice.velocity_eci, vel[0:2])


class TestCircularSatelliteOrbitPropagator:
    """Test the CircularSatelliteOrbitPropagator."""

    def test_basic_propagation(self):
        """Test basic propagation returns correct shapes."""
        start_time = np.datetime64("2026-01-01T00:00:00")
        time_array: npt.NDArray[np.datetime64] = start_time + np.timedelta64(60, "s") * np.arange(10)

        model = CircularSatelliteOrbitModel(
            semi_major_axis=7000e3,  # 7000 km in meters
            inclination=np.radians(0),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=start_time,
        )

        propagator = CircularSatelliteOrbitPropagator(model=model)
        states = propagator.propagate(time_array)

        assert states.position_eci.shape == (10, 3)
        assert states.velocity_eci.shape == (10, 3)

    def test_mean_motion_calculation(self):
        """Test that mean motion is calculated correctly from Kepler's third law."""
        model = CircularSatelliteOrbitModel(
            semi_major_axis=7000e3,  # 7000 km
            inclination=0.0,
            raan=0.0,
            phase_at_epoch=0.0,
            epoch=np.datetime64("2026-01-01T00:00:00"),
        )
        propagator = CircularSatelliteOrbitPropagator(model=model)

        # n = sqrt(mu / a^3)
        expected_mean_motion = np.sqrt(EARTH_MU / (7000e3) ** 3)
        assert np.isclose(propagator.mean_motion, expected_mean_motion)

    def test_position_at_epoch(self):
        """Test that position at epoch matches expected value for zero phase."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        semi_major_axis = 7000e3

        model = CircularSatelliteOrbitModel(
            semi_major_axis=semi_major_axis,
            inclination=np.radians(0),  # Equatorial orbit
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),  # Starting at x-axis
            epoch=epoch,
        )
        propagator = CircularSatelliteOrbitPropagator(model=model)

        states = propagator.propagate(np.array([epoch]))

        # At phase=0, position should be (a, 0, 0) for equatorial orbit
        expected_position = np.array([[semi_major_axis, 0.0, 0.0]])
        assert np.allclose(states.position_eci, expected_position, atol=1e-6)

    def test_position_at_90_degrees(self):
        """Test position after a quarter orbit."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        semi_major_axis = 7000e3

        model = CircularSatelliteOrbitModel(
            semi_major_axis=semi_major_axis,
            inclination=np.radians(0),
            raan=np.radians(0),
            phase_at_epoch=np.radians(90),  # Starting at y-axis
            epoch=epoch,
        )
        propagator = CircularSatelliteOrbitPropagator(model=model)

        states = propagator.propagate(np.array([epoch]))

        # At phase=90°, position should be (0, a, 0)
        expected_position = np.array([[0.0, semi_major_axis, 0.0]])
        assert np.allclose(states.position_eci, expected_position, atol=1e-6)

    def test_velocity_magnitude_constant(self):
        """Test that velocity magnitude is constant for circular orbit."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        semi_major_axis = 7000e3

        model = CircularSatelliteOrbitModel(
            semi_major_axis=semi_major_axis,
            inclination=np.radians(0),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
        )
        propagator = CircularSatelliteOrbitPropagator(model=model)

        time_array = epoch + np.timedelta64(60, "s") * np.arange(100)
        states = propagator.propagate(time_array)

        velocity_magnitudes = np.linalg.norm(states.velocity_eci, axis=1)

        # For circular orbit: v = sqrt(mu/a)
        expected_velocity = np.sqrt(EARTH_MU / semi_major_axis)
        assert np.allclose(velocity_magnitudes, expected_velocity, rtol=1e-10)

    def test_position_magnitude_constant(self):
        """Test that position magnitude (orbital radius) is constant for circular orbit."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        semi_major_axis = 7000e3

        model = CircularSatelliteOrbitModel(
            semi_major_axis=semi_major_axis,
            inclination=np.radians(45),
            raan=np.radians(30),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
        )
        propagator = CircularSatelliteOrbitPropagator(model=model)

        time_array = epoch + np.timedelta64(60, "s") * np.arange(100)
        states = propagator.propagate(time_array)

        position_magnitudes = np.linalg.norm(states.position_eci, axis=1)
        assert np.allclose(position_magnitudes, semi_major_axis, rtol=1e-10)

    def test_velocity_perpendicular_to_position(self):
        """Test that velocity is perpendicular to position for circular orbit."""
        epoch = np.datetime64("2026-01-01T00:00:00")

        model = CircularSatelliteOrbitModel(
            semi_major_axis=7000e3,
            inclination=np.radians(45),
            raan=np.radians(30),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
        )
        propagator = CircularSatelliteOrbitPropagator(model=model)

        time_array = epoch + np.timedelta64(60, "s") * np.arange(100)
        states = propagator.propagate(time_array)

        # Dot product should be zero for perpendicular vectors
        dot_products = np.sum(states.position_eci * states.velocity_eci, axis=1)
        assert np.allclose(dot_products, 0.0, atol=1e-3)

    def test_orbital_period(self):
        """Test that satellite returns to same position after one orbital period."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        semi_major_axis = 7000e3

        model = CircularSatelliteOrbitModel(
            semi_major_axis=semi_major_axis,
            inclination=np.radians(0),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
        )
        propagator = CircularSatelliteOrbitPropagator(model=model)

        # Orbital period: T = 2*pi / n
        period_seconds = 2 * np.pi / propagator.mean_motion

        time_array = np.array(
            [
                epoch,
                epoch + np.timedelta64(int(period_seconds * 1e9), "ns"),
            ],
        )
        states = propagator.propagate(time_array)

        # Position should be the same (within numerical tolerance)
        # Use absolute tolerance for position comparison (meters)
        assert np.allclose(states.position_eci[0], states.position_eci[1], atol=1e-3)
        assert np.allclose(states.velocity_eci[0], states.velocity_eci[1], atol=1e-6)

    def test_inclined_orbit_z_component(self):
        """Test that inclined orbit has non-zero z-component."""
        epoch = np.datetime64("2026-01-01T00:00:00")

        model = CircularSatelliteOrbitModel(
            semi_major_axis=7000e3,
            inclination=np.radians(45),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
        )
        propagator = CircularSatelliteOrbitPropagator(model=model)

        # Propagate for half an orbit
        period_seconds = 2 * np.pi / propagator.mean_motion
        time_array = epoch + np.timedelta64(1, "s") * np.linspace(0, period_seconds, 100).astype(int)
        states = propagator.propagate(time_array)

        # Should have non-zero z components
        assert np.any(np.abs(states.position_eci[:, 2]) > 1e6)  # At least 1000 km

    def test_propagate_from_epoch(self):
        """Test propagate_from_epoch method with timedelta."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        semi_major_axis = 7000e3

        model = CircularSatelliteOrbitModel(
            semi_major_axis=semi_major_axis,
            inclination=np.radians(0),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
        )
        propagator = CircularSatelliteOrbitPropagator(model=model)

        # Propagate using timedelta
        time_deltas = np.timedelta64(60, "s") * np.arange(10)
        states_from_epoch = propagator.propagate_from_epoch(time_deltas)

        # Should match propagate with absolute times
        time_array = epoch + time_deltas
        states_absolute = propagator.propagate(time_array)

        assert np.allclose(states_from_epoch.position_eci, states_absolute.position_eci)
        assert np.allclose(states_from_epoch.velocity_eci, states_absolute.velocity_eci)


class TestEllipticalSatelliteOrbitPropagator:
    """Test the EllipticalSatelliteOrbitPropagator using SGP4."""

    def test_basic_propagation(self):
        """Test basic propagation returns correct shapes."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        time_array = epoch + np.timedelta64(60, "s") * np.arange(10)

        model = EllipticalSatelliteOrbitModel(
            semi_major_axis=7000e3,
            inclination=np.radians(51.6),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
            satnum=12345,
            gravity_model=GravityModel.WGS84,
            drag_coeff=0.0,
            eccentricity=0.001,
            argpo=np.radians(0),
        )

        propagator = EllipticalSatelliteOrbitPropagator(model=model)
        states = propagator.propagate(time_array)

        assert states.position_eci.shape == (10, 3)
        assert states.velocity_eci.shape == (10, 3)

    def test_mean_motion_calculation(self):
        """Test mean motion calculation for elliptical orbit."""
        model = EllipticalSatelliteOrbitModel(
            semi_major_axis=7000e3,
            inclination=np.radians(51.6),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=np.datetime64("2026-01-01T00:00:00"),
            satnum=12345,
            gravity_model=GravityModel.WGS84,
            drag_coeff=0.0,
            eccentricity=0.0,
            argpo=np.radians(0),
        )

        propagator = EllipticalSatelliteOrbitPropagator(model=model)
        expected_mean_motion = np.sqrt(EARTH_MU / (7000e3) ** 3)
        assert np.isclose(propagator.mean_motion, expected_mean_motion)

    def test_circular_case_eccentricity_zero(self):
        """Test that zero eccentricity produces near-circular orbit."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        semi_major_axis = 7000e3

        model = EllipticalSatelliteOrbitModel(
            semi_major_axis=semi_major_axis,
            inclination=np.radians(0),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
            satnum=12345,
            gravity_model=GravityModel.WGS84,
            drag_coeff=0.0,
            eccentricity=0.0,
            argpo=np.radians(0),
        )

        propagator = EllipticalSatelliteOrbitPropagator(model=model)
        time_array = epoch + np.timedelta64(60, "s") * np.arange(100)
        states = propagator.propagate(time_array)

        # Radius should be approximately constant
        radii = np.linalg.norm(states.position_eci, axis=1)
        radius_variation = np.std(radii) / np.mean(radii)
        assert radius_variation < 0.01  # Less than 1% variation

    def test_different_reference_frames(self):
        """Test propagation in different reference frames."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        time_array = epoch + np.timedelta64(60, "s") * np.arange(10)

        model = EllipticalSatelliteOrbitModel(
            semi_major_axis=7000e3,
            inclination=np.radians(51.6),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
            satnum=12345,
            gravity_model=GravityModel.WGS84,
            drag_coeff=0.0,
            eccentricity=0.01,
            argpo=np.radians(0),
        )

        # Test GCRS frame
        propagator_gcrs = EllipticalSatelliteOrbitPropagator(model=model, reference_frame="gcrs")
        states_gcrs = propagator_gcrs.propagate(time_array)
        assert states_gcrs.position_eci.shape == (10, 3)

        # Test TEME frame
        propagator_teme = EllipticalSatelliteOrbitPropagator(model=model, reference_frame="teme")
        states_teme = propagator_teme.propagate(time_array)
        assert states_teme.position_eci.shape == (10, 3)

        # Results should be different (different reference frames)
        assert not np.allclose(states_gcrs.position_eci, states_teme.position_eci, rtol=1e-3)

    def test_invalid_reference_frame_raises(self):
        """Test that invalid reference frame raises ValueError."""
        model = EllipticalSatelliteOrbitModel(
            semi_major_axis=7000e3,
            inclination=np.radians(51.6),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=np.datetime64("2026-01-01T00:00:00"),
            satnum=12345,
            gravity_model=GravityModel.WGS84,
            drag_coeff=0.0,
            eccentricity=0.01,
            argpo=np.radians(0),
        )

        with pytest.raises(ValueError, match="Invalid reference_frame"):
            EllipticalSatelliteOrbitPropagator(model=model, reference_frame="invalid")

    def test_iss_like_orbit(self):
        """Test with ISS-like orbital parameters."""
        epoch = np.datetime64("2026-01-01T00:00:00")

        # ISS-like parameters
        model = EllipticalSatelliteOrbitModel(
            semi_major_axis=6793e3,  # ~420 km altitude
            inclination=np.radians(51.6),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
            satnum=25544,
            gravity_model=GravityModel.WGS84,
            drag_coeff=0.0001,
            eccentricity=0.0001,
            argpo=np.radians(0),
        )

        propagator = EllipticalSatelliteOrbitPropagator(model=model)
        time_array = epoch + np.timedelta64(60, "s") * np.arange(100)
        states = propagator.propagate(time_array)

        # Check altitude is reasonable (near 420 km)
        radii = np.linalg.norm(states.position_eci, axis=1)
        altitudes = radii - EARTH_RADIUS
        assert np.all(altitudes > 400e3)
        assert np.all(altitudes < 450e3)

    def test_perigee_apogee_with_eccentricity(self):
        """Test that elliptical orbit has correct perigee and apogee."""
        epoch = np.datetime64("2026-01-01T00:00:00")
        semi_major_axis = 10000e3
        eccentricity = 0.2

        model = EllipticalSatelliteOrbitModel(
            semi_major_axis=semi_major_axis,
            inclination=np.radians(0),
            raan=np.radians(0),
            phase_at_epoch=np.radians(0),
            epoch=epoch,
            satnum=12345,
            gravity_model=GravityModel.WGS84,
            drag_coeff=0.0,
            eccentricity=eccentricity,
            argpo=np.radians(0),
        )

        propagator = EllipticalSatelliteOrbitPropagator(model=model)

        # Propagate for multiple orbits
        period_seconds = 2 * np.pi / propagator.mean_motion
        time_array = epoch + np.timedelta64(1, "s") * np.linspace(0, period_seconds * 2, 200).astype(int)
        states = propagator.propagate(time_array)

        radii = np.linalg.norm(states.position_eci, axis=1)

        # Calculate expected perigee and apogee
        expected_perigee = semi_major_axis * (1 - eccentricity)
        expected_apogee = semi_major_axis * (1 + eccentricity)

        # Allow some tolerance due to SGP4 perturbations
        assert np.min(radii) < expected_perigee * 1.1
        assert np.max(radii) > expected_apogee * 0.9
