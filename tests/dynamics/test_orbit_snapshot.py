"""Snapshot tests for orbit dynamics calculations.

These tests capture the current behavior of orbit propagators to prevent
unintended changes during refactoring. If you intentionally change the
orbit calculation logic, you should review the snapshot diff and update
the snapshots using: pytest --snapshot-update
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.amber import AmberSnapshotExtension

from cosmica.dynamics import (
    CircularSatelliteOrbitPropagator,
    EllipticalSatelliteOrbitPropagator,
    SatelliteOrbitState,
)
from cosmica.dynamics.sun_dynamics import get_sun_direction_eci
from cosmica.models import CircularSatelliteOrbitModel, EllipticalSatelliteOrbitModel, GravityModel


class NumpySnapshotExtension(AmberSnapshotExtension):
    """Snapshot extension for NumPy arrays with human-readable format."""

    def serialize(
        self,
        data: npt.NDArray[np.floating] | SatelliteOrbitState | dict,
        **kwargs,  # noqa: ANN003
    ) -> str:
        """Serialize data to a human-readable string format."""
        if isinstance(data, SatelliteOrbitState):
            return (
                "SatelliteOrbitState(\n"
                f"  position_eci=\n{self._format_array(data.position_eci)}\n"
                f"  velocity_eci=\n{self._format_array(data.velocity_eci)}\n"
                ")"
            )
        elif isinstance(data, dict):
            result = "{\n"
            for key, value in data.items():
                if isinstance(value, SatelliteOrbitState):
                    result += f"  {key}: {self.serialize(value, **kwargs)}\n"
                elif isinstance(value, np.ndarray):
                    result += f"  {key}:\n{self._format_array(value, indent='    ')}\n"
                else:
                    result += f"  {key}: {value}\n"
            result += "}"
            return result
        elif isinstance(data, np.ndarray):
            return self._format_array(data)
        else:
            return str(data)

    def _format_array(self, arr: npt.NDArray[np.floating], indent: str = "    ") -> str:
        """Format numpy array with proper indentation."""
        # Set print options for consistent formatting
        with np.printoptions(precision=10, suppress=False, threshold=10000, linewidth=100):
            lines = np.array2string(arr, separator=", ").split("\n")
            return "\n".join(indent + line for line in lines)


# Circular Orbit Propagator Snapshot Tests


def test_circular_equatorial_orbit_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for equatorial circular orbit."""
    epoch = np.datetime64("2026-01-01T00:00:00")
    time_array = epoch + np.timedelta64(60, "s") * np.arange(10)

    model = CircularSatelliteOrbitModel(
        semi_major_axis=7000e3,  # 7000 km
        inclination=np.radians(0),  # Equatorial
        raan=np.radians(0),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
    )

    propagator = CircularSatelliteOrbitPropagator(model=model)
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_circular_polar_orbit_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for polar circular orbit."""
    epoch = np.datetime64("2026-01-01T00:00:00")
    time_array = epoch + np.timedelta64(120, "s") * np.arange(10)

    model = CircularSatelliteOrbitModel(
        semi_major_axis=7000e3,
        inclination=np.radians(90),  # Polar
        raan=np.radians(0),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
    )

    propagator = CircularSatelliteOrbitPropagator(model=model)
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_circular_inclined_orbit_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for inclined circular orbit (ISS-like)."""
    epoch = np.datetime64("2026-01-01T12:00:00")
    time_array = epoch + np.timedelta64(90, "s") * np.arange(15)

    model = CircularSatelliteOrbitModel(
        semi_major_axis=6793e3,  # ~420 km altitude
        inclination=np.radians(51.6),  # ISS inclination
        raan=np.radians(30),
        phase_at_epoch=np.radians(45),
        epoch=epoch,
    )

    propagator = CircularSatelliteOrbitPropagator(model=model)
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_circular_full_orbit_period_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for one complete orbital period."""
    epoch = np.datetime64("2026-01-01T00:00:00")
    semi_major_axis = 7000e3

    model = CircularSatelliteOrbitModel(
        semi_major_axis=semi_major_axis,
        inclination=np.radians(45),
        raan=np.radians(60),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
    )

    propagator = CircularSatelliteOrbitPropagator(model=model)

    # Calculate orbital period and sample at 20 points
    period_seconds = 2 * np.pi / propagator.mean_motion
    time_array = epoch + np.timedelta64(1, "s") * np.linspace(0, period_seconds, 20).astype(int)
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_circular_geostationary_orbit_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for geostationary orbit altitude."""
    epoch = np.datetime64("2026-01-01T00:00:00")
    # GEO altitude: ~35,786 km, so semi-major axis = 42,164 km
    time_array = epoch + np.timedelta64(600, "s") * np.arange(10)

    model = CircularSatelliteOrbitModel(
        semi_major_axis=42164e3,
        inclination=np.radians(0),
        raan=np.radians(0),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
    )

    propagator = CircularSatelliteOrbitPropagator(model=model)
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_circular_leo_constellation_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for LEO constellation-like orbit."""
    epoch = np.datetime64("2026-01-01T06:00:00")
    time_array = epoch + np.timedelta64(100, "s") * np.arange(12)

    model = CircularSatelliteOrbitModel(
        semi_major_axis=6928e3,  # ~550 km altitude (Starlink-like)
        inclination=np.radians(53),
        raan=np.radians(120),
        phase_at_epoch=np.radians(90),
        epoch=epoch,
    )

    propagator = CircularSatelliteOrbitPropagator(model=model)
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_circular_propagate_from_epoch_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for propagate_from_epoch method."""
    epoch = np.datetime64("2026-01-01T00:00:00")
    time_deltas = np.timedelta64(60, "s") * np.arange(10)

    model = CircularSatelliteOrbitModel(
        semi_major_axis=7000e3,
        inclination=np.radians(30),
        raan=np.radians(45),
        phase_at_epoch=np.radians(60),
        epoch=epoch,
    )

    propagator = CircularSatelliteOrbitPropagator(model=model)
    states = propagator.propagate_from_epoch(time_deltas)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


# Elliptical Orbit Propagator Snapshot Tests


def test_elliptical_circular_case_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for elliptical propagator with zero eccentricity."""
    epoch = np.datetime64("2026-01-01T00:00:00")
    time_array = epoch + np.timedelta64(60, "s") * np.arange(10)

    model = EllipticalSatelliteOrbitModel(
        semi_major_axis=7000e3,
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

    propagator = EllipticalSatelliteOrbitPropagator(model=model, reference_frame="gcrs")
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_elliptical_eccentric_orbit_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for moderately eccentric orbit."""
    epoch = np.datetime64("2026-01-01T00:00:00")
    time_array = epoch + np.timedelta64(120, "s") * np.arange(15)

    model = EllipticalSatelliteOrbitModel(
        semi_major_axis=10000e3,
        inclination=np.radians(30),
        raan=np.radians(45),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
        satnum=23456,
        gravity_model=GravityModel.WGS84,
        drag_coeff=0.00001,
        eccentricity=0.2,  # Moderate eccentricity
        argpo=np.radians(60),
    )

    propagator = EllipticalSatelliteOrbitPropagator(model=model, reference_frame="gcrs")
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_elliptical_highly_eccentric_orbit_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for highly eccentric orbit (Molniya-like)."""
    epoch = np.datetime64("2026-01-01T00:00:00")
    time_array = epoch + np.timedelta64(300, "s") * np.arange(20)

    model = EllipticalSatelliteOrbitModel(
        semi_major_axis=26600e3,  # Molniya orbit
        inclination=np.radians(63.4),  # Critical inclination
        raan=np.radians(0),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
        satnum=34567,
        gravity_model=GravityModel.WGS72,
        drag_coeff=0.0,
        eccentricity=0.72,  # High eccentricity
        argpo=np.radians(270),
    )

    propagator = EllipticalSatelliteOrbitPropagator(model=model, reference_frame="gcrs")
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_elliptical_iss_like_orbit_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for ISS-like orbit parameters."""
    epoch = np.datetime64("2026-01-01T12:00:00")
    time_array = epoch + np.timedelta64(90, "s") * np.arange(10)

    model = EllipticalSatelliteOrbitModel(
        semi_major_axis=6793e3,  # ~420 km altitude
        inclination=np.radians(51.6),
        raan=np.radians(100),
        phase_at_epoch=np.radians(45),
        epoch=epoch,
        satnum=25544,
        gravity_model=GravityModel.WGS84,
        drag_coeff=0.0001,
        eccentricity=0.0001,  # Nearly circular
        argpo=np.radians(30),
    )

    propagator = EllipticalSatelliteOrbitPropagator(model=model, reference_frame="gcrs")
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_elliptical_teme_reference_frame_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for TEME reference frame."""
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

    propagator = EllipticalSatelliteOrbitPropagator(model=model, reference_frame="teme")
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_elliptical_j2000_reference_frame_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for J2000 reference frame."""
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

    propagator = EllipticalSatelliteOrbitPropagator(model=model, reference_frame="j2000")
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


def test_elliptical_gto_orbit_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for Geostationary Transfer Orbit (GTO)."""
    epoch = np.datetime64("2026-01-01T00:00:00")
    time_array = epoch + np.timedelta64(200, "s") * np.arange(15)

    # GTO typical parameters
    model = EllipticalSatelliteOrbitModel(
        semi_major_axis=24371e3,  # Semi-major axis for GTO
        inclination=np.radians(7),  # Low inclination
        raan=np.radians(0),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
        satnum=45678,
        gravity_model=GravityModel.WGS84,
        drag_coeff=0.00001,
        eccentricity=0.73,  # High eccentricity
        argpo=np.radians(180),
    )

    propagator = EllipticalSatelliteOrbitPropagator(model=model, reference_frame="gcrs")
    states = propagator.propagate(time_array)

    assert states == snapshot(extension_class=NumpySnapshotExtension)


# Sun Dynamics Snapshot Tests


def test_sun_direction_single_time_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for sun direction at a single time."""
    time = np.datetime64("2026-01-01T00:00:00")
    sun_dir = get_sun_direction_eci(time)

    assert snapshot(extension_class=NumpySnapshotExtension) == sun_dir


def test_sun_direction_daily_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for sun direction over 24 hours."""
    start_time = np.datetime64("2026-01-01T00:00:00")
    # Sample every 2 hours for 24 hours
    time_array = start_time + np.timedelta64(2, "h") * np.arange(13)
    sun_dir = get_sun_direction_eci(time_array)

    assert snapshot(extension_class=NumpySnapshotExtension) == sun_dir


def test_sun_direction_seasonal_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for sun direction across seasons."""
    times = np.array(
        [
            np.datetime64("2026-01-01T12:00:00"),  # Winter (Northern Hemisphere)
            np.datetime64("2026-04-01T12:00:00"),  # Spring
            np.datetime64("2026-07-01T12:00:00"),  # Summer
            np.datetime64("2026-10-01T12:00:00"),  # Fall
        ]
    )
    sun_dir = get_sun_direction_eci(times)

    assert snapshot(extension_class=NumpySnapshotExtension) == sun_dir


def test_sun_direction_yearly_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for sun direction over a full year."""
    start_time = np.datetime64("2026-01-01T00:00:00")
    # Sample every 30 days for a year
    time_array = start_time + np.timedelta64(30, "D") * np.arange(13)
    sun_dir = get_sun_direction_eci(time_array)

    assert snapshot(extension_class=NumpySnapshotExtension) == sun_dir


def test_sun_direction_equinox_solstice_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for sun direction at equinoxes and solstices."""
    times = np.array(
        [
            np.datetime64("2026-03-20T09:00:00"),  # Vernal equinox (approx)
            np.datetime64("2026-06-21T06:00:00"),  # Summer solstice (approx)
            np.datetime64("2026-09-22T15:00:00"),  # Autumnal equinox (approx)
            np.datetime64("2026-12-21T18:00:00"),  # Winter solstice (approx)
        ]
    )
    sun_dir = get_sun_direction_eci(times)

    assert snapshot(extension_class=NumpySnapshotExtension) == sun_dir


def test_sun_direction_intraday_snapshot(snapshot: SnapshotAssertion) -> None:
    """Snapshot test for sun direction at different times of day."""
    base_date = np.datetime64("2026-06-15")
    times = base_date + np.timedelta64(1, "h") * np.arange(0, 24, 3)
    sun_dir = get_sun_direction_eci(times)

    assert snapshot(extension_class=NumpySnapshotExtension) == sun_dir
