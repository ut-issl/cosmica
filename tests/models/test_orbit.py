import numpy as np
import pytest

from cosmica.models import CircularSatelliteOrbitModel, EllipticalSatelliteOrbitModel, GravityModel


def test_construct_circular_satellite_orbit_model():
    epoch = np.datetime64("2026-01-01T00:00:00")
    _model = CircularSatelliteOrbitModel(
        semi_major_axis=7000,
        inclination=np.radians(45),
        raan=np.radians(30),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
    )


@pytest.mark.xfail(reason="Serialization of `np.datetime64 not yet implemented")
def test_serialize_circular_satellite_orbit_model():
    epoch = np.datetime64("2026-01-01T00:00:00")
    model = CircularSatelliteOrbitModel(
        semi_major_axis=7000,
        inclination=np.radians(45),
        raan=np.radians(30),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
    )

    _as_json = model.model_dump_json()


def test_construct_elliptical_satellite_orbit_model():
    epoch = np.datetime64("2026-01-01T00:00:00")
    _model = EllipticalSatelliteOrbitModel(
        semi_major_axis=7000,
        inclination=np.radians(45),
        raan=np.radians(30),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
        satnum=12345,
        gravity_model=GravityModel.WGS84,
        drag_coeff=2.2,
        eccentricity=0.01,
        argpo=np.radians(90),
    )


@pytest.mark.xfail(reason="Serialization of `np.datetime64 not yet implemented")
def test_serialize_elliptical_satellite_orbit_model():
    epoch = np.datetime64("2026-01-01T00:00:00")
    model = EllipticalSatelliteOrbitModel(
        semi_major_axis=7000,
        inclination=np.radians(45),
        raan=np.radians(30),
        phase_at_epoch=np.radians(0),
        epoch=epoch,
        satnum=12345,
        gravity_model=GravityModel.WGS84,
        drag_coeff=2.2,
        eccentricity=0.01,
        argpo=np.radians(90),
    )

    _as_json = model.model_dump_json()
