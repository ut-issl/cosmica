import numpy as np

from cosmica.models import (
    CircularSatelliteOrbitModel,
    ConstellationSatellite,
    DirectionCosineMatrix,
    OpticalCommunicationTerminal,
)
from cosmica.utils.constants import EARTH_RADIUS

TEST_EPOCH = np.datetime64("2026-01-01T00:00:00")

_IDENTITY_DCM: DirectionCosineMatrix = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def make_orbit(*, phase_at_epoch: float = 0.0) -> CircularSatelliteOrbitModel:
    return CircularSatelliteOrbitModel(
        semi_major_axis=EARTH_RADIUS + 1_000e3,
        inclination=0.0,
        raan=0.0,
        phase_at_epoch=phase_at_epoch,
        epoch=TEST_EPOCH,
    )


def make_terminal(
    terminal_id: int,
    *,
    azimuth_min: float = -np.pi,
    azimuth_max: float = np.pi,
    elevation_min: float = -np.pi / 2,
    elevation_max: float = np.pi / 2,
    angular_velocity_max: float = 1.0,
    dcm_body2terminal: DirectionCosineMatrix = _IDENTITY_DCM,
) -> OpticalCommunicationTerminal[int]:
    return OpticalCommunicationTerminal(
        id=terminal_id,
        azimuth_min=azimuth_min,
        azimuth_max=azimuth_max,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        angular_velocity_max=angular_velocity_max,
        dcm_body2terminal=dcm_body2terminal,
    )


def make_satellite(
    satellite_id: int,
    *,
    phase_at_epoch: float = 0.0,
    terminals: tuple[OpticalCommunicationTerminal[int], ...] = (),
) -> ConstellationSatellite[int]:
    return ConstellationSatellite(
        id=satellite_id,
        orbit=make_orbit(phase_at_epoch=phase_at_epoch),
        terminals=terminals,
    )
