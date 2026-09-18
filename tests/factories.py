import numpy as np

from cosmica.dtos import CommunicationLinkEndpoint, DirectedCommunicationLink, DynamicsData
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


def make_dynamics_data(time: np.ndarray) -> DynamicsData:
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


def make_attitude_dynamics_data(attitude_dcm: np.ndarray) -> DynamicsData:
    satellite = make_satellite(1)
    sample_count = len(attitude_dcm)
    vectors = np.zeros((sample_count, 3))
    return DynamicsData(
        time=TEST_EPOCH + np.arange(sample_count).astype("timedelta64[s]"),
        dcm_eci2ecef=np.repeat(np.eye(3)[None, :, :], sample_count, axis=0),
        satellite_position_eci={satellite: vectors},
        satellite_velocity_eci={satellite: vectors},
        satellite_position_ecef={satellite: vectors},
        satellite_attitude_angular_velocity_eci={satellite: vectors},
        sun_direction_eci=vectors,
        sun_direction_ecef=vectors,
        satellite_attitude_dcm_eci2body={satellite: attitude_dcm},
    )


def make_link(
    source: ConstellationSatellite[int],
    source_terminal: OpticalCommunicationTerminal[int],
    destination: ConstellationSatellite[int],
    destination_terminal: OpticalCommunicationTerminal[int],
) -> DirectedCommunicationLink:
    return DirectedCommunicationLink(
        source=CommunicationLinkEndpoint(node=source, terminal=source_terminal),
        destination=CommunicationLinkEndpoint(node=destination, terminal=destination_terminal),
    )
