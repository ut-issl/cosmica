from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from cosmica.dtos import CommunicationLinkEndpoint, DirectedCommunicationLink, DynamicsData
from cosmica.models import (
    CircularSatelliteOrbitModel,
    ConstellationSatellite,
    DirectionCosineMatrix,
    OpticalCommunicationTerminal,
)
from cosmica.utils.constants import EARTH_RADIUS

TEST_EPOCH = np.datetime64("2026-01-01T00:00:00")

type OpticalSatelliteEndpoint = CommunicationLinkEndpoint[
    ConstellationSatellite[int],
    OpticalCommunicationTerminal[int],
]
type OpticalSatelliteLink = DirectedCommunicationLink[
    ConstellationSatellite[int],
    OpticalCommunicationTerminal[int],
    ConstellationSatellite[int],
    OpticalCommunicationTerminal[int],
]

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


def make_optical_satellite_endpoint(
    satellite_id: int,
    *,
    angular_velocity_max: float = float("inf"),
    azimuth_min: float = -np.pi,
    azimuth_max: float = np.pi,
    elevation_min: float = -np.pi / 2,
    elevation_max: float = np.pi / 2,
    dcm_body2terminal: DirectionCosineMatrix = _IDENTITY_DCM,
) -> OpticalSatelliteEndpoint:
    terminal = make_terminal(
        satellite_id,
        azimuth_min=azimuth_min,
        azimuth_max=azimuth_max,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        angular_velocity_max=angular_velocity_max,
        dcm_body2terminal=dcm_body2terminal,
    )
    satellite = make_satellite(satellite_id, terminals=(terminal,))
    return CommunicationLinkEndpoint(node=satellite, terminal=terminal)


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


def make_link_from_endpoints(
    source: OpticalSatelliteEndpoint,
    destination: OpticalSatelliteEndpoint,
) -> OpticalSatelliteLink:
    return DirectedCommunicationLink(source=source, destination=destination)


def make_link(
    source: ConstellationSatellite[int],
    source_terminal: OpticalCommunicationTerminal[int],
    destination: ConstellationSatellite[int],
    destination_terminal: OpticalCommunicationTerminal[int],
) -> OpticalSatelliteLink:
    return make_link_from_endpoints(
        CommunicationLinkEndpoint(node=source, terminal=source_terminal),
        CommunicationLinkEndpoint(node=destination, terminal=destination_terminal),
    )


def _repeat_dcm(dcm: DirectionCosineMatrix, sample_count: int) -> npt.NDArray[np.floating]:
    return np.repeat(np.asarray(dcm)[None, :, :], sample_count, axis=0)


def make_optical_link_dynamics_data(
    endpoint_a: OpticalSatelliteEndpoint,
    endpoint_b: OpticalSatelliteEndpoint,
    *,
    time: npt.NDArray[np.datetime64],
    line_of_sight_azimuths: Sequence[float] | npt.NDArray[np.floating] | None = None,
    dcm_eci2body_a: DirectionCosineMatrix = _IDENTITY_DCM,
    dcm_eci2body_b: DirectionCosineMatrix = _IDENTITY_DCM,
    include_endpoint_b_attitude: bool = True,
    sun_direction_eci: npt.NDArray[np.floating] | None = None,
) -> DynamicsData[ConstellationSatellite[int]]:
    sample_count = len(time)
    azimuths = np.asarray(
        line_of_sight_azimuths if line_of_sight_azimuths is not None else [np.pi / 2] * sample_count,
    )
    position_a = np.repeat(np.array([[EARTH_RADIUS + 1_000e3, 0.0, 0.0]]), sample_count, axis=0)
    line_of_sight = 100e3 * np.column_stack((np.cos(azimuths), np.sin(azimuths), np.zeros(sample_count)))
    position_b = position_a + line_of_sight
    zero = np.zeros((sample_count, 3))
    identity = _repeat_dcm(_IDENTITY_DCM, sample_count)
    satellite_a = endpoint_a.node
    satellite_b = endpoint_b.node
    positions = {satellite_a: position_a, satellite_b: position_b}
    attitudes = {satellite_a: _repeat_dcm(dcm_eci2body_a, sample_count)}
    if include_endpoint_b_attitude:
        attitudes[satellite_b] = _repeat_dcm(dcm_eci2body_b, sample_count)
    sun_direction = (
        np.repeat(np.array([[0.0, 0.0, 1.0]]), sample_count, axis=0)
        if sun_direction_eci is None
        else np.repeat(sun_direction_eci[None, :], sample_count, axis=0)
    )
    return DynamicsData(
        time=time,
        dcm_eci2ecef=identity,
        satellite_position_eci=positions,
        satellite_velocity_eci={satellite_a: zero, satellite_b: zero},
        satellite_position_ecef=positions,
        satellite_attitude_angular_velocity_eci={satellite_a: zero, satellite_b: zero},
        sun_direction_eci=sun_direction,
        sun_direction_ecef=sun_direction,
        satellite_attitude_dcm_eci2body=attitudes,
    )
