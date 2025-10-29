__all__ = [
    "angle_between",
    "as_column_vector",
    "azimuth_elevation_to_unit_vector",
    "decompose_wrt_reference_vector",
    "generate_normal_vectors",
    "is_column_vector",
    "is_satellite_in_eclipse",
    "normalize",
    "perturb_vector",
    "project_vector",
    "rowwise_innerdot",
    "rowwise_matmul",
    "unit_vector_to_azimuth_elevation",
]

import logging
import math
from typing import Annotated, Any, Literal, overload

import numpy as np
import numpy.typing as npt
from numpy import linalg as LA  # noqa: N812
from numpy._typing import _64Bit
from typing_extensions import Doc

from .constants import EARTH_RADIUS

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)


def is_column_vector(v: npt.NDArray[np.number]) -> bool:
    return v.shape[-1] == 1 and v.ndim > 1


def as_column_vector[NumberType: np.number](v: npt.NDArray[NumberType]) -> npt.NDArray[NumberType]:
    if is_column_vector(v):  # v is already a column vector.
        return v
    else:
        return v[..., np.newaxis]


def rowwise_matmul(x1: npt.NDArray, x2: npt.NDArray) -> Any:
    return x2 @ x1.T


def rowwise_innerdot[NumberType: np.number](
    x1: npt.NDArray[NumberType],
    x2: npt.NDArray[NumberType],
    *,
    keepdims: bool = False,
) -> npt.NDArray[NumberType]:
    return np.sum(x1 * x2, axis=-1, keepdims=keepdims)


@overload
def normalize[S: npt.NBitBase](
    x: npt.NDArray[np.integer[S]],
    ord: float | Literal["fro", "nuc"] | None = None,
    axis: int | None = None,
) -> npt.NDArray[np.floating[S]]: ...


@overload
def normalize[S: npt.NBitBase](
    x: npt.NDArray[np.floating[S]],
    ord: float | Literal["fro", "nuc"] | None = None,
    axis: int | None = None,
) -> npt.NDArray[np.floating[S]]: ...


def normalize[S: npt.NBitBase](
    x,
    ord=None,  # noqa: A002
    axis=None,
):
    norm = LA.norm(x, ord=ord, axis=axis, keepdims=True)
    return np.where(norm == 0.0, x, x / norm)


def angle_between[S: npt.NBitBase, T: npt.NBitBase](
    x1: npt.NDArray[np.floating[S]] | npt.NDArray[np.integer[S]],
    x2: npt.NDArray[np.floating[T]] | npt.NDArray[np.integer[T]],
) -> npt.NDArray[np.floating[S | T | _64Bit]]:
    """Return the angle in radians between vectors x1 and x2.

    Ref: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249.
    """
    x1_unit = normalize(x1, axis=-1)
    x2_unit = normalize(x2, axis=-1)

    return np.arccos(np.clip(rowwise_innerdot(x1_unit, x2_unit), -1.0, 1.0))


def generate_normal_vectors(
    vec: npt.NDArray[np.floating],
    seed_vec: npt.NDArray[np.floating] | None = None,
    seed_vec_backup: npt.NDArray[np.floating] | None = None,
    backup_threshold: float = 0.017,  # [rad] approx 1 degree.
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    seed_vec = np.array([1.0, 0.0, 0.0]) if seed_vec is None else seed_vec
    seed_vec_backup = np.array([0.0, 1.0, 0.0]) if seed_vec_backup is None else seed_vec_backup
    assert vec.shape == seed_vec.shape == seed_vec_backup.shape == (3,)
    if angle_between(vec, seed_vec) <= backup_threshold:
        seed_vec = seed_vec_backup

    e1 = normalize(np.cross(seed_vec, vec))
    e2 = normalize(np.cross(vec, e1))
    return e1, e2


def perturb_vector(
    vec: npt.NDArray[np.floating],
    phase_angle: float,  # [rad]
    alignment_angle: float,  # [rad]
    seed_vec: npt.NDArray[np.floating] | None = None,
    seed_vec_backup: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    if LA.norm(vec) == 0.0:
        return np.zeros_like(vec, dtype=np.float64)

    e1, e2 = generate_normal_vectors(vec, seed_vec, seed_vec_backup)
    norm = LA.norm(vec)
    e3 = vec / norm

    direction = (
        math.cos(phase_angle) * math.sin(alignment_angle) * e1
        + math.sin(phase_angle) * math.sin(alignment_angle) * e2
        + math.cos(alignment_angle) * e3
    )
    return direction * norm


def project_vector[NumberType: np.number](
    vec: npt.NDArray[NumberType],  # (n_data, dim) or (dim,)
    onto: npt.NDArray[NumberType],  # (n_data, dim) or (dim,)
) -> npt.NDArray[np.floating]:  # (n_data, dim) or (dim,)
    # Make sure vec and onto both have shape (n_data, dim)
    if vec.ndim == 1 and onto.ndim == 1:
        n_data = 1
        assert vec.shape[-1] == onto.shape[-1]
        dim = vec.shape[-1]
        returned_shape = vec.shape
    elif vec.ndim == 1 and onto.ndim == 2:
        n_data = onto.shape[-2]
        assert vec.shape[-1] == onto.shape[-1]
        dim = vec.shape[-1]
        returned_shape = onto.shape
    elif vec.ndim == 2 and onto.ndim == 1:
        n_data = vec.shape[-2]
        assert vec.shape[-1] == onto.shape[-1]
        dim = vec.shape[-1]
        returned_shape = vec.shape
    elif vec.ndim == 2 and onto.ndim == 2:
        assert vec.shape == onto.shape
        n_data, dim = vec.shape
        returned_shape = vec.shape
    else:
        msg = "vec should be either 2- or 1-dimensional vector."
        raise ValueError(msg)
    vec = np.broadcast_to(vec, (n_data, dim))
    onto = np.broadcast_to(onto, (n_data, dim))

    projected_vec = np.sum(vec * onto, axis=-1, keepdims=True) / np.sum(onto * onto, axis=-1, keepdims=True) * onto
    return projected_vec.reshape(returned_shape)


def decompose_wrt_reference_vector[S: npt.NBitBase, T: npt.NBitBase](
    vec: npt.NDArray[np.floating[S]],
    ref: npt.NDArray[np.floating[T]],
) -> tuple[npt.NDArray[np.floating[S | T]], npt.NDArray[np.floating[S | T]]]:
    ref = normalize(ref, axis=-1)

    parallel_element = np.sum(vec * ref, axis=-1)
    normal_element = np.sqrt(np.sum(vec**2, axis=-1) - parallel_element**2)

    return parallel_element, normal_element


def closest_point_to_origin_on_line(
    r1: Annotated[npt.NDArray[np.floating], Doc("Numpy arrays representing the position vectors of the first point.")],
    r2: Annotated[npt.NDArray[np.floating], Doc("Numpy arrays representing the position vectors of the second point.")],
    *,
    extend_at_r1: Annotated[
        bool,
        Doc("If True, extend the line segment at r1. Otherwise, do not extend the line segment at r1."),
    ] = True,
    extend_at_r2: Annotated[
        bool,
        Doc("If True, extend the line segment at r2. Otherwise, do not extend the line segment at r2."),
    ] = True,
) -> Annotated[
    npt.NDArray[np.floating],
    Doc("Numpy array representing the position vector of the point on the line segment closest to the origin."),
]:
    """Compute the point on the line segment between r1 and r2 that is closest to the origin."""
    # Compute t
    numerator = rowwise_innerdot(r2 - r1, r1, keepdims=True)
    denominator = rowwise_innerdot(r2 - r1, r2 - r1, keepdims=True)
    t = -numerator / denominator

    t = np.where(extend_at_r1, t, np.clip(t, 0.0, None))
    t = np.where(extend_at_r2, t, np.clip(t, None, 1.0))

    # Compute r*
    return r1 + t * (r2 - r1)


def azimuth_elevation_to_unit_vector(
    az: Annotated[float, Doc("Float value with azimuth angle in radians")],
    el: Annotated[float, Doc("Float value with elevation angle in radians")],
) -> Annotated[
    npt.NDArray[np.floating],
    Doc("Numpy array with unitary x,y,z directions"),
]:
    z = np.sin(el)
    y = np.cos(el) * np.sin(az)
    x = np.cos(el) * np.cos(az)
    return np.array([x, y, z])


def unit_vector_to_azimuth_elevation(
    u: Annotated[
        npt.NDArray[np.floating],
        Doc("Numpy array of floats with the coordinates of a unit vector in three dimensions"),
    ],
) -> Annotated[
    tuple[float, float],
    Doc(
        "Tuple with two floats. The first one represents the corresponding azimuth value in radians, and the second the elevation value in radians",  # noqa: E501
    ),
]:
    assert u.shape[0] == 3

    el = np.arcsin(u[2])
    az = np.arctan2(u[1], u[0])

    return (az, el)


def is_satellite_in_eclipse(
    satellite_position_eci: npt.NDArray[np.floating],
    sun_direction_eci: npt.NDArray[np.floating],
) -> bool:
    """Check if a single satellite is in Earth's shadow (eclipse).

    Args:
        satellite_position_eci: Position of satellite in ECI frame. Shape: (3,)
        sun_direction_eci: Sun direction vector in ECI frame. Shape: (3,)

    Returns:
        True if the satellite is in Earth's eclipse

    """
    # Vector from satellite to sun (opposite of sun direction)
    sat_to_sun = -normalize(sun_direction_eci) * np.linalg.norm(satellite_position_eci) * 2

    # Check if the line from satellite to sun intersects with Earth
    # Using the minimum distance from Earth center to the line
    sat_pos_norm = np.linalg.norm(satellite_position_eci)
    if sat_pos_norm == 0:
        return False

    # Project Earth center (origin) onto the line from satellite to sun
    sat_to_sun_unit = normalize(sat_to_sun)
    proj_length = np.dot(-satellite_position_eci, sat_to_sun_unit)

    # If projection is behind the satellite or beyond the sun, no eclipse
    sun_distance = np.linalg.norm(sat_to_sun)
    if proj_length < 0 or proj_length > sun_distance:
        return False

    # Find the closest point on the line to Earth center
    closest_point = satellite_position_eci + proj_length * sat_to_sun_unit
    distance_to_earth_center = np.linalg.norm(closest_point)

    # If this distance is less than Earth radius, there's an eclipse
    return bool(distance_to_earth_center < EARTH_RADIUS)
