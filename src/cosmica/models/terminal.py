__all__ = [
    "CommunicationTerminal",
    "DirectionCosineMatrix",
    "OpticalCommunicationTerminal",
    "RFCommunicationTerminal",
    "UserOpticalCommunicationTerminal",
]
import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Annotated, override

import numpy as np
from typing_extensions import Doc

from .node import Node

# Terminal models are frozen, hashable graph nodes and dictionary keys, so mounting
# matrices use immutable value-equality tuples rather than mutable, unhashable NumPy arrays.
type DirectionCosineMatrix = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
"""Immutable 3-by-3 direction-cosine matrix."""


_IDENTITY_DCM: DirectionCosineMatrix = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _validate_direction_cosine_matrix(dcm: DirectionCosineMatrix, *, name: str) -> None:
    matrix = np.asarray(dcm, dtype=float)
    if matrix.shape != (3, 3):
        msg = f"{name} must have shape (3, 3), but got {matrix.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if not np.allclose(matrix @ matrix.T, np.eye(3), rtol=0.0, atol=1e-12) or not np.isclose(
        np.linalg.det(matrix),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        msg = f"{name} must be an orthonormal, right-handed direction-cosine matrix"
        raise ValueError(msg)


def _validate_optical_terminal_fields(
    *,
    azimuth_min: float,
    azimuth_max: float,
    elevation_min: float,
    elevation_max: float,
    angular_velocity_max: float,
    dcm_body2terminal: DirectionCosineMatrix,
) -> None:
    if not -math.pi <= azimuth_min <= azimuth_max <= math.pi:
        msg = "azimuth bounds must define a non-wrapping interval within [-pi, pi]"
        raise ValueError(msg)
    if not -math.pi / 2 <= elevation_min <= elevation_max <= math.pi / 2:
        msg = "elevation bounds must define an interval within [-pi/2, pi/2]"
        raise ValueError(msg)
    if math.isnan(angular_velocity_max) or angular_velocity_max < 0.0:
        msg = "angular_velocity_max must be non-negative"
        raise ValueError(msg)
    _validate_direction_cosine_matrix(dcm_body2terminal, name="dcm_body2terminal")


@dataclass(frozen=True, kw_only=True, slots=True)
class CommunicationTerminal[T: Hashable](Node[T]):
    id: T

    @classmethod
    @override
    def class_name(cls) -> str:
        return "CT"


@dataclass(frozen=True, kw_only=True, slots=True)
class OpticalCommunicationTerminal[T: Hashable](CommunicationTerminal[T]):
    """Optical terminal with field-of-regard, slew-rate, and body-mounting constraints.

    Azimuth and elevation are expressed in the terminal frame. The terminal-frame
    ``+x`` axis has zero azimuth and elevation, positive azimuth rotates toward
    ``+y``, and positive elevation rotates toward ``+z``. Azimuth bounds must be a
    non-wrapping interval in ``[-pi, pi]``; elevation bounds must lie in
    ``[-pi/2, pi/2]``.
    """

    azimuth_min: Annotated[float, Doc("Minimum terminal-frame azimuth in radians, inclusive.")]
    azimuth_max: Annotated[float, Doc("Maximum terminal-frame azimuth in radians, inclusive.")]
    elevation_min: Annotated[float, Doc("Minimum terminal-frame elevation in radians, inclusive.")]
    elevation_max: Annotated[float, Doc("Maximum terminal-frame elevation in radians, inclusive.")]
    angular_velocity_max: Annotated[
        float,
        Doc("Maximum absolute azimuth or elevation slew rate in radians per second, inclusive."),
    ]
    dcm_body2terminal: Annotated[
        DirectionCosineMatrix,
        Doc(
            "Right-handed direction-cosine matrix that transforms vector components "
            "from the satellite body frame to the terminal frame.",
        ),
    ] = _IDENTITY_DCM

    def __post_init__(self) -> None:
        _validate_optical_terminal_fields(
            azimuth_min=self.azimuth_min,
            azimuth_max=self.azimuth_max,
            elevation_min=self.elevation_min,
            elevation_max=self.elevation_max,
            angular_velocity_max=self.angular_velocity_max,
            dcm_body2terminal=self.dcm_body2terminal,
        )

    @classmethod
    @override
    def class_name(cls) -> str:
        return "OCT"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserOpticalCommunicationTerminal[T: Hashable](OpticalCommunicationTerminal[T]):
    @classmethod
    @override
    def class_name(cls) -> str:
        return "UOCT"


@dataclass(frozen=True, kw_only=True, slots=True)
class RFCommunicationTerminal[T: Hashable](CommunicationTerminal[T]):
    @classmethod
    @override
    def class_name(cls) -> str:
        return "RFCT"
