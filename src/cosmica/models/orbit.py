__all__ = [
    "CircularSatelliteOrbitModel",
    "EllipticalSatelliteOrbitModel",
    "GravityModel",
]
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sgp4.model import WGS72, WGS72OLD, WGS84


class GravityModel(Enum):
    WGS72 = WGS72
    WGS72OLD = WGS72OLD
    WGS84 = WGS84


@dataclass(kw_only=True, slots=True, frozen=True)
class SatelliteOrbitModel:
    """Base model for a satellite orbit."""


@dataclass(kw_only=True, slots=True, frozen=True)
class CircularSatelliteOrbitModel(SatelliteOrbitModel):
    semi_major_axis: float
    inclination: float
    raan: float
    phase_at_epoch: float
    epoch: np.datetime64


@dataclass(kw_only=True, slots=True, frozen=True)
class EllipticalSatelliteOrbitModel(SatelliteOrbitModel):
    semi_major_axis: float
    inclination: float
    raan: float
    phase_at_epoch: float
    epoch: np.datetime64
    satnum: int  # Satellite number
    gravity_model: GravityModel = GravityModel.WGS84
    drag_coeff: float  # B star drag coeffient given in [1/earth radii]
    eccentricity: float
    argpo: float  # Argument of Perigee
