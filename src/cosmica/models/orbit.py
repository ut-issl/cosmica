__all__ = [
    "CircularSatelliteOrbitModel",
    "EllipticalSatelliteOrbitModel",
    "GravityModel",
]
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict
from sgp4.model import WGS72, WGS72OLD, WGS84


class GravityModel(Enum):
    WGS72 = WGS72
    WGS72OLD = WGS72OLD
    WGS84 = WGS84


class SatelliteOrbitModel(BaseModel):
    """Base model for a satellite orbit."""


class CircularSatelliteOrbitModel(SatelliteOrbitModel):
    # arbitrary_types_allowed is required to allow numpy.datetime64
    model_config = ConfigDict(arbitrary_types_allowed=True)

    semi_major_axis: float
    inclination: float
    raan: float
    phase_at_epoch: float
    epoch: np.datetime64


class EllipticalSatelliteOrbitModel(SatelliteOrbitModel):
    # arbitrary_types_allowed is required to allow numpy.datetime64
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
