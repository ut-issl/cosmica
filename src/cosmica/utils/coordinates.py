__all__ = [
    "calc_dcm_eci2ecef",
    "ecef2aer",
    "geodetic2ecef",
    "great_circle_distance",
    "greenwichsrt",
    "juliandate",
]
from typing import Annotated, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from pymap3d.aer import ecef2aer  # re-exported as cosmica.utils.coordinates.ecef2aer
from pymap3d.ecef import geodetic2ecef
from pymap3d.sidereal import greenwichsrt as _greenwichsrt
from typing_extensions import Doc

from .constants import EARTH_RADIUS


def great_circle_distance(
    lat1: Annotated[npt.ArrayLike, Doc("Latitude of the first point(s) in radians.")],
    lon1: Annotated[npt.ArrayLike, Doc("Longitude of the first point(s) in radians.")],
    lat2: Annotated[npt.ArrayLike, Doc("Latitude of the second point(s) in radians.")],
    lon2: Annotated[npt.ArrayLike, Doc("Longitude of the second point(s) in radians.")],
    *,
    radius: Annotated[float, Doc("Sphere radius in meters. Defaults to the Earth radius.")] = EARTH_RADIUS,
) -> Annotated[
    npt.NDArray[np.float64],
    Doc("Great-circle distance(s) in meters. Inputs are broadcast against each other."),
]:
    """Calculate the great-circle distance between points on a sphere using the haversine formula.

    All angles are in radians. Inputs broadcast, so a distance matrix between N and M points
    can be computed by passing arrays with shapes (N, 1) and (M,).
    """
    lat1, lon1, lat2, lon2 = (np.asarray(x, dtype=np.float64) for x in (lat1, lon1, lat2, lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return radius * c


@overload
def juliandate(time: np.datetime64) -> float: ...


@overload
def juliandate(time: npt.NDArray[np.datetime64]) -> npt.NDArray[np.float64]: ...


def juliandate(time):
    return pd.to_datetime(time).to_julian_date()


@overload
def greenwichsrt(time: np.datetime64) -> float: ...


@overload
def greenwichsrt(time: npt.NDArray[np.datetime64]) -> npt.NDArray[np.float64]: ...


def greenwichsrt(time):
    return _greenwichsrt(juliandate(time))


def calc_dcm_eci2ecef(time: np.datetime64 | npt.NDArray[np.datetime64]) -> npt.NDArray[np.float64]:
    """Calculate the direction cosine matrix from ECI to ECEF.

    Input is UTC time.
    If the input is a single time, the output is a 3x3 matrix.
    If the input is an array of times, the output is a Nx3x3 array.
    """
    gst = greenwichsrt(time)
    if isinstance(gst, float):
        return np.array(
            [
                [np.cos(gst), np.sin(gst), 0],
                [-np.sin(gst), np.cos(gst), 0],
                [0, 0, 1],
            ],
        )
    else:
        return np.stack(
            [
                np.stack(
                    [
                        np.cos(gst),
                        np.sin(gst),
                        np.zeros_like(gst),
                    ],
                    axis=-1,
                ),
                np.stack(
                    [
                        -np.sin(gst),
                        np.cos(gst),
                        np.zeros_like(gst),
                    ],
                    axis=-1,
                ),
                np.stack(
                    [
                        np.zeros_like(gst),
                        np.zeros_like(gst),
                        np.ones_like(gst),
                    ],
                    axis=-1,
                ),
            ],
            axis=-2,
        )
