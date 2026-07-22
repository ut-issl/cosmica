__all__ = [
    "calc_dcm_eci2ecef",
    "ecef2aer",
    "geodetic2ecef",
    "greenwichsrt",
    "juliandate",
]
from typing import overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from pymap3d.aer import ecef2aer  # re-exported as cosmica.utils.coordinates.ecef2aer
from pymap3d.ecef import geodetic2ecef


@overload
def juliandate(time: np.datetime64) -> float: ...


@overload
def juliandate(time: npt.NDArray[np.datetime64]) -> npt.NDArray[np.float64]: ...


def juliandate(
    time: np.datetime64 | npt.NDArray[np.datetime64],
) -> float | npt.NDArray[np.float64]:
    if isinstance(time, np.datetime64):
        return float(pd.Timestamp(time).to_julian_date())
    converted = pd.DatetimeIndex(time)
    return np.asarray(converted.to_julian_date(), dtype=np.float64)


@overload
def greenwichsrt(time: np.datetime64) -> float: ...


@overload
def greenwichsrt(time: npt.NDArray[np.datetime64]) -> npt.NDArray[np.float64]: ...


def greenwichsrt(
    time: np.datetime64 | npt.NDArray[np.datetime64],
) -> float | npt.NDArray[np.float64]:
    # Vallado, Fundamentals of Astrodynamics and Applications, 4th ed., Eq. 3-47.
    julian_date = juliandate(time)
    centuries_since_j2000 = (julian_date - 2451545.0) / 36525.0
    mean_sidereal_time_seconds = (
        67310.54841
        + (876600 * 3600 + 8640184.812866) * centuries_since_j2000
        + 0.093104 * centuries_since_j2000**2
        - 6.2e-6 * centuries_since_j2000**3
    )
    sidereal_time = mean_sidereal_time_seconds * (2 * np.pi) / 86400.0 % (2 * np.pi)
    if isinstance(time, np.datetime64):
        return float(sidereal_time)
    return np.asarray(sidereal_time, dtype=np.float64)


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
