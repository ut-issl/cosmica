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
from pymap3d.sidereal import greenwichsrt as _greenwichsrt


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
    if isinstance(time, np.datetime64):
        return float(_greenwichsrt(juliandate(time)))

    julian_dates = juliandate(time)
    result = np.fromiter(
        (_greenwichsrt(float(date)) for date in julian_dates.flat),
        dtype=np.float64,
        count=julian_dates.size,
    )
    return result.reshape(julian_dates.shape)


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
