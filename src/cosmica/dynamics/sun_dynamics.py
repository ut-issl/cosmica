from __future__ import annotations

__all__ = [
    "get_sun_direction_eci",
]
from typing import TYPE_CHECKING, Annotated, overload

import numpy as np
from typing_extensions import Doc

from cosmica.utils.coordinates import juliandate
from cosmica.utils.vector import normalize

if TYPE_CHECKING:
    import numpy.typing as npt


@overload
def get_sun_direction_eci(time: npt.NDArray[np.datetime64]) -> npt.NDArray[np.float64]: ...


@overload
def get_sun_direction_eci(time: np.datetime64) -> npt.NDArray[np.float64]: ...


def get_sun_direction_eci(
    time: Annotated[
        np.datetime64 | npt.NDArray[np.datetime64],
        Doc("single element or array of UTC time values in np.datetime64 format."),
    ],
) -> Annotated[
    npt.NDArray[np.float64],
    Doc(
        "Corresponding Sun direction vector(s) (x,y,z) in eci."
        " If the input is a single element, the output is a 1D array."
        " If the input is an array, the output is a 2D array with the shape (len(time), 3).",
    ),
]:
    """Sun direction approximation function.

    The approximation considers an unperturbed motion of the Earth around the Sun with mean orbital elements
    that approximate the Sun's elliptic orbit wrt Earth around the year 2000.
    The algorithm is based on Oliver Montenbruck, Eberhard Grill; "Satellite Orbits: Models, Methods and Applications";
    Springer-Verlag Berlin. The alterations to the original model follow JAXA's Attitude Control Handbook
    JERG-2-510-HB001.
    """
    # tdt: Number of Julian Centuries. J2000 epoch -> 2451545.0 TT
    tdt = (juliandate(time) - 2451545.0) / 36525.0

    # m: mean anomaly
    m = np.deg2rad(357.5256 + 35999.045 * tdt)

    # right-ascension + argument of perigee = 282.94 deg
    # s: Sun's ecliptic longitude
    s = m + np.deg2rad(282.94 + (6892.0 * np.sin(m) + 72.0 * np.sin(2.0 * m) + 1250.09115 * tdt) / 3600.0 - 0.002652)

    # Obliquity of the ecliptic in radians
    epsilon_deg = 23.43929111
    epsilon = np.deg2rad(epsilon_deg)

    # Compute Sun Vector @ J2000. Shape: (3,) or (3, len(time))
    sun_vec_eci = np.array([np.cos(s), np.sin(s) * np.cos(epsilon), np.sin(s) * np.sin(epsilon)], dtype=np.float64)

    # normalize
    sun_vec_eci = sun_vec_eci.T  # shape: (3,) or (len(time), 3)

    # shape: (3,) or (len(time), 3)
    return normalize(sun_vec_eci, axis=-1)  # type: ignore[return-value]
