__all__ = [
    "calc_gauss_beam_radius",
    "calc_gauss_rx_power",
]

import logging
from typing import Annotated

import numpy as np
from typing_extensions import Doc

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)


def calc_gauss_rx_power(
    power_tx: Annotated[float, Doc("Float value with transmitted power in Watts.")],
    aperture: Annotated[float, Doc("Float value with receiving aperture diameter in meters.")],
    distance: Annotated[float, Doc("Float value with the link distance in meters.")],
    wavelength: Annotated[float, Doc("Float value with the wavelength of the beam in meters.")] = 1550 * 1e-9,
    beam_waist: Annotated[float, Doc("Float value with the beam waist radius in meters.")] = 0.04,
) -> float:
    return power_tx * (np.exp(-(aperture**2) / 2 / calc_gauss_beam_radius(beam_waist, wavelength, distance) ** 2) - 1)


def calc_gauss_beam_radius(
    w0: Annotated[float, Doc("Float value with the beam waist radius in meters.")],
    wvl: Annotated[float, Doc("Float value with the beam wavelength in meters.")],
    z: Annotated[float, Doc("Float value with the distance 'z' from the source.")],
) -> float:
    return w0 * np.sqrt(1 + (z * wvl / (np.pi * w0**2)))
