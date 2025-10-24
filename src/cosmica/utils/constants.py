__all__ = [
    "BOLTZ_CONST",
    "EARTH_MU",
    "EARTH_RADIUS",
    "EARTH_ROTATION_RATE",
]
import math
from typing import Final

# Earth gravitational constant from GRS80 model: 3.986005e14 m3/s2.
EARTH_MU: Final[float] = 3.986005e14  # [m3/s2]

# Earth equatorial radius as defined by IAU 2015 resolution B3: 6.3781e6 (m).
EARTH_RADIUS: Final[float] = 6.378137e6  # [m]

# Earth rotation rate
EARTH_ROTATION_RATE: Final[float] = 2 * math.pi / 86164.0905  # [rad/s]

SPEED_OF_LIGHT: Final[float] = 299_792_458.0  # [m/s]

# Boltzmann Constant
BOLTZ_CONST: Final[float] = 1.380649e-23  # [J/K]
