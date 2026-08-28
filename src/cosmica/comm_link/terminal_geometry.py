__all__ = [
    "TerminalAngularRate",
    "TerminalPointing",
    "calc_terminal_angular_rate",
    "calc_terminal_pointing",
    "is_pointing_rate_within_limit",
    "is_pointing_within_field_of_regard",
    "shortest_angular_difference",
]

import math
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import numpy.typing as npt
from typing_extensions import Doc

from cosmica.models import OpticalCommunicationTerminal
from cosmica.utils.vector import unit_vector_to_azimuth_elevation


@dataclass(frozen=True, slots=True)
class TerminalPointing:
    """Azimuth and elevation of a line of sight in terminal coordinates."""

    azimuth: Annotated[float, Doc("Terminal-frame azimuth in radians.")]
    elevation: Annotated[float, Doc("Terminal-frame elevation in radians.")]


@dataclass(frozen=True, slots=True)
class TerminalAngularRate:
    """Absolute terminal-axis slew rates between consecutive snapshots."""

    azimuth: Annotated[float, Doc("Absolute azimuth slew rate in radians per second.")]
    elevation: Annotated[float, Doc("Absolute elevation slew rate in radians per second.")]


def calc_terminal_pointing(
    line_of_sight_eci: Annotated[
        npt.NDArray[np.floating],
        Doc("Non-zero line-of-sight vector expressed in ECI coordinates. Shape: (3,)."),
    ],
    *,
    dcm_eci2body: Annotated[
        npt.NDArray[np.floating],
        Doc("Direction-cosine matrix transforming ECI components to satellite body components. Shape: (3, 3)."),
    ],
    terminal: Annotated[
        OpticalCommunicationTerminal,
        Doc("Body-mounted optical terminal whose mounting transform defines the terminal frame."),
    ],
) -> Annotated[TerminalPointing, Doc("Line-of-sight azimuth and elevation in the terminal frame.")]:
    """Transform an ECI line of sight into terminal-frame azimuth and elevation."""
    if line_of_sight_eci.shape != (3,):
        msg = f"line_of_sight_eci must have shape (3,), but got {line_of_sight_eci.shape}"
        raise ValueError(msg)
    if dcm_eci2body.shape != (3, 3):
        msg = f"dcm_eci2body must have shape (3, 3), but got {dcm_eci2body.shape}"
        raise ValueError(msg)

    line_of_sight_norm = float(np.linalg.norm(line_of_sight_eci))
    if not np.isfinite(line_of_sight_norm) or line_of_sight_norm == 0.0:
        msg = "line_of_sight_eci must be finite and non-zero"
        raise ValueError(msg)

    line_of_sight_unit_eci = line_of_sight_eci / line_of_sight_norm
    dcm_body2terminal = np.asarray(terminal.dcm_body2terminal, dtype=float)
    line_of_sight_unit_terminal = dcm_body2terminal @ dcm_eci2body @ line_of_sight_unit_eci
    azimuth, elevation = unit_vector_to_azimuth_elevation(line_of_sight_unit_terminal)
    return TerminalPointing(azimuth=float(azimuth), elevation=float(elevation))


def shortest_angular_difference(
    current: Annotated[float, Doc("Current angle in radians.")],
    previous: Annotated[float, Doc("Previous angle in radians.")],
) -> Annotated[float, Doc("Signed shortest rotation from the previous angle to the current angle, in radians.")]:
    """Return the signed shortest angular difference in ``[-pi, pi]``."""
    return math.remainder(current - previous, math.tau)


def calc_terminal_angular_rate(
    current: Annotated[TerminalPointing, Doc("Current terminal pointing angles.")],
    previous: Annotated[TerminalPointing, Doc("Terminal pointing angles at the preceding snapshot.")],
    *,
    time_delta: Annotated[float, Doc("Positive elapsed time between snapshots in seconds.")],
) -> Annotated[TerminalAngularRate, Doc("Absolute azimuth and elevation slew rates.")]:
    """Calculate absolute terminal-axis rates using a wrapped azimuth difference."""
    if not np.isfinite(time_delta) or time_delta <= 0.0:
        msg = "time_delta must be finite and positive"
        raise ValueError(msg)

    return TerminalAngularRate(
        azimuth=abs(shortest_angular_difference(current.azimuth, previous.azimuth)) / time_delta,
        elevation=abs(current.elevation - previous.elevation) / time_delta,
    )


def is_pointing_within_field_of_regard(
    pointing: Annotated[TerminalPointing, Doc("Pointing angles expressed in the terminal frame.")],
    terminal: Annotated[OpticalCommunicationTerminal, Doc("Terminal providing inclusive field-of-regard bounds.")],
) -> bool:
    """Check whether pointing lies within both terminal-axis bounds."""
    return bool(
        terminal.azimuth_min <= pointing.azimuth <= terminal.azimuth_max
        and terminal.elevation_min <= pointing.elevation <= terminal.elevation_max,
    )


def is_pointing_rate_within_limit(
    rate: Annotated[TerminalAngularRate, Doc("Absolute terminal-axis slew rates.")],
    terminal: Annotated[OpticalCommunicationTerminal, Doc("Terminal providing the inclusive slew-rate limit.")],
) -> bool:
    """Check both terminal-axis rates against the terminal slew-rate limit."""
    return bool(rate.azimuth <= terminal.angular_velocity_max and rate.elevation <= terminal.angular_velocity_max)
