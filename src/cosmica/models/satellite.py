__all__ = [
    "ConstellationSatellite",
    "Satellite",
    "SatelliteTerminal",
    "UserSatellite",
]
from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Annotated, Any, override

from typing_extensions import Doc

from .node import Node
from .orbit import SatelliteOrbitModel
from .terminal import DirectionCosineMatrix, OpticalCommunicationTerminal


class Satellite[T: Hashable](Node[T], ABC): ...


@dataclass(frozen=True, kw_only=True, slots=True)
class ConstellationSatellite[T: Hashable, O: SatelliteOrbitModel = Any](Satellite[T]):
    id: T

    orbit: O = field(hash=False, compare=False)

    @classmethod
    @override
    def class_name(cls) -> str:
        return "CSAT"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserSatellite[T: Hashable, O: SatelliteOrbitModel = Any](Satellite[T]):
    id: T

    orbit: O = field(hash=False, compare=False)

    @classmethod
    @override
    def class_name(cls) -> str:
        return "USAT"


@dataclass(frozen=True, kw_only=True, slots=True)
class SatelliteTerminal[T: Hashable](Satellite[T]):
    """Satellite node representing one body-mounted optical terminal."""

    id: T
    terminal_id: T
    azimuth_min: float
    azimuth_max: float
    elevation_min: float
    elevation_max: float
    angular_velocity_max: float
    dcm_body2terminal: Annotated[
        DirectionCosineMatrix,
        Doc(
            "Right-handed direction-cosine matrix that transforms vector components "
            "from the satellite body frame to the terminal frame.",
        ),
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        _ = self._make_terminal()

    def _make_terminal(self) -> OpticalCommunicationTerminal[T]:
        return OpticalCommunicationTerminal(
            id=self.terminal_id,
            azimuth_min=self.azimuth_min,
            azimuth_max=self.azimuth_max,
            elevation_min=self.elevation_min,
            elevation_max=self.elevation_max,
            angular_velocity_max=self.angular_velocity_max,
            dcm_body2terminal=self.dcm_body2terminal,
        )

    @property
    def terminal(self) -> OpticalCommunicationTerminal[T]:
        """Return the optical-terminal model carried by this satellite node."""
        return self._make_terminal()
