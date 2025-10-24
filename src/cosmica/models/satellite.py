__all__ = [
    "ConstellationSatellite",
    "Satellite",
    "SatelliteTerminal",
    "UserSatellite",
]
from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

from .node import Node
from .orbit import SatelliteOrbitModel
from .terminal import OpticalCommunicationTerminal

_T = TypeVar("_T", bound=Hashable)


class Satellite(Node[_T], ABC): ...


@dataclass(frozen=True, kw_only=True, slots=True)
class ConstellationSatellite(Satellite[_T]):
    id: _T

    # Allow none for backward compatibility.
    orbit: SatelliteOrbitModel | None = None

    @classmethod
    def class_name(cls) -> str:
        return "CSAT"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserSatellite(Satellite[_T]):
    id: _T

    @classmethod
    def class_name(cls) -> str:
        return "USAT"


@dataclass(frozen=True, kw_only=True, slots=True)
class SatelliteTerminal(Satellite[_T]):
    id: _T
    terminal_id: _T
    azimuth_min: float
    azimuth_max: float
    elevation_min: float
    elevation_max: float
    angular_velocity_max: float

    @property
    def terminal(self) -> OpticalCommunicationTerminal:
        return OpticalCommunicationTerminal(
            id=self.terminal_id,
            azimuth_min=self.azimuth_min,
            azimuth_max=self.azimuth_max,
            elevation_min=self.elevation_min,
            elevation_max=self.elevation_max,
            angular_velocity_max=self.angular_velocity_max,
        )
