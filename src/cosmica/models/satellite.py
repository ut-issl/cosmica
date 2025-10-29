__all__ = [
    "ConstellationSatellite",
    "Satellite",
    "SatelliteTerminal",
    "UserSatellite",
]
from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass
from typing import override

from .node import Node
from .orbit import SatelliteOrbitModel
from .terminal import OpticalCommunicationTerminal


class Satellite[T: Hashable](Node[T], ABC): ...


@dataclass(frozen=True, kw_only=True, slots=True)
class ConstellationSatellite[T: Hashable](Satellite[T]):
    id: T

    # Allow none for backward compatibility.
    orbit: SatelliteOrbitModel | None = None

    @classmethod
    @override
    def class_name(cls) -> str:
        return "CSAT"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserSatellite[T: Hashable](Satellite[T]):
    id: T

    @classmethod
    @override
    def class_name(cls) -> str:
        return "USAT"


@dataclass(frozen=True, kw_only=True, slots=True)
class SatelliteTerminal[T: Hashable](Satellite[T]):
    id: T
    terminal_id: T
    azimuth_min: float
    azimuth_max: float
    elevation_min: float
    elevation_max: float
    angular_velocity_max: float

    @property
    def terminal(self) -> OpticalCommunicationTerminal[T]:
        return OpticalCommunicationTerminal(
            id=self.terminal_id,
            azimuth_min=self.azimuth_min,
            azimuth_max=self.azimuth_max,
            elevation_min=self.elevation_min,
            elevation_max=self.elevation_max,
            angular_velocity_max=self.angular_velocity_max,
        )
