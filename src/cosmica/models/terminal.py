__all__ = [
    "CommunicationTerminal",
    "OpticalCommunicationTerminal",
    "RFCommunicationTerminal",
    "UserOpticalCommunicationTerminal",
]
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

from .node import Node

_T = TypeVar("_T", bound=Hashable)


@dataclass(frozen=True, kw_only=True, slots=True)
class CommunicationTerminal(Node[_T]):
    id: _T

    @classmethod
    def class_name(cls) -> str:
        return "CT"


@dataclass(frozen=True, kw_only=True, slots=True)
class OpticalCommunicationTerminal(CommunicationTerminal[_T]):
    azimuth_min: float
    azimuth_max: float
    elevation_min: float
    elevation_max: float
    angular_velocity_max: float

    @classmethod
    def class_name(cls) -> str:
        return "OCT"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserOpticalCommunicationTerminal(OpticalCommunicationTerminal[_T]):
    @classmethod
    def class_name(cls) -> str:
        return "UOCT"


@dataclass(frozen=True, kw_only=True, slots=True)
class RFCommunicationTerminal(CommunicationTerminal[_T]):
    @classmethod
    def class_name(cls) -> str:
        return "RFCT"
