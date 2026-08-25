__all__ = [
    "CommunicationTerminal",
    "OpticalCommunicationTerminal",
    "RFCommunicationTerminal",
    "TerminalOwner",
    "UserOpticalCommunicationTerminal",
]
from collections.abc import Collection, Hashable
from dataclasses import dataclass
from typing import Protocol, override, runtime_checkable

from .node import Node


@dataclass(frozen=True, kw_only=True, slots=True)
class CommunicationTerminal[T: Hashable](Node[T]):
    id: T

    @classmethod
    @override
    def class_name(cls) -> str:
        return "CT"


@dataclass(frozen=True, kw_only=True, slots=True)
class OpticalCommunicationTerminal[T: Hashable](CommunicationTerminal[T]):
    azimuth_min: float
    azimuth_max: float
    elevation_min: float
    elevation_max: float
    angular_velocity_max: float

    @classmethod
    @override
    def class_name(cls) -> str:
        return "OCT"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserOpticalCommunicationTerminal[T: Hashable](OpticalCommunicationTerminal[T]):
    @classmethod
    @override
    def class_name(cls) -> str:
        return "UOCT"


@dataclass(frozen=True, kw_only=True, slots=True)
class RFCommunicationTerminal[T: Hashable](CommunicationTerminal[T]):
    @classmethod
    @override
    def class_name(cls) -> str:
        return "RFCT"


@runtime_checkable
class TerminalOwner(Protocol):
    """Forwarding entity that owns communication-terminal resources."""

    terminals: Collection[CommunicationTerminal]
