__all__ = [
    "StationaryOnGroundUser",
    "User",
]
from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass, field

from .node import Node
from .terminal import CommunicationTerminal


@dataclass(frozen=True, kw_only=True, slots=True)
class User[T: Hashable](Node[T], ABC):
    """Base model for a user."""

    id: T
    terminals: list[CommunicationTerminal[Hashable]] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True, slots=True)
class StationaryOnGroundUser[T: Hashable](User[T]):
    """Model for a stationary user on the ground."""

    id: T
    latitude: float = field(compare=False)
    longitude: float = field(compare=False)
    altitude: float = field(default=0.0, compare=False)
    minimum_elevation: float = field(compare=False)
    terminals: list[CommunicationTerminal[Hashable]] = field(default_factory=list, compare=False)
