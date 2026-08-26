__all__ = [
    "ConstellationSatellite",
    "Satellite",
    "UserSatellite",
]
from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, override

from .node import Node
from .orbit import SatelliteOrbitModel
from .terminal import CommunicationTerminal, TerminalOwner


@dataclass(frozen=True, kw_only=True, slots=True)
class Satellite[T: Hashable](Node[T], TerminalOwner, ABC):
    """Physical satellite that owns communication-terminal resources."""

    id: T
    terminals: tuple[CommunicationTerminal, ...] = field(
        default_factory=tuple,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        terminal_ids = tuple(terminal.global_id for terminal in self.terminals)
        if len(terminal_ids) != len(set(terminal_ids)):
            msg = "satellite terminal global IDs must be unique"
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True, slots=True)
class ConstellationSatellite[T: Hashable, O: SatelliteOrbitModel = Any](Satellite[T]):
    orbit: O = field(hash=False, compare=False)

    @classmethod
    @override
    def class_name(cls) -> str:
        return "CSAT"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserSatellite[T: Hashable, O: SatelliteOrbitModel = Any](Satellite[T]):
    orbit: O = field(hash=False, compare=False)

    @classmethod
    @override
    def class_name(cls) -> str:
        return "USAT"
