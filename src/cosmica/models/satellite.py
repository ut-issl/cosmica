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
from .terminal import (
    CommunicationTerminal,
    DirectionCosineMatrix,
    OpticalCommunicationTerminal,
    TerminalOwner,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class Satellite[T: Hashable](Node[T], TerminalOwner, ABC):
    """Physical satellite that owns communication-terminal resources."""

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


@dataclass(frozen=True, kw_only=True, slots=True)
class SatelliteTerminal[T: Hashable](Satellite[T]):
    """Legacy satellite node representing one body-mounted optical terminal."""

    id: T
    terminal_id: T
    azimuth_min: float
    azimuth_max: float
    elevation_min: float
    elevation_max: float
    angular_velocity_max: float
    dcm_body2terminal: Annotated[
        DirectionCosineMatrix,
        Doc("Direction-cosine matrix transforming body components to terminal components."),
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        Satellite.__post_init__(self)
        _ = self.terminal

    @property
    def terminal(self) -> OpticalCommunicationTerminal[T]:
        return OpticalCommunicationTerminal(
            id=self.terminal_id,
            azimuth_min=self.azimuth_min,
            azimuth_max=self.azimuth_max,
            elevation_min=self.elevation_min,
            elevation_max=self.elevation_max,
            angular_velocity_max=self.angular_velocity_max,
            dcm_body2terminal=self.dcm_body2terminal,
        )
