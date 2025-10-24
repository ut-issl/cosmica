__all__ = [
    "StationaryOnGroundUser",
    "User",
    "load_users_from_toml_file",
    "parse_user_item",
]
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, TypeVar

import numpy as np

from .node import Node
from .terminal import CommunicationTerminal

_T = TypeVar("_T", bound=Hashable)


@dataclass(frozen=True, kw_only=True, slots=True)
class User(Node[_T], ABC):
    """Base model for a user."""

    id: _T
    terminals: list[CommunicationTerminal] = field(default_factory=list)

    @classmethod
    @abstractmethod
    def parse_user_item(cls, item: Mapping[str, Any]) -> Self:
        """Parse a user item."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class StationaryOnGroundUser(User[_T]):
    """Model for a stationary user on the ground."""

    id: _T
    latitude: float = field(compare=False)
    longitude: float = field(compare=False)
    altitude: float = field(default=0.0, compare=False)
    minimum_elevation: float = field(compare=False)
    terminals: list[CommunicationTerminal] = field(default_factory=list, compare=False)

    @classmethod
    def parse_user_item(cls, item: Mapping[str, Any]) -> Self:
        return cls(
            id=item["id"],
            latitude=np.deg2rad(item["lat_deg"]),
            longitude=np.deg2rad(item["lon_deg"]),
            altitude=item["alt_m"],
            minimum_elevation=np.deg2rad(item["min_el_deg"]),
        )


_USER_TYPES: dict[str, type[User]] = {cls.class_name(): cls for cls in (StationaryOnGroundUser,)}


def parse_user_item(item: Mapping[str, Any]) -> User:
    """Parse a user item."""
    user_type = item["type"]
    return _USER_TYPES[user_type].parse_user_item(item)


def load_users_from_toml_file(toml_file_path: str | Path) -> list[User]:
    """Load users from a TOML file."""
    toml_file_path = Path(toml_file_path)
    with toml_file_path.open("rb") as f:
        toml_data = tomllib.load(f)
    return list(map(parse_user_item, toml_data["users"]))
