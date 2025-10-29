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
from typing import Any, Self

import numpy as np
from typing_extensions import deprecated

from .node import Node
from .terminal import CommunicationTerminal


@dataclass(frozen=True, kw_only=True, slots=True)
class User[T: Hashable](Node[T], ABC):
    """Base model for a user."""

    id: T
    terminals: list[CommunicationTerminal[Hashable]] = field(default_factory=list)

    @classmethod
    @abstractmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def parse_user_item(cls, item: Mapping[str, Any]) -> Self:
        """Parse a user item."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class StationaryOnGroundUser[T: Hashable](User[T]):
    """Model for a stationary user on the ground."""

    id: T
    latitude: float = field(compare=False)
    longitude: float = field(compare=False)
    altitude: float = field(default=0.0, compare=False)
    minimum_elevation: float = field(compare=False)
    terminals: list[CommunicationTerminal[Hashable]] = field(default_factory=list, compare=False)

    @classmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def parse_user_item(cls, item: Mapping[str, Any]) -> Self:
        return cls(
            id=item["id"],
            latitude=np.deg2rad(item["lat_deg"]),
            longitude=np.deg2rad(item["lon_deg"]),
            altitude=item["alt_m"],
            minimum_elevation=np.deg2rad(item["min_el_deg"]),
        )


_USER_TYPES: dict[str, type[User[Hashable]]] = {cls.class_name(): cls for cls in (StationaryOnGroundUser[Hashable],)}


@deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
def parse_user_item(item: Mapping[str, Any]) -> User[Hashable]:
    """Parse a user item."""
    user_type = item["type"]
    return _USER_TYPES[user_type].parse_user_item(item)


@deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
def load_users_from_toml_file(toml_file_path: str | Path) -> list[User[Hashable]]:
    """Load users from a TOML file."""
    toml_file_path = Path(toml_file_path)
    with toml_file_path.open("rb") as f:
        toml_data = tomllib.load(f)
    return list(map(parse_user_item, toml_data["users"]))
