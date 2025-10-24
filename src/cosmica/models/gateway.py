from __future__ import annotations

__all__ = [
    "Gateway",
    "GatewayOGS",
]

import tomllib
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import MutableMapping


import numpy as np

from .node import Node

_T = TypeVar("_T", bound=Hashable)


@dataclass(frozen=True, kw_only=True, slots=True)
class Gateway(Node[_T]):
    id: _T
    latitude: float = field(compare=False)
    longitude: float = field(compare=False)
    minimum_elevation: float = field(compare=False)
    altitude: float = field(default=0.0, compare=False)
    n_terminals: int = field(default=1, compare=False)

    @classmethod
    def from_toml_file(cls, toml_file_path: Path | str) -> list[Self]:
        toml_file_path = Path(toml_file_path)
        with toml_file_path.open("rb") as f:
            toml_data = tomllib.load(f)
        return list(map(cls.parse_gateway_item, toml_data["gateways"]))

    @classmethod
    def parse_gateway_item(cls, item: MutableMapping[str, Any]) -> Self:
        item["latitude"] = np.deg2rad(item.pop("lat_deg"))
        item["longitude"] = np.deg2rad(item.pop("lon_deg"))
        item["minimum_elevation"] = np.deg2rad(item.pop("min_el_deg"))
        return cls(**item)

    @classmethod
    def class_name(cls) -> str:
        return "GW"


@dataclass(frozen=True, kw_only=True, slots=True)
class GatewayOGS(Node[_T]):
    id: _T
    latitude: float = field(compare=False)
    longitude: float = field(compare=False)
    minimum_elevation: float = field(compare=False)
    altitude: float = field(default=0.0, compare=False)
    n_terminals: int = field(default=1, compare=False)
    aperture_size: float = field(default=1.0, compare=False)
    rytov_variance: float = field(default=0.5, compare=False)

    @classmethod
    def from_toml_file(cls, toml_file_path: Path | str) -> list[Self]:
        toml_file_path = Path(toml_file_path)
        with toml_file_path.open("rb") as f:
            toml_data = tomllib.load(f)
        return list(map(cls.parse_gateway_item, toml_data["gateways"]))

    @classmethod
    def parse_gateway_item(cls, item: MutableMapping[str, Any]) -> Self:
        item["latitude"] = np.deg2rad(item.pop("lat_deg"))
        item["longitude"] = np.deg2rad(item.pop("lon_deg"))
        item["minimum_elevation"] = np.deg2rad(item.pop("min_el_deg"))
        item["aperture_size"] = item.pop("aperture_size_m")
        item["rytov_variance"] = item.pop("rytov_variance")

        return cls(**item)

    @classmethod
    def class_name(cls) -> str:
        return "GW_OGS"
