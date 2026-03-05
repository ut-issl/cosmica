from __future__ import annotations

__all__ = [
    "Gateway",
    "GatewayOGS",
]
import math
import tomllib
from collections.abc import Hashable
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, override

from typing_extensions import deprecated

if TYPE_CHECKING:
    from collections.abc import MutableMapping


import numpy as np

from .node import Node


def _assert_finite(value: float, *, name: str) -> None:
    assert math.isfinite(value), f"{name} must be finite."


def _assert_in_range(value: float, *, name: str, lower: float, upper: float) -> None:
    assert lower <= value <= upper, f"{name} must be in [{lower}, {upper}]."


def _validate_gateway_fields(
    *,
    latitude: float,
    longitude: float,
    minimum_elevation: float,
    altitude: float,
    n_terminals: int,
) -> None:
    _assert_finite(latitude, name="latitude")
    _assert_finite(longitude, name="longitude")
    _assert_finite(minimum_elevation, name="minimum_elevation")
    _assert_finite(altitude, name="altitude")

    _assert_in_range(latitude, name="latitude", lower=-math.pi / 2, upper=math.pi / 2)
    _assert_in_range(longitude, name="longitude", lower=-math.pi, upper=math.pi)
    _assert_in_range(minimum_elevation, name="minimum_elevation", lower=0.0, upper=math.pi / 2)
    assert isinstance(n_terminals, Integral), "n_terminals must be an integer."
    assert n_terminals > 0, "n_terminals must be positive."


@dataclass(frozen=True, kw_only=True, slots=True)
class Gateway[T: Hashable](Node[T]):
    id: T
    latitude: float = field(compare=False)
    longitude: float = field(compare=False)
    minimum_elevation: float = field(compare=False)
    altitude: float = field(default=0.0, compare=False)
    n_terminals: int = field(default=1, compare=False)

    def __post_init__(self) -> None:
        _validate_gateway_fields(
            latitude=self.latitude,
            longitude=self.longitude,
            minimum_elevation=self.minimum_elevation,
            altitude=self.altitude,
            n_terminals=self.n_terminals,
        )

    @classmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def from_toml_file(cls, toml_file_path: Path | str) -> list[Self]:
        toml_file_path = Path(toml_file_path)
        with toml_file_path.open("rb") as f:
            toml_data = tomllib.load(f)
        return list(map(cls.parse_gateway_item, toml_data["gateways"]))

    @classmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def parse_gateway_item(cls, item: MutableMapping[str, Any]) -> Self:
        item["latitude"] = np.deg2rad(item.pop("lat_deg"))
        item["longitude"] = np.deg2rad(item.pop("lon_deg"))
        item["minimum_elevation"] = np.deg2rad(item.pop("min_el_deg"))
        return cls(**item)

    @classmethod
    @override
    def class_name(cls) -> str:
        return "GW"


@dataclass(frozen=True, kw_only=True, slots=True)
class GatewayOGS[T: Hashable](Node[T]):
    id: T
    latitude: float = field(compare=False)
    longitude: float = field(compare=False)
    minimum_elevation: float = field(compare=False)
    altitude: float = field(default=0.0, compare=False)
    n_terminals: int = field(default=1, compare=False)
    aperture_size: float = field(default=1.0, compare=False)
    rytov_variance: float = field(default=0.5, compare=False)

    def __post_init__(self) -> None:
        _validate_gateway_fields(
            latitude=self.latitude,
            longitude=self.longitude,
            minimum_elevation=self.minimum_elevation,
            altitude=self.altitude,
            n_terminals=self.n_terminals,
        )
        _assert_finite(self.aperture_size, name="aperture_size")
        _assert_finite(self.rytov_variance, name="rytov_variance")
        assert self.aperture_size > 0.0, "aperture_size must be positive."
        assert self.rytov_variance >= 0.0, "rytov_variance must be non-negative."

    @classmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def from_toml_file(cls, toml_file_path: Path | str) -> list[Self]:
        toml_file_path = Path(toml_file_path)
        with toml_file_path.open("rb") as f:
            toml_data = tomllib.load(f)
        return list(map(cls.parse_gateway_item, toml_data["gateways"]))

    @classmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def parse_gateway_item(cls, item: MutableMapping[str, Any]) -> Self:
        item["latitude"] = np.deg2rad(item.pop("lat_deg"))
        item["longitude"] = np.deg2rad(item.pop("lon_deg"))
        item["minimum_elevation"] = np.deg2rad(item.pop("min_el_deg"))
        item["aperture_size"] = item.pop("aperture_size_m")
        item["rytov_variance"] = item.pop("rytov_variance")

        return cls(**item)

    @classmethod
    @override
    def class_name(cls) -> str:
        return "GW_OGS"
