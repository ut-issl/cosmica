from __future__ import annotations

__all__ = [
    "Gateway",
    "GatewayOGS",
]
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import override

from .node import Node


@dataclass(frozen=True, kw_only=True, slots=True)
class Gateway[T: Hashable](Node[T]):
    id: T
    latitude: float = field(compare=False)
    longitude: float = field(compare=False)
    minimum_elevation: float = field(compare=False)
    altitude: float = field(default=0.0, compare=False)
    n_terminals: int = field(default=1, compare=False)

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

    @classmethod
    @override
    def class_name(cls) -> str:
        return "GW_OGS"
