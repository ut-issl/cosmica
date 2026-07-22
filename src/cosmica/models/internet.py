__all__ = [
    "Internet",
]
from collections.abc import Hashable
from dataclasses import dataclass
from typing import override

from .node import Node


@dataclass(frozen=True, kw_only=True, slots=True)
class Internet[T: Hashable](Node[T | None]):
    """The Internet node."""

    id: T | None = None

    @classmethod
    @override
    def class_name(cls) -> str:
        return "Internet"
