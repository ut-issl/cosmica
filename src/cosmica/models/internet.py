__all__ = [
    "Internet",
]
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

from .node import Node

_T = TypeVar("_T", bound=Hashable)


@dataclass(frozen=True, kw_only=True, slots=True)
class Internet(Node[_T | None]):
    """The Internet node."""

    id: _T | None = None

    @classmethod
    def class_name(cls) -> str:
        return "Internet"
