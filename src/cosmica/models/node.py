__all__ = [
    "Node",
    "NodeGID",
]
from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass
from typing import NewType

NodeGID = NewType("NodeGID", str)


@dataclass(frozen=True, kw_only=True, slots=True)
class Node[T: Hashable](ABC):
    id: T

    @classmethod
    def class_name(cls) -> str:
        """Class name as a string used for the key generation.

        Override this method if you want to use a different class name for the key generation.
        """
        return cls.__name__

    @property
    def global_id(self) -> NodeGID:
        """The universally unique identifier of the node."""
        if self.id is None:
            return NodeGID(f"{self.class_name()}")
        return NodeGID(f"{self.class_name()}-{self.id}")

    def __str__(self) -> str:
        return self.global_id
