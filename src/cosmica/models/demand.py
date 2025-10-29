__all__ = [
    "ConstantCommunicationDemand",
    "Demand",
    "OneTimeCommunicationDemand",
]
from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .node import NodeGID


@dataclass(frozen=True, kw_only=True, slots=True)
class Demand[T: Hashable](ABC):
    """Base model for a demand."""

    id: T


@dataclass(frozen=True, kw_only=True, slots=True)
class ConstantCommunicationDemand[T: Hashable](Demand[T]):
    """Constant communication demand model."""

    id: T
    source: NodeGID
    destination: NodeGID
    distribution: Literal["uniform", "poisson"]
    transmission_rate: float


@dataclass(frozen=True, kw_only=True, slots=True)
class TemporaryCommunicationDemand[T: Hashable](Demand[T]):
    """Temporary communication demand model."""

    id: T
    source: NodeGID
    destination: NodeGID
    transmission_rate: float
    distribution: Literal["uniform", "poisson"]
    start_time: np.datetime64
    end_time: np.datetime64

    def is_active(
        self,
        current_time: np.datetime64,
    ) -> np.bool:
        return self.start_time <= current_time < self.end_time


@dataclass(frozen=True, kw_only=True, slots=True)
class OneTimeCommunicationDemand[T: Hashable](Demand[T]):
    """One-time communication demand model.

    This model is used for the communication demand to transfer a certain amount of data
    from a source to a destination. The data is generated at the source at a certain
    time and the communication demand is created at the same time. The time by which the
    data transfer should be completed is also given.
    """

    id: T
    source: NodeGID
    destination: NodeGID
    data_size: float
    generation_time: np.datetime64
    deadline: np.datetime64
