__all__ = [
    "ConstantCommunicationDemand",
    "Demand",
    "OneTimeCommunicationDemand",
    "load_demands_from_toml_file",
    "parse_demand_item",
]
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from typing_extensions import deprecated

from .node import NodeGID


@dataclass(frozen=True, kw_only=True, slots=True)
class Demand[T: Hashable](ABC):
    """Base model for a demand."""

    id: T

    @classmethod
    @abstractmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def parse_demand_item(cls, item: Mapping[str, Any]) -> Self:
        """Parse a demand item."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class ConstantCommunicationDemand[T: Hashable](Demand[T]):
    """Constant communication demand model."""

    id: T
    source: NodeGID
    destination: NodeGID
    distribution: Literal["uniform", "poisson"]
    transmission_rate: float

    @classmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def parse_demand_item(cls, item: Mapping[str, Any]) -> Self:
        return cls(
            id=item["id"],
            source=item["src_gid"],
            destination=item["dst_gid"],
            transmission_rate=item["transmission_rate_bps"],
            distribution=item["distribution"],
        )


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

    @classmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def parse_demand_item(cls, item: Mapping[str, Any]) -> Self:
        return cls(
            id=item["id"],
            source=item["src_gid"],
            destination=item["dst_gid"],
            transmission_rate=item["transmission_rate_bps"],
            distribution=item["distribution"],
            start_time=np.datetime64(item["start_time"].astimezone(tz=UTC).replace(tzinfo=None)),
            end_time=np.datetime64(item["end_time"].astimezone(tz=UTC).replace(tzinfo=None)),
        )


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

    @classmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def parse_demand_item(cls, item: Mapping[str, Any]) -> Self:
        return cls(
            id=item["id"],
            source=item["src_gid"],
            destination=item["dst_gid"],
            data_size=item["data_size"],
            generation_time=item["generation_time"],
            deadline=item["deadline"],
        )


_DEMAND_TYPES: dict[str, type[Demand[Hashable]]] = {
    "constant": ConstantCommunicationDemand,
    "temporary": TemporaryCommunicationDemand,
    "one_time": OneTimeCommunicationDemand,
}


@deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
def parse_demand_item(item: Mapping[str, Any]) -> Demand[Hashable]:
    """Parse a demand item."""
    demand_type = item["type"]
    return _DEMAND_TYPES[demand_type].parse_demand_item(item)


@deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
def load_demands_from_toml_file(toml_file_path: str | Path) -> list[Demand[Hashable]]:
    """Load demands from a TOML file."""
    toml_file_path = Path(toml_file_path)
    with toml_file_path.open("rb") as f:
        toml_data = tomllib.load(f)
    return list(map(parse_demand_item, toml_data["demands"]))
