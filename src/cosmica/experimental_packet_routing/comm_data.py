__all__ = [
    "CommData",
    "CommDataDemand",
    "CommDataLSA",
]

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from cosmica.models.node import Node


@dataclass(kw_only=True, slots=True)
class CommData:
    """Base model for a communication data."""

    data_size: int
    packet_size: int = 10000
    packet_num: int
    dst_node: Node
    next_node: Node
    current_position: Node | tuple[Node, Node]
    path: list[Node]
    generated_time: np.datetime64
    time: np.datetime64
    time_from_generated: float
    time_remaining_for_current_position: float
    delay: float = np.nan
    reach_dst: bool = False
    packet_loss: bool = False


@dataclass(kw_only=True, slots=True)
class CommDataDemand(CommData):
    """Model for a communication data for demand."""

    demand_id: Hashable


@dataclass(kw_only=True, slots=True)
class CommDataLSA(CommData):
    """Model for a communication data for LSA."""

    failure_position: Node | tuple[Node, Node]
