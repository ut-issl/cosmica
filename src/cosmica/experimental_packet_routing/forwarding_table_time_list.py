__all__ = [
    "ForwardingTableTimeList",
]
from dataclasses import dataclass, field

import numpy as np

from cosmica.models.node import Node

from .forwarding_table import ForwardingTable


@dataclass(kw_only=True, slots=True)
class ForwardingTableTimeList:
    time_for_snapshots: list[np.datetime64]
    nominal_forwarding_table_for_snapshots: list[ForwardingTable]
    backup_forwarding_tables_for_snapshots: list[dict[frozenset[frozenset[Node]], ForwardingTable]] = field(
        default_factory=list,
    )

    # TODO(Takashima): ある時刻以降のforwarding_table_time_listを更新
