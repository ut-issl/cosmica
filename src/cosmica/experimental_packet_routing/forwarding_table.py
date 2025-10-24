__all__ = [
    "ForwardingTable",
    "ForwardingTableInformation",
]

from dataclasses import dataclass, field

from cosmica.models.node import Node


@dataclass(kw_only=True, slots=True)
class ForwardingTableInformation:
    destination: Node
    next_node: Node
    cost: float = 0.0


@dataclass(kw_only=True, slots=True)
class ForwardingTable:
    entries: dict[Node, ForwardingTableInformation] = field(default_factory=dict)

    def update_entry(
        self,
        destination: Node,
        next_node: Node,
        cost: float = 0.0,
    ) -> None:
        if destination in self.entries:
            entry: ForwardingTableInformation = self.entries[destination]
            entry.next_node = next_node
            entry.cost = cost
        else:
            self.entries[destination] = ForwardingTableInformation(
                destination=destination,
                next_node=next_node,
                cost=cost,
            )

    def remove_entry(
        self,
        destination: Node,
    ) -> None:
        if destination in self.entries:
            del self.entries[destination]

    def find_entry(
        self,
        destination: Node,
    ) -> ForwardingTableInformation | None:
        return self.entries.get(destination, None)
