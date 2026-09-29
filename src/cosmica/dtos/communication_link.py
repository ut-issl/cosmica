from __future__ import annotations

__all__ = [
    "CommunicationLinkEndpoint",
    "DirectedCommunicationLink",
]

from dataclasses import dataclass

from cosmica.models import CommunicationTerminal, Node, TerminalOwner


@dataclass(frozen=True, slots=True)
class CommunicationLinkEndpoint[N: Node, T: CommunicationTerminal]:
    """A forwarding node and one of the communication terminals it owns."""

    node: N
    terminal: T

    def __post_init__(self) -> None:
        if not isinstance(self.node, TerminalOwner):
            msg = f"{self.node!r} does not own communication terminals"
            raise TypeError(msg)
        if self.terminal not in self.node.terminals:
            msg = f"terminal {self.terminal!r} is not owned by node {self.node!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DirectedCommunicationLink[
    SourceNode: Node,
    SourceTerminal: CommunicationTerminal,
    DestinationNode: Node,
    DestinationTerminal: CommunicationTerminal,
]:
    """Directed topology link with an explicitly assigned terminal at each endpoint."""

    source: CommunicationLinkEndpoint[SourceNode, SourceTerminal]
    destination: CommunicationLinkEndpoint[DestinationNode, DestinationTerminal]

    @property
    def node_pair(self) -> tuple[SourceNode, DestinationNode]:
        """Return the directed graph edge represented by this link."""
        return self.source.node, self.destination.node

    def reversed(
        self,
    ) -> DirectedCommunicationLink[
        DestinationNode,
        DestinationTerminal,
        SourceNode,
        SourceTerminal,
    ]:
        """Return the opposite direction using the same endpoint terminals."""
        return DirectedCommunicationLink(
            source=self.destination,
            destination=self.source,
        )
