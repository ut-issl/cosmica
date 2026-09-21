from __future__ import annotations

__all__ = [
    "COMMUNICATION_LINK_ATTRIBUTE",
    "assign_communication_links",
    "get_terminal_assigned_communication_links",
]

from typing import TYPE_CHECKING

from cosmica.dtos import CommunicationLinkEndpoint, DirectedCommunicationLink

if TYPE_CHECKING:
    from collections.abc import Iterable

    import networkx as nx

    from cosmica.models import Node

COMMUNICATION_LINK_ATTRIBUTE = "communication_link"
"""NetworkX edge attribute containing the assigned directed communication link."""


def _validate_terminal_assignments(links: Iterable[DirectedCommunicationLink]) -> None:
    """Ensure each terminal endpoint is paired with at most one peer endpoint.

    For example, reject ``A.terminal1 -> B.terminal1`` together with
    ``A.terminal1 -> C.terminal1`` because ``A.terminal1`` would have two peers. The
    invariant is independent of link direction, so reverse links using the same endpoint
    pair are valid.
    """
    peer_by_endpoint: dict[CommunicationLinkEndpoint, CommunicationLinkEndpoint] = {}

    for link in links:
        for endpoint, peer in (
            (link.source, link.destination),
            (link.destination, link.source),
        ):
            existing_peer = peer_by_endpoint.get(endpoint)
            if existing_peer is not None and existing_peer != peer:
                msg = (
                    f"terminal endpoint {endpoint!r} is already assigned to peer "
                    f"{existing_peer!r}, so it cannot be assigned to {peer!r}"
                )
                raise ValueError(msg)
            peer_by_endpoint[endpoint] = peer


def _validate_node_pair_assignments(links: Iterable[DirectedCommunicationLink]) -> None:
    """Ensure each directed node pair has at most one distinct terminal assignment.

    For example, reject ``A.terminal1 -> B.terminal1`` together with
    ``A.terminal2 -> B.terminal2`` because both assign the directed node pair ``A -> B``.
    Repeated equal links are valid, and the reverse node pair is validated independently.
    """
    link_by_node_pair: dict[tuple[Node, Node], DirectedCommunicationLink] = {}

    for link in links:
        existing_link = link_by_node_pair.get(link.node_pair)
        if existing_link is not None and existing_link != link:
            msg = f"directed edge {link.node_pair!r} already has a different terminal assignment"
            raise ValueError(msg)
        link_by_node_pair[link.node_pair] = link


def assign_communication_links(
    graph: nx.DiGraph,
    links: Iterable[DirectedCommunicationLink],
) -> None:
    """Add terminal assignments to edges in a simple directed graph.

    Assigning the same link repeatedly is idempotent, while assigning different
    terminal pairs to the same directed node pair is rejected. Each terminal endpoint
    may be paired with only one peer endpoint, independent of link direction. All
    assignments are validated before the graph is modified.
    """
    links = list(links)
    existing_assigned_links = get_terminal_assigned_communication_links(graph)

    links_after_update = existing_assigned_links + links
    _validate_terminal_assignments(links_after_update)
    _validate_node_pair_assignments(links_after_update)

    for link in links:
        graph.add_edge(*link.node_pair, **{COMMUNICATION_LINK_ATTRIBUTE: link})


def get_terminal_assigned_communication_links(
    graph: nx.DiGraph,
) -> list[DirectedCommunicationLink]:
    """Recover terminal assignments from a simple directed topology graph.

    Ordinary node-only edges are ignored, preserving compatibility with existing
    topology builders and communication-link calculators. Conflicting peer assignments
    for the same terminal endpoint are rejected, independent of link direction.
    """
    links = []

    for source, destination, data in graph.edges(data=True):
        link = data.get(COMMUNICATION_LINK_ATTRIBUTE)
        if link is None:
            continue
        if not isinstance(link, DirectedCommunicationLink):
            msg = f"{COMMUNICATION_LINK_ATTRIBUTE!r} edge metadata must be a DirectedCommunicationLink"
            raise TypeError(msg)
        if link.node_pair != (source, destination):
            msg = "communication-link endpoint nodes must match the topology edge"
            raise ValueError(msg)
        links.append(link)

    _validate_terminal_assignments(links)
    return links
