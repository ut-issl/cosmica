__all__ = [
    "COMMUNICATION_LINK_ATTRIBUTE",
    "assign_communication_link",
    "get_terminal_assigned_communication_links",
]

from collections.abc import Iterable

import networkx as nx

from cosmica.dtos import CommunicationLinkEndpoint, DirectedCommunicationLink

COMMUNICATION_LINK_ATTRIBUTE = "communication_link"
"""NetworkX edge attribute containing the assigned directed communication link."""


def _validate_terminal_assignments(links: Iterable[DirectedCommunicationLink]) -> None:
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


def assign_communication_link(
    graph: nx.DiGraph,
    link: DirectedCommunicationLink,
) -> None:
    """Add one terminal assignment to an edge in a simple directed graph.

    Assigning the same link again is idempotent, while assigning a different terminal
    pair to the same directed node pair is rejected. Each terminal endpoint may be
    paired with only one peer endpoint, independent of link direction.
    """
    source, destination = link.node_pair
    existing_data = graph.get_edge_data(source, destination)
    if existing_data is not None:
        existing_link = existing_data.get(COMMUNICATION_LINK_ATTRIBUTE)
        if existing_link is not None and existing_link != link:
            msg = f"directed edge {link.node_pair!r} already has a different terminal assignment"
            raise ValueError(msg)

    assigned_links = get_terminal_assigned_communication_links(graph)
    _validate_terminal_assignments((*assigned_links, link))

    graph.add_edge(source, destination, **{COMMUNICATION_LINK_ATTRIBUTE: link})


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
