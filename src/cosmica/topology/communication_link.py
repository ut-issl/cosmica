__all__ = [
    "COMMUNICATION_LINK_ATTRIBUTE",
    "assign_communication_link",
    "get_assigned_communication_links",
]

from collections.abc import Iterator, Mapping

import networkx as nx

from cosmica.dtos import DirectedCommunicationLink
from cosmica.models import Node

COMMUNICATION_LINK_ATTRIBUTE = "communication_link"
"""NetworkX edge attribute containing the assigned directed communication link."""

type CommunicationLinkGraph = nx.DiGraph | nx.MultiDiGraph


def _validate_directed_graph(graph: CommunicationLinkGraph) -> None:
    if not graph.is_directed():
        msg = "communication links can only be assigned to a directed topology graph"
        raise ValueError(msg)


def assign_communication_link(
    graph: CommunicationLinkGraph,
    link: DirectedCommunicationLink,
) -> None:
    """Add a terminal-assigned link while retaining its forwarding nodes as graph nodes.

    A ``MultiDiGraph`` can hold several links, using different terminal pairs, between
    the same directed node pair. A ``DiGraph`` can hold at most one such assignment;
    assigning the same link again is idempotent.
    """
    _validate_directed_graph(graph)
    source, destination = link.node_pair

    if graph.is_multigraph():
        graph.add_edge(source, destination, **{COMMUNICATION_LINK_ATTRIBUTE: link})
        return

    existing_data = graph.get_edge_data(source, destination)
    if existing_data is not None:
        existing_link = existing_data.get(COMMUNICATION_LINK_ATTRIBUTE)
        if existing_link is not None and existing_link != link:
            msg = (
                "a DiGraph cannot represent multiple terminal assignments for "
                f"the directed node pair {link.node_pair!r}; use nx.MultiDiGraph"
            )
            raise ValueError(msg)

    graph.add_edge(source, destination, **{COMMUNICATION_LINK_ATTRIBUTE: link})


def _iter_edges_with_data(
    graph: CommunicationLinkGraph,
) -> Iterator[tuple[Node, Node, Mapping[str, object]]]:
    if isinstance(graph, nx.MultiDiGraph):
        for source, destination, _key, data in graph.edges(keys=True, data=True):
            yield source, destination, data
        return

    yield from graph.edges(data=True)


def get_assigned_communication_links(
    graph: CommunicationLinkGraph,
) -> tuple[DirectedCommunicationLink, ...]:
    """Recover terminal assignments from a directed topology graph.

    Ordinary node-only edges are ignored, preserving compatibility with existing
    topology builders and communication-link calculators.
    """
    _validate_directed_graph(graph)
    links = []

    for source, destination, data in _iter_edges_with_data(graph):
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

    return tuple(links)
