__all__ = [
    "COMMUNICATION_LINK_ATTRIBUTE",
    "assign_communication_link",
    "get_assigned_communication_links",
]

import networkx as nx

from cosmica.dtos import DirectedCommunicationLink

COMMUNICATION_LINK_ATTRIBUTE = "communication_link"
"""NetworkX edge attribute containing the assigned directed communication link."""


def _validate_directed_graph(graph: nx.DiGraph) -> None:
    if graph.is_multigraph():
        msg = "communication-link assignments do not support MultiDiGraph topologies"
        raise ValueError(msg)
    if not graph.is_directed():
        msg = "communication links can only be assigned to a directed topology graph"
        raise ValueError(msg)


def assign_communication_link(
    graph: nx.DiGraph,
    link: DirectedCommunicationLink,
) -> None:
    """Add one terminal assignment to a directed edge.

    Assigning the same link again is idempotent, while assigning a different terminal
    pair to the same directed node pair is rejected.
    """
    _validate_directed_graph(graph)
    source, destination = link.node_pair
    existing_data = graph.get_edge_data(source, destination)
    if existing_data is not None:
        existing_link = existing_data.get(COMMUNICATION_LINK_ATTRIBUTE)
        if existing_link is not None and existing_link != link:
            msg = f"directed edge {link.node_pair!r} already has a different terminal assignment"
            raise ValueError(msg)

    graph.add_edge(source, destination, **{COMMUNICATION_LINK_ATTRIBUTE: link})


def get_assigned_communication_links(
    graph: nx.DiGraph,
) -> tuple[DirectedCommunicationLink, ...]:
    """Recover terminal assignments from a directed topology graph.

    Ordinary node-only edges are ignored, preserving compatibility with existing
    topology builders and communication-link calculators.
    """
    _validate_directed_graph(graph)
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

    return tuple(links)
