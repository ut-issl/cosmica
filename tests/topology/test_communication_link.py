import networkx as nx
import pytest

from cosmica.topology import (
    COMMUNICATION_LINK_ATTRIBUTE,
    assign_communication_link,
    get_terminal_assigned_communication_links,
)
from tests.factories import make_link, make_satellite, make_terminal


def test_terminal_assignment_extraction_ignores_node_only_edges() -> None:
    source = make_satellite(1)
    destination = make_satellite(2)
    graph = nx.DiGraph([(source, destination)])

    assert get_terminal_assigned_communication_links(graph) == []


def test_terminal_assigned_link_round_trips_through_graph() -> None:
    source_terminal = make_terminal(1)
    destination_terminal = make_terminal(2)
    source = make_satellite(1, terminals=(source_terminal,))
    destination = make_satellite(2, terminals=(destination_terminal,))
    graph = nx.DiGraph()
    link = make_link(source, source_terminal, destination, destination_terminal)

    assign_communication_link(graph, link)
    assign_communication_link(graph, link)

    assert graph.number_of_edges(source, destination) == 1
    assert get_terminal_assigned_communication_links(graph) == [link]


def test_digraph_allows_reverse_link_for_the_same_terminal_pair() -> None:
    source_terminal = make_terminal(1)
    destination_terminal = make_terminal(2)
    source = make_satellite(1, terminals=(source_terminal,))
    destination = make_satellite(2, terminals=(destination_terminal,))
    graph = nx.DiGraph()
    link = make_link(source, source_terminal, destination, destination_terminal)

    assign_communication_link(graph, link)
    assign_communication_link(graph, link.reversed())

    assert set(get_terminal_assigned_communication_links(graph)) == {link, link.reversed()}


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_digraph_rejects_terminal_assignment_to_a_different_peer(direction: str) -> None:
    assigned_terminal = make_terminal(1)
    first_peer_terminal = make_terminal(2)
    other_peer_terminal = make_terminal(3)
    assigned_node = make_satellite(1, terminals=(assigned_terminal,))
    first_peer_node = make_satellite(2, terminals=(first_peer_terminal,))
    other_peer_node = make_satellite(3, terminals=(other_peer_terminal,))
    graph = nx.DiGraph()
    assign_communication_link(
        graph,
        make_link(assigned_node, assigned_terminal, first_peer_node, first_peer_terminal),
    )
    conflicting_link = (
        make_link(assigned_node, assigned_terminal, other_peer_node, other_peer_terminal)
        if direction == "outgoing"
        else make_link(other_peer_node, other_peer_terminal, assigned_node, assigned_terminal)
    )

    with pytest.raises(ValueError, match="is already assigned to peer"):
        assign_communication_link(graph, conflicting_link)

    assert not graph.has_edge(*conflicting_link.node_pair)


def test_link_extraction_rejects_terminal_assignment_to_a_different_peer() -> None:
    assigned_terminal = make_terminal(1)
    first_peer_terminal = make_terminal(2)
    other_peer_terminal = make_terminal(3)
    assigned_node = make_satellite(1, terminals=(assigned_terminal,))
    first_peer_node = make_satellite(2, terminals=(first_peer_terminal,))
    other_peer_node = make_satellite(3, terminals=(other_peer_terminal,))
    first_link = make_link(assigned_node, assigned_terminal, first_peer_node, first_peer_terminal)
    conflicting_link = make_link(other_peer_node, other_peer_terminal, assigned_node, assigned_terminal)
    graph = nx.DiGraph()
    graph.add_edge(*first_link.node_pair, **{COMMUNICATION_LINK_ATTRIBUTE: first_link})
    graph.add_edge(*conflicting_link.node_pair, **{COMMUNICATION_LINK_ATTRIBUTE: conflicting_link})

    with pytest.raises(ValueError, match="is already assigned to peer"):
        get_terminal_assigned_communication_links(graph)


def test_digraph_rejects_a_second_assignment_for_the_same_node_pair() -> None:
    source_terminals = (make_terminal(1), make_terminal(2))
    destination_terminals = (make_terminal(3), make_terminal(4))
    source = make_satellite(1, terminals=source_terminals)
    destination = make_satellite(2, terminals=destination_terminals)
    graph = nx.DiGraph()
    assign_communication_link(
        graph,
        make_link(source, source_terminals[0], destination, destination_terminals[0]),
    )

    with pytest.raises(ValueError, match="already has a different terminal assignment"):
        assign_communication_link(
            graph,
            make_link(source, source_terminals[1], destination, destination_terminals[1]),
        )
