import networkx as nx
import pytest

from cosmica.dtos import CommunicationLinkEndpoint, DirectedCommunicationLink
from cosmica.models import ConstellationSatellite, OpticalCommunicationTerminal
from cosmica.topology import assign_communication_link, get_assigned_communication_links
from tests.factories import make_satellite, make_terminal


def _make_link(
    source: ConstellationSatellite[int],
    source_terminal: OpticalCommunicationTerminal[int],
    destination: ConstellationSatellite[int],
    destination_terminal: OpticalCommunicationTerminal[int],
) -> DirectedCommunicationLink:
    return DirectedCommunicationLink(
        source=CommunicationLinkEndpoint(node=source, terminal=source_terminal),
        destination=CommunicationLinkEndpoint(node=destination, terminal=destination_terminal),
    )


def test_multidigraph_round_trips_multiple_terminal_assignments_for_one_node_pair() -> None:
    source_terminals = (make_terminal(1), make_terminal(2))
    destination_terminals = (make_terminal(3), make_terminal(4))
    source = make_satellite(1, terminals=source_terminals)
    destination = make_satellite(2, terminals=destination_terminals)
    links = (
        _make_link(source, source_terminals[0], destination, destination_terminals[0]),
        _make_link(source, source_terminals[1], destination, destination_terminals[1]),
    )
    graph = nx.MultiDiGraph()

    for link in links:
        assign_communication_link(graph, link)

    assert set(graph.nodes) == {source, destination}
    assert graph.number_of_edges(source, destination) == 2
    assert set(get_assigned_communication_links(graph)) == set(links)


def test_node_only_edges_remain_compatible_with_assignment_extraction() -> None:
    source_terminal = make_terminal(1)
    destination_terminal = make_terminal(2)
    source = make_satellite(1, terminals=(source_terminal,))
    destination = make_satellite(2, terminals=(destination_terminal,))
    graph = nx.DiGraph([(source, destination)])

    assert get_assigned_communication_links(graph) == ()

    link = _make_link(source, source_terminal, destination, destination_terminal)
    assign_communication_link(graph, link)

    assert get_assigned_communication_links(graph) == (link,)


def test_digraph_rejects_a_second_assignment_for_the_same_node_pair() -> None:
    source_terminals = (make_terminal(1), make_terminal(2))
    destination_terminals = (make_terminal(3), make_terminal(4))
    source = make_satellite(1, terminals=source_terminals)
    destination = make_satellite(2, terminals=destination_terminals)
    graph = nx.DiGraph()
    assign_communication_link(
        graph,
        _make_link(source, source_terminals[0], destination, destination_terminals[0]),
    )

    with pytest.raises(ValueError, match=r"use nx\.MultiDiGraph"):
        assign_communication_link(
            graph,
            _make_link(source, source_terminals[1], destination, destination_terminals[1]),
        )
