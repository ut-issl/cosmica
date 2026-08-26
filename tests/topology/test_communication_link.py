import networkx as nx
import numpy as np
import pytest

from cosmica.dtos import CommunicationLinkEndpoint, DirectedCommunicationLink
from cosmica.models import CircularSatelliteOrbitModel, ConstellationSatellite, OpticalCommunicationTerminal
from cosmica.topology import assign_communication_link, get_assigned_communication_links
from cosmica.utils.constants import EARTH_RADIUS

EPOCH = np.datetime64("2026-01-01T00:00:00")


def _make_terminal(terminal_id: int) -> OpticalCommunicationTerminal[int]:
    return OpticalCommunicationTerminal(
        id=terminal_id,
        azimuth_min=-np.pi,
        azimuth_max=np.pi,
        elevation_min=-np.pi / 2,
        elevation_max=np.pi / 2,
        angular_velocity_max=1.0,
    )


def _make_satellite(
    satellite_id: int,
    terminals: tuple[OpticalCommunicationTerminal[int], ...],
) -> ConstellationSatellite[int]:
    return ConstellationSatellite(
        id=satellite_id,
        orbit=CircularSatelliteOrbitModel(
            semi_major_axis=EARTH_RADIUS + 1_000e3,
            inclination=0.0,
            raan=0.0,
            phase_at_epoch=0.0,
            epoch=EPOCH,
        ),
        terminals=terminals,
    )


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
    source_terminals = (_make_terminal(1), _make_terminal(2))
    destination_terminals = (_make_terminal(3), _make_terminal(4))
    source = _make_satellite(1, source_terminals)
    destination = _make_satellite(2, destination_terminals)
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
    source_terminal = _make_terminal(1)
    destination_terminal = _make_terminal(2)
    source = _make_satellite(1, (source_terminal,))
    destination = _make_satellite(2, (destination_terminal,))
    graph = nx.DiGraph([(source, destination)])

    assert get_assigned_communication_links(graph) == ()

    link = _make_link(source, source_terminal, destination, destination_terminal)
    assign_communication_link(graph, link)

    assert get_assigned_communication_links(graph) == (link,)


def test_digraph_rejects_a_second_assignment_for_the_same_node_pair() -> None:
    source_terminals = (_make_terminal(1), _make_terminal(2))
    destination_terminals = (_make_terminal(3), _make_terminal(4))
    source = _make_satellite(1, source_terminals)
    destination = _make_satellite(2, destination_terminals)
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
