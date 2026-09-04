import pytest

from cosmica.dtos import CommunicationLinkEndpoint, DirectedCommunicationLink
from cosmica.models import Gateway
from tests.factories import make_satellite, make_terminal


def test_directed_link_identifies_nodes_and_assigned_terminals() -> None:
    source_terminal = make_terminal(1)
    destination_terminal = make_terminal(2)
    source_satellite = make_satellite(1, terminals=(source_terminal,))
    destination_satellite = make_satellite(2, terminals=(destination_terminal,))

    link = DirectedCommunicationLink(
        source=CommunicationLinkEndpoint(node=source_satellite, terminal=source_terminal),
        destination=CommunicationLinkEndpoint(node=destination_satellite, terminal=destination_terminal),
    )

    assert link.node_pair == (source_satellite, destination_satellite)
    assert link.reversed().source == link.destination
    assert link.reversed().destination == link.source
    assert link.reversed().node_pair == (destination_satellite, source_satellite)


def test_endpoint_rejects_terminal_owned_by_another_node() -> None:
    owned_terminal = make_terminal(1)
    other_terminal = make_terminal(2)
    satellite = make_satellite(1, terminals=(owned_terminal,))

    with pytest.raises(ValueError, match="is not owned by node"):
        CommunicationLinkEndpoint(node=satellite, terminal=other_terminal)


def test_endpoint_rejects_node_without_terminal_inventory() -> None:
    gateway = Gateway(
        id=1,
        latitude=0.0,
        longitude=0.0,
        minimum_elevation=0.0,
    )

    with pytest.raises(TypeError, match="does not own communication terminals"):
        CommunicationLinkEndpoint(node=gateway, terminal=make_terminal(1))
