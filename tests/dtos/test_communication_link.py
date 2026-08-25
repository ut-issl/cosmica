import numpy as np
import pytest

from cosmica.dtos import CommunicationLinkEndpoint, DirectedCommunicationLink
from cosmica.models import (
    CircularSatelliteOrbitModel,
    ConstellationSatellite,
    Gateway,
    OpticalCommunicationTerminal,
)
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


def test_directed_link_identifies_nodes_and_assigned_terminals() -> None:
    source_terminal = _make_terminal(1)
    destination_terminal = _make_terminal(2)
    source_satellite = _make_satellite(1, (source_terminal,))
    destination_satellite = _make_satellite(2, (destination_terminal,))

    link = DirectedCommunicationLink(
        source=CommunicationLinkEndpoint(node=source_satellite, terminal=source_terminal),
        destination=CommunicationLinkEndpoint(node=destination_satellite, terminal=destination_terminal),
    )

    assert link.node_pair == (source_satellite, destination_satellite)
    assert link.reversed().source == link.destination
    assert link.reversed().destination == link.source
    assert link.reversed().node_pair == (destination_satellite, source_satellite)


def test_endpoint_rejects_terminal_owned_by_another_node() -> None:
    owned_terminal = _make_terminal(1)
    other_terminal = _make_terminal(2)
    satellite = _make_satellite(1, (owned_terminal,))

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
        CommunicationLinkEndpoint(node=gateway, terminal=_make_terminal(1))
