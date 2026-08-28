from dataclasses import FrozenInstanceError

import pytest

from cosmica.models import ConstellationSatellite
from tests.factories import make_orbit, make_terminal


def test_satellite_owns_immutable_terminal_inventory_without_changing_identity() -> None:
    terminal = make_terminal(1)
    satellite_with_terminal = ConstellationSatellite(
        id=1,
        orbit=make_orbit(),
        terminals=(terminal,),
    )
    same_satellite_without_payload = ConstellationSatellite(id=1, orbit=make_orbit())

    assert satellite_with_terminal.terminals == (terminal,)
    assert satellite_with_terminal == same_satellite_without_payload
    assert hash(satellite_with_terminal) == hash(same_satellite_without_payload)

    field_name = "terminals"
    with pytest.raises(FrozenInstanceError):
        setattr(satellite_with_terminal, field_name, ())


def test_satellite_rejects_duplicate_terminal_global_ids() -> None:
    with pytest.raises(ValueError, match="terminal global IDs must be unique"):
        ConstellationSatellite(
            id=1,
            orbit=make_orbit(),
            terminals=(
                make_terminal(1, angular_velocity_max=1.0),
                make_terminal(1, angular_velocity_max=2.0),
            ),
        )
