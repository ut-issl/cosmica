from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cosmica.models import CircularSatelliteOrbitModel, ConstellationSatellite, OpticalCommunicationTerminal
from cosmica.utils.constants import EARTH_RADIUS

EPOCH = np.datetime64("2026-01-01T00:00:00")


def _make_orbit() -> CircularSatelliteOrbitModel:
    return CircularSatelliteOrbitModel(
        semi_major_axis=EARTH_RADIUS + 1_000e3,
        inclination=0.0,
        raan=0.0,
        phase_at_epoch=0.0,
        epoch=EPOCH,
    )


def _make_terminal(terminal_id: int, *, angular_velocity_max: float = 1.0) -> OpticalCommunicationTerminal[int]:
    return OpticalCommunicationTerminal(
        id=terminal_id,
        azimuth_min=-np.pi,
        azimuth_max=np.pi,
        elevation_min=-np.pi / 2,
        elevation_max=np.pi / 2,
        angular_velocity_max=angular_velocity_max,
    )


def test_satellite_owns_immutable_terminal_inventory_without_changing_identity() -> None:
    terminal = _make_terminal(1)
    satellite_with_terminal = ConstellationSatellite(
        id=1,
        orbit=_make_orbit(),
        terminals=(terminal,),
    )
    same_satellite_without_payload = ConstellationSatellite(id=1, orbit=_make_orbit())

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
            orbit=_make_orbit(),
            terminals=(
                _make_terminal(1, angular_velocity_max=1.0),
                _make_terminal(1, angular_velocity_max=2.0),
            ),
        )
