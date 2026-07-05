import dataclasses

import numpy as np
import pytest

from cosmica.models import (
    ConstantCommunicationDemand,
    OneTimeCommunicationDemand,
    TemporaryCommunicationDemand,
)
from cosmica.models.node import NodeGID


def test_constant_demand_defaults_traffic_class_and_priority() -> None:
    demand = ConstantCommunicationDemand(
        id=1,
        source=NodeGID("GW-0"),
        destination=NodeGID("GW-1"),
        distribution="uniform",
        transmission_rate=1e9,
    )
    assert demand.traffic_class == "default"
    assert demand.priority == 0


def test_constant_demand_accepts_traffic_class_and_priority() -> None:
    demand = ConstantCommunicationDemand(
        id=1,
        source=NodeGID("GW-0"),
        destination=NodeGID("GW-1"),
        distribution="poisson",
        transmission_rate=1e6,
        traffic_class="financial",
        priority=2,
    )
    assert demand.traffic_class == "financial"
    assert demand.priority == 2


def test_demand_is_frozen() -> None:
    demand = ConstantCommunicationDemand(
        id=1,
        source=NodeGID("GW-0"),
        destination=NodeGID("GW-1"),
        distribution="uniform",
        transmission_rate=1e9,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        demand.traffic_class = "video"  # type: ignore[misc]


def test_temporary_demand_is_active_within_window() -> None:
    demand = TemporaryCommunicationDemand(
        id=1,
        source=NodeGID("GW-0"),
        destination=NodeGID("GW-1"),
        transmission_rate=1e9,
        distribution="uniform",
        start_time=np.datetime64("2026-01-01T00:00:10"),
        end_time=np.datetime64("2026-01-01T00:00:20"),
        traffic_class="video",
    )
    assert not demand.is_active(np.datetime64("2026-01-01T00:00:09"))
    assert demand.is_active(np.datetime64("2026-01-01T00:00:10"))
    assert demand.is_active(np.datetime64("2026-01-01T00:00:19"))
    assert not demand.is_active(np.datetime64("2026-01-01T00:00:20"))


def test_onetime_demand_construction() -> None:
    demand = OneTimeCommunicationDemand(
        id=("imagery", "UserSatellite-0", "GW-0", 0),
        source=NodeGID("UserSatellite-0"),
        destination=NodeGID("GW-0"),
        data_size=1e9,
        generation_time=np.datetime64("2026-01-01T00:00:30"),
        deadline=np.datetime64("2026-01-01T00:01:00"),
        traffic_class="imagery",
    )
    assert demand.data_size == 1e9
    assert demand.traffic_class == "imagery"
