from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from cosmica.comm_link import (
    GatewayToSatBinaryMemoryCommLinkCalculator,
    MemorylessCommLinkCalculator,
    SatToGatewayBinaryMemoryCommLinkCalculator,
    SatToSatBinaryMemoryCommLinkCalculator,
)
from cosmica.dtos import DynamicsData
from cosmica.models import Gateway, Node
from tests.factories import make_satellite


@pytest.mark.parametrize("boolean_type", [bool, np.bool_], ids=["python", "numpy"])
@pytest.mark.parametrize(
    "calculator_type",
    [
        GatewayToSatBinaryMemoryCommLinkCalculator,
        SatToGatewayBinaryMemoryCommLinkCalculator,
        SatToSatBinaryMemoryCommLinkCalculator,
    ],
)
def test_unavailable_link_restarts_acquisition(
    calculator_type: Any,
    boolean_type: type[bool] | type[np.bool_],
):
    satellite = make_satellite(1)
    gateway = Gateway(id=1, latitude=0.0, longitude=0.0, minimum_elevation=0.0)
    edge: tuple[Node, Node]
    if calculator_type is GatewayToSatBinaryMemoryCommLinkCalculator:
        edge = (gateway, satellite)
    elif calculator_type is SatToGatewayBinaryMemoryCommLinkCalculator:
        edge = (satellite, gateway)
    else:
        edge = (satellite, make_satellite(2))

    memoryless = Mock(spec=MemorylessCommLinkCalculator)
    memoryless.calc.side_effect = [
        {
            edge: {
                "link_available": boolean_type(available),
                "link_capacity": 10.0 if available else 0.0,
                "delay": 0.01,
            },
        }
        for available in [True, False, False, True, True]
    ]
    calculator = calculator_type(memoryless_calculator=memoryless, link_acquisition_time=20.0)
    time = np.datetime64("2026-01-01") + np.array([0, 10, 20, 30, 40], dtype="timedelta64[s]")
    dynamics_data: DynamicsData = DynamicsData(
        time=time,
        dcm_eci2ecef=np.broadcast_to(np.eye(3), (5, 3, 3)),
        satellite_position_eci={},
        satellite_velocity_eci={},
        satellite_position_ecef={},
        satellite_attitude_angular_velocity_eci={},
        sun_direction_eci=np.zeros((5, 3)),
        sun_direction_ecef=np.zeros((5, 3)),
    )

    results = calculator.calc(
        [[edge]] * len(time),
        dynamics_data=dynamics_data,
        rng=np.random.default_rng(0),
    )

    assert [bool(result[edge]["link_available"]) for result in results] == [True, False, False, False, True]
    assert [result[edge]["link_capacity"] for result in results] == [10.0, 0.0, 0.0, 0.0, 10.0]
    assert [result[edge]["delay"] for result in results] == [0.01, np.inf, np.inf, np.inf, 0.01]
