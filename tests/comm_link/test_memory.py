from collections.abc import Callable, Collection
from typing import Any, cast

import numpy as np
import pytest

from cosmica.comm_link import (
    CommLinkCalculator,
    CommLinkPerformance,
    GatewayToSatBinaryMemoryCommLinkCalculator,
    MemorylessCommLinkCalculator,
    SatToGatewayBinaryMemoryCommLinkCalculator,
    SatToSatBinaryMemoryCommLinkCalculator,
)
from cosmica.dtos import DynamicsData
from cosmica.models import Node


class _AvailabilitySequenceCalculator(MemorylessCommLinkCalculator[Node[int], Node[int]]):
    def __init__(self) -> None:
        self.availability = iter((np.bool_(0), True))

    def calc(
        self,
        edges: Collection[tuple[Node[int], Node[int]]],
        *,
        dynamics_data: DynamicsData,  # noqa: ARG002
        rng: np.random.Generator,  # noqa: ARG002
    ) -> dict[tuple[Node[int], Node[int]], CommLinkPerformance]:
        link_available = next(self.availability)
        return {
            edge: CommLinkPerformance(
                link_capacity=1.0 if link_available else 0.0,
                delay=0.0,
                link_available=cast("bool", link_available),
            )
            for edge in edges
        }


def _dynamics_data() -> DynamicsData:
    time = np.array(
        [
            np.datetime64("2026-01-01T00:00:00"),
            np.datetime64("2026-01-01T00:00:30"),
        ],
    )
    return DynamicsData(
        time=time,
        dcm_eci2ecef=np.broadcast_to(np.eye(3), (len(time), 3, 3)),
        satellite_position_eci={},
        satellite_velocity_eci={},
        satellite_position_ecef={},
        satellite_attitude_angular_velocity_eci={},
        sun_direction_eci=np.zeros((len(time), 3)),
        sun_direction_ecef=np.zeros((len(time), 3)),
    )


type _MemoryCalculatorFactory = Callable[..., CommLinkCalculator[Any, Any]]


@pytest.mark.parametrize(
    "calculator_factory",
    [
        GatewayToSatBinaryMemoryCommLinkCalculator,
        SatToGatewayBinaryMemoryCommLinkCalculator,
        SatToSatBinaryMemoryCommLinkCalculator,
    ],
)
def test_numpy_false_restarts_link_acquisition(
    calculator_factory: _MemoryCalculatorFactory,
) -> None:
    edge = (Node(id=1), Node(id=2))
    calculator = calculator_factory(
        memoryless_calculator=cast("Any", _AvailabilitySequenceCalculator()),
        link_acquisition_time=60.0,
        skip_link_acquisition_at_simulation_start=True,
    )

    performance = calculator.calc(
        edges_time_series=[{edge}, {edge}],
        dynamics_data=_dynamics_data(),
        rng=np.random.default_rng(0),
    )

    assert not performance[0][edge]["link_available"]
    assert performance[1][edge] == CommLinkPerformance(
        link_capacity=0.0,
        delay=np.inf,
        link_available=False,
    )
