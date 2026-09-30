import json
import os
import subprocess
import sys
import textwrap

import pytest

# Run a seeded stochastic link simulation with string node IDs, whose hashes depend on PYTHONHASHSEED.
_SIMULATION = textwrap.dedent(
    """
    import json
    import sys

    import numpy as np

    from cosmica.comm_link import (
        BinaryCloudModel,
        CommLinkCalculationCoordinator,
        MemorylessCommLinkCalculatorWrapper,
        SatToGatewayBinaryCommLinkCalculator,
        SatToGatewayBinaryCommLinkCalculatorWithScintillation,
        SatToGatewayStochasticBinaryCommLinkCalculator,
    )
    from cosmica.dtos import DynamicsData
    from cosmica.models import CircularSatelliteOrbitModel, ConstellationSatellite, Gateway, GatewayOGS

    n_time = 20
    time = np.datetime64("2026-01-01T00:00:00") + np.arange(n_time).astype("timedelta64[s]")
    orbit = CircularSatelliteOrbitModel(
        semi_major_axis=7_000_000.0, inclination=0.0, raan=0.0, phase_at_epoch=0.0, epoch=time[0]
    )
    satellites = [ConstellationSatellite(id=f"SAT-{name}", orbit=orbit) for name in "ABCDE"]
    ground = {"latitude": 0.0, "longitude": 0.0, "minimum_elevation": 0.0}
    gateways = [Gateway(id=name, **ground) for name in ("Tokyo", "Oslo", "Lima")]
    stations = [GatewayOGS(id=name, **ground) for name in ("Kona", "Perth")]

    # Every satellite is directly above every ground node, so all links are geometrically available.
    overhead = {satellite: np.tile([7_000_000.0, 0.0, 0.0], (n_time, 1)) for satellite in satellites}
    zero = {satellite: np.zeros((n_time, 3)) for satellite in satellites}
    dynamics_data = DynamicsData(
        time=time,
        dcm_eci2ecef=np.repeat(np.eye(3)[None, :, :], n_time, axis=0),
        satellite_position_eci=overhead,
        satellite_velocity_eci=zero,
        satellite_position_ecef=overhead,
        satellite_attitude_angular_velocity_eci=zero,
        sun_direction_eci=np.tile([0.0, 0.0, 1.0], (n_time, 1)),
        sun_direction_ecef=np.tile([0.0, 0.0, 1.0], (n_time, 1)),
    )

    cloud_calculator = SatToGatewayStochasticBinaryCommLinkCalculator(
        memoryless_calculator=SatToGatewayBinaryCommLinkCalculator(link_capacity=10e9),
        stochastic_model_factory=lambda src, dst: BinaryCloudModel(),
    )
    scintillation_calculator = MemorylessCommLinkCalculatorWrapper(
        SatToGatewayBinaryCommLinkCalculatorWithScintillation(
            satellite_to_gateway_link_capacity=10e9,
            link_capacity=1e9,
            noise_figure=3.0,
            lna_gain=30.0,
            lct_p0=1.0,
        ),
    )
    gateway_edges = {(satellite, gateway) for satellite in satellites for gateway in gateways}
    station_edges = {(satellite, station) for satellite in satellites for station in stations}
    rng = np.random.default_rng(0)

    if sys.argv[1] == "coordinator":
        coordinator = CommLinkCalculationCoordinator(
            calculator_assignment={
                (ConstellationSatellite, Gateway): cloud_calculator,
                (ConstellationSatellite, GatewayOGS): scintillation_calculator,
            },
        )
        performance = coordinator.calc(
            [gateway_edges | station_edges] * n_time, dynamics_data=dynamics_data, rng=rng
        )
    else:
        # Call each calculator directly with unordered sets, bypassing the coordinator.
        cloud = cloud_calculator.calc([gateway_edges] * n_time, dynamics_data=dynamics_data, rng=rng)
        scintillation = scintillation_calculator.calc([station_edges] * n_time, dynamics_data=dynamics_data, rng=rng)
        performance = [a | b for a, b in zip(cloud, scintillation, strict=True)]

    print(json.dumps(sorted(
        [time_index, src.global_id, dst.global_id, perf["link_available"], perf["link_capacity"]]
        for time_index, snapshot in enumerate(performance)
        for (src, dst), perf in snapshot.items()
    )))
    """,
)


def _run_simulation(mode: str, hash_seed: str) -> list[list[object]]:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", _SIMULATION, mode],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("mode", ["coordinator", "direct"])
def test_seeded_link_simulation_is_reproducible_across_hash_seeds(mode: str) -> None:
    results = [_run_simulation(mode, hash_seed) for hash_seed in ("0", "1", "2", "3")]

    assert all(result == results[0] for result in results[1:])
