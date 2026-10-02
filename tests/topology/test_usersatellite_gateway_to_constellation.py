import numpy as np
import numpy.typing as npt
import pytest

from cosmica.dtos import DynamicsData
from cosmica.models import CircularSatelliteOrbitModel, Constellation, ConstellationSatellite, Gateway
from cosmica.topology import build_hybrid_us2c_g2c_topology

_OVERHEAD_SATELLITE_POSITION_ECEF = np.array([7_000_000.0, 0.0, 0.0])
_TOWARD_SATELLITE = np.array([1.0, 0.0, 0.0])
_AWAY_FROM_SATELLITE = np.array([-1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("sun_direction_eci", "sun_direction_ecef", "expected_linked"),
    [
        pytest.param(_AWAY_FROM_SATELLITE, _TOWARD_SATELLITE, False, id="ecef-sun-behind-satellite"),
        pytest.param(_TOWARD_SATELLITE, _AWAY_FROM_SATELLITE, True, id="only-eci-sun-behind-satellite"),
    ],
)
def test_hybrid_ground_sun_exclusion_uses_ecef_sun_direction(
    sun_direction_eci: npt.NDArray[np.floating],
    sun_direction_ecef: npt.NDArray[np.floating],
    *,
    expected_linked: bool,
) -> None:
    n_time = 2
    time = np.datetime64("2026-01-01") + np.arange(n_time).astype("timedelta64[s]")
    satellite = ConstellationSatellite(
        id="satellite",
        orbit=CircularSatelliteOrbitModel(
            semi_major_axis=7_000_000.0,
            inclination=0.0,
            raan=0.0,
            phase_at_epoch=0.0,
            epoch=time[0],
        ),
    )
    gateway = Gateway(id="gateway", latitude=0.0, longitude=0.0, minimum_elevation=0.0)
    zero_vectors: dict[ConstellationSatellite[str, CircularSatelliteOrbitModel], npt.NDArray[np.floating]] = {
        satellite: np.zeros((n_time, 3)),
    }
    dynamics_data = DynamicsData(
        time=time,
        dcm_eci2ecef=np.repeat(np.eye(3)[None, :, :], n_time, axis=0),
        satellite_position_eci=zero_vectors,
        satellite_velocity_eci=zero_vectors,
        satellite_position_ecef={satellite: np.tile(_OVERHEAD_SATELLITE_POSITION_ECEF, (n_time, 1))},
        satellite_attitude_angular_velocity_eci=zero_vectors,
        sun_direction_eci=np.tile(sun_direction_eci, (n_time, 1)),
        sun_direction_ecef=np.tile(sun_direction_ecef, (n_time, 1)),
    )

    graphs = build_hybrid_us2c_g2c_topology(
        Constellation(satellites={0: satellite}),
        user_satellites=(),
        ground_nodes=(gateway,),
        dynamics_data=dynamics_data,
        sun_exclusion_angle=np.deg2rad(10.0),
    )

    assert len(graphs) == n_time
    for graph in graphs:
        assert graph.has_edge(gateway, satellite) is expected_linked
        assert graph.has_edge(satellite, gateway) is expected_linked
