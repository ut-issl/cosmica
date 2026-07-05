from __future__ import annotations

__all__ = [
    "build_city_gateway_network",
    "build_default_gateway_network",
]

from typing import Annotated, Final

import numpy as np
from typing_extensions import Doc

from cosmica.models import Gateway

# Major cities ordered roughly by metropolitan population, used as candidate gateway sites.
# Each entry is (name, latitude [deg], longitude [deg]).
_MAJOR_CITY_LOCATIONS: Final[tuple[tuple[str, float, float], ...]] = (
    ("Tokyo", 35.68, 139.65),
    ("Delhi", 28.61, 77.21),
    ("Shanghai", 31.23, 121.47),
    ("Sao Paulo", -23.55, -46.63),
    ("Mexico City", 19.43, -99.13),
    ("Cairo", 30.04, 31.24),
    ("Mumbai", 19.08, 72.88),
    ("Beijing", 39.90, 116.41),
    ("Dhaka", 23.81, 90.41),
    ("Osaka", 34.69, 135.50),
    ("New York", 40.71, -74.01),
    ("Karachi", 24.86, 67.01),
    ("Buenos Aires", -34.60, -58.38),
    ("Istanbul", 41.01, 28.98),
    ("Lagos", 6.52, 3.38),
    ("Kinshasa", -4.44, 15.27),
    ("Manila", 14.60, 120.98),
    ("Rio de Janeiro", -22.91, -43.17),
    ("Guangzhou", 23.13, 113.26),
    ("Moscow", 55.76, 37.62),
    ("Los Angeles", 34.05, -118.24),
    ("Paris", 48.86, 2.35),
    ("Bangkok", 13.76, 100.50),
    ("Jakarta", -6.21, 106.85),
    ("London", 51.51, -0.13),
    ("Lima", -12.05, -77.04),
    ("Bogota", 4.71, -74.07),
    ("Chicago", 41.88, -87.63),
    ("Johannesburg", -26.20, 28.05),
    ("Tehran", 35.69, 51.39),
    ("Seoul", 37.57, 126.98),
    ("Hong Kong", 22.32, 114.17),
    ("Chennai", 13.08, 80.27),
    ("Baghdad", 33.31, 44.36),
    ("Riyadh", 24.71, 46.68),
    ("Singapore", 1.35, 103.82),
    ("Sydney", -33.87, 151.21),
    ("Madrid", 40.42, -3.70),
    ("Toronto", 43.65, -79.38),
    ("Nairobi", -1.29, 36.82),
)


def build_default_gateway_network() -> list[Gateway]:
    """Build a list of default gateways."""
    return [
        Gateway(id=0, latitude=np.deg2rad(36.0), longitude=np.deg2rad(139.0), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=1, latitude=np.deg2rad(40.0), longitude=np.deg2rad(-120.0), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=2, latitude=np.deg2rad(33.0), longitude=np.deg2rad(130.0), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=3, latitude=np.deg2rad(47.0), longitude=np.deg2rad(9.0), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=4, latitude=np.deg2rad(47.0), longitude=np.deg2rad(-70.0), minimum_elevation=np.deg2rad(30.0)),
    ]


def build_city_gateway_network(
    n_gateways: Annotated[
        int,
        Doc("Number of gateways to build. Must not exceed the number of predefined cities."),
    ] = 30,
    *,
    minimum_elevation: Annotated[
        float,
        Doc("Minimum elevation angle of every gateway in radians."),
    ] = np.deg2rad(30.0),
) -> Annotated[list[Gateway], Doc("Gateways placed at major cities, with ids 0 to n_gateways - 1.")]:
    """Build a deterministic gateway network placed at major cities.

    Cities are ordered roughly by metropolitan population, so the first `n_gateways`
    cities are used. The result is deterministic: repeated calls return equal gateways.
    """
    assert 0 < n_gateways <= len(_MAJOR_CITY_LOCATIONS), (
        f"n_gateways must be in [1, {len(_MAJOR_CITY_LOCATIONS)}], but got {n_gateways}"
    )
    return [
        Gateway(
            id=gateway_id,
            latitude=np.deg2rad(latitude_deg),
            longitude=np.deg2rad(longitude_deg),
            minimum_elevation=minimum_elevation,
        )
        for gateway_id, (_, latitude_deg, longitude_deg) in enumerate(_MAJOR_CITY_LOCATIONS[:n_gateways])
    ]
