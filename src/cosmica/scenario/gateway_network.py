from __future__ import annotations

__all__ = [
    "N_MAJOR_CITIES",
    "DemandWeightModel",
    "build_city_gateway_network",
    "build_default_gateway_network",
    "build_geographically_distributed_gateway_network",
    "get_gateway_city_name",
    "get_gateway_econ_multiplier",
]

from typing import TYPE_CHECKING, Annotated, Final, Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import Doc

from cosmica.models import Gateway
from cosmica.utils.coordinates import great_circle_distance

if TYPE_CHECKING:
    from collections.abc import Sequence

type DemandWeightModel = Literal["population", "uniform", "gdp", "penetration", "subscriber"]
"""Which attractiveness to use when weighting demand.

"population"/"gdp"/"penetration"/"subscriber" reshape the population gravity model via a
per-gateway multiplier (see `get_gateway_econ_multiplier`); "uniform" gives every gateway the
same weight `1/N` regardless of population or economy (handled in `compute_gateway_demand_weights`).
"""

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

# Country and economic attributes for each candidate city, aligned by index with
# `_MAJOR_CITY_LOCATIONS`. Each entry is
# (iso3, gdp_per_capita_ppp_usd, internet_penetration, fixed_broadband_subs_per_100).
# Values are approximate ~2023 figures: GDP per capita (PPP, current international $) and
# fixed-broadband subscriptions per 100 people from the World Bank, individuals-using-the-
# Internet from ITU Facts & Figures 2024. They are intended as relative demand-weighting
# factors for scenarios, not for precise economic analysis. Attributes are country-level
# (a city inherits its country's figures), so several cities share the same values.
_MAJOR_CITY_ECON: Final[tuple[tuple[str, float, float, float], ...]] = (
    ("JPN", 51800.0, 0.85, 35.6),  # Tokyo
    ("IND", 9180.0, 0.52, 3.0),  # Delhi
    ("CHN", 23300.0, 0.77, 43.0),  # Shanghai
    ("BRA", 19900.0, 0.84, 22.6),  # Sao Paulo
    ("MEX", 23800.0, 0.78, 19.8),  # Mexico City
    ("EGY", 16600.0, 0.72, 11.4),  # Cairo
    ("IND", 9180.0, 0.52, 3.0),  # Mumbai
    ("CHN", 23300.0, 0.77, 43.0),  # Beijing
    ("BGD", 9000.0, 0.39, 7.9),  # Dhaka
    ("JPN", 51800.0, 0.85, 35.6),  # Osaka
    ("USA", 80400.0, 0.92, 37.9),  # New York
    ("PAK", 6400.0, 0.33, 1.4),  # Karachi
    ("ARG", 28700.0, 0.88, 22.5),  # Buenos Aires
    ("TUR", 41900.0, 0.86, 22.5),  # Istanbul
    ("NGA", 6100.0, 0.55, 0.05),  # Lagos
    ("COD", 1570.0, 0.27, 0.03),  # Kinshasa
    ("PHL", 11300.0, 0.83, 7.5),  # Manila
    ("BRA", 19900.0, 0.84, 22.6),  # Rio de Janeiro
    ("CHN", 23300.0, 0.77, 43.0),  # Guangzhou
    ("RUS", 38300.0, 0.90, 23.5),  # Moscow
    ("USA", 80400.0, 0.92, 37.9),  # Los Angeles
    ("FRA", 58800.0, 0.93, 46.6),  # Paris
    ("THA", 22700.0, 0.88, 19.0),  # Bangkok
    ("IDN", 15900.0, 0.68, 4.9),  # Jakarta
    ("GBR", 59000.0, 0.96, 41.0),  # London
    ("PER", 16600.0, 0.74, 9.5),  # Lima
    ("COL", 21600.0, 0.74, 16.9),  # Bogota
    ("USA", 80400.0, 0.92, 37.9),  # Chicago
    ("ZAF", 16000.0, 0.72, 2.5),  # Johannesburg
    ("IRN", 19600.0, 0.79, 12.5),  # Tehran
    ("KOR", 54000.0, 0.98, 44.5),  # Seoul
    ("HKG", 69000.0, 0.93, 37.0),  # Hong Kong
    ("IND", 9180.0, 0.52, 3.0),  # Chennai
    ("IRQ", 11000.0, 0.79, 2.0),  # Baghdad
    ("SAU", 59000.0, 0.99, 23.0),  # Riyadh
    ("SGP", 127000.0, 0.96, 28.0),  # Singapore
    ("AUS", 62600.0, 0.96, 34.9),  # Sydney
    ("ESP", 50000.0, 0.95, 35.0),  # Madrid
    ("CAN", 60200.0, 0.93, 42.0),  # Toronto
    ("KEN", 6600.0, 0.40, 0.6),  # Nairobi
)

assert len(_MAJOR_CITY_ECON) == len(_MAJOR_CITY_LOCATIONS), (
    "_MAJOR_CITY_ECON must be aligned one-to-one with _MAJOR_CITY_LOCATIONS"
)

N_MAJOR_CITIES: Final[int] = len(_MAJOR_CITY_LOCATIONS)
"""Number of candidate cities available to the city-based gateway builders."""


def get_gateway_city_name(gateway: Gateway) -> str | None:
    """Return the major-city name for a gateway's id, or None if the id is outside the city table.

    `build_city_gateway_network` and `build_geographically_distributed_gateway_network` assign each
    gateway an id equal to its index in `_MAJOR_CITY_LOCATIONS`, so this recovers the city they were
    placed at. The lookup is purely id-based: a gateway built by other means whose id happens to fall
    in `[0, len(_MAJOR_CITY_LOCATIONS))` still returns the city at that index, so use this only for
    gateways created by the city-based builders above.
    """
    gid = gateway.id
    if isinstance(gid, int) and 0 <= gid < len(_MAJOR_CITY_LOCATIONS):
        return _MAJOR_CITY_LOCATIONS[gid][0]
    return None


def get_gateway_econ_multiplier(
    gateways: Annotated[Sequence[Gateway], Doc("Gateways built by city-based builders (ids index the econ table).")],
    model: Annotated[DemandWeightModel, Doc("Which per-gateway attractiveness to return.")],
) -> Annotated[
    npt.NDArray[np.floating],
    Doc("Per-gateway demand-weight multiplier m_i, shape (len(gateways),)."),
]:
    """Return the per-gateway multiplier `m_i` used to reshape population weights into a demand model.

    Demand weights combine the population aggregated to each gateway `P_i` with this multiplier as
    `w_i ∝ P_i · m_i` (see `compute_gateway_demand_weights`). The multiplier encodes the attractiveness
    of the gateway's country:

    - `"population"`: `m_i = 1` (pure population gravity, the historical default).
    - `"gdp"`: `m_i = g_i` (GDP per capita), so `w_i ∝ P_i·g_i` is the region's total GDP.
    - `"penetration"`: `m_i = r_i` (internet penetration), so `w_i ∝ P_i·r_i` is the online population.
    - `"subscriber"`: `m_i = s_i` (fixed-broadband subscribers per capita), so `w_i ∝ P_i·s_i`.

    The lookup is id-based like `get_gateway_city_name`: every gateway id must index `_MAJOR_CITY_ECON`,
    so only gateways created by the city-based builders are supported for the non-population models.
    The `"uniform"` model has no per-gateway multiplier and is handled by `compute_gateway_demand_weights`.
    """
    if model == "population":
        return np.ones(len(gateways), dtype=np.float64)
    if model == "uniform":
        msg = "The 'uniform' demand model has no per-gateway multiplier; use compute_gateway_demand_weights."
        raise ValueError(msg)

    multipliers = np.empty(len(gateways), dtype=np.float64)
    for i, gateway in enumerate(gateways):
        gid = gateway.id
        if not (isinstance(gid, int) and 0 <= gid < len(_MAJOR_CITY_ECON)):
            msg = (
                f"Gateway id {gid!r} is outside the city economic table [0, {len(_MAJOR_CITY_ECON)}); "
                f"the {model!r} demand model only supports gateways built by the city-based builders."
            )
            raise ValueError(msg)
        _, gdp_per_capita, penetration, broadband_per_100 = _MAJOR_CITY_ECON[gid]
        if model == "gdp":
            multipliers[i] = gdp_per_capita
        elif model == "penetration":
            multipliers[i] = penetration
        elif model == "subscriber":
            multipliers[i] = broadband_per_100 / 100.0
        else:
            msg = f"Unknown demand weight model: {model!r}"
            raise ValueError(msg)
    return multipliers


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


def build_geographically_distributed_gateway_network(
    n_gateways: Annotated[
        int,
        Doc("Number of gateways to build. Must not exceed the number of predefined cities."),
    ] = 20,
    *,
    minimum_elevation: Annotated[
        float,
        Doc("Minimum elevation angle of every gateway in radians."),
    ] = np.deg2rad(30.0),
) -> Annotated[
    list[Gateway],
    Doc("Gateways at major cities chosen to be spread around the globe (farthest-point selection)."),
]:
    """Build a gateway network whose cities are spread as evenly as possible over the globe.

    Unlike `build_city_gateway_network`, which simply takes the most populous cities and is
    therefore biased toward Asia and South America, this greedily selects cities by
    farthest-point sampling on the sphere: starting from the first city, each next city is the
    candidate whose minimum great-circle distance to the already-selected cities is largest.
    This pulls in geographically underrepresented sites (Western Europe, the US west coast,
    Australia, ...) so the gateways cover the world more uniformly.

    Each gateway keeps the id equal to its index in `_MAJOR_CITY_LOCATIONS` (so the ids are not
    contiguous), which keeps the id-to-city-name mapping identical to `build_city_gateway_network`.
    The result is deterministic: repeated calls return equal gateways.
    """
    assert 0 < n_gateways <= len(_MAJOR_CITY_LOCATIONS), (
        f"n_gateways must be in [1, {len(_MAJOR_CITY_LOCATIONS)}], but got {n_gateways}"
    )
    latitudes = np.deg2rad(np.array([lat for _, lat, _ in _MAJOR_CITY_LOCATIONS]))
    longitudes = np.deg2rad(np.array([lon for _, _, lon in _MAJOR_CITY_LOCATIONS]))

    selected: list[int] = [0]  # 決定論性のため人口最大都市 (index 0) を起点に固定
    while len(selected) < n_gateways:
        distance_to_selected = great_circle_distance(
            latitudes[:, None],
            longitudes[:, None],
            latitudes[selected][None, :],
            longitudes[selected][None, :],
        )
        min_distance = distance_to_selected.min(axis=1)
        min_distance[selected] = -np.inf  # 既選択都市は再選択しない
        selected.append(int(np.argmax(min_distance)))

    return [
        Gateway(
            id=city_index,
            latitude=float(latitudes[city_index]),
            longitude=float(longitudes[city_index]),
            minimum_elevation=minimum_elevation,
        )
        for city_index in sorted(selected)
    ]
