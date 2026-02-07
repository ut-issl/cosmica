from __future__ import annotations

__all__ = [
    "DEFAULT_GATEWAYS",
    "build_default_gateway_network",
    "build_gateway_network",
]
from collections.abc import Collection, Hashable, Mapping
from numbers import Integral, Real
from typing import Annotated

import numpy as np
from typing_extensions import Doc

from cosmica.models import Gateway

DEFAULT_GATEWAYS: dict[int, dict[str, float | int]] = {
    0: {  # Tokyo, Japan
        "lat_deg": 36.0,
        "lon_deg": 139.0,
        "min_el_deg": 30.0,
        "n_terminals": 1,
    },
    1: {  # California, USA
        "lat_deg": 40.0,
        "lon_deg": -120,
        "min_el_deg": 30.0,
        "n_terminals": 1,
    },
    2: {  # Nagasaki, Japan
        "lat_deg": 33,
        "lon_deg": 130,
        "min_el_deg": 30.0,
        "n_terminals": 1,
    },
    3: {  # Switzerland
        "lat_deg": 47.0,
        "lon_deg": 9.0,
        "min_el_deg": 30.0,
        "n_terminals": 1,
    },
    4: {  # Quebec, Canada
        "lat_deg": 47.0,
        "lon_deg": -70.0,
        "min_el_deg": 30.0,
        "n_terminals": 1,
    },
}

_REQUIRED_FIELDS: frozenset[str] = frozenset({"lat_deg", "lon_deg", "min_el_deg"})


def build_default_gateway_network(
    n_stations: Annotated[int, Doc("Number of default gateways to include. Ignored if indexes is provided.")] = 5,
    indexes: Annotated[
        Collection[int] | None,
        Doc("Optional subset of default gateway IDs to include."),
    ] = None,
) -> list[Gateway]:
    """Build a list of default gateways.

    If `indexes` is provided, only those IDs are included. Otherwise the first
    `n_stations` default gateways are returned in key order.
    """
    if indexes is not None:
        assert len(indexes) > 0, "indexes must be non-empty when provided."
        invalid_ids = [idx for idx in indexes if idx not in DEFAULT_GATEWAYS]
        assert not invalid_ids, f"Unknown gateway ids: {sorted(invalid_ids)}."
        selected_ids = list(indexes)
    else:
        assert 1 <= n_stations <= len(DEFAULT_GATEWAYS), f"n_stations must be between 1 and {len(DEFAULT_GATEWAYS)}."
        selected_ids = sorted(DEFAULT_GATEWAYS)[:n_stations]

    gateway_map: Mapping[Hashable, Mapping[str, float | int]] = {
        gateway_id: DEFAULT_GATEWAYS[gateway_id] for gateway_id in selected_ids
    }
    return build_gateway_network(gateway_map)


def build_gateway_network(
    gateway_map: Annotated[
        Mapping[Hashable, Mapping[str, float | int]],
        Doc("Mapping of gateway ID to gateway parameters."),
    ],
) -> list[Gateway]:
    """Build a list of gateways from a mapping.

    Required fields for each gateway:
    - `lat_deg`: Latitude in degrees (-90 to 90).
    - `lon_deg`: Longitude in degrees (-180 to 180).
    - `min_el_deg`: Minimum elevation angle in degrees (0 to 90).

    Optional fields:
    - `altitude` or `altitude_m`: Gateway altitude in meters.
    - `n_terminals`: Number of terminals (positive integer).
    """
    _validate_gateway_map(gateway_map)

    gateway_list: list[Gateway] = []
    for gateway_id, specs in gateway_map.items():
        altitude = specs.get("altitude", specs.get("altitude_m", 0.0))
        n_terminals_raw = specs.get("n_terminals", 1)
        assert isinstance(n_terminals_raw, Integral)
        n_terminals = int(n_terminals_raw)
        gateway_list.append(
            Gateway(
                id=gateway_id,
                latitude=np.deg2rad(float(specs["lat_deg"])),
                longitude=np.deg2rad(float(specs["lon_deg"])),
                minimum_elevation=np.deg2rad(float(specs["min_el_deg"])),
                altitude=float(altitude),
                n_terminals=int(n_terminals),
            ),
        )
    return gateway_list


def _validate_gateway_map(gateway_map: Mapping[Hashable, Mapping[str, float | int]]) -> None:
    assert len(gateway_map) > 0, "gateway_map must contain at least one gateway."
    for gateway_id, specs in gateway_map.items():
        assert isinstance(specs, Mapping), f"Gateway {gateway_id} parameters must be a mapping."
        missing = _REQUIRED_FIELDS.difference(specs.keys())
        assert not missing, f"Gateway {gateway_id} missing required fields: {sorted(missing)}."

        lat_deg = specs["lat_deg"]
        lon_deg = specs["lon_deg"]
        min_el_deg = specs["min_el_deg"]
        n_terminals = specs.get("n_terminals")

        _assert_real_in_range(lat_deg, "lat_deg", -90.0, 90.0, gateway_id)
        _assert_real_in_range(lon_deg, "lon_deg", -180.0, 180.0, gateway_id)
        _assert_real_in_range(min_el_deg, "min_el_deg", 0.0, 90.0, gateway_id)
        if n_terminals is not None:
            assert isinstance(n_terminals, Integral), f"Gateway {gateway_id} n_terminals must be an integer."
            assert int(n_terminals) > 0, f"Gateway {gateway_id} n_terminals must be a positive integer."

        if "altitude" in specs:
            _assert_real_in_range(specs["altitude"], "altitude", -1e6, 1e6, gateway_id)
        if "altitude_m" in specs:
            _assert_real_in_range(specs["altitude_m"], "altitude_m", -1e6, 1e6, gateway_id)


def _assert_real_in_range(
    value: float,
    name: str,
    min_value: float,
    max_value: float,
    gateway_id: Hashable,
) -> None:
    assert isinstance(value, Real), f"Gateway {gateway_id} {name} must be a real number."
    assert np.isfinite(float(value)), f"Gateway {gateway_id} {name} must be finite."
    assert min_value <= float(value) <= max_value, (
        f"Gateway {gateway_id} {name} must be between {min_value} and {max_value}."
    )
