from __future__ import annotations

__all__ = [
    "DEFAULT_GATEWAYS",
    "build_default_gateway_network",
    "build_gateway_network",
]
from collections.abc import Collection, Hashable, Mapping
from typing import Annotated, cast

import numpy as np
from typing_extensions import Doc

from cosmica.models import Gateway

DEFAULT_GATEWAYS: list[Gateway[int]] = [
    Gateway(id=0, latitude=np.deg2rad(36.0), longitude=np.deg2rad(139.0), minimum_elevation=np.deg2rad(30.0)),
    Gateway(id=1, latitude=np.deg2rad(40.0), longitude=np.deg2rad(-120.0), minimum_elevation=np.deg2rad(30.0)),
    Gateway(id=2, latitude=np.deg2rad(33.0), longitude=np.deg2rad(130.0), minimum_elevation=np.deg2rad(30.0)),
    Gateway(id=3, latitude=np.deg2rad(47.0), longitude=np.deg2rad(9.0), minimum_elevation=np.deg2rad(30.0)),
    Gateway(id=4, latitude=np.deg2rad(47.0), longitude=np.deg2rad(-70.0), minimum_elevation=np.deg2rad(30.0)),
]

_REQUIRED_FIELDS: frozenset[str] = frozenset({"lat_deg", "lon_deg", "min_el_deg"})


def build_default_gateway_network(
    n_stations: Annotated[
        int,
        Doc("Number of default gateways to include. Ignored when indexes is provided."),
    ] = 5,
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
        default_gateway_by_id = {gateway.id: gateway for gateway in DEFAULT_GATEWAYS}
        invalid_ids = [idx for idx in indexes if idx not in default_gateway_by_id]
        assert not invalid_ids, f"Unknown gateway ids: {sorted(invalid_ids)}."
        return [default_gateway_by_id[gateway_id] for gateway_id in indexes]
    else:
        assert 1 <= n_stations <= len(DEFAULT_GATEWAYS), f"n_stations must be between 1 and {len(DEFAULT_GATEWAYS)}."
        return list(DEFAULT_GATEWAYS[:n_stations])


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
    assert len(gateway_map) > 0, "gateway_map must contain at least one gateway."
    gateway_list: list[Gateway] = []
    for gateway_id, specs in gateway_map.items():
        assert isinstance(specs, Mapping), f"Gateway {gateway_id} parameters must be a mapping."
        missing = _REQUIRED_FIELDS.difference(specs.keys())
        assert not missing, f"Gateway {gateway_id} missing required fields: {sorted(missing)}."
        n_terminals = cast("int", specs.get("n_terminals", 1))
        gateway_list.append(
            Gateway(
                id=gateway_id,
                latitude=np.deg2rad(float(specs["lat_deg"])),
                longitude=np.deg2rad(float(specs["lon_deg"])),
                minimum_elevation=np.deg2rad(float(specs["min_el_deg"])),
                altitude=float(specs.get("altitude", specs.get("altitude_m", 0.0))),
                n_terminals=n_terminals,
            ),
        )
    return gateway_list
