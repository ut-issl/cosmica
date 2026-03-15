from __future__ import annotations

__all__ = [
    "build_default_gateway_network",
]

import numpy as np

from cosmica.models import Gateway


def build_default_gateway_network() -> list[Gateway]:
    """Build a list of default gateways."""
    return [
        Gateway(id=0, latitude=np.deg2rad(36.0), longitude=np.deg2rad(139.0), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=1, latitude=np.deg2rad(40.0), longitude=np.deg2rad(-120.0), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=2, latitude=np.deg2rad(33.0), longitude=np.deg2rad(130.0), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=3, latitude=np.deg2rad(47.0), longitude=np.deg2rad(9.0), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=4, latitude=np.deg2rad(47.0), longitude=np.deg2rad(-70.0), minimum_elevation=np.deg2rad(30.0)),
    ]
