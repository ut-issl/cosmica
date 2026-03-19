"""Constellation models.

A constellation has two distinct ID concepts:

- **Structural ID** (`SatelliteId`, the dict key in `Constellation.satellites`):
  Describes the satellite's role within the constellation structure.
  For example, `tuple[int, int]` might encode `(plane_id, in_plane_index)`.
  Topology builders and plotting functions read structural information
  **only** from these dict keys — never from `satellite.id`.

- **Node ID** (`ConstellationSatellite.id`, inherited from `Node`):
  The satellite's identity for use as a graph node, `DynamicsData` key,
  and display (`__str__`). This is an opaque identifier that has nothing
  to do with constellation structure.

These two IDs may happen to share the same value, but they serve
fundamentally different purposes. Consumers of `Constellation` must
use the dict key for structural queries and the satellite object itself
(not `.id`) for graph/dynamics operations.
"""

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from .orbit import CircularSatelliteOrbitModel
from .satellite import ConstellationSatellite


@dataclass(frozen=True, slots=True, kw_only=True)
class Constellation[SatelliteId: Hashable]:
    """A constellation of satellites.

    `satellites` maps a structural identifier (`SatelliteId`) to each
    satellite object. The key type encodes whatever structure the
    constellation has — e.g., `tuple[int, int]` for `(plane_id, in_plane_index)`
    in a multi-plane constellation, or plain `int` for a flat constellation.

    Consumers that need structural information (like topology builders)
    should declare the `SatelliteId` type they expect in their signature,
    e.g., `Constellation[tuple[int, int]]`.
    """

    satellites: dict[SatelliteId, ConstellationSatellite]


def build_walker_delta_constellation(
    semi_major_axis: float,
    inclination: float,
    n_total_sats: int,
    n_geometry_planes: int,
    phasing_factor: int,
    epoch: np.datetime64,
) -> Constellation[tuple[int, int]]:
    """Create a multi-plane Walker Delta constellation."""
    assert semi_major_axis > 0, "Semi-major axis must be positive."
    assert 0 <= inclination <= np.pi, "Inclination must be between 0 and pi radians."
    assert n_total_sats > 0, "Total number of satellites must be positive."
    assert 0 < n_geometry_planes <= n_total_sats, (
        "Number of geometry planes must be positive and less than or equal to total number of satellites."
    )
    assert n_total_sats % n_geometry_planes == 0, "Total number of satellites must be divisible by number of planes."
    assert 0 <= phasing_factor < n_geometry_planes, (
        "Phasing factor must be less than number of planes and non-negative."
    )

    n_sats_per_plane = n_total_sats // n_geometry_planes

    satellites: dict[tuple[int, int], ConstellationSatellite] = {}

    for plane_id in range(1, n_geometry_planes + 1):
        raan = plane_id * (2 * np.pi / n_geometry_planes)

        for sat_id_in_plane in range(1, n_sats_per_plane + 1):
            phase = (sat_id_in_plane - 1) * (2 * np.pi / n_sats_per_plane) + (plane_id - 1) * phasing_factor * (
                2 * np.pi / n_total_sats
            )
            sat_id = (plane_id, sat_id_in_plane)
            sat_orbit = CircularSatelliteOrbitModel(
                semi_major_axis=semi_major_axis,
                inclination=inclination,
                raan=raan,
                phase_at_epoch=phase,
                epoch=epoch,
            )
            sat = ConstellationSatellite(
                id=(plane_id, sat_id_in_plane),
                orbit=sat_orbit,
            )
            satellites[sat_id] = sat

    return Constellation(satellites=satellites)
