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
