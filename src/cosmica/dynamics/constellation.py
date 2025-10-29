from __future__ import annotations

__all__ = [
    "MOPCSatelliteKey",
    "MultiOrbitalPlaneConstellation",
    "SatelliteConstellation",
]

import logging
from abc import ABC
from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from cosmica.models import ConstellationSatellite

from .orbit import SatelliteOrbit, SatelliteOrbitState

logger = logging.getLogger(__name__)

# Type of the key used to identify a satellite in a constellation


class SatelliteConstellation[T: Hashable, TOrbit: SatelliteOrbit](ABC):
    """A constellation of satellites.

    `satellite_orbits` is a dictionary mapping satellite keys to satellite orbits.
    """

    satellite_orbits: Mapping[ConstellationSatellite[T], TOrbit]
    satellites: Sequence[ConstellationSatellite[T]]

    def propagate(self, time: npt.NDArray[np.datetime64]) -> dict[ConstellationSatellite[T], SatelliteOrbitState]:
        return {sat: orbit.propagate(time) for sat, orbit in self.satellite_orbits.items()}


class MOPCSatelliteKey(NamedTuple):
    plane_id: int
    satellite_id: int

    def __str__(self) -> str:
        return f"({self.plane_id},{self.satellite_id})"


class MultiOrbitalPlaneConstellation[TOrbit: SatelliteOrbit](SatelliteConstellation[MOPCSatelliteKey, TOrbit]):
    """A constellation of satellites in multiple orbital planes.

    The satellite key is a tuple of the plane ID and the satellite ID.
    """

    def __init__(
        self,
        satellite_orbits: Mapping[ConstellationSatellite[MOPCSatelliteKey], TOrbit],
    ) -> None:
        self.satellite_orbits = satellite_orbits

        self.satellites = tuple(self.satellite_orbits.keys())
        self.plane_ids = sorted({sat.id.plane_id for sat in self.satellites})
        self.satellite_ids = sorted({sat.id.satellite_id for sat in self.satellites})

        self.plane_id_to_satellites: dict[int, list[ConstellationSatellite[MOPCSatelliteKey]]] = {
            plane_id: [sat for sat in self.satellites if sat.id.plane_id == plane_id] for plane_id in self.plane_ids
        }

        # Check if all planes have the same number of satellites
        self._all_planes_have_same_n_satellites = (
            len({len(self.plane_id_to_satellites[plane_id]) for plane_id in self.plane_ids}) == 1
        )
        if not self._all_planes_have_same_n_satellites:
            logger.warning("Not all planes have the same number of satellites.")

    @property
    def n_satellites_per_plane(self) -> int:
        """Number of satellites per orbital plane.

        It is assumed that all planes have the same number of satellites. Otherwise, it will raise an exception.
        """
        # Check that all planes have the same number of satellites
        if not self._all_planes_have_same_n_satellites:
            msg = "Not all planes have the same number of satellites."
            raise ValueError(msg)
        return len(self.plane_id_to_satellites[self.plane_ids[0]])
