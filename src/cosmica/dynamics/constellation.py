from __future__ import annotations

__all__ = [
    "MOPCSatelliteKey",
    "MultiOrbitalPlaneConstellation",
    "SatelliteConstellation",
]

import logging
import tomllib
from abc import ABC
from collections.abc import Hashable, Mapping, Sequence
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from typing_extensions import deprecated

if TYPE_CHECKING:
    import numpy.typing as npt


from cosmica.models import ConstellationSatellite

from .orbit import SatelliteOrbit, SatelliteOrbitState, make_satellite_orbit

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

    @classmethod
    @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
    def from_toml_file(cls, toml_file_path: Path | str) -> MultiOrbitalPlaneConstellation:
        toml_file_path = Path(toml_file_path)
        with toml_file_path.open("rb") as f:
            toml_data = tomllib.load(f)

        epoch = np.datetime64(toml_data["epoch"].astimezone(tz=UTC).replace(tzinfo=None))

        @deprecated("Construction of objects from TOML files is deprecated and will be removed in future versions.")
        def parse_satellite_item(
            item: dict[str, Any],
        ) -> tuple[ConstellationSatellite[MOPCSatelliteKey], TOrbit]:
            plane_id = item.pop("plane_id")
            satellite_id = item.pop("id")

            # Convert degrees to radians
            item["semi_major_axis"] = item.pop("sma_m")
            item["inclination"] = np.radians(item.pop("inc_deg"))
            item["raan"] = np.radians(item.pop("raan_deg"))
            item["phase_at_epoch"] = np.radians(item.pop("phase_at_epoch_deg"))
            item["epoch"] = epoch

            orbit_type = item.pop("orbit_type")
            return ConstellationSatellite(  # type: ignore[return-value]
                id=MOPCSatelliteKey(plane_id, satellite_id),
            ), make_satellite_orbit(
                orbit_type=orbit_type,
                **item,
            )

        satellite_orbits = dict(map(parse_satellite_item, toml_data["satellites"]))

        return cls(satellite_orbits)
