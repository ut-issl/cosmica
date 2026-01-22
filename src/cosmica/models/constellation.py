import numpy as np
from pydantic import BaseModel, ConfigDict

from .satellite import ConstellationSatellite


class ConstellationModel(BaseModel):
    """Constellation model."""


class MultiOrbitalPlaneConstellationModel(ConstellationModel):
    """Multi-orbital plane constellation model."""

    # arbitrary_types_allowed is required to allow numpy.datetime64
    model_config = ConfigDict(arbitrary_types_allowed=True)

    epoch: np.datetime64
    satellites: list[list[ConstellationSatellite]]


class WalkerDeltaConstellationModel(ConstellationModel):
    """Walker Delta constellation model."""

    # arbitrary_types_allowed is required to allow numpy.datetime64
    model_config = ConfigDict(arbitrary_types_allowed=True)

    radius: float  # meters (semi_major_axis)
    inclination: float  # radians
    total_satellites_num: int
    geometry_planes_num: int
    phasing_factor: int
    epoch: np.datetime64

    @property
    def satellites_per_plane(self) -> int:
        """Number of satellites per orbital plane."""
        return self.total_satellites_num // self.geometry_planes_num
