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
