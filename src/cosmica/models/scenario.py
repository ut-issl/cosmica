__all__ = [
    "Scenario",
]
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

# from cosmica.dynamics import SatelliteConstellation # This leads to circular import.
# Maybe we should move constellation to the `models` subpackage?
from .demand import Demand
from .gateway import Gateway
from .user import User


@dataclass(frozen=True, kw_only=True, slots=True)
class Scenario:
    """The simulation scenario."""

    time: npt.NDArray[np.datetime64]
    # constellation: SatelliteConstellation
    gateways: list[Gateway] = field(default_factory=list)
    users: list[User] = field(default_factory=list)
    demands: list[Demand] = field(default_factory=list)

    def build_topology(self) -> None:
        """Build the topology of the scenario."""
        # self.constellation.build_topology(self.gateways, self.users, self.demands)
