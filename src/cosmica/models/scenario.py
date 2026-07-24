__all__ = [
    "Scenario",
]
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .constellation import Constellation
from .demand import Demand
from .gateway import Gateway
from .user import User


@dataclass(frozen=True, kw_only=True, slots=True)
class Scenario:
    """The simulation scenario."""

    time: npt.NDArray[np.datetime64]
    constellation: Constellation[Any, Any, Any]
    gateways: list[Gateway[Any]] = field(default_factory=list)
    users: list[User[Any]] = field(default_factory=list)
    demands: list[Demand[Any]] = field(default_factory=list)

    def build_topology(self) -> None:
        """Build the topology of the scenario."""
        # self.constellation.build_topology(self.gateways, self.users, self.demands)
