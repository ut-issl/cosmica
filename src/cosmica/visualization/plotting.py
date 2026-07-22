from __future__ import annotations

__all__ = [
    "visualize_grouped_constellation",
]
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from cosmica.utils.constants import EARTH_RADIUS

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from cosmica.dynamics import SatelliteOrbitState
    from cosmica.models import Constellation, ConstellationSatellite


type PlaneId = int
type InPlaneIndex = int


def visualize_grouped_constellation(
    constellation: Constellation[tuple[PlaneId, InPlaneIndex]],
    propagation_result: Mapping[ConstellationSatellite, SatelliteOrbitState],
    *,
    time_index: int = 0,
) -> None:
    """Visualize a grouped constellation in 3D.

    The constellation must be parameterized as `Constellation[tuple[int, int]]`
    where each key is `(plane_id, in_plane_index)`. Plane structure is derived
    entirely from the dict keys — not from orbital parameters or `satellite.id`.

    Plots one orbital trajectory per plane (using the first satellite in each
    plane) and marks all satellite positions at the given `time_index`.

    Args:
        constellation: Constellation with `(plane_id, in_plane_index)` keys.
        propagation_result: Mapping from satellite objects to propagation results.
        time_index: Time step index at which to plot satellite positions.

    """
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    cmap = mpl.colormaps["tab20"]

    # Plot the Earth
    ax.plot_surface(*_ms(0, 0, 0, EARTH_RADIUS), color="blue", alpha=0.2)

    # Group satellites by plane_id (first element of the structural key).
    # All structural information comes from dict keys, not from satellite.id.
    planes_dd: defaultdict[PlaneId, list[tuple[InPlaneIndex, ConstellationSatellite]]] = defaultdict(list)
    for (plane_id, in_plane_index), satellite in constellation.satellites.items():
        planes_dd[plane_id].append((in_plane_index, satellite))

    planes = dict(planes_dd)

    for i, plane_id in enumerate(sorted(planes)):  # sort by plane_id
        # Sort by in_plane_index to find the first satellite
        sats_in_plane = sorted(planes[plane_id])
        first_satellite = sats_in_plane[0][1]

        # Plot trajectory of the first satellite in this plane
        ax.plot(
            propagation_result[first_satellite].position_eci[:, 0],
            propagation_result[first_satellite].position_eci[:, 1],
            propagation_result[first_satellite].position_eci[:, 2],
            color=cmap(i),
            linewidth=0.5,
            label=f"Plane {plane_id}",
        )

    # Plot all satellite positions at the given time index
    for satellite in constellation.satellites.values():
        ax.plot(
            propagation_result[satellite].position_eci[time_index, 0],
            propagation_result[satellite].position_eci[time_index, 1],
            propagation_result[satellite].position_eci[time_index, 2],
            "ro",
            markersize=2,
        )

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_box_aspect([1, 1, 1])

    ax.legend()

    plt.show()


def _ms(
    x: float,
    y: float,
    z: float,
    radius: float,
    resolution: float = 20,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the coordinates for plotting a sphere centered at (x,y,z)."""
    u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
    xx = radius * np.cos(u) * np.sin(v) + x
    yy = radius * np.sin(u) * np.sin(v) + y
    zz = radius * np.cos(v) + z
    return (xx, yy, zz)
