from __future__ import annotations

__all__ = [
    "visualize_multi_orbital_plane_constellation",
]
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from cosmica.models import ConstellationSatellite

if TYPE_CHECKING:
    from cosmica.dynamics import (
        CircularSatelliteOrbit,
        MultiOrbitalPlaneConstellation,
        SatelliteOrbitState,
    )

from cosmica.utils.constants import EARTH_RADIUS


def visualize_multi_orbital_plane_constellation(
    constellation: MultiOrbitalPlaneConstellation[CircularSatelliteOrbit],
    propagation_result: Mapping[ConstellationSatellite, SatelliteOrbitState],
    time_index: int = 0,
) -> None:
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    cmap = mpl.colormaps["tab20"]

    # Plot the Earth
    ax.plot_surface(*_ms(0, 0, 0, EARTH_RADIUS), color="blue", alpha=0.2)

    # Plot the orbit trajectory of the first satellite in each orbital plane
    for i, plane_id in enumerate(constellation.plane_ids):
        satellite = constellation.plane_id_to_satellites[plane_id][0]
        ax.plot(
            propagation_result[satellite].position_eci[:, 0],
            propagation_result[satellite].position_eci[:, 1],
            propagation_result[satellite].position_eci[:, 2],
            color=cmap(i),
            linewidth=0.5,
            label=f"Plane {plane_id}",
        )
        for satellite in constellation.satellites:
            # Plot the position of each satellite at the `time_index`
            ax.plot(
                propagation_result[satellite].position_eci[time_index, 0],
                propagation_result[satellite].position_eci[time_index, 1],
                propagation_result[satellite].position_eci[time_index, 2],
                "ro",
                markersize=2,
            )

    # Set plot labels and aspect ratio
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
    u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]  # type: ignore[misc]
    xx = radius * np.cos(u) * np.sin(v) + x
    yy = radius * np.sin(u) * np.sin(v) + y
    zz = radius * np.cos(v) + z
    return (xx, yy, zz)
