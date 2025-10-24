import importlib.resources
import logging
from functools import cache
from typing import Annotated

import numpy as np
import numpy.typing as npt
from typing_extensions import Doc

from cosmica.utils.constants import EARTH_RADIUS

logger = logging.getLogger(__name__)


@cache
def _load_numpy_array_from_assets_dir(filename: str) -> npt.NDArray[np.floating]:
    rel_path = f"assets/{filename}"
    resource = importlib.resources.files("cosmica.scenario").joinpath(rel_path)
    with importlib.resources.as_file(resource) as f:
        logger.debug(f"Loading NumPy array from {f}")
        return np.load(f)


def get_ais_density_data() -> Annotated[
    npt.NDArray[np.floating],
    Doc(
        "AIS density data [seconds/m2] in Oct 2023 as a NumPy array with shape (180, 359)."
        " The rows are latitude (+90 deg to -90 deg) and the columns are longitude (-179.5 deg to +179.5 deg).",
    ),
]:
    """Get AIS density data."""
    # Conversion [hours/km2] -> [seconds/m2]
    return _load_numpy_array_from_assets_dir("maritime_20231001_non-loitering.npy") * 3600 / 1e6


def sample_demand_locations(
    n_samples: Annotated[int, Doc("Number of demand locations to sample.")],
    *,
    rng: Annotated[np.random.Generator | None, Doc("NumPy random number generator. If None, use default.")] = None,
) -> Annotated[
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    Doc("Sampled demand locations as a tuple of (longitude, latitude) in radians, each with shape (n_samples,)."),
]:
    """Sample maritime demand locations based on AIS density data.

    The probability of sampling a location is proportional to the AIS density at that location,
    with the correction for the area difference between the lower and higher latitudes.
    """
    rng = rng if rng is not None else np.random.default_rng()

    data = get_ais_density_data()
    lon = np.deg2rad(np.arange(-179, 180))
    lat = np.deg2rad(np.arange(89.5, -90.5, -1))
    longitude, latitude = np.meshgrid(lon, lat)

    d_lon = abs(float(lon[1] - lon[0]))
    d_lat = abs(float(lat[1] - lat[0]))

    area = EARTH_RADIUS**2 * np.cos(latitude) * d_lon * d_lat
    volume = data * area

    is_valid = ~np.isnan(volume)

    probability = volume[is_valid].flatten() / np.nansum(volume)
    assert np.isclose(np.nansum(probability), 1)

    sample_indices = rng.choice(np.sum(is_valid), size=n_samples, replace=False, p=probability)

    return longitude[is_valid].flatten()[sample_indices], latitude[is_valid].flatten()[sample_indices]
