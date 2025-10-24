import importlib.resources
import logging
from functools import cache
from typing import Annotated

import numpy as np
import numpy.typing as npt
from typing_extensions import Doc

logger = logging.getLogger(__name__)


@cache
def _load_numpy_array_from_assets_dir(filename: str) -> npt.NDArray[np.floating]:
    rel_path = f"assets/{filename}"
    resource = importlib.resources.files("cosmica.scenario").joinpath(rel_path)
    with importlib.resources.as_file(resource) as f:
        logger.debug(f"Loading NumPy array from {f}")
        return np.load(f)


def get_population_data() -> Annotated[
    npt.NDArray[np.floating],
    Doc(
        "Global population count data with shape (180, 360)."
        " The rows are latitude (+90 deg to -90 deg) and the columns are longitude (-180 deg to +180 deg)."
        " The data is based on the Global Human Settlement Layer (GHSL) population grid data (GHS-POP) for 2025.",
    ),
]:
    """Get population data."""
    return _load_numpy_array_from_assets_dir("aggregated_population_data.npy")


def sample_demand_locations(
    n_samples: Annotated[int, Doc("Number of demand locations to sample.")],
    *,
    rng: Annotated[np.random.Generator | None, Doc("NumPy random number generator. If None, use default.")] = None,
) -> Annotated[
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    Doc("Sampled demand locations as a tuple of (longitude, latitude) in radians, each with shape (n_samples,)."),
]:
    """Sample on-ground demand locations based on global population distribution data.

    The probability of sampling a location is proportional to the population count at that location.
    """
    rng = rng if rng is not None else np.random.default_rng()

    population = get_population_data()
    lon = np.deg2rad(np.arange(-179.5, 180.5))
    lat = np.flip(np.deg2rad(np.arange(-89.5, 90.5)))
    longitude, latitude = np.meshgrid(lon, lat)

    probability = population.flatten() / np.nansum(population)
    assert np.isclose(np.nansum(probability), 1)

    sample_indices = rng.choice(len(probability), size=n_samples, replace=False, p=probability)

    return longitude.flatten()[sample_indices], latitude.flatten()[sample_indices]
