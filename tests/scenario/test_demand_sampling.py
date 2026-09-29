from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from cosmica.scenario import maritime, population

type _Sampler = Callable[..., tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]


def _n_nonzero_cells(data: npt.NDArray[np.floating]) -> int:
    return int(np.sum(np.nan_to_num(data) > 0))


def _n_unique_locations(longitude: npt.NDArray[np.floating], latitude: npt.NDArray[np.floating]) -> int:
    return len(np.unique(np.column_stack((longitude, latitude)), axis=0))


@pytest.mark.parametrize(
    ("sample_demand_locations", "n_nonzero_cells"),
    [
        pytest.param(
            maritime.sample_demand_locations,
            _n_nonzero_cells(maritime.get_ais_density_data()),
            id="maritime",
        ),
        pytest.param(
            population.sample_demand_locations,
            _n_nonzero_cells(population.get_population_data()),
            id="population",
        ),
    ],
)
def test_sampling_allows_more_samples_than_nonzero_cells(
    sample_demand_locations: _Sampler,
    n_nonzero_cells: int,
) -> None:
    n_samples = n_nonzero_cells + 1

    longitude, latitude = sample_demand_locations(n_samples, rng=np.random.default_rng(0))

    assert longitude.shape == latitude.shape == (n_samples,)
    assert np.all(np.abs(longitude) <= np.pi)
    assert np.all(np.abs(latitude) <= np.pi / 2)
    # Pigeonhole: some cell must repeat.
    assert _n_unique_locations(longitude, latitude) < n_samples


@pytest.mark.parametrize(
    "sample_demand_locations",
    [
        pytest.param(maritime.sample_demand_locations, id="maritime"),
        pytest.param(population.sample_demand_locations, id="population"),
    ],
)
def test_sampling_repeats_high_density_cells(sample_demand_locations: _Sampler) -> None:
    longitude, latitude = sample_demand_locations(1_000, rng=np.random.default_rng(0))

    assert _n_unique_locations(longitude, latitude) < len(longitude)
