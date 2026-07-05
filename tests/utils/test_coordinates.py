import numpy as np

from cosmica.utils.constants import EARTH_RADIUS
from cosmica.utils.coordinates import great_circle_distance


def test_great_circle_distance_is_zero_for_identical_points() -> None:
    lat = np.deg2rad(35.68)
    lon = np.deg2rad(139.65)
    assert np.isclose(great_circle_distance(lat, lon, lat, lon), 0.0)


def test_great_circle_distance_tokyo_to_new_york() -> None:
    tokyo = (np.deg2rad(35.68), np.deg2rad(139.65))
    new_york = (np.deg2rad(40.71), np.deg2rad(-74.01))
    distance = great_circle_distance(tokyo[0], tokyo[1], new_york[0], new_york[1])
    assert np.isclose(distance, 1.084e7, rtol=1e-2)


def test_great_circle_distance_antipodal_points() -> None:
    distance = great_circle_distance(0.0, 0.0, 0.0, np.pi)
    assert np.isclose(distance, np.pi * EARTH_RADIUS)


def test_great_circle_distance_broadcasts_to_distance_matrix() -> None:
    lat1 = np.deg2rad(np.array([0.0, 10.0, 20.0]))
    lon1 = np.deg2rad(np.array([0.0, 10.0, 20.0]))
    lat2 = np.deg2rad(np.array([30.0, 40.0]))
    lon2 = np.deg2rad(np.array([30.0, 40.0]))

    distance_matrix = great_circle_distance(lat1[:, None], lon1[:, None], lat2[None, :], lon2[None, :])

    assert distance_matrix.shape == (3, 2)
    assert np.isclose(
        distance_matrix[1, 0],
        great_circle_distance(lat1[1], lon1[1], lat2[0], lon2[0]),
    )


def test_great_circle_distance_custom_radius() -> None:
    distance = great_circle_distance(0.0, 0.0, 0.0, np.pi / 2, radius=1.0)
    assert np.isclose(distance, np.pi / 2)
