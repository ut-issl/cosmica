import numpy as np
import pytest

from cosmica.models import Gateway
from cosmica.scenario.gateway_network import (
    build_city_gateway_network,
    build_default_gateway_network,
    get_gateway_econ_multiplier,
)


def test_build_default_gateway_network_returns_expected_gateways() -> None:
    gateways = build_default_gateway_network()

    expected_gateways = [
        (0, 36.0, 139.0, 30.0),
        (1, 40.0, -120.0, 30.0),
        (2, 33.0, 130.0, 30.0),
        (3, 47.0, 9.0, 30.0),
        (4, 47.0, -70.0, 30.0),
    ]

    assert len(gateways) == len(expected_gateways)
    for gateway, (gateway_id, lat_deg, lon_deg, min_el_deg) in zip(gateways, expected_gateways, strict=True):
        assert isinstance(gateway, Gateway)
        assert gateway.id == gateway_id
        assert np.isclose(gateway.latitude, np.deg2rad(lat_deg))
        assert np.isclose(gateway.longitude, np.deg2rad(lon_deg))
        assert np.isclose(gateway.minimum_elevation, np.deg2rad(min_el_deg))
        assert gateway.altitude == 0.0
        assert gateway.n_terminals == 1


def test_build_default_gateway_network_returns_a_new_list_each_time() -> None:
    gateways = build_default_gateway_network()
    gateways.pop()

    rebuilt_gateways = build_default_gateway_network()

    assert len(rebuilt_gateways) == 5
    assert [gateway.id for gateway in rebuilt_gateways] == [0, 1, 2, 3, 4]


def test_build_city_gateway_network_returns_requested_number_of_gateways() -> None:
    gateways = build_city_gateway_network(20)

    assert len(gateways) == 20
    for gateway in gateways:
        assert isinstance(gateway, Gateway)
        assert np.isclose(gateway.minimum_elevation, np.deg2rad(30.0))


def test_build_city_gateway_network_ids_are_unique_and_sequential() -> None:
    gateways = build_city_gateway_network(30)

    assert [gateway.id for gateway in gateways] == list(range(30))
    assert len({gateway.global_id for gateway in gateways}) == 30


def test_build_city_gateway_network_is_deterministic() -> None:
    first = build_city_gateway_network(25)
    second = build_city_gateway_network(25)

    assert first == second


def test_build_city_gateway_network_first_gateway_is_tokyo() -> None:
    gateways = build_city_gateway_network(1)

    assert np.isclose(gateways[0].latitude, np.deg2rad(35.68))
    assert np.isclose(gateways[0].longitude, np.deg2rad(139.65))


def test_build_city_gateway_network_rejects_too_many_gateways() -> None:
    with pytest.raises(AssertionError):
        build_city_gateway_network(1000)


def test_get_gateway_econ_multiplier_population_is_all_ones() -> None:
    gateways = build_city_gateway_network(5)

    multipliers = get_gateway_econ_multiplier(gateways, "population")

    assert multipliers.shape == (5,)
    assert np.all(multipliers == 1.0)


def test_get_gateway_econ_multiplier_models_are_positive() -> None:
    gateways = build_city_gateway_network(10)

    for model in ("gdp", "penetration", "subscriber"):
        multipliers = get_gateway_econ_multiplier(gateways, model)
        assert multipliers.shape == (10,)
        assert np.all(multipliers > 0)


def test_get_gateway_econ_multiplier_gdp_matches_table_for_tokyo() -> None:
    gateways = build_city_gateway_network(1)  # Tokyo has id 0

    multipliers = get_gateway_econ_multiplier(gateways, "gdp")

    assert np.isclose(multipliers[0], 51800.0)


def test_get_gateway_econ_multiplier_rejects_unknown_gateway_id() -> None:
    gateway = Gateway(id=999, latitude=0.0, longitude=0.0, minimum_elevation=np.deg2rad(30.0))

    with pytest.raises(ValueError, match="outside the city economic table"):
        get_gateway_econ_multiplier([gateway], "gdp")
