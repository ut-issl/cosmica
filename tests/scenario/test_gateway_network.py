import numpy as np

from cosmica.models import Gateway
from cosmica.scenario.gateway_network import build_default_gateway_network


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
