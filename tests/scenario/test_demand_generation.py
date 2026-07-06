import numpy as np
import pytest

from cosmica.models import (
    ConstantCommunicationDemand,
    Gateway,
    OneTimeCommunicationDemand,
    TemporaryCommunicationDemand,
)
from cosmica.scenario.demand_generation import (
    ConstantTrafficProfile,
    OneTimeTrafficProfile,
    assign_locations_to_nearest_gateway,
    build_gdp_gateway_network,
    compute_gateway_demand_weights,
    compute_gateway_population_weights,
    generate_downlink_demands,
    generate_gateway_od_demands,
    make_default_traffic_profiles,
)
from cosmica.scenario.gateway_network import build_city_gateway_network


def _make_test_gateways() -> list[Gateway]:
    return [
        Gateway(id=0, latitude=np.deg2rad(35.68), longitude=np.deg2rad(139.65), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=1, latitude=np.deg2rad(40.71), longitude=np.deg2rad(-74.01), minimum_elevation=np.deg2rad(30.0)),
        Gateway(id=2, latitude=np.deg2rad(48.86), longitude=np.deg2rad(2.35), minimum_elevation=np.deg2rad(30.0)),
    ]


def test_assign_locations_to_nearest_gateway() -> None:
    gateways = _make_test_gateways()
    # Osaka (near Tokyo), Boston (near New York), Madrid (near Paris)
    longitude = np.deg2rad(np.array([135.50, -71.06, -3.70]))
    latitude = np.deg2rad(np.array([34.69, 42.36, 40.42]))

    indices = assign_locations_to_nearest_gateway(longitude, latitude, gateways)

    assert indices.tolist() == [0, 1, 2]


def test_compute_gateway_population_weights_is_normalized_and_deterministic() -> None:
    gateways = _make_test_gateways()

    weights = compute_gateway_population_weights(gateways)
    weights_again = compute_gateway_population_weights(gateways)

    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)
    assert np.array_equal(weights, weights_again)


def test_compute_gateway_demand_weights_is_normalized_and_deterministic() -> None:
    gateways = _make_test_gateways()

    for model in ("population", "gdp", "penetration", "subscriber"):
        weights = compute_gateway_demand_weights(gateways, model)
        weights_again = compute_gateway_demand_weights(gateways, model)

        assert weights.shape == (3,)
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights > 0)
        assert np.array_equal(weights, weights_again)


def test_compute_gateway_demand_weights_population_matches_population_weights() -> None:
    gateways = _make_test_gateways()

    demand_weights = compute_gateway_demand_weights(gateways, "population")
    population_weights = compute_gateway_population_weights(gateways)

    assert np.array_equal(demand_weights, population_weights)


def test_compute_gateway_demand_weights_uniform_is_equal_for_all_gateways() -> None:
    gateways = _make_test_gateways()

    weights = compute_gateway_demand_weights(gateways, "uniform")

    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights, 1.0 / 3.0)


def test_compute_gateway_demand_weights_gdp_lifts_developed_economies() -> None:
    # First 11 cities include Delhi (id 1, developing) and New York (id 10, developed).
    gateways = build_city_gateway_network(11)
    idx_delhi = next(i for i, gateway in enumerate(gateways) if gateway.id == 1)
    idx_new_york = next(i for i, gateway in enumerate(gateways) if gateway.id == 10)

    population = compute_gateway_demand_weights(gateways, "population")
    gdp = compute_gateway_demand_weights(gateways, "gdp")

    # GDP weighting lifts New York (high GDP per capita) relative to Delhi (low).
    assert gdp[idx_new_york] / gdp[idx_delhi] > population[idx_new_york] / population[idx_delhi]


def test_generate_gateway_od_demands_total_rate_and_endpoints() -> None:
    gateways = _make_test_gateways()
    profile = ConstantTrafficProfile(traffic_class="video", total_rate=10e9)

    demands = generate_gateway_od_demands(gateways, profile)

    assert len(demands) == 6  # 3 gateways -> 3 * 2 directed pairs
    assert np.isclose(sum(demand.transmission_rate for demand in demands), 10e9)
    for demand in demands:
        assert isinstance(demand, ConstantCommunicationDemand)
        assert demand.source != demand.destination
        assert demand.traffic_class == "video"
        assert demand.priority == 0
    assert len({demand.id for demand in demands}) == len(demands)


def test_generate_gateway_od_demands_top_k_limits_pairs() -> None:
    gateways = _make_test_gateways()
    profile = ConstantTrafficProfile(traffic_class="video", total_rate=10e9)

    demands = generate_gateway_od_demands(gateways, profile, top_k=2)

    assert len(demands) == 2
    assert np.isclose(sum(demand.transmission_rate for demand in demands), 10e9)


def test_build_gdp_gateway_network_selects_requested_number_and_is_deterministic() -> None:
    gateways = build_gdp_gateway_network(15)

    assert len(gateways) == 15
    assert len({gateway.id for gateway in gateways}) == 15
    assert gateways == build_gdp_gateway_network(15)


def test_build_gdp_gateway_network_pulls_in_high_gdp_cities_over_population() -> None:
    # New York has id 10, so the population top-10 (ids 0-9) excludes it, but GDP ranking should not.
    population_ids = {gateway.id for gateway in build_city_gateway_network(10)}
    gdp_ids = {gateway.id for gateway in build_gdp_gateway_network(10)}

    assert 10 not in population_ids
    assert 10 in gdp_ids


def test_build_gdp_gateway_network_rejects_too_many_gateways() -> None:
    with pytest.raises(AssertionError):
        build_gdp_gateway_network(1000)


def test_generate_gateway_od_demands_gdp_model_preserves_total_rate() -> None:
    gateways = build_city_gateway_network(11)
    profile = ConstantTrafficProfile(traffic_class="video", total_rate=10e9)

    demands = generate_gateway_od_demands(gateways, profile, weight_model="gdp")

    assert np.isclose(sum(demand.transmission_rate for demand in demands), 10e9)
    assert len({demand.id for demand in demands}) == len(demands)


def test_generate_gateway_od_demands_temporary_when_window_given() -> None:
    gateways = _make_test_gateways()
    profile = ConstantTrafficProfile(
        traffic_class="financial",
        total_rate=1e9,
        distribution="poisson",
        priority=1,
        start_time=np.datetime64("2026-01-01T00:00:10"),
        end_time=np.datetime64("2026-01-01T00:00:50"),
    )

    demands = generate_gateway_od_demands(gateways, profile)

    for demand in demands:
        assert isinstance(demand, TemporaryCommunicationDemand)
        assert demand.distribution == "poisson"
        assert demand.priority == 1


def test_generate_gateway_od_demands_sampling_is_reproducible_with_seed() -> None:
    gateways = _make_test_gateways()
    profile = ConstantTrafficProfile(traffic_class="video", total_rate=10e9)

    demands_a = generate_gateway_od_demands(
        gateways, profile, method="sampling", n_samples=100, rng=np.random.default_rng(42),
    )
    demands_b = generate_gateway_od_demands(
        gateways, profile, method="sampling", n_samples=100, rng=np.random.default_rng(42),
    )

    assert demands_a == demands_b


def test_generate_downlink_demands() -> None:
    gateways = _make_test_gateways()
    window = (np.datetime64("2026-01-01T00:00:00"), np.datetime64("2026-01-01T00:01:00"))
    profile = OneTimeTrafficProfile(
        traffic_class="imagery",
        data_size_per_event=8e9,
        n_events=5,
        generation_window=window,
        deadline_offset=np.timedelta64(30, "s"),
    )

    demands = generate_downlink_demands("UserSatellite-0", gateways, profile, rng=np.random.default_rng(0))  # type: ignore[arg-type]

    assert len(demands) == 5
    gateway_gids = {str(gateway.global_id) for gateway in gateways}
    for demand in demands:
        assert isinstance(demand, OneTimeCommunicationDemand)
        assert str(demand.source) == "UserSatellite-0"
        assert str(demand.destination) in gateway_gids
        assert window[0] <= demand.generation_time < window[1]
        assert demand.deadline == demand.generation_time + np.timedelta64(30, "s")
        assert demand.traffic_class == "imagery"
    assert len({demand.id for demand in demands}) == len(demands)


def test_generate_downlink_demands_is_reproducible_with_seed() -> None:
    gateways = _make_test_gateways()
    window = (np.datetime64("2026-01-01T00:00:00"), np.datetime64("2026-01-01T00:01:00"))
    profile = OneTimeTrafficProfile(
        traffic_class="imagery",
        data_size_per_event=8e9,
        n_events=3,
        generation_window=window,
        deadline_offset=np.timedelta64(30, "s"),
    )

    demands_a = generate_downlink_demands("UserSatellite-0", gateways, profile, rng=np.random.default_rng(7))  # type: ignore[arg-type]
    demands_b = generate_downlink_demands("UserSatellite-0", gateways, profile, rng=np.random.default_rng(7))  # type: ignore[arg-type]

    assert demands_a == demands_b


def test_make_default_traffic_profiles() -> None:
    window = (np.datetime64("2026-01-01T00:00:00"), np.datetime64("2026-01-01T00:01:00"))

    profiles = make_default_traffic_profiles(window)

    assert set(profiles) == {"video", "financial", "imagery"}
    financial = profiles["financial"]
    assert isinstance(financial, ConstantTrafficProfile)
    assert financial.distribution == "poisson"
    imagery = profiles["imagery"]
    assert isinstance(imagery, OneTimeTrafficProfile)
    assert imagery.generation_window == window


def test_constant_traffic_profile_rejects_half_open_window() -> None:
    with pytest.raises(AssertionError):
        ConstantTrafficProfile(
            traffic_class="video",
            total_rate=1e9,
            start_time=np.datetime64("2026-01-01T00:00:00"),
        )


def test_onetime_traffic_profile_rejects_inverted_window() -> None:
    with pytest.raises(AssertionError):
        OneTimeTrafficProfile(
            traffic_class="imagery",
            data_size_per_event=1e9,
            n_events=1,
            generation_window=(np.datetime64("2026-01-01T00:01:00"), np.datetime64("2026-01-01T00:00:00")),
            deadline_offset=np.timedelta64(30, "s"),
        )
