"""Generate communication demands aggregated at gateways from geographic demand distributions.

Ground demand is modeled by aggregating the global population distribution to the nearest
gateway, which turns the gateway set into the geographic resolution of the demand. The
resulting origin-destination (OD) pairs between gateways are converted to `Demand` objects
that the packet simulator can consume directly.
"""

__all__ = [
    "ConstantTrafficProfile",
    "DemandWeightModel",
    "OneTimeTrafficProfile",
    "TrafficProfile",
    "assign_locations_to_nearest_gateway",
    "build_demand_dataframe",
    "build_gdp_gateway_network",
    "compute_gateway_demand_weights",
    "compute_gateway_population_weights",
    "generate_downlink_demands",
    "generate_gateway_od_demands",
    "make_default_traffic_profiles",
]

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import Doc

from cosmica.models import (
    ConstantCommunicationDemand,
    Demand,
    Gateway,
    NodeGID,
    OneTimeCommunicationDemand,
    TemporaryCommunicationDemand,
)
from cosmica.utils.coordinates import great_circle_distance

from .gateway_network import (
    N_MAJOR_CITIES,
    DemandWeightModel,
    build_city_gateway_network,
    get_gateway_city_name,
    get_gateway_econ_multiplier,
)
from .population import get_population_data, sample_demand_locations

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class ConstantTrafficProfile:
    """Profile of a continuously generated traffic type (e.g. video streaming, financial data).

    If both `start_time` and `end_time` are given, demands are generated as
    `TemporaryCommunicationDemand`; otherwise as `ConstantCommunicationDemand`.
    """

    traffic_class: str
    total_rate: Annotated[float, Doc("Total worldwide transmission rate of this traffic class in bit/s.")]
    distribution: Literal["uniform", "poisson"] = "uniform"
    priority: int = 0
    start_time: np.datetime64 | None = None
    end_time: np.datetime64 | None = None

    def __post_init__(self) -> None:
        assert self.total_rate > 0, f"total_rate must be positive, but got {self.total_rate}"
        assert (self.start_time is None) == (self.end_time is None), (
            "start_time and end_time must be either both set or both None"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class OneTimeTrafficProfile:
    """Profile of a bulk-transfer traffic type (e.g. satellite imagery downlink)."""

    traffic_class: str
    data_size_per_event: Annotated[float, Doc("Data size of a single transfer event in bit.")]
    n_events: int
    generation_window: Annotated[
        tuple[np.datetime64, np.datetime64],
        Doc("Half-open time window [start, end) in which generation times are drawn."),
    ]
    deadline_offset: Annotated[np.timedelta64, Doc("Deadline of each event relative to its generation time.")]
    priority: int = 0

    def __post_init__(self) -> None:
        assert self.data_size_per_event > 0, f"data_size_per_event must be positive, but got {self.data_size_per_event}"
        assert self.n_events > 0, f"n_events must be positive, but got {self.n_events}"
        assert self.generation_window[0] < self.generation_window[1], (
            f"generation_window must be an increasing interval, but got {self.generation_window}"
        )


type TrafficProfile = ConstantTrafficProfile | OneTimeTrafficProfile


def make_default_traffic_profiles(
    generation_window: Annotated[
        tuple[np.datetime64, np.datetime64],
        Doc(
            "Half-open time window [start, end) for one-time (imagery) transfer events."
            " Keep the end before the last simulation step so the simulator can schedule the events.",
        ),
    ],
) -> Annotated[dict[str, TrafficProfile], Doc("Default traffic profiles keyed by traffic class.")]:
    """Make default traffic profiles for the three representative traffic classes.

    Video is a constant high-rate stream, financial is a low-rate stream with Poisson
    arrivals (a uniform distribution would generate zero packets per time step at low
    rates because the packet count is truncated to an integer), and imagery is a bulk
    one-time transfer.
    """
    return {
        "video": ConstantTrafficProfile(
            traffic_class="video",
            total_rate=20e9,
            distribution="uniform",
        ),
        "financial": ConstantTrafficProfile(
            traffic_class="financial",
            total_rate=0.5e9,
            distribution="poisson",
            priority=1,
        ),
        "imagery": OneTimeTrafficProfile(
            traffic_class="imagery",
            data_size_per_event=8e9,
            n_events=3,
            generation_window=generation_window,
            deadline_offset=np.timedelta64(60, "s"),
        ),
    }


def assign_locations_to_nearest_gateway(
    longitude: Annotated[npt.NDArray[np.floating], Doc("Longitudes of the locations in radians, shape (n,).")],
    latitude: Annotated[npt.NDArray[np.floating], Doc("Latitudes of the locations in radians, shape (n,).")],
    gateways: Annotated[Sequence[Gateway], Doc("Candidate gateways.")],
) -> Annotated[npt.NDArray[np.intp], Doc("Index into `gateways` of the nearest gateway, shape (n,).")]:
    """Assign each location to the nearest gateway by great-circle distance.

    The output of `cosmica.scenario.population.sample_demand_locations` can be passed directly.
    """
    assert len(gateways) > 0, "gateways must not be empty"
    gateway_latitudes = np.array([gateway.latitude for gateway in gateways])
    gateway_longitudes = np.array([gateway.longitude for gateway in gateways])

    distance_matrix = great_circle_distance(
        np.asarray(latitude)[:, None],
        np.asarray(longitude)[:, None],
        gateway_latitudes[None, :],
        gateway_longitudes[None, :],
    )
    return np.argmin(distance_matrix, axis=1)


def compute_gateway_population_weights(
    gateways: Annotated[Sequence[Gateway], Doc("Gateways to aggregate the population to.")],
) -> Annotated[
    npt.NDArray[np.floating],
    Doc("Normalized population weight of each gateway (sums to 1), shape (len(gateways),)."),
]:
    """Aggregate the global population grid to the nearest gateway of each grid cell.

    This is deterministic: every grid cell contributes its full population to its nearest
    gateway, so it does not suffer from the sampling noise (and the no-replacement
    constraint) of `sample_demand_locations`.
    """
    population = get_population_data()
    lon = np.deg2rad(np.arange(-179.5, 180.5))
    lat = np.flip(np.deg2rad(np.arange(-89.5, 90.5)))
    longitude, latitude = np.meshgrid(lon, lat)

    population_flat = np.nan_to_num(population.flatten(), nan=0.0)
    nearest_gateway_indices = assign_locations_to_nearest_gateway(
        longitude.flatten(),
        latitude.flatten(),
        gateways,
    )

    weights = np.bincount(nearest_gateway_indices, weights=population_flat, minlength=len(gateways))
    total = weights.sum()
    assert total > 0, "Total population must be positive"
    return weights / total


def compute_gateway_demand_weights(
    gateways: Annotated[Sequence[Gateway], Doc("Gateways acting as demand endpoints.")],
    weight_model: Annotated[
        DemandWeightModel,
        Doc('Attractiveness model. "population" is pure population gravity; the others reshape it.'),
    ] = "population",
) -> Annotated[
    npt.NDArray[np.floating],
    Doc("Normalized demand weight of each gateway (sums to 1), shape (len(gateways),)."),
]:
    """Compute normalized per-gateway demand weights for a given attractiveness model.

    The base population weight `P_i` (from `compute_gateway_population_weights`) is combined with a
    per-gateway multiplier `m_i` (from `get_gateway_econ_multiplier`) as `w_i ∝ P_i · m_i`, then
    renormalized to sum to 1. With `weight_model="population"` the multiplier is 1 everywhere, so
    this reduces to the historical population gravity model. GDP / penetration / subscriber models
    lift economically stronger regions relative to sheer population. `weight_model="uniform"` ignores
    both population and economy and gives every gateway the same weight `1/N` (a demand-agnostic
    baseline, e.g. for stress tests).
    """
    if weight_model == "uniform":
        n_gateways = len(gateways)
        return np.full(n_gateways, 1.0 / n_gateways)

    population_weights = compute_gateway_population_weights(gateways)
    if weight_model == "population":
        return population_weights

    multipliers = get_gateway_econ_multiplier(gateways, weight_model)
    weights = population_weights * multipliers
    total = weights.sum()
    assert total > 0, f"Total demand weight must be positive for model {weight_model!r}"
    return weights / total


def build_gdp_gateway_network(
    n_gateways: Annotated[int, Doc("Number of gateways to build. Must not exceed the number of candidate cities.")],
    *,
    minimum_elevation: Annotated[float, Doc("Minimum elevation angle of every gateway in radians.")] = np.deg2rad(30.0),
) -> Annotated[
    list[Gateway],
    Doc("Gateways at the `n_gateways` cities with the largest GDP-weighted demand."),
]:
    """Build a gateway network placed at the cities with the largest total GDP.

    Where `build_city_gateway_network` selects the most populous cities and
    `build_geographically_distributed_gateway_network` spreads them over the globe, this ranks the
    candidate cities by their GDP demand weight `w_i ∝ P_i · g_i` (population aggregated to the city
    times its country's GDP per capita, see `compute_gateway_demand_weights`) and keeps the top
    `n_gateways`. This favors economically dominant regions (North America, Europe, developed Asia)
    over sheer-population megacities. Each gateway keeps the id equal to its index in the city table
    (so ids are not contiguous), keeping the id-to-city-name mapping identical to the other builders.
    The result is deterministic: repeated calls return equal gateways.
    """
    assert 0 < n_gateways <= N_MAJOR_CITIES, f"n_gateways must be in [1, {N_MAJOR_CITIES}], but got {n_gateways}"

    all_gateways = build_city_gateway_network(N_MAJOR_CITIES, minimum_elevation=minimum_elevation)
    gdp_weights = compute_gateway_demand_weights(all_gateways, "gdp")
    top_indices = np.argsort(gdp_weights)[::-1][:n_gateways]
    return [all_gateways[index] for index in sorted(top_indices)]


def _compute_gateway_weights(
    gateways: Sequence[Gateway],
    method: Literal["deterministic", "sampling"],
    n_samples: int,
    rng: np.random.Generator | None,
    weight_model: DemandWeightModel = "population",
) -> npt.NDArray[np.floating]:
    if weight_model == "uniform":
        n_gateways = len(gateways)
        return np.full(n_gateways, 1.0 / n_gateways)
    if method == "deterministic":
        return compute_gateway_demand_weights(gateways, weight_model)

    # Sampling estimates population weights from sampled locations; the economic multiplier is then
    # applied on top so the requested demand model is honored regardless of the method.
    rng = rng if rng is not None else np.random.default_rng()
    longitude, latitude = sample_demand_locations(n_samples, rng=rng)
    nearest_gateway_indices = assign_locations_to_nearest_gateway(longitude, latitude, gateways)
    counts = np.bincount(nearest_gateway_indices, minlength=len(gateways)).astype(np.float64)
    population_weights = counts / counts.sum()
    if weight_model == "population":
        return population_weights
    weights = population_weights * get_gateway_econ_multiplier(gateways, weight_model)
    return weights / weights.sum()


def generate_gateway_od_demands(
    gateways: Annotated[Sequence[Gateway], Doc("Gateways acting as demand endpoints.")],
    profile: Annotated[ConstantTrafficProfile, Doc("Traffic profile shared by all generated demands.")],
    *,
    method: Annotated[
        Literal["deterministic", "sampling"],
        Doc(
            '"deterministic" aggregates the full population grid to gateways;'
            ' "sampling" estimates gateway weights from sampled demand locations.',
        ),
    ] = "deterministic",
    n_samples: Annotated[int, Doc('Number of sampled demand locations (used only when method="sampling").')] = 500,
    top_k: Annotated[
        int | None,
        Doc("Keep only the largest `top_k` OD pairs to bound the number of demands. None keeps all pairs."),
    ] = 50,
    weight_model: Annotated[
        DemandWeightModel,
        Doc('Gateway attractiveness model. "population" is population gravity; "gdp"/"penetration"/'
            '"subscriber" reshape it by the region\'s economy (see `compute_gateway_demand_weights`).'),
    ] = "population",
    rng: Annotated[np.random.Generator | None, Doc("NumPy random number generator. If None, use default.")] = None,
) -> Annotated[
    list[ConstantCommunicationDemand | TemporaryCommunicationDemand],
    Doc("Gateway-to-gateway OD demands whose rates sum to `profile.total_rate`."),
]:
    """Generate gateway-to-gateway OD demands from the geographic demand distribution.

    Gateway weights `w` are computed from `weight_model` (population gravity by default), the OD
    share of a pair (i, j), i != j, is proportional to `w_i * w_j`, and `profile.total_rate` is
    split over the kept pairs in proportion to their shares.
    """
    weights = _compute_gateway_weights(gateways, method, n_samples, rng, weight_model)

    od_matrix = np.outer(weights, weights)
    np.fill_diagonal(od_matrix, 0.0)

    flat_shares = od_matrix.flatten()
    pair_indices = np.argsort(flat_shares)[::-1]
    if top_k is not None:
        pair_indices = pair_indices[:top_k]
    pair_indices = pair_indices[flat_shares[pair_indices] > 0]
    dropped_share = 1 - flat_shares[pair_indices].sum() / flat_shares.sum()
    if dropped_share > 0:
        logger.info(
            f"Keeping {len(pair_indices)} OD pairs for traffic class '{profile.traffic_class}'"
            f" ({dropped_share:.1%} of the OD share is folded into the kept pairs by renormalization)",
        )

    kept_shares = flat_shares[pair_indices]
    rates = profile.total_rate * kept_shares / kept_shares.sum()

    n_gateways = len(gateways)
    demands: list[ConstantCommunicationDemand | TemporaryCommunicationDemand] = []
    for pair_index, rate in zip(pair_indices, rates, strict=True):
        source = gateways[pair_index // n_gateways].global_id
        destination = gateways[pair_index % n_gateways].global_id
        demand_id = (profile.traffic_class, str(source), str(destination))
        if profile.start_time is not None and profile.end_time is not None:
            demands.append(
                TemporaryCommunicationDemand(
                    id=demand_id,
                    source=source,
                    destination=destination,
                    transmission_rate=float(rate),
                    distribution=profile.distribution,
                    start_time=profile.start_time,
                    end_time=profile.end_time,
                    traffic_class=profile.traffic_class,
                    priority=profile.priority,
                ),
            )
        else:
            demands.append(
                ConstantCommunicationDemand(
                    id=demand_id,
                    source=source,
                    destination=destination,
                    distribution=profile.distribution,
                    transmission_rate=float(rate),
                    traffic_class=profile.traffic_class,
                    priority=profile.priority,
                ),
            )
    return demands


def generate_downlink_demands(
    source: Annotated[NodeGID, Doc("Global id of the source node (e.g. a user satellite).")],
    gateways: Annotated[Sequence[Gateway], Doc("Candidate destination gateways.")],
    profile: Annotated[OneTimeTrafficProfile, Doc("Traffic profile shared by all generated demands.")],
    *,
    weight_model: Annotated[
        DemandWeightModel,
        Doc("Gateway attractiveness model for the destination distribution (see `generate_gateway_od_demands`)."),
    ] = "population",
    rng: Annotated[np.random.Generator | None, Doc("NumPy random number generator. If None, use default.")] = None,
) -> Annotated[
    list[OneTimeCommunicationDemand],
    Doc("One-time downlink demands from `source` to demand-weighted random gateways."),
]:
    """Generate one-time bulk-transfer demands from a source node to gateways.

    The destination gateway of each event is drawn with probability proportional to the gateway
    demand weights for `weight_model` (population gravity by default), and the generation time is
    drawn uniformly from `profile.generation_window`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    weights = compute_gateway_demand_weights(gateways, weight_model)

    window_start, window_end = profile.generation_window
    window_ns = (window_end - window_start) / np.timedelta64(1, "ns")

    demands: list[OneTimeCommunicationDemand] = []
    for event_index in range(profile.n_events):
        destination = gateways[rng.choice(len(gateways), p=weights)].global_id
        generation_time = window_start + np.timedelta64(int(rng.uniform(0, window_ns)), "ns")
        demands.append(
            OneTimeCommunicationDemand(
                id=(profile.traffic_class, str(source), str(destination), event_index),
                source=source,
                destination=destination,
                data_size=profile.data_size_per_event,
                generation_time=generation_time,
                deadline=generation_time + profile.deadline_offset,
                traffic_class=profile.traffic_class,
                priority=profile.priority,
            ),
        )
    return demands


_DEMAND_DATAFRAME_COLUMNS: Final[tuple[str, ...]] = (
    "demand_id",
    "traffic_class",
    "priority",
    "source",
    "destination",
    "source_city",
    "destination_city",
    "demand_type",
    "transmission_rate_bps",
    "distribution",
    "data_size_bit",
    "generation_time",
    "deadline",
    "start_time",
    "end_time",
)


def build_demand_dataframe(
    demands: Annotated[Sequence[Demand], Doc("Demands to summarize, one row per demand.")],
    gateways: Annotated[
        Sequence[Gateway] | None,
        Doc("Gateways used to resolve source_city/destination_city columns. If None, city columns stay empty."),
    ] = None,
) -> Annotated[pd.DataFrame, Doc("One row per demand with the fixed column layout of `_DEMAND_DATAFRAME_COLUMNS`.")]:
    """Summarize generated demands into a one-row-per-demand DataFrame.

    Lets one inspect after the fact which source/destination each demand connects and at what rate.
    The three demand classes carry different fields (Constant/Temporary hold an instantaneous
    `transmission_rate`, OneTime holds a total `data_size` plus a time window), so columns that do not
    apply to a demand are left blank via the records approach. When `gateways` is given, the
    `source_city`/`destination_city` columns are filled from `get_gateway_city_name` (non-gateway
    endpoints such as user satellites stay blank).
    """
    # global_id ("GW-3" 等) -> 都市名 の逆引き辞書。gateways 未指定なら都市列は空欄のまま。
    gid_to_city: dict[str, str] = {}
    if gateways is not None:
        for gw in gateways:
            gid_to_city[gw.global_id] = get_gateway_city_name(gw) or ""

    records: list[dict] = []
    for d in demands:
        # 基底 Demand は source/destination を持たないが, 現状の全具象クラスは持つ。
        # 将来の非 OD 需要にも備えて getattr でフォールバックする。
        source = getattr(d, "source", "")
        destination = getattr(d, "destination", "")
        row: dict = {
            "demand_id": str(d.id),
            "traffic_class": d.traffic_class,
            "priority": d.priority,
            "source": source,
            "destination": destination,
            "source_city": gid_to_city.get(source, ""),
            "destination_city": gid_to_city.get(destination, ""),
            "demand_type": type(d).__name__,
            "transmission_rate_bps": "",
            "distribution": "",
            "data_size_bit": "",
            "generation_time": "",
            "deadline": "",
            "start_time": "",
            "end_time": "",
        }
        if isinstance(d, ConstantCommunicationDemand | TemporaryCommunicationDemand):
            row["transmission_rate_bps"] = d.transmission_rate
            row["distribution"] = d.distribution
        if isinstance(d, TemporaryCommunicationDemand):
            row["start_time"] = np.datetime_as_string(d.start_time, unit="ms")
            row["end_time"] = np.datetime_as_string(d.end_time, unit="ms")
        if isinstance(d, OneTimeCommunicationDemand):
            row["data_size_bit"] = d.data_size
            row["generation_time"] = np.datetime_as_string(d.generation_time, unit="ms")
            row["deadline"] = np.datetime_as_string(d.deadline, unit="ms")
        records.append(row)

    return pd.DataFrame(records, columns=list(_DEMAND_DATAFRAME_COLUMNS))
