"""Models for the uncertainty parameters in the communication link."""

__all__ = [
    "ApertureAveragedLogNormalScintillationModel",
    "AtmosphericScintillationModel",
    "BinaryCloudModel",
    "CloudStates",
    "EdgeFailureModel",
    "ExpEdgeModel",
]
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Annotated

import numpy as np
import numpy.typing as npt
from typing_extensions import Doc


class CloudStates[T: np.number | np.bool_](ABC):
    @abstractmethod
    def simulate(self, time: npt.NDArray[np.datetime64], rng: np.random.Generator) -> npt.NDArray[T]: ...


@dataclass(frozen=False, kw_only=True, slots=True)
class BinaryCloudModel(CloudStates[np.bool_]):
    """Binary Cloud Model.

    Generate the state of clouds described by a Markov Chain by calling the 'simulate' method.
    """

    initial_cloud: Annotated[
        bool,
        Doc("Initial state. True indicates cloudy and False indicates free-of-clouds."),
    ] = False
    transition_p_0_to_1: float = 0.15  # probability of state transition from 0 to 1
    transition_p_1_to_0: float = 0.40  # probability of state transition from 1 to 0

    @cached_property
    def _state_tm(self) -> npt.NDArray[np.floating]:
        return np.array(
            [
                [1 - self.transition_p_0_to_1, self.transition_p_0_to_1],
                [self.transition_p_1_to_0, 1 - self.transition_p_1_to_0],
            ],
        )

    def simulate(
        self,
        time: Annotated[
            npt.NDArray[np.datetime64],
            Doc("time_array as a np.ndarray with elements in the np.datetime64 format"),
        ],
        rng: Annotated[np.random.Generator, Doc("NumPy random number generator.")],
    ) -> Annotated[
        npt.NDArray[np.bool_],
        Doc("time series of cloud states on given time frame (1|True = cloudy, 0|False = Free-of-clouds)"),
    ]:
        sampled = np.empty(time.shape[0], dtype=bool)
        current_state = self.initial_cloud

        for i in range(time.shape[0]):
            sampled[i] = current_state
            next_state = rng.choice(np.array([0, 1]), size=1, p=self._state_tm[int(current_state)])[0]
            current_state = next_state
        return sampled


class EdgeFailureModel[T: np.number | np.bool_](ABC):
    @abstractmethod
    def simulate(self, time: npt.NDArray[np.datetime64], rng: np.random.Generator) -> npt.NDArray[T]: ...


_SECOND = np.timedelta64(1, "s")


@dataclass(frozen=True, kw_only=True, slots=True)
class ExpEdgeModel(EdgeFailureModel[np.bool_]):
    """Exponential Edge Model.

    Generate the state of an edge following an exponential distribution by calling the 'simulate' method.
    An edge can be thought as the link between two nodes (e.g. terminals, satellites)
    """

    # Exponential distribution with mean = reliability
    reliability: Annotated[
        np.timedelta64,
        Doc("time length in days which the edge is expected not to fail (Expected value of exponential distribution)"),
    ] = field(
        default_factory=lambda: np.timedelta64(15, "D"),
    )
    # Deterministic recovery time
    recovery_time: Annotated[np.timedelta64, Doc("Recovery time in seconds (expected)")] = field(
        default_factory=lambda: np.timedelta64(1800, "s"),
    )

    def simulate(
        self,
        time: Annotated[
            npt.NDArray[np.datetime64],
            Doc("time_array as a np.ndarray with elements in the np.datetime64 format"),
        ],
        rng: Annotated[np.random.Generator, Doc("NumPy random number generator")],
    ) -> Annotated[
        npt.NDArray[np.bool_],
        Doc("time series of edge states on given time frame (True = Failure, False= No-failure)"),
    ]:
        state_changed = np.zeros(time.shape, dtype=np.bool_)
        time_step = time[1] - time[0]
        total_time = time[-1] - time[0]

        failure_time: np.timedelta64 = rng.exponential(self.reliability / _SECOND) * _SECOND
        while failure_time <= total_time:
            failure_idx: int = np.where(time > (time[0] + failure_time))[0][0]
            state_changed[failure_idx] = True
            recovery_idx = failure_idx + int(self.recovery_time / time_step)
            if recovery_idx < state_changed.shape[0]:  # type: ignore[misc] # Possibly a typing bug in NumPy
                state_changed[recovery_idx] = True
            else:
                return np.logical_xor.accumulate(state_changed)
            failure_time += self.recovery_time + rng.exponential(self.reliability / _SECOND) * _SECOND
        return np.logical_xor.accumulate(state_changed)


class AtmosphericScintillationModel[T: np.number | np.bool_](ABC):
    @abstractmethod
    def sample(
        self,
        rng: np.random.Generator,
        link_distance: float,
        rytov_variance: float | None = None,
    ) -> float: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class ApertureAveragedLogNormalScintillationModel(AtmosphericScintillationModel[np.bool_]):
    default_rytov_variance: float
    wavelength: float
    aperture_diameter: float

    @cached_property
    def k_number(self) -> float:
        return 2 * np.pi / self.wavelength

    def scaled_aperture(self, link_distance: float) -> float:
        return np.sqrt((self.k_number * self.aperture_diameter**2) / (4 * link_distance))

    def sigma2_scintillation(
        self,
        link_distance: float,
        ryotv_variance: float | None = None,
    ) -> Annotated[
        float,
        Doc(
            "Float approximation for aperture averaged scintillation for a plane-wave in the absence of inner scale and"
            'outer scale effects as described by Larry C. Andrews and Ronald L. Phillips in the book "Laser Beam'
            'Propagation through Random Media" Chapter 10.',
        ),
    ]:
        if ryotv_variance is not None and self.default_rytov_variance != ryotv_variance:
            ryotv_variance = self.default_rytov_variance
        scaled_aperture = self.scaled_aperture(link_distance=link_distance)
        num1 = 0.49 * self.default_rytov_variance
        den1 = (1 + 0.65 * scaled_aperture**2 + 1.11 * self.default_rytov_variance ** (6 / 5)) ** (7 / 6)
        num2 = 0.51 * self.default_rytov_variance * (1 + 0.69 * self.default_rytov_variance ** (6 / 5)) ** (-5 / 6)
        den2 = 1 + 0.90 * scaled_aperture**2 + 0.62 * scaled_aperture**2 * self.default_rytov_variance ** (6 / 5)
        return np.exp(num1 / den1 + num2 / den2) - 1

    def sample(
        self,
        rng: np.random.Generator,
        link_distance: float,
        rytov_variance: float | None = None,
    ) -> float:
        sigma2_scintillation = self.sigma2_scintillation(link_distance, rytov_variance)
        return rng.lognormal(-sigma2_scintillation / 2, sigma2_scintillation)
