"""Models for the uncertainty parameters in the communication link."""

__all__ = [
    "ApertureAveragedLogNormalScintillationModel",
    "AtmosphericScintillationModel",
    "BinaryCloudModel",
    "CloudStates",
    "EdgeFailureModel",
    "ExpEdgeModel",
]
import math
import warnings
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


def _validate_time_grid(time: npt.NDArray[np.datetime64]) -> None:
    if time.ndim != 1 or time.shape[0] == 0:
        msg = f"time must be a non-empty 1-D array, got shape {time.shape}."
        raise ValueError(msg)
    if np.any(np.isnat(time)):
        msg = "time must be free of NaT."
        raise ValueError(msg)
    if np.any(np.diff(time) <= np.timedelta64(0, "s")):
        msg = "time must be strictly increasing."
        raise ValueError(msg)


@dataclass(frozen=True, kw_only=True, slots=True)
class ExpEdgeModel(EdgeFailureModel[np.bool_]):
    """Exponential Edge Model.

    Generate the state of an edge following an exponential distribution by calling the 'simulate' method.
    An edge can be thought as the link between two nodes (e.g. terminals, satellites)

    Failures occur after exponentially distributed operating times (whole seconds), and each failure lasts
    exactly `recovery_time`. A failure at elapsed time `t_f` marks the samples in `(t_f, t_f + recovery_time]`
    as failed, so a failure that starts and ends between two samples is not observed.
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

    def _draw_operating_time(self, rng: np.random.Generator) -> np.timedelta64:
        return rng.exponential(self.reliability / _SECOND) * _SECOND

    def simulate(
        self,
        time: Annotated[
            npt.NDArray[np.datetime64],
            Doc("Non-empty, strictly increasing 1-D array of np.datetime64 sample times; spacing may be irregular."),
        ],
        rng: Annotated[np.random.Generator, Doc("NumPy random number generator")],
    ) -> Annotated[
        npt.NDArray[np.bool_],
        Doc("time series of edge states on given time frame (True = Failure, False= No-failure)"),
    ]:
        _validate_time_grid(time)
        elapsed = time - time[0]
        total_time = elapsed[-1]
        failed = np.zeros(time.shape, dtype=np.bool_)

        failure_time = self._draw_operating_time(rng)
        while failure_time <= total_time:
            recovered_time = failure_time + self.recovery_time
            start = np.searchsorted(elapsed, failure_time, side="right")
            stop = np.searchsorted(elapsed, recovered_time, side="right")
            failed[start:stop] = True
            if recovered_time >= total_time:
                break
            failure_time = recovered_time + self._draw_operating_time(rng)
        return failed


class AtmosphericScintillationModel[T: np.number | np.bool_](ABC):
    @abstractmethod
    def sample(
        self,
        rng: np.random.Generator,
        zenith_angle: Annotated[float, Doc("Zenith angle of the link at the ground station in radians.")],
    ) -> float: ...


# Altitude grid in meters above the receiver for the Eq. 12.39 quadrature. The geometric spacing resolves the
# 100 m ground layer of the H-V profile, and 50 km is far above the last layer, which peaks near 10 km.
# 1000 points agree with 4000 points to 4e-5 relative at zenith angles up to 80 deg (measured 2026-11-06).
_ALTITUDE_GRID = np.concatenate([[0.0], np.geomspace(1e-2, 50e3, 1000)])


@dataclass(frozen=True, kw_only=True, slots=True)
class ApertureAveragedLogNormalScintillationModel(AtmosphericScintillationModel[np.bool_]):
    """Aperture-averaged log-normal scintillation of a satellite-to-ground downlink.

    Plane-wave downlink model with a Hufnagel-Valley turbulence profile, as described by Larry C. Andrews and
    Ronald L. Phillips in the book "Laser Beam Propagation through Random Media", 2nd ed., Section 12.5.2.
    Weak-fluctuation theory: valid while the Rytov variance is below 1. Derivation and validation in issue #354.
    """

    wavelength: Annotated[float, Doc("Optical wavelength in meters.")]
    aperture_diameter: Annotated[float, Doc("Receiver aperture diameter in meters. Zero is a point receiver.")]
    ground_cn2: Annotated[float, Doc("Hufnagel-Valley ground-level structure parameter Cn2(0) in m^-2/3.")] = 1.7e-14
    rms_wind_speed: Annotated[float, Doc("Hufnagel-Valley rms (pseudo) wind speed in m/s.")] = 21.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.wavelength) or self.wavelength <= 0.0:
            msg = f"wavelength must be finite and positive, got {self.wavelength}."
            raise ValueError(msg)
        for name in ("aperture_diameter", "ground_cn2", "rms_wind_speed"):
            if not np.isfinite(value := getattr(self, name)) or value < 0.0:
                msg = f"{name} must be finite and non-negative, got {value}."
                raise ValueError(msg)

    def cn2(
        self,
        altitude: npt.NDArray[np.floating],
        ground_cn2: Annotated[float | None, Doc("Override of ground_cn2, e.g. for a change in weather.")] = None,
        rms_wind_speed: Annotated[float | None, Doc("Override of rms_wind_speed.")] = None,
    ) -> npt.NDArray[np.floating]:
        """Hufnagel-Valley profile Cn2(h) in m^-2/3 at altitude h in meters above the receiver (Eq. 12.1)."""
        ground_cn2, rms_wind_speed = self._conditions(ground_cn2, rms_wind_speed)
        return (
            0.00594 * (rms_wind_speed / 27) ** 2 * (1e-5 * altitude) ** 10 * np.exp(-altitude / 1000)
            + 2.7e-16 * np.exp(-altitude / 1500)
            + ground_cn2 * np.exp(-altitude / 100)
        )

    def _conditions(self, ground_cn2: float | None, rms_wind_speed: float | None) -> tuple[float, float]:
        ground_cn2 = self.ground_cn2 if ground_cn2 is None else ground_cn2
        rms_wind_speed = self.rms_wind_speed if rms_wind_speed is None else rms_wind_speed
        for name, value in (("ground_cn2", ground_cn2), ("rms_wind_speed", rms_wind_speed)):
            if not np.isfinite(value) or value < 0.0:
                msg = f"{name} must be finite and non-negative, got {value}."
                raise ValueError(msg)
        return ground_cn2, rms_wind_speed

    def rytov_variance(
        self,
        zenith_angle: Annotated[float, Doc("Zenith angle of the link at the ground station in radians.")],
        ground_cn2: Annotated[float | None, Doc("Override of ground_cn2, e.g. for a change in weather.")] = None,
        rms_wind_speed: Annotated[float | None, Doc("Override of rms_wind_speed.")] = None,
    ) -> Annotated[float, Doc("Plane-wave Rytov variance, the point-receiver scintillation index (Eq. 12.38).")]:
        _validate_zenith_angle(zenith_angle)
        ground_cn2, rms_wind_speed = self._conditions(ground_cn2, rms_wind_speed)
        k = 2 * np.pi / self.wavelength
        # Integral of Cn2(h) h^(5/6) from 0 to infinity, term by term: int h^p exp(-h/s) dh = s^(p+1) Gamma(p+1).
        tropopause = 0.00594 * (rms_wind_speed / 27) ** 2 * 1e-50 * 1000 ** (71 / 6) * math.gamma(71 / 6)
        background = 2.7e-16 * 1500 ** (11 / 6) * math.gamma(11 / 6)
        ground = ground_cn2 * 100 ** (11 / 6) * math.gamma(11 / 6)
        integral = tropopause + background + ground
        return 2.25 * k ** (7 / 6) * np.cos(zenith_angle) ** (-11 / 6) * integral

    def sigma2_scintillation(
        self,
        zenith_angle: Annotated[float, Doc("Zenith angle of the link at the ground station in radians.")],
        ground_cn2: Annotated[float | None, Doc("Override of ground_cn2, e.g. for a change in weather.")] = None,
        rms_wind_speed: Annotated[float | None, Doc("Override of rms_wind_speed.")] = None,
    ) -> Annotated[float, Doc("Aperture-averaged irradiance flux variance sigma_I^2(D) (Eq. 12.39).")]:
        if (rytov_variance := self.rytov_variance(zenith_angle, ground_cn2, rms_wind_speed)) >= 1.0:
            warnings.warn(
                f"Rytov variance {rytov_variance:.2f} is outside the weak-fluctuation range (< 1) of Eq. 12.39, "
                "so the scintillation index is unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
        k = 2 * np.pi / self.wavelength
        # Eq. 12.39 with L = (H - h0) sec(zenith) and the (H - h0)^(5/6) factor moved inside the bracket.
        fresnel_term = k * self.aperture_diameter**2 * np.cos(zenith_angle) / 16
        h = _ALTITUDE_GRID
        bracket = np.real((fresnel_term + 1j * h) ** (5 / 6)) - fresnel_term ** (5 / 6)
        integral = np.trapezoid(self.cn2(h, ground_cn2, rms_wind_speed) * bracket, h)
        return float(8.70 * k ** (7 / 6) * np.cos(zenith_angle) ** (-11 / 6) * integral)

    def sample(
        self,
        rng: np.random.Generator,
        zenith_angle: Annotated[float, Doc("Zenith angle of the link at the ground station in radians.")],
        ground_cn2: Annotated[float | None, Doc("Override of ground_cn2, e.g. for a change in weather.")] = None,
        rms_wind_speed: Annotated[float | None, Doc("Override of rms_wind_speed.")] = None,
    ) -> float:
        """Sample a unit-mean intensity multiplier with the aperture-averaged scintillation index.

        For a unit-mean log-normal variable, the scintillation index is
        exp(log_variance) - 1 (Eq. 9.12). NumPy expects the underlying normal distribution's
        standard deviation, not its variance.
        """
        log_variance = np.log1p(self.sigma2_scintillation(zenith_angle, ground_cn2, rms_wind_speed))
        return rng.lognormal(-log_variance / 2, np.sqrt(log_variance))


def _validate_zenith_angle(zenith_angle: float) -> None:
    if not 0.0 <= zenith_angle < np.pi / 2:
        msg = f"zenith_angle must be in [0, pi/2) radians, got {zenith_angle}."
        raise ValueError(msg)
