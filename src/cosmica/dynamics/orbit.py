from __future__ import annotations

__all__ = [
    "CircularSatelliteOrbit",
    "CircularSatelliteOrbitPropagator",
    "EllipticalSatelliteOrbit",
    "EllipticalSatelliteOrbitPropagator",
    "SatelliteOrbit",
    "SatelliteOrbitState",
    "make_satellite_orbit",
]

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, override

import numpy as np
from sgp4.model import Satrec
from skyfield.api import EarthSatellite, Time, Timescale, load, utc
from typing_extensions import deprecated

from cosmica.models import CircularSatelliteOrbitModel, EllipticalSatelliteOrbitModel, GravityModel
from cosmica.utils.constants import EARTH_MU
from cosmica.utils.vector import rowwise_matmul

if TYPE_CHECKING:
    import numpy.typing as npt


logger = logging.getLogger(__name__)

ReferenceFrame = Literal["teme", "j2000", "gcrs"]


@dataclass(frozen=True, kw_only=True, slots=True)
class SatelliteOrbitState:
    position_eci: npt.NDArray[np.floating]
    velocity_eci: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        assert self.position_eci.shape[-1] == 3, self.position_eci.shape
        assert self.velocity_eci.shape[-1] == 3, self.velocity_eci.shape
        assert self.position_eci.shape[:-1] == self.velocity_eci.shape[:-1], (
            self.position_eci.shape,
            self.velocity_eci.shape,
        )

    def __getitem__(self, item: int | slice) -> SatelliteOrbitState:
        return SatelliteOrbitState(
            position_eci=self.position_eci[item],
            velocity_eci=self.velocity_eci[item],
        )

    def calc_position_ecef(self, dcm_eci_to_ecef: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return np.einsum("ijk,ik->ij", dcm_eci_to_ecef, self.position_eci)


class SatelliteOrbitPropagator(ABC):
    @abstractmethod
    def propagate(self, time: npt.NDArray[np.datetime64]) -> SatelliteOrbitState: ...


class CircularSatelliteOrbitPropagator(SatelliteOrbitPropagator):
    def __init__(self, model: CircularSatelliteOrbitModel) -> None:
        self._model = model

    @cached_property
    def mean_motion(self) -> float:
        return float(np.sqrt(EARTH_MU / self._model.semi_major_axis**3))  # pyright: ignore [reportAny]

    @cached_property
    def dcm_orbit_to_eci(self) -> npt.NDArray[np.floating]:
        return np.array(
            [
                [np.cos(self._model.raan), -np.sin(self._model.raan), 0],
                [np.sin(self._model.raan), np.cos(self._model.raan), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        ) @ np.array(
            [
                [1, 0, 0],
                [0, np.cos(self._model.inclination), -np.sin(self._model.inclination)],
                [0, np.sin(self._model.inclination), np.cos(self._model.inclination)],
            ],
            dtype=np.float64,
        )

    def propagate(self, time: npt.NDArray[np.datetime64]) -> SatelliteOrbitState:
        time_from_epoch = time - self._model.epoch
        return self.propagate_from_epoch(time_from_epoch)

    def propagate_from_epoch(self, time_from_epoch: npt.NDArray[np.timedelta64]) -> SatelliteOrbitState:
        time_from_epoch_in_seconds = time_from_epoch / np.timedelta64(1, "s")
        phase = self._model.phase_at_epoch + self.mean_motion * time_from_epoch_in_seconds
        in_plane_position = self._model.semi_major_axis * np.array([np.cos(phase), np.sin(phase), np.zeros_like(phase)])
        in_plane_velocity = (
            self.mean_motion
            * self._model.semi_major_axis
            * np.array([-np.sin(phase), np.cos(phase), np.zeros_like(phase)])
        )

        position_eci: npt.NDArray[np.floating] = rowwise_matmul(self.dcm_orbit_to_eci, in_plane_position.T)
        velocity_eci: npt.NDArray[np.floating] = rowwise_matmul(self.dcm_orbit_to_eci, in_plane_velocity.T)

        return SatelliteOrbitState(position_eci=position_eci, velocity_eci=velocity_eci)


class EllipticalSatelliteOrbitPropagator(SatelliteOrbitPropagator):
    def __init__(self, model: EllipticalSatelliteOrbitModel, reference_frame: ReferenceFrame = "gcrs") -> None:
        self._model = model
        rf = reference_frame.lower()
        if rf not in {"teme", "j2000", "gcrs"}:
            logger.error("Invalid reference frame: %s", reference_frame)
            msg = f"Invalid reference_frame: {reference_frame!r}"
            raise ValueError(msg)
        self.reference_frame = rf

    @cached_property
    def ts(self) -> Timescale:
        return load.timescale()

    @cached_property
    def mean_motion(self) -> float:
        return np.sqrt(EARTH_MU / self._model.semi_major_axis**3)

    def datetime64_utc_to_skytime(self, t_datetime64: npt.NDArray[np.datetime64]) -> Time:
        t_datetimes = [time.astype(datetime).replace(tzinfo=utc) for time in t_datetime64]
        return self.ts.from_datetimes(t_datetimes)

    @cached_property
    def satellite(self) -> EarthSatellite:
        satrec = Satrec()
        sgp4_ref_epoch = np.datetime64("1949-12-31T00:00:00")
        epoch = (self._model.epoch - sgp4_ref_epoch) / np.timedelta64(1, "D")
        rad_per_min = 60
        satrec.sgp4init(
            self._model.gravity_model.value,
            "i",
            self._model.satnum,
            epoch,
            self._model.drag_coeff,
            0.0,
            0.0,
            self._model.eccentricity,
            self._model.argpo,
            self._model.inclination,
            self._model.phase_at_epoch,
            self.mean_motion * rad_per_min,
            self._model.raan,
        )
        return EarthSatellite.from_satrec(satrec, self.ts)

    @override
    def propagate(self, time: npt.NDArray[np.datetime64]) -> SatelliteOrbitState:
        time_from_epoch = self.datetime64_utc_to_skytime(time)
        if self.reference_frame in {"j2000", "gcrs"}:
            geoc = self.satellite.at(time_from_epoch)  # Geocentric position in GCRS/ICRF axes
            position_eci_km = geoc.position.km
            velocity_eci_km_s = geoc.velocity.km_per_s
            return SatelliteOrbitState(position_eci=position_eci_km.T * 1e3, velocity_eci=velocity_eci_km_s.T * 1e3)
        else:  # TEME
            [position_eci, velocity_eci, _] = self.satellite._position_and_velocity_TEME_km(time_from_epoch)  # noqa: SLF001
            return SatelliteOrbitState(position_eci=position_eci.T * 1e3, velocity_eci=velocity_eci.T * 1e3)


@deprecated("Use SatelliteOrbitModel and SatelliteOrbitPropagator instead.")
class SatelliteOrbit(ABC):
    @abstractmethod
    def propagate(self, time: npt.NDArray[np.datetime64]) -> SatelliteOrbitState: ...


@deprecated("Use CircularSatelliteOrbitModel and CircularSatelliteOrbitPropagator instead.")
@dataclass(frozen=True, kw_only=True, slots=True)
class CircularSatelliteOrbit(SatelliteOrbit):
    semi_major_axis: float
    inclination: float
    raan: float
    phase_at_epoch: float
    epoch: np.datetime64

    _model: CircularSatelliteOrbitModel = field(init=False, repr=False)
    _propagator: CircularSatelliteOrbitPropagator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_model",
            CircularSatelliteOrbitModel(
                semi_major_axis=self.semi_major_axis,
                inclination=self.inclination,
                raan=self.raan,
                phase_at_epoch=self.phase_at_epoch,
                epoch=self.epoch,
            ),
        )
        object.__setattr__(self, "_propagator", CircularSatelliteOrbitPropagator(self._model))

    @cached_property
    def mean_motion(self) -> float:
        return self._propagator.mean_motion

    @cached_property
    def dcm_orbit_to_eci(self) -> npt.NDArray[np.floating]:
        return self._propagator.dcm_orbit_to_eci

    def propagate(self, time: npt.NDArray[np.datetime64]) -> SatelliteOrbitState:
        return self._propagator.propagate(time)

    def propagate_from_epoch(self, time_from_epoch: npt.NDArray[np.timedelta64]) -> SatelliteOrbitState:
        return self._propagator.propagate_from_epoch(time_from_epoch)


@deprecated(
    "Use EllipticalSatelliteOrbitModel and EllipticalSatelliteOrbitPropagator instead.",
)
@dataclass(frozen=True, kw_only=True, slots=True)
class EllipticalSatelliteOrbit(SatelliteOrbit):
    semi_major_axis: float
    inclination: float
    raan: float
    phase_at_epoch: float
    epoch: np.datetime64
    satnum: int  # Satellite number
    gravity_model: GravityModel = GravityModel.WGS84
    drag_coeff: float  # B star drag coefficient given in [1/earth radii]
    eccentricity: float
    argpo: float  # Argument of Perigee

    _model: EllipticalSatelliteOrbitModel = field(init=False, repr=False)
    _propagator: EllipticalSatelliteOrbitPropagator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_model",
            EllipticalSatelliteOrbitModel(
                semi_major_axis=self.semi_major_axis,
                inclination=self.inclination,
                raan=self.raan,
                phase_at_epoch=self.phase_at_epoch,
                epoch=self.epoch,
                satnum=self.satnum,
                gravity_model=self.gravity_model,
                drag_coeff=self.drag_coeff,
                eccentricity=self.eccentricity,
                argpo=self.argpo,
            ),
        )
        object.__setattr__(self, "_propagator", EllipticalSatelliteOrbitPropagator(self._model))

    @cached_property
    def ts(self) -> Timescale:
        return self._propagator.ts

    @cached_property
    def mean_motion(self) -> float:
        return self._propagator.mean_motion

    def datetime64_utc_to_skytime(self, t_datetime64: npt.NDArray[np.datetime64]) -> Time:
        return self._propagator.datetime64_utc_to_skytime(t_datetime64)

    @cached_property
    def satellite(self) -> EarthSatellite:
        return self._propagator.satellite

    def propagate(self, time: npt.NDArray[np.datetime64]) -> SatelliteOrbitState:
        return self._propagator.propagate(time)


def make_satellite_orbit(
    orbit_type: Literal["circular"],
    *args: Any,
    **kwargs: Any,
) -> SatelliteOrbit:
    if orbit_type == "circular":
        return CircularSatelliteOrbit(*args, **kwargs)
    else:
        msg = f"Unknown orbit type: {orbit_type}"
        raise ValueError(msg)
