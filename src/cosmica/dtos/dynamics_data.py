from __future__ import annotations

__all__ = [
    "DynamicsData",
    "DynamicsDataSnapshot",
]

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

from cosmica.models import Satellite


def _check_item_shape_if_any[T: Satellite](
    data: dict[T, npt.NDArray[np.floating]],
    target_shape: tuple[int, ...],
    slice_: int | slice | None = None,
) -> bool:
    try:
        slice_ = slice_ or slice(None)
        return next(iter(data.values())).shape[slice_] == target_shape
    except StopIteration:
        return True


@dataclass(frozen=True, kw_only=True, slots=True)
class DynamicsDataSnapshot[T: Satellite]:
    time: np.datetime64
    dcm_eci2ecef: npt.NDArray[np.floating]
    satellite_position_eci: dict[T, npt.NDArray[np.floating]]
    satellite_velocity_eci: dict[T, npt.NDArray[np.floating]]
    satellite_position_ecef: dict[T, npt.NDArray[np.floating]]
    satellite_attitude_angular_velocity_eci: dict[T, npt.NDArray[np.floating]]
    sun_direction_eci: npt.NDArray[np.floating]
    sun_direction_ecef: npt.NDArray[np.floating]

    @property
    def data_shape(self) -> tuple[()]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class DynamicsData[T: Satellite]:
    time: npt.NDArray[np.datetime64]
    dcm_eci2ecef: npt.NDArray[np.floating]
    satellite_position_eci: dict[T, npt.NDArray[np.floating]]
    satellite_velocity_eci: dict[T, npt.NDArray[np.floating]]
    satellite_position_ecef: dict[T, npt.NDArray[np.floating]]
    satellite_attitude_angular_velocity_eci: dict[T, npt.NDArray[np.floating]]
    sun_direction_eci: npt.NDArray[np.floating]
    sun_direction_ecef: npt.NDArray[np.floating]

    @property
    def data_shape(self) -> tuple[int, ...]:
        return self.time.shape

    def __post_init__(self) -> None:
        if any(current_time <= previous_time for previous_time, current_time in pairwise(self.time.flat)):
            msg = "time must be strictly increasing"
            raise ValueError(msg)

        data_shape = self.time.shape
        assert self.dcm_eci2ecef.shape[:-2] == data_shape
        assert _check_item_shape_if_any(self.satellite_position_eci, data_shape, slice(-1))
        assert _check_item_shape_if_any(self.satellite_velocity_eci, data_shape, slice(-1))
        assert _check_item_shape_if_any(self.satellite_position_ecef, data_shape, slice(-1))
        assert _check_item_shape_if_any(self.satellite_attitude_angular_velocity_eci, data_shape, slice(-1))
        assert self.sun_direction_eci.shape[:-1] == data_shape
        assert self.sun_direction_ecef.shape[:-1] == data_shape

    def __iter__(self) -> Iterator[DynamicsDataSnapshot[T]]:
        return (self._snapshot(index) for index in range(len(self.time)))

    def _snapshot(self, index: int) -> DynamicsDataSnapshot[T]:
        snapshot_data = self[index]
        assert isinstance(snapshot_data.time, np.datetime64)
        return DynamicsDataSnapshot(
            time=snapshot_data.time,
            dcm_eci2ecef=snapshot_data.dcm_eci2ecef,
            satellite_position_eci=snapshot_data.satellite_position_eci,
            satellite_velocity_eci=snapshot_data.satellite_velocity_eci,
            satellite_position_ecef=snapshot_data.satellite_position_ecef,
            satellite_attitude_angular_velocity_eci=snapshot_data.satellite_attitude_angular_velocity_eci,
            sun_direction_eci=snapshot_data.sun_direction_eci,
            sun_direction_ecef=snapshot_data.sun_direction_ecef,
        )

    def __getitem__(self, item: int | slice) -> DynamicsData[T]:
        return DynamicsData(
            time=self.time[item],
            dcm_eci2ecef=self.dcm_eci2ecef[item],
            satellite_position_eci={key: value[item] for key, value in self.satellite_position_eci.items()},
            satellite_velocity_eci={key: value[item] for key, value in self.satellite_velocity_eci.items()},
            satellite_position_ecef={key: value[item] for key, value in self.satellite_position_ecef.items()},
            satellite_attitude_angular_velocity_eci={
                key: value[item] for key, value in self.satellite_attitude_angular_velocity_eci.items()
            },
            sun_direction_eci=self.sun_direction_eci[item],
            sun_direction_ecef=self.sun_direction_ecef[item],
        )
