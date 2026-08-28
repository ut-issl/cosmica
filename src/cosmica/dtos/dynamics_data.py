from __future__ import annotations

__all__ = [
    "DynamicsData",
    "DynamicsDataSnapshot",
]

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated

import numpy as np
from typing_extensions import Doc

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
    satellite_attitude_dcm_eci2body: dict[T, npt.NDArray[np.floating]] = field(default_factory=dict)

    @property
    def data_shape(self) -> tuple[()]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class DynamicsData[T: Satellite]:
    """Time-indexed orbital, attitude, and solar dynamics data.

    ``satellite_attitude_dcm_eci2body`` may contain body attitudes for only the
    satellites needed by attitude-aware consumers. Calculators that require an
    attitude validate that their endpoint satellites have entries.
    """

    time: npt.NDArray[np.datetime64]
    dcm_eci2ecef: npt.NDArray[np.floating]
    satellite_position_eci: dict[T, npt.NDArray[np.floating]]
    satellite_velocity_eci: dict[T, npt.NDArray[np.floating]]
    satellite_position_ecef: dict[T, npt.NDArray[np.floating]]
    satellite_attitude_angular_velocity_eci: dict[T, npt.NDArray[np.floating]]
    sun_direction_eci: npt.NDArray[np.floating]
    sun_direction_ecef: npt.NDArray[np.floating]
    satellite_attitude_dcm_eci2body: Annotated[
        dict[T, npt.NDArray[np.floating]],
        Doc(
            "Per-satellite right-handed direction-cosine matrices transforming "
            "vector components from ECI to satellite body coordinates. Shape per "
            "satellite: time shape followed by (3, 3).",
        ),
    ] = field(default_factory=dict)

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
        self._validate_attitude_dcms(data_shape)

    def _validate_attitude_dcms(self, data_shape: tuple[int, ...]) -> None:
        for satellite, dcm in self.satellite_attitude_dcm_eci2body.items():
            expected_shape = (*data_shape, 3, 3)
            if dcm.shape != expected_shape:
                msg = (
                    "satellite_attitude_dcm_eci2body entries must have shape "
                    f"{expected_shape}, but {satellite!r} has shape {dcm.shape}"
                )
                raise ValueError(msg)
            if not np.all(np.isfinite(dcm)):
                msg = f"satellite_attitude_dcm_eci2body[{satellite!r}] must contain only finite values"
                raise ValueError(msg)
            orthogonality = dcm @ np.swapaxes(dcm, -1, -2)
            if not np.allclose(orthogonality, np.eye(3), rtol=0.0, atol=1e-12) or not np.allclose(
                np.linalg.det(dcm),
                1.0,
                rtol=0.0,
                atol=1e-12,
            ):
                msg = (
                    f"satellite_attitude_dcm_eci2body[{satellite!r}] must contain "
                    "orthonormal, right-handed direction-cosine matrices"
                )
                raise ValueError(msg)

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
            satellite_attitude_dcm_eci2body=snapshot_data.satellite_attitude_dcm_eci2body,
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
            satellite_attitude_dcm_eci2body={
                key: value[item] for key, value in self.satellite_attitude_dcm_eci2body.items()
            },
        )
