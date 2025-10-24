__all__ = [
    "preserve_tick_params",
]
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

from matplotlib.axes import Axes

logger = logging.getLogger(__name__)
_WHICH_LIST: tuple[Literal["major", "minor"], Literal["major", "minor"]] = ("major", "minor")


def _save_tick_params(ax: Axes) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "x": {which: ax.xaxis.get_tick_params(which=which) for which in _WHICH_LIST},
        "y": {which: ax.yaxis.get_tick_params(which=which) for which in _WHICH_LIST},
    }


def _restore_tick_params(params: dict[str, dict[str, dict[str, Any]]], ax: Axes) -> None:
    for which in _WHICH_LIST:
        ax.xaxis.set_tick_params(which=which, **params["x"][which])
        ax.yaxis.set_tick_params(which=which, **params["y"][which])


@contextmanager
def preserve_tick_params(ax: Axes) -> Generator[None, Any, None]:
    params = _save_tick_params(ax)
    try:
        yield
    finally:
        _restore_tick_params(params, ax)
