"""Tests for communication-link uncertainty models."""

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmica.comm_link import ExpEdgeModel

EPOCH = np.datetime64("2026-01-01T00:00:00", "s")


class _SequenceRng:
    """Return predetermined exponential samples for deterministic transition tests."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)

    def exponential(self, _scale: float) -> float:
        return next(self._values)


def _rng(*values: float) -> np.random.Generator:
    return cast("np.random.Generator", _SequenceRng(values))


def _time_grid(*seconds: int) -> npt.NDArray[np.datetime64]:
    return EPOCH + np.asarray(seconds) * np.timedelta64(1, "s")


@pytest.mark.parametrize("seconds", [(), (0,)])
def test_exp_edge_model_requires_at_least_two_samples(seconds: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="at least two"):
        ExpEdgeModel().simulate(_time_grid(*seconds), _rng(100.0))


@pytest.mark.parametrize("seconds", [(0, 0, 10), (0, 10, 5)])
def test_exp_edge_model_requires_strictly_increasing_samples(seconds: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ExpEdgeModel().simulate(_time_grid(*seconds), _rng(100.0))


def test_exp_edge_model_requires_a_regular_time_grid() -> None:
    with pytest.raises(ValueError, match="regular"):
        ExpEdgeModel().simulate(_time_grid(0, 10, 25), _rng(100.0))


def test_exp_edge_model_ignores_failure_at_final_sample_boundary() -> None:
    states = ExpEdgeModel(
        reliability=np.timedelta64(30, "s"),
    ).simulate(_time_grid(0, 10, 20, 30), _rng(30.0))

    np.testing.assert_array_equal(states, [False, False, False, False])


def test_exp_edge_model_ignores_recovery_at_final_sample_boundary() -> None:
    states = ExpEdgeModel(
        reliability=np.timedelta64(1, "s"),
        recovery_time=np.timedelta64(29, "s"),
    ).simulate(_time_grid(0, 10, 20, 30), _rng(1.0, 100.0))

    np.testing.assert_array_equal(states, [False, True, True, True])


def test_exp_edge_model_cancels_failure_and_recovery_in_one_sample() -> None:
    states = ExpEdgeModel(
        reliability=np.timedelta64(1, "s"),
        recovery_time=np.timedelta64(1, "s"),
    ).simulate(_time_grid(0, 10, 20, 30), _rng(1.0, 100.0))

    np.testing.assert_array_equal(states, [False, False, False, False])


def test_exp_edge_model_preserves_colliding_recovery_and_later_failure() -> None:
    states = ExpEdgeModel(
        reliability=np.timedelta64(1, "s"),
        recovery_time=np.timedelta64(15, "s"),
    ).simulate(_time_grid(0, 10, 20, 30, 40), _rng(1.0, 1.0, 100.0))

    np.testing.assert_array_equal(states, [False, True, True, True, False])
