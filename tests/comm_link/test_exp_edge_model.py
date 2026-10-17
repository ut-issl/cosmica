from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmica.comm_link import ExpEdgeModel

_START = np.datetime64("2026-01-01T00:00:00", "s")


class _ScriptedExponentials:
    """Stand-in random generator whose exponential draws (in seconds) are scripted."""

    def __init__(self, draws: Sequence[float]) -> None:
        self._draws = list(draws)

    def exponential(self, scale: float) -> float:  # noqa: ARG002
        # After the script ends, the next failure falls far beyond any test time grid.
        return self._draws.pop(0) if self._draws else 1e12


def _simulate(
    elapsed_seconds: Sequence[int],
    *,
    draws: Sequence[float],
    recovery_seconds: int,
) -> npt.NDArray[np.bool_]:
    time = _START + np.asarray(elapsed_seconds).astype("timedelta64[s]")
    model = ExpEdgeModel(recovery_time=np.timedelta64(recovery_seconds, "s"))
    return model.simulate(time, cast("np.random.Generator", _ScriptedExponentials(draws)))


@pytest.mark.parametrize(
    "time",
    [
        pytest.param(np.array([], dtype="datetime64[s]"), id="empty"),
        pytest.param(np.full((2, 2), _START), id="two-dimensional"),
        pytest.param(_START + np.array([0, 60, 60]).astype("timedelta64[s]"), id="repeated"),
        pytest.param(_START + np.array([0, 120, 60]).astype("timedelta64[s]"), id="decreasing"),
    ],
)
def test_exp_edge_model_rejects_unsupported_time_grids(time: npt.NDArray[np.datetime64]) -> None:
    with pytest.raises(ValueError, match="time must be"):
        ExpEdgeModel().simulate(time, np.random.default_rng(0))


def test_exp_edge_model_accepts_one_sample() -> None:
    assert _simulate([0], draws=[0.0], recovery_seconds=60).tolist() == [False]


def test_exp_edge_model_marks_samples_after_failure_through_recovery() -> None:
    # Failure at 90 s lasting 120 s: samples in (90 s, 210 s] are failed.
    failed = _simulate([0, 60, 120, 180, 240, 300], draws=[90.0], recovery_seconds=120)

    assert failed.tolist() == [False, False, True, True, False, False]


def test_exp_edge_model_supports_irregular_time_grid() -> None:
    # Failure at 5 s lasting 10 s: samples in (5 s, 15 s] are failed.
    failed = _simulate([0, 1, 3, 7, 15, 16, 30], draws=[5.0], recovery_seconds=10)

    assert failed.tolist() == [False, False, False, True, True, False, False]


def test_exp_edge_model_ignores_failure_shorter_than_time_step() -> None:
    # Failure at 90 s recovers at 120 s, so no later sample stays failed.
    failed = _simulate([0, 60, 120, 180, 240], draws=[90.0], recovery_seconds=20)

    assert not failed.any()


def test_exp_edge_model_keeps_later_failure_separate_from_earlier_recovery() -> None:
    # Failures at 30 s and 100 s, each lasting 60 s: (30 s, 90 s] and (100 s, 160 s].
    # Both the first recovery and the second failure map to the 120 s sample.
    failed = _simulate([0, 60, 120, 180, 240], draws=[30.0, 10.0], recovery_seconds=60)

    assert failed.tolist() == [False, True, True, False, False]


@pytest.mark.parametrize(
    ("failure_seconds", "expected"),
    [
        pytest.param(60.0, [False, False, True], id="on-inner-sample"),
        pytest.param(120.0, [False, False, False], id="on-final-sample"),
    ],
)
def test_exp_edge_model_handles_failures_on_sample_times(failure_seconds: float, expected: list[bool]) -> None:
    assert _simulate([0, 60, 120], draws=[failure_seconds], recovery_seconds=600).tolist() == expected
