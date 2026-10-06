import numpy as np
import pytest

from cosmica.comm_link import ApertureAveragedLogNormalScintillationModel

_LINK_DISTANCE = 1_000e3
_DEFAULT_RYTOV_VARIANCE = 0.5


def _make_model(default_rytov_variance: float = _DEFAULT_RYTOV_VARIANCE) -> ApertureAveragedLogNormalScintillationModel:
    return ApertureAveragedLogNormalScintillationModel(
        default_rytov_variance=default_rytov_variance,
        wavelength=1550e-9,
        aperture_diameter=0.4,
    )


def test_scintillation_variance_uses_default_rytov_variance_without_override() -> None:
    model = _make_model()

    assert model.sigma2_scintillation(_LINK_DISTANCE) == model.sigma2_scintillation(
        _LINK_DISTANCE,
        rytov_variance=_DEFAULT_RYTOV_VARIANCE,
    )


@pytest.mark.parametrize("rytov_variance", [0.1, 1.0, 2.0])
def test_scintillation_variance_honors_rytov_variance_override(rytov_variance: float) -> None:
    model = _make_model()

    overridden = model.sigma2_scintillation(_LINK_DISTANCE, rytov_variance=rytov_variance)

    assert overridden != pytest.approx(model.sigma2_scintillation(_LINK_DISTANCE))
    assert overridden == pytest.approx(
        _make_model(default_rytov_variance=rytov_variance).sigma2_scintillation(_LINK_DISTANCE),
    )


def test_scintillation_variance_honors_zero_rytov_variance_override() -> None:
    assert _make_model().sigma2_scintillation(_LINK_DISTANCE, rytov_variance=0.0) == 0.0


def test_scintillation_variance_increases_with_rytov_variance() -> None:
    model = _make_model()

    variances = [model.sigma2_scintillation(_LINK_DISTANCE, rytov_variance=v) for v in (0.1, 0.5, 1.0, 2.0)]

    assert variances == sorted(variances)
    assert len(set(variances)) == len(variances)


@pytest.mark.parametrize("rytov_variance", [0.1, 2.0])
def test_scintillation_sample_honors_rytov_variance_override(rytov_variance: float) -> None:
    overridden = _make_model().sample(np.random.default_rng(0), _LINK_DISTANCE, rytov_variance=rytov_variance)
    expected = _make_model(default_rytov_variance=rytov_variance).sample(np.random.default_rng(0), _LINK_DISTANCE)

    assert overridden == pytest.approx(expected)
