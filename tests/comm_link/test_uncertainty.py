"""Tests for the downlink scintillation model, see docs/models/atmospheric-scintillation.md.

Reference values are read by eye from Andrews and Phillips, Laser Beam Propagation through Random Media,
2nd ed. (2005), Figures 12.6 and 12.7, which use the H-V5/7 profile (ground Cn2 1.7e-14 m^-2/3, rms wind 21 m/s).
"""

import numpy as np
import pytest

from cosmica.comm_link import ApertureAveragedLogNormalScintillationModel


def _make_model(
    *,
    wavelength: float = 1550e-9,
    aperture_diameter: float = 0.4,
    ground_cn2: float = 1.7e-14,
) -> ApertureAveragedLogNormalScintillationModel:
    return ApertureAveragedLogNormalScintillationModel(
        wavelength=wavelength,
        aperture_diameter=aperture_diameter,
        ground_cn2=ground_cn2,
        rms_wind_speed=21.0,
    )


# Figure 12.6 at D = 0 (point receiver): 1.55 um, zenith 45 and 60 deg.
# Figure 12.7, weak-fluctuation curve: 1.06 um, ground Cn2 1.7e-14 at zenith 0 and 40 deg, 3e-13 at zenith 0 deg.
# Tolerance: half a minor tick of each figure (0.005 for Fig. 12.6, 0.01 for Fig. 12.7).
@pytest.mark.parametrize(
    ("wavelength", "ground_cn2", "zenith_deg", "expected", "tolerance"),
    [
        pytest.param(1550e-9, 1.7e-14, 45.0, 0.117, 0.005, id="fig12.6-zenith45"),
        pytest.param(1550e-9, 1.7e-14, 60.0, 0.225, 0.005, id="fig12.6-zenith60"),
        pytest.param(1.06e-6, 1.7e-14, 0.0, 0.10, 0.01, id="fig12.7-hv57-zenith0"),
        pytest.param(1.06e-6, 1.7e-14, 40.0, 0.16, 0.01, id="fig12.7-hv57-zenith40"),
        pytest.param(1.06e-6, 3e-13, 0.0, 0.32, 0.01, id="fig12.7-strong-ground-zenith0"),
    ],
)
def test_rytov_variance_matches_andrews_figures(
    wavelength: float,
    ground_cn2: float,
    zenith_deg: float,
    expected: float,
    tolerance: float,
) -> None:
    model = _make_model(wavelength=wavelength, ground_cn2=ground_cn2)

    assert model.rytov_variance(np.deg2rad(zenith_deg)) == pytest.approx(expected, abs=tolerance)


# Figure 12.6: 1.55 um, H-V5/7. Tolerance 0.005 is half a minor tick (0.01) of the figure's vertical axis.
@pytest.mark.parametrize(
    ("zenith_deg", "aperture_diameter", "expected"),
    [
        (45.0, 0.0, 0.117),
        (45.0, 0.1, 0.056),
        (45.0, 0.2, 0.025),
        (45.0, 0.3, 0.013),
        (45.0, 0.5, 0.004),
        (60.0, 0.0, 0.225),
        (60.0, 0.1, 0.114),
        (60.0, 0.2, 0.061),
        (60.0, 0.3, 0.034),
        (60.0, 0.5, 0.014),
    ],
)
def test_aperture_averaged_scintillation_matches_andrews_figure_12_6(
    zenith_deg: float,
    aperture_diameter: float,
    expected: float,
) -> None:
    model = _make_model(aperture_diameter=aperture_diameter)

    assert model.sigma2_scintillation(np.deg2rad(zenith_deg)) == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize("zenith_deg", [0.0, 45.0, 60.0, 70.0])
def test_aperture_averaged_scintillation_reduces_to_rytov_variance_for_point_receiver(zenith_deg: float) -> None:
    """Eq. 12.39 reduces to Eq. 12.38 at D = 0.

    The quadrature of Eq. 12.39 is checked against the closed-form Rytov variance. The prefactors differ by
    8.70 cos(5 pi / 12) / 2.25 - 1 = 0.08 %, so the tolerance is 0.2 %.
    """
    zenith = np.deg2rad(zenith_deg)
    model = _make_model(aperture_diameter=0.0)

    assert model.sigma2_scintillation(zenith) == pytest.approx(model.rytov_variance(zenith), rel=2e-3)


def test_aperture_averaged_scintillation_decreases_with_aperture_and_increases_with_zenith() -> None:
    zenith = np.deg2rad([0.0, 30.0, 60.0, 75.0])
    diameters = [0.05, 0.2, 1.0]
    variances = np.array(
        [[_make_model(aperture_diameter=d).sigma2_scintillation(z) for z in zenith] for d in diameters],
    )

    assert np.all(np.diff(variances, axis=0) < 0)
    assert np.all(np.diff(variances, axis=1) > 0)


@pytest.mark.parametrize("zenith_angle", [-0.1, np.pi / 2, np.pi, np.nan])
def test_scintillation_rejects_zenith_angles_outside_the_upper_hemisphere(zenith_angle: float) -> None:
    model = _make_model()

    with pytest.raises(ValueError, match="zenith_angle"):
        model.sigma2_scintillation(zenith_angle)
    with pytest.raises(ValueError, match="zenith_angle"):
        model.rytov_variance(zenith_angle)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"wavelength": 0.0}, id="wavelength"),
        pytest.param({"aperture_diameter": -1.0}, id="aperture-diameter"),
        pytest.param({"ground_cn2": -1e-14}, id="ground-cn2"),
    ],
)
def test_scintillation_model_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _make_model(**kwargs)


@pytest.mark.parametrize("zenith_deg", [0.0, 60.0])
def test_scintillation_sample_uses_log_normal_parameters(zenith_deg: float) -> None:
    zenith = np.deg2rad(zenith_deg)
    model = _make_model()
    log_variance = np.log1p(model.sigma2_scintillation(zenith))
    expected = np.random.default_rng(42).lognormal(-log_variance / 2, np.sqrt(log_variance))

    assert model.sample(np.random.default_rng(42), zenith) == pytest.approx(expected)


def test_scintillation_samples_have_unit_mean_and_expected_variance() -> None:
    """Check unit mean and scintillation-index variance over a reproducible sample population.

    At zenith 75 deg with a 5 cm aperture the index is about 0.6. With 50 000 samples the standard error of the
    sample variance is about 2 % and that of the mean about 0.003, so the tolerances are about three standard errors.
    """
    zenith = np.deg2rad(75.0)
    model = _make_model(aperture_diameter=0.05)
    rng = np.random.default_rng(42)
    samples = np.array([model.sample(rng, zenith) for _ in range(50_000)])

    assert np.mean(samples) == pytest.approx(1.0, abs=0.01)
    assert np.var(samples) == pytest.approx(model.sigma2_scintillation(zenith), rel=0.05)
