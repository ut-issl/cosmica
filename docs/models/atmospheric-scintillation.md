# Atmospheric scintillation on downlinks

This page explains how COSMICA models the irradiance fluctuations (scintillation) that atmospheric turbulence causes on a satellite-to-ground optical link, and how the model was checked. It is written for an engineer who uses or changes `ApertureAveragedLogNormalScintillationModel` or `SatToGatewayBinaryCommLinkCalculatorWithScintillation` and knows basic link budgets, but not turbulence theory.

Every equation number refers to L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random Media*, 2nd ed., SPIE Press, 2005.

## What the model computes

Turbulence makes the received power fluctuate around its mean. The calculator multiplies the mean received power by a random factor $I$ with mean 1, drawn once per link and time step. The size of the fluctuation is the scintillation index $\sigma_I^2$, the variance of $I$. A large receiver aperture averages over many independent bright and dark patches of the beam, so it sees a smaller index than a point receiver. This reduction is called aperture averaging, and it is the main reason ground stations use apertures of tens of centimeters or more.

On a downlink the turbulence sits in the lowest ~20 km of the atmosphere, close to the receiver, while the satellite is hundreds of kilometers away. The wave arriving at the top of the atmosphere is therefore close to a plane wave, and Section 12.5.2 shows that beam wander is negligible. The model follows that section.

## Turbulence profile

The strength of turbulence at altitude $h$ is the refractive-index structure parameter $C_n^2(h)$, in m$^{-2/3}$. The model uses the Hufnagel-Valley profile (Eq. 12.1):

$$
C_n^2(h) = 0.00594 \left(\frac{w}{27}\right)^2 \left(10^{-5} h\right)^{10} e^{-h/1000} + 2.7 \times 10^{-16}\, e^{-h/1500} + A\, e^{-h/100}
$$

Here $h$ is the altitude above the receiver in meters, $w$ is the rms wind speed in m/s (`rms_wind_speed`), and $A = C_n^2(0)$ is the ground-level turbulence (`ground_cn2`). The first term is the tropopause layer near 10 km, which the high-altitude wind drives. The last term is the ground layer, which dominates near the station. The defaults $w = 21$ m/s and $A = 1.7 \times 10^{-14}$ m$^{-2/3}$ are the H-V5/7 model used throughout Chapter 12. Each `GatewayOGS` carries its own $A$ and $w$, because both depend on the site and the season.

## Point-receiver index: the Rytov variance

For a point receiver under weak turbulence, the index equals the plane-wave Rytov variance (Eq. 12.38):

$$
\sigma_R^2 = 2.25\, k^{7/6} \sec^{11/6}(\zeta) \int_0^\infty C_n^2(h)\, h^{5/6}\, dh
$$

where $k = 2\pi/\lambda$ is the wavenumber and $\zeta$ is the zenith angle of the link at the station. The $\sec^{11/6}(\zeta)$ factor is why scintillation grows quickly as the satellite moves toward the horizon. Each H-V term has the form $c\, h^p e^{-h/s}$, so the integral has the closed form $\int_0^\infty h^p e^{-h/s}\, dh = s^{p+1}\Gamma(p+1)$, and `rytov_variance` evaluates it exactly. The book integrates up to the satellite altitude $H$. Integrating to infinity instead is the same for any satellite, because $C_n^2$ above 30 km is negligible. Stopping at 20 km would lower the result by about 1 %.

As a worked example, at 1550 nm with H-V5/7 the Rytov variance is 0.063 at zenith and 0.119 at $\zeta = 45°$.

## Aperture-averaged index

For a receiver of diameter $D$, the index is the irradiance flux variance (Eq. 12.39). With $L = (H - h_0)\sec\zeta$, the $(H - h_0)^{5/6}$ factor of Eq. 12.39 cancels inside the bracket, and the receiver at $h_0 = 0$ gives

$$
\sigma_I^2(D) = 8.70\, k^{7/6} \sec^{11/6}(\zeta)\, \mathrm{Re} \int_0^\infty C_n^2(h) \left[ \left(\frac{k D^2 \cos\zeta}{16} + i h\right)^{5/6} - \left(\frac{k D^2 \cos\zeta}{16}\right)^{5/6} \right] dh
$$

This form no longer depends on the satellite altitude. When $D \to 0$ it reduces to Eq. 12.38, because $8.70 \cos(5\pi/12) = 2.25$. `sigma2_scintillation` evaluates the integral with the trapezoidal rule on a geometric altitude grid from 1 cm to 50 km with 1000 points. The geometric spacing resolves the 100 m ground layer, and 1000 points agree with 4000 points to within $4 \times 10^{-5}$ relative.

The aperture effect is large. At $\zeta = 45°$ and 1550 nm, a 20 cm aperture reduces the index from 0.119 to 0.025, and a 1 m aperture reduces it to 0.0008.

## Distribution of the fluctuations

The intensity factor is log-normal with unit mean. With $\sigma_{\ln I}^2 = \ln(1 + \sigma_I^2)$ (Eq. 9.12), the sample is $I = e^{X}$ with $X \sim \mathcal{N}(-\sigma_{\ln I}^2/2,\ \sigma_{\ln I}^2)$, which gives $E[I] = 1$ and $\mathrm{Var}[I] = \sigma_I^2$.

## Assumptions and validity

- **Weak fluctuations.** Eq. 12.38 and 12.39 come from weak-fluctuation theory, which holds while $\sigma_R^2 < 1$. With H-V5/7 at 1550 nm this covers zenith angles up to about 77° (elevation above about 13°). With stronger ground turbulence, the limit moves toward the zenith: Section 12.5.2 puts it near $\zeta = 40°$ for $A = 3 \times 10^{-13}$ m$^{-2/3}$ at 1.06 µm. Past the limit, the weak-fluctuation index overestimates the point-receiver scintillation, and the model makes no correction.
- **Log-normal irradiance.** Section 12.7 notes that the gamma-gamma distribution fits better in moderate to strong turbulence. With aperture averaging, the downlink index is small and the two distributions agree closely.
- **Plane-parallel atmosphere.** The $\sec\zeta$ path stretching ignores Earth curvature, so the model rejects zenith angles at or beyond 90°.
- **Receiver at the bottom of the profile.** $h$ is measured from the station, so a high-altitude site is modeled by lowering $A$ rather than by shifting the profile.
- **Independent samples.** Each time step draws a new factor. Scintillation decorrelates within milliseconds, so this is consistent with time steps of a second or more.

## Why this model

The earlier model used Eq. 10.69, which describes a horizontal path with constant $C_n^2$ over the whole path length, and passed the slant range to the satellite as that length. That placed the turbulence along hundreds of kilometers instead of the lowest 20 km. As a result it predicted almost no aperture averaging, and its Rytov variance was a fixed number per station that ignored elevation. For a 1 m aperture at 1550 nm it gave 0.15 to 0.24 over a pass, while Eq. 12.39 gives 0.0003 to 0.007 (issue #354).

## Verification

| Claim | Test | Reference | Tolerance |
|---|---|---|---|
| Rytov variance of the H-V profile matches the book | `test_rytov_variance_matches_andrews_figures` | Fig. 12.6 at $D = 0$ (1.55 µm), Fig. 12.7 weak-fluctuation curves (1.06 µm) | half a minor tick: 0.005 (Fig. 12.6), 0.01 (Fig. 12.7) |
| Aperture-averaged index matches the book | `test_aperture_averaged_scintillation_matches_andrews_figure_12_6` | Fig. 12.6, $\zeta$ = 45° and 60°, $D$ = 0.1 to 0.5 m | 0.005, half a minor tick |
| Eq. 12.39 quadrature reduces to the closed-form Eq. 12.38 | `test_aperture_averaged_scintillation_tends_to_rytov_variance_for_point_receiver` | analytic limit $D \to 0$ | 0.2 %, from the 0.08 % prefactor difference |
| Index falls with aperture and rises with zenith angle | `test_aperture_averaged_scintillation_decreases_with_aperture_and_increases_with_zenith` | property of Eq. 12.39 | strict ordering |
| Samples have unit mean and variance $\sigma_I^2$ | `test_scintillation_samples_have_unit_mean_and_expected_variance` | Eq. 9.12 | about three standard errors |
| The calculator passes each link's zenith angle | `test_scintillation_downlink_passes_link_zenith_angle_to_model` | link geometry: $\zeta = 90° - $ elevation | $10^{-9}$ rad |

The figure values were read by eye from the printed figures.
