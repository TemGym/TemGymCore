"""Truth consistency of the bounded multilayer estimator, not all specimens."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.periodic_metrology import fit_periodic_frequency, periodic_template

jax.config.update("jax_enable_x64", True)


def image_fixture(shape=(64, 88), bins=(8.37, 4.28), defocus_phase=0., finite=True):
    """Independent analytic contrast-transfer image, not periodic_template.

    A finite measured illumination envelope multiplies a periodic three-order
    contrast image. A spatially invariant weak-object contrast transfer changes
    the coefficients/sign with defocus, but not the periodic wave vectors.
    This does not model propagation-induced beam-envelope coupling.
    """
    ny, nx = shape
    y, x = np.indices(shape)
    x, y = (x - (nx - 1) / 2) / nx, (y - (ny - 1) / 2) / ny
    envelope = np.exp(-((x - .12) / .38)**2 - ((y + .09) / .43)**2) if finite else np.ones(shape)
    phase = 2 * np.pi * (bins[0] * x + bins[1] * y)
    signal = 1000 + 25 * x - 40 * y + 16 * x * y
    for order, amplitude in enumerate((260, 90, 30), start=1):
        signal += amplitude * np.cos(defocus_phase * order**2) * np.cos(order * phase + .32 * order)
    return envelope * signal, envelope, np.asarray(bins) / [nx, ny]


def fit(image, frequency, envelope=None, variance=100., **kwargs):
    initial = frequency + np.array([.19, -.23]) / image.shape[::-1]
    return fit_periodic_frequency(image, initial, variance, illumination_yx=envelope, **kwargs)


@pytest.mark.parametrize("bins", [(8.37, 4.28), (9.62, -3.71), (-7.83, 5.43)])
@pytest.mark.parametrize("defocus_phase", [0., .8, 2.4])
def test_off_grid_frequency_truth_with_rotated_envelope_and_defocus_contrast(bins, defocus_phase):
    image, envelope, frequency = image_fixture(bins=bins, defocus_phase=defocus_phase)
    result = fit(image, frequency, envelope)
    assert result.accepted, result.reasons
    np.testing.assert_allclose(result.frequency_xy_per_pixel, frequency, atol=1e-11)
    np.testing.assert_allclose(result.fitted_image_yx, image, atol=1e-7)
    assert result.reduced_chi_squared < 1e-14
    assert np.linalg.eigvalsh(result.covariance_xy_per_pixel2).min() > 0


def test_smooth_gaussian_does_not_become_a_dc_mask_peak():
    y, x = np.indices((128, 128))
    image = 1000 * np.exp(-((x - 63.5)**2 + (y - 63.5)**2) / (2 * 45**2))
    # A smooth envelope known from the perforation image has exactly no modulation.
    result = fit_periodic_frequency(image, np.array([5.5, 0.]) / 128, 100.,
                                    illumination_yx=image / 1000, harmonics=1)
    assert not result.accepted
    assert "no significant fundamental modulation" in result.reasons
    assert result.fundamental_delta_chi_squared < 1e-12
    # The original spectrum has no local feature maxima in the search annulus.
    assert result.fft_candidates_xy_per_pixel.shape == (0, 2)
    with pytest.raises(ValueError, match="Unresolved periodic feature"):
        result.require_accepted()


def test_broad_smooth_gaussian_without_known_envelope_is_also_rejected():
    y, x = np.indices((128, 128))
    image = 1000 * np.exp(-((x - 63.5)**2 + (y - 63.5)**2) / (2 * 90**2))
    # A quadratic background can spuriously lend a >6 sigma sinusoid to this
    # smooth tail even while total residuals are small. The *unmodified*
    # spectrum's lack of a local maximum must still veto the measurement.
    result = fit_periodic_frequency(image, [5.5 / 128, 0.], 100., harmonics=1)
    assert not result.accepted
    assert "no corresponding maximum in the unmasked Fourier spectrum" in result.reasons


@pytest.mark.parametrize("mtf_sigma", [.65, 1.3])
def test_finite_envelope_detector_mtf_is_applied_before_frequency_measurement(mtf_sigma):
    from scipy.ndimage import gaussian_filter

    image, envelope, frequency = image_fixture()
    blurred = gaussian_filter(image, mtf_sigma, mode="nearest")
    result = fit(blurred, frequency, envelope, mtf_sigma_px=mtf_sigma)
    assert result.accepted, result.reasons
    np.testing.assert_allclose(result.frequency_xy_per_pixel, frequency, atol=1e-10)
    np.testing.assert_allclose(result.fitted_image_yx, blurred, atol=1e-6)


def test_noise_only_rejected_without_clipping_signed_pixels():
    rng = np.random.default_rng(82)
    image = rng.normal(0, 10, (48, 64))
    result = fit_periodic_frequency(image, [8 / 64, 4 / 48], 100., harmonics=1)
    assert not result.accepted
    assert "no significant fundamental modulation" in result.reasons


def test_missing_fundamental_is_not_confused_with_its_harmonic():
    shape = (64, 88)
    y, x = np.indices(shape)
    frequency = np.array([8.37 / 88, 4.28 / 64])
    image = 1000 + 200 * np.cos(4 * np.pi * (frequency[0] * x + frequency[1] * y))
    result = fit(image, frequency)
    assert not result.accepted
    assert "no significant fundamental modulation" in result.reasons


def test_unmodelled_nonperiodic_specimen_feature_fails_noise_gate():
    image, envelope, frequency = image_fixture()
    image[20:34, 31:48] += 800
    result = fit(image, frequency, envelope)
    assert not result.accepted
    assert "harmonic image model is inconsistent with supplied detector noise" in result.reasons


def test_frequency_outside_frozen_registration_is_not_silently_reassigned():
    image, envelope, frequency = image_fixture()
    result = fit_periodic_frequency(image, frequency + np.array([2., 0]) / image.shape[::-1],
                                    100., illumination_yx=envelope, search_radius_bins=.7)
    assert not result.accepted


def test_registered_raw_pixel_mask_excludes_unmodelled_structure_without_resampling():
    image, envelope, frequency = image_fixture()
    mask = np.ones(image.shape, dtype=bool)
    mask[:8] = False
    image[:8] += 200
    result = fit(image, frequency, envelope, valid_mask_yx=mask)
    assert result.accepted, result.reasons
    np.testing.assert_allclose(result.frequency_xy_per_pixel, frequency, atol=1e-10)
    assert result.chi_squared < 1e-12


def test_mask_cannot_hide_insufficient_visible_periods():
    image, envelope, frequency = image_fixture()
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:30, 30:40] = True
    with pytest.raises(ValueError, match="three specimen cycles"):
        fit(image, frequency, envelope, valid_mask_yx=mask)


def test_high_count_noise_covariance_and_frequency_bias_are_consistent():
    image, envelope, frequency = image_fixture(shape=(48, 64), bins=(6.31, 3.42))
    rng = np.random.default_rng(314)
    whitened = []
    chi_squared = []
    # Independent Poisson + signed Gaussian read noise. Variance is fixed at
    # the synthetic expected image, not estimated from each noisy observation.
    variance = image + 9
    for _ in range(12):
        observed = rng.poisson(image) + rng.normal(0, 3, image.shape)
        result = fit(observed, frequency, envelope, variance=variance)
        assert result.accepted, result.reasons
        whitened.append(np.linalg.solve(np.linalg.cholesky(result.covariance_xy_per_pixel2),
                                         result.frequency_xy_per_pixel - frequency))
        chi_squared.append(result.reduced_chi_squared)
    whitened = np.asarray(whitened)
    assert .5 < np.mean(whitened**2) < 1.8
    assert np.max(np.abs(whitened.mean(axis=0))) < .65
    assert .9 < np.mean(chi_squared) < 1.1


def test_template_derivatives_jit_and_vmap_match_analytic_frequency_derivative():
    shape = (23, 31)
    frequency = np.array([.123, -.091])
    coefficient = np.array([200., 2., 3., 0., 0., 0., 25., -17.])
    function = lambda f: periodic_template(f, coefficient, shape, harmonics=1)
    derivative = np.asarray(jax.jacfwd(function)(frequency))
    y, x = np.indices(shape)
    x, y = x - (shape[1] - 1) / 2, y - (shape[0] - 1) / 2
    phase = 2 * np.pi * (frequency[0] * x + frequency[1] * y)
    factor = 2 * np.pi * (-25 * np.sin(phase) - 17 * np.cos(phase))
    np.testing.assert_allclose(derivative[..., 0], factor * x, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(derivative[..., 1], factor * y, rtol=1e-12, atol=1e-10)
    batch = jax.jit(jax.vmap(function))(jnp.stack((frequency, frequency * 1.01)))
    assert batch.shape == (2, *shape)


@pytest.mark.parametrize("kwargs", [
    {"variance_yx": 0.}, {"harmonics": 0}, {"search_radius_bins": np.nan},
    {"expected_frequency_xy_per_pixel": [0., 0.]},
    {"expected_frequency_xy_per_pixel": [.49, 0.]},
    {"illumination_yx": np.zeros((48, 64))},
])
def test_invalid_inputs_rejected(kwargs):
    parameters = dict(image_yx=np.ones((48, 64)), expected_frequency_xy_per_pixel=[.13, .07],
                      variance_yx=1.)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        fit_periodic_frequency(**parameters)
