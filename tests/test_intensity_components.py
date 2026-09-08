import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from temgym_core.intensity import (
    GaussianIntensityDetector,
    GaussianIntensitySample,
    normal_interval_probability,
)


def test_detector_matches_independent_pixel_integrals_and_loses_offscreen_flux():
    detector = GaussianIntensityDetector((8, 10), psf_sigma_pixels=0.3)
    centres = jnp.array([[1.0, -0.5], [20.0, 0.0]])
    weights = jnp.array([0.4, 0.6])
    actual = detector.image_pixels(centres, 0.7**2, weights)
    x, y = np.arange(11) - 5, np.arange(9) - 4
    width = np.sqrt(0.7**2 + 0.3**2)
    px = np.diff(ndtr((x[:, None] - np.asarray(centres)[:, 0]) / width), axis=0)
    py = np.diff(ndtr((y[:, None] - np.asarray(centres)[:, 1]) / width), axis=0)
    np.testing.assert_allclose(actual, (py * weights) @ px.T, atol=1e-16)
    assert 0.399 < float(actual.sum()) < 0.4


def test_specimen_gaussian_product_matches_direct_product():
    sample = GaussianIntensitySample(jnp.array([[1.0, -2.0]]), jnp.array([0.7]), 0.5)
    mean = jnp.array([-0.3, 0.1])
    centres, variance, weights = sample.illuminate(mean, 1.2)
    points = jnp.array([[0.0, 0.0], [1.0, -1.0], [2.0, -3.0]])
    direct = -0.5 * jnp.sum((points - mean) ** 2, axis=1) / 1.2
    direct += -0.5 * jnp.sum((points - sample.centres[0]) ** 2, axis=1) / 0.5
    product = -0.5 * jnp.sum((points - centres[0]) ** 2, axis=1) / variance
    np.testing.assert_allclose(direct - direct[0], product - product[0], atol=2e-14)
    np.testing.assert_allclose(weights, [1.0])


def test_nonnegative_exposure_background_matches_active_set_reference():
    from scipy.optimize import nnls

    from temgym_core.intensity import fit_intensity_scale_and_background

    shape = np.array([[0.0, 0.1], [0.5, 1.0]])
    noise = np.array([[1.0, 2.0], [1.0, 3.0]])
    for coefficients in ([3.0, 2.0], [-2.0, 4.0], [3.0, -1.0]):
        observed = coefficients[0] * shape + coefficients[1]
        design = np.stack((shape.ravel(), np.ones(4)), axis=1) / noise.ravel()[:, None]
        expected = nnls(design, observed.ravel() / noise.ravel())[0]
        np.testing.assert_allclose(
            fit_intensity_scale_and_background(
                jnp.asarray(shape), jnp.asarray(observed), jnp.asarray(noise)
            ),
            expected,
            atol=1e-12,
        )


def test_normal_interval_preserves_small_positive_tail_probability():
    probability = float(normal_interval_probability(jnp.asarray(9.0), jnp.asarray(10.0)))
    assert probability > 0.0
    np.testing.assert_allclose(probability, ndtr(-9.0) - ndtr(-10.0), rtol=1e-13)
