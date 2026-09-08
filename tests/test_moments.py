"""Wave-moment physics checked independently of the numerical microscope fit."""

import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from temgym_core.components import solenoid_column
from temgym_core.constants import energy2wavelength
from temgym_core.calibration.condenser_images import illuminate_gaussian_spots
from temgym_core.moments import SampledTransmission
from temgym_core.evaluate import gaussian_intensity_image, normal_interval_probability
from temgym_core.run import column_matrix, propagate_gaussian_mixture
from temgym_core.source import gaussian_waist


def test_transparent_sample_preserves_mixed_gaussian_and_phase_lens():
    wavelength = 2.5e-12
    axis = jnp.linspace(-150e-9, 150e-9, 241)
    xy = jnp.stack(jnp.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    size, h, spread = 18e-9, 1.2e4, 8e-5
    spatial = size**2 * jnp.eye(2)
    mean = jnp.array([3e-9, -4e-9, 2e-5, -3e-5])
    covariance = jnp.block(
        [
            [spatial, h * spatial],
            [h * spatial, (h * h * size**2 + spread**2) * jnp.eye(2)],
        ]
    )
    for phase_power in (0.0, -2e4):
        sample = SampledTransmission(
            xy, jnp.ones(len(xy)), jnp.zeros_like(xy), phase_power * xy
        )
        actual_mean, actual_cov = sample.gaussian_moments(mean, covariance, wavelength)
        lens = jnp.eye(4).at[2:, :2].set(phase_power * jnp.eye(2))
        np.testing.assert_allclose(actual_mean, lens @ mean, rtol=1e-9, atol=1e-18)
        expected = np.asarray(lens @ covariance @ lens.T)
        units = np.sqrt(np.diag(expected))
        scale = units[:, None] * units[None, :]
        np.testing.assert_allclose(
            actual_cov / scale, expected / scale, rtol=1e-9, atol=1e-12
        )


def test_native_gaussian_mixture_matches_matrix_transport():
    from dataclasses import replace

    voltage = 200000.0
    size, spread = 8e-9, 8e-5
    source = gaussian_waist(z=0.0, voltage=voltage, x=2e-9, theta_x=1e-5, rms_size=size)
    column = tuple(
        replace(c, include_axial_path=False) if hasattr(c, "q") else c
        for c in solenoid_column(
            jnp.array([0.2, 0.035, 0.05]),
            jnp.array([0.005, 0.005]),
            jnp.array([0.2, 0.37]),
        )
    )
    floor = (energy2wavelength(voltage) / (4 * jnp.pi * size)) ** 2
    extra = jnp.diag(jnp.array([0.0, 0.0, spread**2 - floor, spread**2 - floor]))
    mean, covariance = propagate_gaussian_mixture(source, column, extra)
    matrix = column_matrix(column)[:4, :4]
    expected = (
        matrix
        @ jnp.diag(jnp.array([size**2, size**2, spread**2, spread**2]))
        @ matrix.T
    )
    np.testing.assert_allclose(
        mean, matrix @ jnp.array([2e-9, 0.0, 1e-5, 0.0]), rtol=1e-9, atol=1e-18
    )
    np.testing.assert_allclose(covariance, expected, rtol=1e-8, atol=1e-22)


def test_measured_moment_covariance_matches_repeated_detector_noise():
    from temgym_core.moments import measure_detector_moments

    y, x = np.indices((24, 24), dtype=float) - 11.5
    signal = np.exp(-((x - 0.7) ** 2 / 8 + (y + 0.4) ** 2 / 12))
    expected = 4e6 * signal / signal.sum() + 3
    rng = np.random.default_rng(7103)
    images = rng.poisson(expected, size=(800, *expected.shape)) + rng.normal(
        0, 2, (800, *expected.shape)
    )
    result = measure_detector_moments(
        images=images, background_counts=3, read_noise_counts=2
    )
    target = measure_detector_moments(
        images=expected[None], background_counts=3, read_noise_counts=2
    )["values"][0]
    white = np.linalg.inv(np.linalg.cholesky(result["covariance"].mean(axis=0)))
    residual = (result["values"] - target) @ white.T
    # Sampling error of each normalized variance is sqrt(2/799), about 5%.
    np.testing.assert_allclose(residual.mean(axis=0), 0, atol=0.12)
    np.testing.assert_allclose(np.cov(residual.T), np.eye(5), atol=0.16)


def test_pixel_bin_correction_matches_exact_integrated_gaussian_pixels():
    from scipy.special import ndtr

    from temgym_core.moments import pixel_bin_correction

    mean = jnp.array([0.17, -0.22])
    variance = jnp.array([0.52**2, 0.67**2])
    continuous = jnp.array([*mean, variance[0] + 1 / 12, 0.0, variance[1] + 1 / 12])

    def characteristic(frequency):
        return jnp.exp(1j * frequency @ mean - 0.5 * jnp.sum(frequency**2 * variance))

    corrected = np.asarray(pixel_bin_correction(continuous, characteristic))
    edges = np.arange(-16, 17)
    centres = (edges[1:] + edges[:-1]) / 2
    counts = np.diff(
        ndtr((edges[:, None] - np.asarray(mean)) / np.sqrt(variance)), axis=0
    )
    expected_mean = centres @ counts
    expected_var = centres**2 @ counts - expected_mean**2
    np.testing.assert_allclose(
        corrected, [*expected_mean, expected_var[0], 0.0, expected_var[1]], atol=3e-9
    )
    assert abs(continuous[0] - expected_mean[0]) > 0.001


def test_characteristic_integral_agrees_with_gaussian_matrix_transport():
    from temgym_core.moments import gaussian_image_characteristic

    wavelength = 2.5e-12
    axis = jnp.linspace(-180e-9, 180e-9, 361)
    xy = jnp.stack(jnp.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    sample = SampledTransmission(
        xy, jnp.ones(len(xy)), jnp.zeros_like(xy), jnp.zeros_like(xy)
    )
    mean = jnp.array([3e-9, -4e-9, 2e-5, -3e-5])
    size, h, spread = 18e-9, -1.2e4, 8e-5
    spatial = size**2 * jnp.eye(2)
    covariance = jnp.block(
        [
            [spatial, h * spatial],
            [h * spatial, (h * h * size**2 + spread**2) * jnp.eye(2)],
        ]
    )
    matrix = (
        jnp.eye(5).at[:2, :2].set(4000 * jnp.eye(2)).at[:2, 2:4].set(0.2 * jnp.eye(2))
    )
    mapping = matrix[:2, :4] / 220e-6
    expected_mean = mapping @ mean
    expected_covariance = mapping @ covariance @ mapping.T + 0.5**2 * jnp.eye(2)
    for frequency in (
        jnp.array([0.1, 0.3]),
        jnp.array([2 * jnp.pi, 0.0]),
        jnp.array([2 * jnp.pi, -2 * jnp.pi]),
    ):
        actual = gaussian_image_characteristic(
            sample,
            jnp.zeros(len(xy)),
            mean,
            covariance,
            wavelength,
            matrix,
            frequency,
            pixel_pitch_m=220e-6,
            psf_sigma_pixels=0.5,
        )
        expected = jnp.exp(
            1j * frequency @ expected_mean
            - 0.5 * frequency @ expected_covariance @ frequency
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_detector_matches_independent_pixel_integrals_and_loses_offscreen_flux():
    centres = jnp.array([[1.0, -0.5], [20.0, 0.0]])
    weights = jnp.array([0.4, 0.6])
    actual = gaussian_intensity_image(
        centres, 0.7**2, weights, shape_yx=(8, 10), psf_sigma_pixels=0.3
    )
    x, y = np.arange(11) - 5, np.arange(9) - 4
    width = np.sqrt(0.7**2 + 0.3**2)
    px = np.diff(ndtr((x[:, None] - np.asarray(centres)[:, 0]) / width), axis=0)
    py = np.diff(ndtr((y[:, None] - np.asarray(centres)[:, 1]) / width), axis=0)
    np.testing.assert_allclose(actual, (py * weights) @ px.T, atol=1e-16)
    assert 0.399 < float(actual.sum()) < 0.4


def test_specimen_gaussian_product_matches_direct_product():
    spot_centres = jnp.array([[1.0, -2.0]])
    mean = jnp.array([-0.3, 0.1])
    centres, variance, weights = illuminate_gaussian_spots(
        spot_centres, jnp.array([0.7]), 0.5, mean, 1.2
    )
    points = jnp.array([[0.0, 0.0], [1.0, -1.0], [2.0, -3.0]])
    direct = -0.5 * jnp.sum((points - mean) ** 2, axis=1) / 1.2
    direct += -0.5 * jnp.sum((points - spot_centres[0]) ** 2, axis=1) / 0.5
    product = -0.5 * jnp.sum((points - centres[0]) ** 2, axis=1) / variance
    np.testing.assert_allclose(direct - direct[0], product - product[0], atol=2e-14)
    np.testing.assert_allclose(weights, [1.0])


def test_nonnegative_exposure_background_matches_active_set_reference():
    from scipy.optimize import nnls

    from temgym_core.optimization import fit_intensity_scale_and_background

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
    probability = float(
        normal_interval_probability(jnp.asarray(9.0), jnp.asarray(10.0))
    )
    assert probability > 0.0
    np.testing.assert_allclose(probability, ndtr(-9.0) - ndtr(-10.0), rtol=1e-13)


def test_mixture_uses_source_voltage_for_current_controlled_lenses():
    from temgym_core.components import ElectromagneticLens, Plane
    from temgym_core.constants import effective_accelerating_potential
    from temgym_core.gaussian import FreeSpacePropagator
    from temgym_core.run import run_to_end

    voltage = 100_000.
    source = gaussian_waist(z=0., voltage=voltage, rms_size=1e-6)
    lens = ElectromagneticLens(z=.1, turns=100., current=1.,
                              Gc=float(effective_accelerating_potential(voltage)) / (.08 * 100**2))
    column = (lens, Plane(z=.3))
    extra = jnp.diag(jnp.array((0., 0., 1e-6, 2e-6)))
    _, mixed_covariance = propagate_gaussian_mixture(source, column, extra)
    coherent = run_to_end(source, column, propagator=FreeSpacePropagator(include_axial_path=False))
    angle = float(lens.rotation_angle(voltage))
    rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    identity, zero = np.eye(2), np.zeros((2, 2))
    before = np.block([[identity, .1 * identity], [zero, identity]])
    after = np.block([[identity, .2 * identity], [zero, identity]])
    kick = np.block([[rotation, zero], [-rotation / .08, rotation]])
    matrix = after @ kick @ before
    expected = coherent.moments()[1] + matrix @ extra @ matrix.T
    np.testing.assert_allclose(mixed_covariance, expected, rtol=1e-10, atol=1e-18)
