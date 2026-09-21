"""Native Gaussian acquisition and matrix fitting must share the same optics."""

import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from temgym_core.calibration.condenser_model import initial_parameters, specimen_moments
from temgym_core.calibration.gaussian import (
    condenser_gaussian_state,
    make_gaussian_renderer,
)
from temgym_core.components import Plane
from temgym_core.gaussian import make_gaussian


def test_condenser_gaussian_and_matrix_states_agree():
    parameters = initial_parameters()
    controls = jnp.array([1.6, 0.8, 1.02])
    expected_mean, expected_covariance = specimen_moments(
        parameters.tolist(), controls.tolist()
    )
    mean, covariance = condenser_gaussian_state(
        parameters=parameters, controls=controls, voltage_v=200000.0
    )
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-10, atol=1e-18)
    scale = np.sqrt(np.diag(expected_covariance))
    units = scale[:, None] * scale[None, :]
    np.testing.assert_allclose(
        covariance / units, expected_covariance / units, rtol=1e-9, atol=1e-10
    )


def test_gaussian_camera_matches_analytic_pixel_integrals():
    pitch, waist = 2e-9, 12e-9
    source = make_gaussian(waist_x=waist, waist_y=waist, voltage=200000.0)
    render = make_gaussian_renderer(
        shape_yx=(32, 32),
        pixel_pitch_m=pitch,
        samples_per_pixel=8,
        psf_sigma_pixels=0.5,
    )
    actual = np.asarray(render(source, (Plane(z=0.0),), 0.0, jnp.zeros(2)))
    sigma = np.sqrt((waist / 2) ** 2 + (0.5 * pitch) ** 2)
    probability = np.diff(ndtr(np.arange(-16, 17) * pitch / sigma))
    expected = probability[:, None] * probability[None, :]
    np.testing.assert_allclose(
        actual / actual.sum(), expected / expected.sum(), rtol=2e-3, atol=1e-6
    )


def test_camera_blur_uses_the_actual_source_plane():
    """Independent pixel integrals catch a hidden source_z=0 assumption."""
    from temgym_core.constants import energy2wavelength

    pitch, waist, distance, spread = 2e-9, 12e-9, 10e-6, 4e-4
    voltage = 80000.0
    source = make_gaussian(
        z=-distance, x=jnp.zeros(2), waist_x=waist, waist_y=waist, voltage=voltage
    )
    render = make_gaussian_renderer(
        shape_yx=(32, 32),
        pixel_pitch_m=pitch,
        samples_per_pixel=8,
        psf_sigma_pixels=0.5,
    )
    actual = np.asarray(render(source, (Plane(z=0.0),), spread**2, jnp.zeros(2)))
    wavelength = float(energy2wavelength(voltage))
    sigma = np.sqrt(
        (waist / 2) ** 2
        + (distance * wavelength / (2 * np.pi * waist)) ** 2
        + (spread * distance) ** 2
        + (0.5 * pitch) ** 2
    )
    probability = np.diff(ndtr(np.arange(-16, 17) * pitch / sigma))
    expected = probability[:, None] * probability[None, :]
    np.testing.assert_allclose(
        actual / actual.sum(), expected / expected.sum(), rtol=2e-3, atol=1e-6
    )


def test_off_axis_gaussian_image_moments_match_native_ray_coordinates():
    """An asymmetric source detects row/y reflection and signed tilt errors."""
    from temgym_core.moments import measure_detector_moments, predict_detector_moments
    from temgym_core.run import column_matrix
    from temgym_core.source import gaussian_waist

    source = gaussian_waist(
        z=-10e-6,
        voltage=200000.0,
        rms_size=4e-9,
        x=3e-9,
        y=-5e-9,
        theta_x=1e-4,
        theta_y=-2e-4,
    )
    column = (Plane(z=0.0),)
    pitch = 2e-9
    render = make_gaussian_renderer(
        shape_yx=(48, 48),
        pixel_pitch_m=pitch,
        samples_per_pixel=8,
    )
    frame = np.asarray(render(source, column, 0.0, jnp.zeros(2)))
    measured = measure_detector_moments(
        images=frame[None],
        background_counts=0.0,
        read_noise_counts=0.0,
    )["values"][0]
    mean, covariance = source.moments()
    predicted = predict_detector_moments(
        column_matrix(column, source_z=source.z),
        mean,
        covariance,
        pixel_pitch_m=pitch,
        psf_sigma_pixels=0.0,
    )
    np.testing.assert_allclose(measured[:2], [2.0, -3.5], atol=1e-5)
    # Midpoint field quadrature contributes a small finite-sampling error.
    np.testing.assert_allclose(measured, predicted, rtol=5e-4, atol=2e-3)
