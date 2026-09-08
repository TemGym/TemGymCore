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
    expected_mean, expected_covariance = specimen_moments(parameters, controls)
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
