from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))

from neoarm_last_lens_geometry_profile import VolumeChart, magnetic_twoform_diagnostic
from neoarm_last_lens_tomography import geometry_coordinates, predict_response_parameters


def seed_parameters():
    # General positive-volume coefficients; no optical generating truth is
    # needed for the algebraic chart and derivative tests.
    matrix = np.array([[2., .1, .2, .05], [.1, 1.7, -.1, .1],
                       [.2, .02, .5, .07], [-.03, .1, .05, .6]])
    return np.r_[np.log(4.), -1.4, -.8, matrix.ravel()]


def test_volume_chart_round_trip_and_changed_span():
    p = seed_parameters()
    chart = VolumeChart.from_response_parameters(p, np.ones(4), 1.)
    theta = jnp.asarray(chart.encode(p))
    np.testing.assert_allclose(chart.decode(theta), p, rtol=2e-15, atol=2e-15)
    changed = theta.at[0].set(np.log(.9)).at[1].set(np.log(2.))
    decoded = chart.decode(changed)
    geometry = np.asarray(geometry_coordinates(decoded, jnp.ones(4), 1.))
    np.testing.assert_allclose(np.exp(geometry[:2]), [.3, .6], rtol=2e-15)
    assert np.linalg.det(np.asarray(decoded[3:]).reshape(4, 4)) == pytest.approx(.6**2, rel=3e-15)


def test_jit_vmap_and_first_second_derivatives():
    p = seed_parameters()
    chart = VolumeChart.from_response_parameters(p, np.ones(4), 1.)
    theta = jnp.asarray(chart.encode(p))
    jac = jax.jit(jax.jacfwd(chart.decode))(theta)
    assert np.linalg.matrix_rank(np.asarray(jac)) == 19
    hessian = jax.jit(jax.jacfwd(jax.jacfwd(chart.decode)))(theta)
    assert np.all(np.isfinite(hessian))
    step = 1e-5
    indices = [0, 1, 5, 12]
    for index in indices:
        d = jnp.eye(19)[index]*step
        fd = (chart.decode(theta+d)-chart.decode(theta-d))/(2*step)
        np.testing.assert_allclose(jac[:, index], fd, rtol=2e-8, atol=1e-9)
    batched = jax.jit(jax.vmap(chart.decode))(jnp.stack((theta, theta)))
    np.testing.assert_allclose(batched[0], p, rtol=2e-15, atol=2e-15)


def test_polynomial_derivatives_at_singular_pivot_zero_matrix():
    p = np.r_[np.log(3.), -1., -.5, np.eye(4).ravel()]
    chart = VolumeChart(0, 1.)
    theta = jnp.asarray(chart.encode(p))
    assert np.all(np.isfinite(jax.jacfwd(jax.jacfwd(chart.decode))(theta)))
    np.testing.assert_allclose(chart.decode(theta), p, atol=1e-15)


def test_degenerate_cofactor_is_not_clipped_or_pseudoinverted():
    chart = VolumeChart(0, 1.)
    theta = jnp.zeros(19)
    decoded = jax.jit(chart.decode)(theta)
    assert np.isnan(decoded[3])


def test_exact_magnetic_upstream_diagnostic():
    # Identity canonical upstream followed by a mechanical shear. This test
    # verifies units/signs without asserting this model for the noisy fit.
    kappa, length, gap = 800., .02, .3
    upstream = np.eye(4);upstream[2:, :2] = kappa*np.array([[0., -1.], [1., 0.]])
    steps = np.array([5e-9, 5e-9, 1e-3, 1e-3]);pixel = 55e-6
    ratio = gap/length
    coefficients = np.concatenate((-ratio*upstream[:2], length*upstream[2:]))*(steps/pixel)
    p = np.r_[np.log(ratio), -1.4, -.8, coefficients.ravel()]
    result = magnetic_twoform_diagnostic(p, steps, pixel)
    assert result['inferred_kappa_m_inv'] == pytest.approx(kappa, rel=3e-15)
    assert result['maximum_balanced_twoform_defect'] < 1e-14
    assert result['length_m'] == pytest.approx(length, rel=3e-15)
