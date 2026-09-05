"""Small quadrature/derivative checks, not full detector image simulations."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from neoarm_aperture_image_fit import (
    AperturePixelOperator, encode_spatial, decode_spatial,
    _thermal_polynomial_sum, local_covariance, require_gpu,
    data_only_initializer, accepted_image_fit,
)
from temgym_core.neoarm_calibration import DetectorGeometry

jax.config.update("jax_enable_x64", True)


def test_spatial_covariance_roundtrip():
    mean = np.array([.7, -1.2]); covariance = np.array([[9., -2.1], [-2.1, 4.]])
    actual_mean, actual_covariance = decode_spatial(encode_spatial(mean, covariance))
    np.testing.assert_allclose(actual_mean, mean)
    np.testing.assert_allclose(actual_covariance, covariance, rtol=1e-14)


def test_circle_quadrature_area_and_translation():
    operator = AperturePixelOperator(DetectorGeometry(shape_yx=(13, 17), mtf_sigma_px=0), 16)
    for centre in ([0., 0.], [.3, -.7]):
        x, weight, lower, upper = operator.geometry(3.2, jnp.array(centre))
        area = jnp.sum(weight*(upper-lower))
        np.testing.assert_allclose(area, np.pi*3.2**2, rtol=2e-13)


def test_circle_area_scale_gradient():
    operator = AperturePixelOperator(DetectorGeometry(shape_yx=(13, 17), mtf_sigma_px=0), 12)
    def area(radius):
        _, weight, lower, upper = operator.geometry(radius, jnp.array([.31, -.27]))
        return jnp.sum(weight*(upper-lower))
    actual = jax.jit(jax.grad(area))(3.2)
    np.testing.assert_allclose(actual, 2*np.pi*3.2, rtol=1e-9)


def test_circle_finite_crop_not_normalized():
    operator = AperturePixelOperator(DetectorGeometry(shape_yx=(9, 7), mtf_sigma_px=0), 12)
    _, weight, lower, upper = operator.geometry(50., jnp.zeros(2))
    np.testing.assert_allclose(jnp.sum(weight*(upper-lower)), 63., rtol=1e-12)


def test_thermal_sum_is_not_renormalized():
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    ratio = .3
    integrated = np.sum(weights*np.asarray(_thermal_polynomial_sum(jnp.asarray(nodes), ratio, 4)))/np.sqrt(np.pi)
    np.testing.assert_allclose(integrated, 1-ratio**4, rtol=1e-13)


def test_local_covariance_reports_missing_scale_rank():
    jac = np.column_stack([np.eye(4), np.zeros(4)])
    covariance, _, rank = local_covariance(jac)
    assert rank == 4
    assert np.isnan(covariance).all()


def test_profiled_covariance_retains_nuisance_correlation():
    jac = np.array([[2., 1.], [1., 3.], [0., 2.]])
    covariance, _, rank = local_covariance(jac)
    information = jac.T@jac
    profiled = 1/(information[0, 0]-information[0, 1]**2/information[1, 1])
    assert rank == 2
    np.testing.assert_allclose(covariance[0, 0], profiled)
    assert covariance[0, 0] > 1/information[0, 0]


def test_common_radius_scale_gauge_is_exact():
    operator = AperturePixelOperator(DetectorGeometry(shape_yx=(13, 17), mtf_sigma_px=0), 12)
    def area(log_scales):
        radius = 3.2*jnp.exp(log_scales[0]+log_scales[1])
        _, weight, lower, upper = operator.geometry(radius, jnp.array([.31, -.27]))
        return jnp.sum(weight*(upper-lower))
    derivative = jax.grad(area)(jnp.array([.02, -.02]))
    np.testing.assert_allclose(derivative@jnp.array([1., -1.]), 0., atol=1e-13)


def test_physical_simulation_requires_gpu():
    if jax.default_backend() != "gpu":
        with pytest.raises(RuntimeError, match="require GPU"):
            require_gpu()


def test_initializer_uses_data_counts_and_common_seed_only():
    detector = DetectorGeometry(shape_yx=(13, 17), mtf_sigma_px=0)
    # A numerical moment fixture, not a physical forward simulation.
    observed = np.ones((4, 13, 17))*np.array([3., 2., 1., 4.])[:, None, None]
    coverage = np.array([.8, .2, .4, .6])
    def fixture_prediction(parameters):
        assert np.all(parameters[12:16] == 0)
        return np.ones_like(observed)*coverage[:, None, None]/(13*17)
    initial = data_only_initializer(observed, detector, predictor=fixture_prediction)
    np.testing.assert_allclose(initial[:2], 0., atol=1e-14)
    np.testing.assert_allclose(np.exp(initial[5]), 100.)
    np.testing.assert_array_equal(initial[6:12], np.zeros(6))
    np.testing.assert_allclose(np.exp(initial[12:16]), observed.sum(axis=(-1, -2))/coverage)
    doubled = data_only_initializer(2*observed, detector, predictor=fixture_prediction)
    np.testing.assert_allclose(doubled[:12], initial[:12])
    np.testing.assert_allclose(doubled[12:16]-initial[12:16], np.log(2))


def test_initializer_rejects_invalid_signal():
    detector = DetectorGeometry(shape_yx=(13, 17), mtf_sigma_px=0)
    with pytest.raises(ValueError, match="positive total"):
        data_only_initializer(np.zeros((4, 13, 17)), detector)


@pytest.mark.parametrize("chi_square_z,accepted", [(0., True), (4.9, True), (-4.9, True), (5.1, False), (-5.1, False)])
def test_noisy_acceptance_is_two_sided(chi_square_z, accepted):
    report = dict(optimizer_success=True, jacobian_rank=16, parameter_count=16,
        chi_square_z=chi_square_z, reference_induced_maximum_parameter_shift_sigma=0.,
        fit_quadrature_refinement_maximum_parameter_shift_sigma=0.,
        two_successive_complete_refinement_fits=[dict(optimizer_success=True, maximum_parameter_change_sigma=0.)]*2)
    assert accepted_image_fit(report) is accepted
