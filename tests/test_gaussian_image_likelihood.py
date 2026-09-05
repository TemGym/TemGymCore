import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr

from temgym_core.gaussian_image_likelihood import (
    apply_detector_psf, detector_psf_kernel, fixed_variance_image_residual,
    make_gaussian_detector_operator, normal_interval_probability,
    transported_spatial_moments, validate_phase_space_covariance,
    decode_gaussian_image_parameters, encode_gaussian_image_parameters, fit_gaussian_image,
    gaussian_image_information_factor,
)
from temgym_core.neoarm_calibration import DetectorGeometry

jax.config.update("jax_enable_x64", True)


def detector(**kwargs):
    defaults = dict(z_m=1., shape_yx=(23, 31), pixel_size_yx_m=(2e-6, 3e-6),
                    mtf_sigma_px=0.)
    return DetectorGeometry(**(defaults | kwargs))


def test_pixel_integrals_match_separable_normal_cdf_with_clipping():
    d = detector()
    mean = np.array([35e-6, -20e-6])
    sigma = np.array([14e-6, 8e-6])
    x = (np.arange(32) - 31/2) * 3e-6
    y = (np.arange(24) - 23/2) * 2e-6
    expected = np.outer(np.diff(ndtr((y-mean[1])/sigma[1])),
                        np.diff(ndtr((x-mean[0])/sigma[0]))) * 123456
    result = make_gaussian_detector_operator(d)(mean, np.diag(sigma**2), 123456.)
    np.testing.assert_allclose(result, expected, atol=2e-11, rtol=1e-12)
    assert np.asarray(result).sum() < 90000  # lost electrons are not renormalized


def test_affine_centre_and_correlated_pixels_against_independent_2d_quadrature():
    d = detector(centre_xy_m=(4e-6, -7e-6),
                 affine_xy=np.array([[1.05, .2], [-.1, .95]]))
    mean = np.array([11e-6, -8e-6])
    covariance = np.array([[1.5, .4], [.4, .8]]) * 1e-10
    nodes, weights = np.polynomial.legendre.leggauss(12)
    offsets = np.stack(np.meshgrid(nodes*1.5e-6, nodes*1e-6), axis=-1).reshape(-1, 2)
    weights = np.outer(weights, weights).ravel() * 1.5e-12 / abs(np.linalg.det(d.affine_xy))
    physical_centres = np.einsum('ij,...j->...i', d.affine_xy,
                                d.optical_coords_xy_m - np.asarray(d.centre_xy_m))
    optical = np.einsum('ij,qyxj->qyxi', np.linalg.inv(d.affine_xy),
                        physical_centres[None] + offsets[:, None, None]) + d.centre_xy_m
    delta = optical - mean
    density = np.exp(-.5*np.einsum('...i,ij,...j->...', delta, np.linalg.inv(covariance), delta))
    density /= 2*np.pi*np.sqrt(np.linalg.det(covariance))
    expected = np.einsum('q,qyx->yx', weights, density)
    actual = make_gaussian_detector_operator(d)(mean, covariance)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-16)


@pytest.mark.parametrize('sigma', [0., .65, 1.2])
def test_psf_uses_illumination_halo_and_preserves_uniform_counts(sigma):
    d = detector(mtf_sigma_px=sigma)
    radius = (detector_psf_kernel(sigma).size-1)//2
    result = apply_detector_psf(jnp.ones((23+2*radius, 31+2*radius)), d)
    np.testing.assert_allclose(result, 1., atol=3e-15)
    if radius:
        halo = jnp.zeros((23+2*radius, 31+2*radius)).at[radius-1, radius+8].set(1.)
        assert float(apply_detector_psf(halo, d).sum()) > 0


def test_positive_tail_interval_does_not_cancel():
    result = normal_interval_probability(jnp.array(9.), jnp.array(10.))
    assert float(result) > 0
    np.testing.assert_allclose(result, ndtr(-9.)-ndtr(-10.), rtol=1e-13)


def test_jit_vmap_and_autodiff_pixel_counts():
    operator = make_gaussian_detector_operator(detector(mtf_sigma_px=.65))
    def predict(p):
        lower = jnp.array([[jnp.exp(p[2]), 0.], [p[3], jnp.exp(p[4])]]) * 1e-5
        return operator(p[:2]*1e-5, lower@lower.T, jnp.exp(p[5])*1e5)
    p = jnp.array([.3, -.7, .1, .3, -.1, .2])
    jacobian = jax.jacfwd(predict)(p)
    step = 1e-5
    numerical = np.stack([(predict(p+step*e)-predict(p-step*e))/(2*step)
                          for e in np.eye(6)], axis=-1)
    np.testing.assert_allclose(jacobian, numerical, atol=1e-7, rtol=3e-6)
    result = jax.jit(jax.vmap(predict))(jnp.stack([p, p+.01]))
    assert result.shape == (2, 23, 31)
    assert result.dtype == jnp.float64


def test_transport_has_no_independent_orientation_summary():
    matrix = jnp.eye(4).at[0, 1].set(.01).at[2, 3].set(.01)
    covariance = np.diag([2., 3., 4., 5.])
    mean = np.arange(4.)
    m, c = transported_spatial_moments(matrix, mean, covariance)
    np.testing.assert_allclose(m, np.asarray(matrix@mean)[[0, 2]])
    np.testing.assert_allclose(c, (matrix@covariance@matrix.T)[np.ix_([0, 2], [0, 2])])


def test_read_noise_values_remain_signed_and_flux_is_not_normalized():
    data = jnp.array([[-.3, 5.]])
    result = fixed_variance_image_residual(jnp.array([[0., 6.]]), data, jnp.array([[.36, 4.]]))
    np.testing.assert_allclose(result, [.5, .5])
    assert not np.isfinite(fixed_variance_image_residual(jnp.ones((1,)), jnp.zeros((1,)), jnp.zeros((1,)))).all()


def test_wave_uncertainty_rejects_old_illegal_covariance_and_accepts_pure_and_mixed():
    wavelength = 2.508e-12
    epsilon = wavelength/(4*np.pi)
    illegal = np.diag([60e-9**2, .5e-6**2]*2)
    with pytest.raises(ValueError, match='uncertainty'):
        validate_phase_space_covariance(illegal, wavelength)
    covariance = np.diag([10e-9**2, (epsilon/10e-9)**2]*2)
    validate_phase_space_covariance(covariance, wavelength)
    validate_phase_space_covariance(covariance * 10, wavelength)
    transfer = np.eye(4)
    transfer[0, 1] = transfer[2, 3] = .001
    validate_phase_space_covariance(transfer@covariance@transfer.T, wavelength)


@pytest.mark.parametrize('order', [0, -1, 1.5, True])
def test_invalid_quadrature_rejected(order):
    with pytest.raises(ValueError):
        make_gaussian_detector_operator(detector(), order)


def test_invalid_shapes_and_nonsymmetric_source_rejected():
    operator = make_gaussian_detector_operator(detector())
    with pytest.raises(ValueError):
        operator(jnp.zeros(4), jnp.eye(4))
    with pytest.raises(ValueError):
        apply_detector_psf(jnp.ones((2, 3)), detector())
    invalid = np.eye(4)*1e-12
    invalid[0, 1] = .1e-12
    with pytest.raises(ValueError, match='symmetric'):
        validate_phase_space_covariance(invalid, 2.5e-12)


def test_operator_itself_can_be_jitted():
    operator = make_gaussian_detector_operator(detector())
    result = jax.jit(operator)(jnp.zeros(2), jnp.eye(2)*1e-10, 100.)
    assert result.shape == (23, 31)


def test_three_isotropic_projections_cannot_observe_antisymmetric_cross_covariance():
    # The unobservable x-theta_y versus y-theta_x covariance is retained as
    # a physical nuisance. Three spatial ellipses determine nine, not ten,
    # second-moment combinations, regardless of redundant ellipse angles.
    lower = np.tril_indices(4)
    def projected(p):
        covariance = jnp.zeros((4, 4)).at[lower].set(p)
        covariance = covariance + covariance.T - jnp.diag(jnp.diag(covariance))
        def one(distance):
            matrix = jnp.eye(4).at[0, 1].set(distance).at[2, 3].set(distance)
            _, spatial = transported_spatial_moments(matrix, jnp.zeros(4), covariance)
            return spatial[jnp.tril_indices(2)]
        return jax.vmap(one)(jnp.array([-1., 0., 1.])).reshape(-1)
    jacobian = np.asarray(jax.jacfwd(projected)(jnp.ones(10)))
    assert np.linalg.matrix_rank(jacobian) == 9
    delta = np.zeros((4, 4))
    delta[3, 0], delta[2, 1] = 1., -1.
    np.testing.assert_allclose(jacobian@delta[lower], 0., atol=1e-14)


def test_image_parameter_roundtrip_with_affine():
    d = detector(affine_xy=np.array([[1.1, .2], [-.15, .8]]), centre_xy_m=(1e-6, -2e-6))
    p = jnp.array([2., -5., 1.4, 1.7, -.3, np.log(1e6)])
    mean, covariance, dose = decode_gaussian_image_parameters(p, d)
    np.testing.assert_allclose(encode_gaussian_image_parameters(mean, covariance, dose, d), p, atol=2e-15)


def test_raw_image_fit_recovers_clipped_gaussian_from_perturbed_initial_state():
    d = detector(mtf_sigma_px=.65)
    p = jnp.array([7., -5., np.log(9.), np.log(6.), .3, np.log(2e7)])
    operator = make_gaussian_detector_operator(d)
    expected = np.asarray(operator(*decode_gaussian_image_parameters(p, d)))
    variance = expected + .6**2
    fit = fit_gaussian_image(expected, variance, d, initial_parameters=np.asarray(p)+.05)
    fit.require_accepted()
    np.testing.assert_allclose(fit.parameters, p, atol=1e-7)
    assert fit.residual_rms_sigma < 1e-6
    assert fit.numerical_rank == 6
    assert np.linalg.eigvalsh(fit.parameter_covariance).min() > 0
    assert fit.quadrature_parameter_shift_sigma < 1e-7


def test_noisy_image_fit_retains_full_covariance_and_rejects_mismatch():
    d = detector(mtf_sigma_px=.65)
    p = jnp.array([1., -2., np.log(8.), np.log(5.), -.25, np.log(2e7)])
    expected = np.asarray(make_gaussian_detector_operator(d)(*decode_gaussian_image_parameters(p, d)))
    rng = np.random.default_rng(817)
    observed = rng.poisson(expected).astype(float) + rng.normal(0., .6, expected.shape)
    fit = fit_gaussian_image(observed, expected+.36, d).require_accepted()
    assert .85 < fit.residual_rms_sigma < 1.15
    assert np.max(np.abs(fit.parameters-p) / np.sqrt(np.diag(fit.parameter_covariance))) < 4.
    mismatch = observed.copy()
    mismatch[::2, ::2] += np.sqrt(expected[::2, ::2]) * 20
    rejected = fit_gaussian_image(mismatch, expected+.36, d, initial_parameters=fit.parameters)
    with pytest.raises(ValueError, match='admission'):
        rejected.require_accepted()


def test_information_compression_preserves_pixel_jacobian_metric_without_angle_duplicate():
    d = detector(mtf_sigma_px=.65)
    p = jnp.array([8., -6., np.log(12.), np.log(8.), -.25, np.log(2e7)])
    operator = make_gaussian_detector_operator(d)
    predict = lambda q: operator(*decode_gaussian_image_parameters(q, d))
    expected = predict(p)
    variance = expected + .36
    jacobian = jax.jacfwd(predict)(p).reshape(-1, 6) / jnp.sqrt(variance.reshape(-1, 1))
    factor = gaussian_image_information_factor(p, d)
    delta = jnp.array([.01, -.02, .001, .002, -.003, .001])
    np.testing.assert_allclose(jnp.linalg.norm(factor@delta), jnp.linalg.norm(jacobian@delta), rtol=2e-13)
    # Release incident flux before assessing the shape. Dose/shape correlation
    # from a truncated detector is retained, unlike independent summary errors.
    assert abs(np.asarray(factor.T@factor)[2, 5]) > 1e4


def test_coarse_pixel_quadrature_cannot_self_certify_by_fitting_its_own_data():
    d = detector(mtf_sigma_px=.65)
    p = jnp.array([.3, -.2, np.log(.6), np.log(1.2), .2, np.log(2e7)])
    low_order = make_gaussian_detector_operator(d, quadrature_order=1)
    expected = np.asarray(low_order(*decode_gaussian_image_parameters(p, d)))
    fit = fit_gaussian_image(expected, expected+.36, d,
                             initial_parameters=p, quadrature_order=1)
    assert fit.residual_rms_sigma < 1e-7  # same-formula closure alone is inadequate
    assert fit.quadrature_parameter_shift_sigma > .2
    with pytest.raises(ValueError, match='quadrature admission'):
        fit.require_accepted()
