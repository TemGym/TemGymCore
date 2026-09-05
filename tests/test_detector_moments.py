import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from temgym_core.detector_moments import finite_gaussian_centroid, finite_gaussian_raw_moments
from temgym_core.neoarm_calibration import DetectorGeometry

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("sigma", [0., .65, 1.2])
def test_matches_full_detector_with_anisotropy_rotation_and_offset(sigma):
    detector = DetectorGeometry(shape_yx=(23, 31), pixel_size_yx_m=(2e-6, 3e-6),
                                centre_xy_m=(4e-6, -3e-6), z_m=1., mtf_sigma_px=sigma,
                                affine_xy=np.array([[1.02, .12], [-.10, .99]]))
    mean = np.array([24e-6, -30e-6])
    covariance = np.array([[2., .7], [.7, 1.]])*1e-9
    delta = detector.optical_coords_xy_m-mean
    intensity = np.exp(-.5*np.einsum('...i,ij,...j->...', delta, np.linalg.inv(covariance), delta))
    if sigma:
        intensity = gaussian_filter(intensity, sigma, mode="nearest")
    yy, xx = np.indices(detector.shape_yx)
    expected = np.array([(intensity*xx).sum(), (intensity*yy).sum()])/intensity.sum()
    np.testing.assert_allclose(finite_gaussian_centroid(mean, covariance, detector), expected, atol=1e-12)
    expected_raw = np.array([(intensity*w).sum()/intensity.sum() for w in (xx, yy, xx**2, xx*yy, yy**2)])
    np.testing.assert_allclose(finite_gaussian_raw_moments(mean, covariance, detector), expected_raw, atol=1e-10)


def test_custom_derivatives_match_finite_difference_and_reverse_mode():
    detector = DetectorGeometry(shape_yx=(27, 35), pixel_size_yx_m=(1e-6, 1.5e-6), z_m=1.)
    def f(p):
        mean = p[:2]*1e-5
        covariance = jnp.array([[jnp.exp(p[2]), p[3]], [p[3], jnp.exp(p[4])]])*1e-9
        return finite_gaussian_centroid(mean, covariance, detector)
    p = np.array([1.2, -.8, .2, .3, -.1])
    jacobian = np.asarray(jax.jacfwd(f)(p))
    step = 1e-4
    numerical = np.column_stack([(np.asarray(f(p+step*d))-np.asarray(f(p-step*d)))/(2*step)
                                 for d in np.eye(5)])
    np.testing.assert_allclose(jacobian, numerical, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(jacobian, jax.jacrev(f)(p), rtol=1e-12, atol=1e-12)
    result = jax.jit(jax.vmap(f))(jnp.stack((p, p+.001)))
    assert result.shape == (2, 2)


def test_clipped_centroid_is_not_unbounded_beam_position():
    detector = DetectorGeometry(shape_yx=(32, 32), pixel_size_yx_m=(1., 1.), z_m=1.)
    mean = np.array([12., 0.])
    measured = np.asarray(finite_gaussian_centroid(mean, np.eye(2)*200., detector))
    assert measured[0] < 15.5+mean[0]-5.
    np.testing.assert_allclose(measured[1], 15.5, atol=1e-12)
