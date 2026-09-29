import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.components import Detector
from temgym_core.evaluate import gaussian_image
from temgym_core.gaussian import FreeSpacePropagator, make_gaussian
from temgym_core.optimization import least_squares


def test_native_detector_pose_and_rectangular_pixels_match_analytic_integrals():
    from scipy.special import ndtr

    width, blur = 4e-9, 1.3e-9
    beam = make_gaussian(x=3e-9, y=-2e-9, waist_x=width, waist_y=width, voltage=200e3)
    detector = Detector(
        z=0.0,
        shape=(16, 20),
        pixel_size=(1e-9, 1.2e-9),
        centre=(5e-9, 1e-9),
        rotation=27.0,
        flip_y=True,
    )
    actual = gaussian_image(
        beam, detector, samples_per_pixel=8, blur_sigma_m=blur, padding_pixels=8
    )
    mean_y, mean_x = detector.metres_to_pixels((beam.x, beam.y), cast=False)
    sigma = np.sqrt(width**2 / 4 + blur**2)
    py = np.diff(ndtr((np.arange(17) - 0.5 - mean_y) * 1e-9 / sigma))
    px = np.diff(ndtr((np.arange(21) - 0.5 - mean_x) * 1.2e-9 / sigma))
    expected = np.pi * width**2 / 2 * np.outer(py, px)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-30)


def test_opposite_gaussian_fields_cancel_before_pixel_integration():
    source = make_gaussian(
        x=jnp.zeros(2),
        y=jnp.zeros(2),
        amp=jnp.array([1.0, -1.0]),
        waist_x=5e-9,
        waist_y=5e-9,
        voltage=200e3,
    )
    detector = Detector(z=0.0, shape=(12, 12), pixel_size=(2e-9, 2e-9))
    np.testing.assert_allclose(gaussian_image(source, detector), 0.0, atol=1e-40)


def test_wave_image_gradient_and_jax_fit_unknown_propagation_distance():
    source = make_gaussian(
        x=jnp.array([-3e-9, 3e-9]),
        y=jnp.zeros(2),
        amp=jnp.array([1.0, 1j]),
        waist_x=2e-9,
        waist_y=3e-9,
        voltage=200e3,
    )

    detector = Detector(z=0.0, shape=(16, 16), pixel_size=(2e-9, 2e-9))

    def image(p):
        beam = FreeSpacePropagator()(source, p[0] * 1e-6)
        return gaussian_image(beam, detector) / 1e-18

    truth = jnp.array([18.0])
    target = image(truth)
    derivative = jax.jacfwd(image)(truth)[..., 0]
    finite = (image(truth + 1e-3) - image(truth - 1e-3)) / 0.002
    np.testing.assert_allclose(derivative, finite, rtol=2e-4, atol=2e-7)
    result = least_squares(
        lambda p: (image(p) - target).ravel(),
        [13.0],
        bounds=([5.0], [30.0]),
        max_nfev=100,
        gtol=1e-8,
        ftol=1e-10,
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, truth, atol=1e-5)


def test_pixel_quadrature_conserves_gaussian_flux():
    source = make_gaussian(waist_x=5e-9, waist_y=5e-9, voltage=200e3)
    image = gaussian_image(
        source,
        Detector(z=0.0, shape=(40, 40), pixel_size=(2e-9, 2e-9)),
        samples_per_pixel=4,
    )
    q = np.asarray(source.Q_inv)
    expected = np.pi / (float(source.k) * np.sqrt(np.linalg.det(q.imag)))
    np.testing.assert_allclose(image.sum(), expected, rtol=1e-10)


def test_real_space_blur_matches_analytic_gaussian_pixel_integral():
    import numpy as np
    from scipy.special import ndtr

    from temgym_core.gaussian import make_gaussian

    width = 4e-9
    blur = 1.3e-9
    pitch = 1e-9
    beam = make_gaussian(waist_x=width, waist_y=width, voltage=200000.0)
    detector = Detector(z=0.0, shape=(16, 16), pixel_size=(pitch, pitch))
    actual = np.asarray(
        gaussian_image(
            beam, detector, samples_per_pixel=8, blur_sigma_m=blur, padding_pixels=8
        )
    )
    variance = width**2 / 4 + blur**2
    edges = (np.arange(17) - 8) * pitch / np.sqrt(variance)
    marginal = np.diff(ndtr(edges))
    expected = np.pi * width**2 / 2 * np.outer(marginal, marginal)
    np.testing.assert_allclose(actual, expected, atol=1e-30, rtol=2e-10)


def test_real_space_blur_has_finite_correct_derivative():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from temgym_core.gaussian import make_gaussian

    beam = make_gaussian(
        x=jnp.array([-2e-9, 2e-9]),
        amp=jnp.array([1.0, 0.7j]),
        waist_x=3e-9,
        waist_y=3e-9,
        voltage=200000.0,
    )
    detector = Detector(z=0.0, shape=(8, 8), pixel_size=(1e-9, 1e-9))

    def image(s):
        return (
            gaussian_image(beam, detector, samples_per_pixel=4, blur_sigma_m=s * 1e-9)
            / 1e-18
        )

    derivative = jax.jacfwd(image)(1.2)
    finite = (image(1.2001) - image(1.1999)) / 0.0002
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, finite, atol=1e-8, rtol=1e-5)
