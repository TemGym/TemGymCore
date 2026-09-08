import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.coherent_image import (
    CoherentDetector,
    ensemble_image,
    fit_images,
    microscope_image,
)
from temgym_core.gaussian import make_gaussian


def test_fields_interfere_within_mode_and_intensities_add_between_modes():
    source = make_gaussian(
        x=jnp.zeros(2),
        y=jnp.zeros(2),
        amp=jnp.array([1.0, -1.0]),
        waist_x=5e-9,
        waist_y=5e-9,
        voltage=200e3,
    )
    detector = CoherentDetector((12, 12), 2e-9, z=0.0)
    np.testing.assert_allclose(microscope_image(source, (), detector), 0.0, atol=1e-40)
    single = make_gaussian(waist_x=5e-9, waist_y=5e-9, voltage=200e3)
    image = microscope_image(single, (), detector)
    mixed = ensemble_image((single, single), (0.3, 0.7), (), detector)
    np.testing.assert_allclose(image, mixed, rtol=1e-13)


def test_wave_image_gradient_and_jax_fit_unknown_propagation_distance():
    source = make_gaussian(
        x=jnp.array([-3e-9, 3e-9]),
        y=jnp.zeros(2),
        amp=jnp.array([1.0, 1j]),
        waist_x=2e-9,
        waist_y=3e-9,
        voltage=200e3,
    )

    def image(p):
        return (
            microscope_image(
                source, (), CoherentDetector((16, 16), 2e-9, z=p[0] * 1e-6)
            )
            / 1e-18
        )

    truth = jnp.array([18.0])
    target = image(truth)
    derivative = jax.jacfwd(image)(truth)[..., 0]
    finite = (image(truth + 1e-3) - image(truth - 1e-3)) / 0.002
    np.testing.assert_allclose(derivative, finite, rtol=2e-4, atol=2e-7)
    result = fit_images(
        predict_images=image,
        images=target,
        noise_std=1.0,
        starting_parameters=[13.0],
        bounds=([5.0], [30.0]),
        max_evaluations=100,
        gradient_tolerance=1e-8,
        relative_cost_tolerance=1e-10,
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, truth, atol=1e-5)


def test_pixel_quadrature_conserves_gaussian_flux():
    source = make_gaussian(waist_x=5e-9, waist_y=5e-9, voltage=200e3)
    image = microscope_image(source, (), CoherentDetector((40, 40), 2e-9, 0.0, 4))
    q = np.asarray(source.Q_inv)
    expected = np.pi / (float(source.k) * np.sqrt(np.linalg.det(q.imag)))
    np.testing.assert_allclose(image.sum(), expected, rtol=1e-10)


def test_real_space_blur_matches_analytic_gaussian_pixel_integral():
    import numpy as np
    from scipy.special import ndtr

    from temgym_core.coherent_image import CoherentDetector, blurred_gaussian_image
    from temgym_core.gaussian import make_gaussian

    width = 4e-9
    blur = 1.3e-9
    pitch = 1e-9
    beam = make_gaussian(waist_x=width, waist_y=width, voltage=200000.0)
    detector = CoherentDetector((16, 16), pitch, 0.0, 8)
    actual = np.asarray(
        blurred_gaussian_image(beam, detector, blur_sigma_m=blur, padding_pixels=8)
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

    from temgym_core.coherent_image import CoherentDetector, blurred_gaussian_image
    from temgym_core.gaussian import make_gaussian

    beam = make_gaussian(
        x=jnp.array([-2e-9, 2e-9]),
        amp=jnp.array([1.0, 0.7j]),
        waist_x=3e-9,
        waist_y=3e-9,
        voltage=200000.0,
    )
    detector = CoherentDetector((8, 8), 1e-9, 0.0, 4)

    def image(s):
        return blurred_gaussian_image(beam, detector, blur_sigma_m=s * 1e-9) / 1e-18

    derivative = jax.jacfwd(image)(1.2)
    finite = (image(1.2001) - image(1.1999)) / 0.0002
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, finite, atol=1e-8, rtol=1e-5)
