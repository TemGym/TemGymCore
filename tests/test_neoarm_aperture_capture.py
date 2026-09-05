"""Mathematical primitive tests; camera-image propagation is GPU-only."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
from neoarm_aperture_capture import (
    camera_acceptance, circle_cell_fraction, czt_axis, normalized_hermite_modes,
    thermal_source_modes, wave_capture,
)
from temgym_core.constants import energy2wavelength

jax.config.update("jax_enable_x64", True)


def test_physical_czt_matches_direct_complex_matrix_for_nonsquare_axes():
    rng = np.random.default_rng(1804)
    field = rng.normal(size=(3, 17))+1j*rng.normal(size=(3, 17))
    x0, dx, y0, dy, product = -.7, .13, -.9, .07, 2.3
    x = x0+dx*np.arange(17); y = y0+dy*np.arange(23)
    direct = dx*np.einsum("bj,kj->bk", field, np.exp(-2j*np.pi*x[None, :]*y[:, None]/product))
    actual = czt_axis(jnp.asarray(field), input_start=x0, input_step=dx,
        output_start=y0, output_step=dy, output_count=23, wavelength_times_b=product)
    np.testing.assert_allclose(actual, direct, rtol=2e-13, atol=2e-13)
    transposed = czt_axis(jnp.asarray(field.T), input_start=x0, input_step=dx,
        output_start=y0, output_step=dy, output_count=23, wavelength_times_b=product, axis=0)
    np.testing.assert_allclose(transposed, direct.T, rtol=2e-13, atol=2e-13)


def test_thermal_modes_restore_covariance_without_renormalizing_tail():
    source = np.diag(np.array([5e-9, 50e-6, 6e-9, 45e-6])**2)
    wavelength = float(energy2wavelength(200e3))
    pure, weights, omitted = thermal_source_modes(source, wavelength, modes_per_axis=12)
    epsilon = wavelength/(4*np.pi)
    np.testing.assert_allclose(np.sqrt(np.diag(pure)[::2]*np.diag(pure)[1::2]), epsilon)
    reconstructed = np.zeros((4, 4))
    for nx in range(12):
        for ny in range(12):
            reconstructed += weights[nx, ny]*pure*np.repeat([2*nx+1, 2*ny+1], 2)
    np.testing.assert_allclose(reconstructed, source, rtol=2e-8, atol=1e-35)
    assert omitted > 0
    assert weights.sum() < 1.


def test_subquantum_source_is_rejected():
    with pytest.raises(ValueError):
        thermal_source_modes(np.diag([1e-18, 1e-12, 1e-18, 1e-12]), float(energy2wavelength(200e3)))


def test_hermite_recurrence_uses_correct_field_not_intensity_polynomials():
    x = jnp.array([-.3, 0., .7])
    modes = normalized_hermite_modes(x, 3)
    np.testing.assert_allclose(modes[0], 1.)
    np.testing.assert_allclose(modes[1], np.sqrt(2)*x)
    np.testing.assert_allclose(modes[2], (2*x*x-1)/np.sqrt(2))


def test_sharp_circle_area_and_translation_are_resolved():
    count = 128; step = 4/count
    coords = (jnp.arange(count)+.5)*step-2
    for centre in (jnp.zeros(2), jnp.array([.21, -.13])):
        fraction = circle_cell_fraction(coords, coords, step, 1., centre, subpixels=8)
        assert abs(float(fraction.sum())*step**2/np.pi-1) < .001
        assert float(fraction.min()) == 0 and float(fraction.max()) == 1


def test_camera_probability_is_not_renormalized_outside_window():
    points = jnp.array([[0., 0.], [1., 0.], [4., 0.]])
    actual = np.asarray(camera_acceptance(points, half_width_m=1., psf_sigma_m=.01))
    np.testing.assert_allclose(actual, [1., .5, 0.], atol=1e-14)


def test_wave_renderer_refuses_cpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="GPU"):
        wave_capture(None, None)
