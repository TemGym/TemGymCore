"""Actual ellipse extraction, crossover integration and remaining symmetry."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar

EXAMPLES = Path(__file__).parents[1] / "examples/microscope_models"
sys.path.insert(0, str(EXAMPLES))
jax.config.update("jax_enable_x64", True)

from condenser_aperture_experiment import (
    aperture_moments,
    aperture_specification,
    equivalent_aperture_model,
)
from condenser_calibration_model import generating_parameters, matrix5
from condenser_rotation_measurements import (
    make_ellipse_renderer,
    make_focus_transmission,
    measure_rotation_and_focus,
)


def test_ellipse_is_measured_from_pixels_without_lens_truth(tmp_path, monkeypatch):
    import condenser_rotation_measurements as module

    p = np.r_[31.3, 31.7, np.log([8.0, 5.0]), 0.8, np.log([2e7, 3.0])]
    renderer = make_ellipse_renderer()
    rng = np.random.default_rng(47)
    image = rng.poisson(np.asarray(renderer(p))) + rng.normal(0, 2.0, (64, 64))
    np.savez(
        tmp_path / "rotation_images.npz",
        images=image[None],
        controls=[[1.0, 1.0, 1.0]],
        projector_rotation=0.2,
        read_noise_e=2.0,
    )
    np.savez(
        tmp_path / "calibration_counts.npz",
        controls=[[1.0, 1.0, 1.0]],
        transmitted_electrons=[[10.0, 20.0, 30.0]],
        reference_electrons=[40.0],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Analysis must not access the microscope")

    monkeypatch.setattr(module, "generating_parameters", forbidden)
    monkeypatch.setattr(module, "matrix5", forbidden)
    result = measure_rotation_and_focus(input_directory=tmp_path)
    measured = np.load(result["rotation"])
    assert abs(measured["angles"][0] - 0.6) < 5 * np.sqrt(measured["variance"][0])
    assert 0.9 < measured["pixel_rms"][0] < 1.1


def test_rotation_law_follows_full_anisotropic_covariance_transport():
    p = generating_parameters()
    phi = 0.35
    r = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    shape = r @ np.diag([1.6, 1 / 1.6]) @ r.T
    cov = np.zeros((4, 4))
    cov[:2, :2] = np.exp(2 * p[17]) * shape
    cov[2:, 2:] = np.exp(2 * p[18]) * shape
    for c in np.random.default_rng(29).uniform(0.6, 1.4, (12, 3)):
        m = np.asarray(matrix5(p, c))[:4, :4]
        out = (m @ cov @ m.T)[:2, :2]
        _, v = np.linalg.eigh(out)
        observed = np.arctan2(v[1, -1], v[0, -1])
        expected = phi + np.exp(p[12]) - np.exp(p[9:12]) @ c
        assert abs(np.sin(2 * (observed - expected))) < 1e-10


def test_crossover_current_integration_and_exact_fixed_objective_family():
    p = np.r_[generating_parameters(), np.log(0.0025 / (0.056 - 0.0025)), 0.2, -0.15]
    peak = minimize_scalar(
        lambda x: float(aperture_moments(p, jnp.array([x, 1.0, 1.0]), "after")[1]),
        bounds=(1.3, 3.0),
        method="bounded",
        options={"xatol": 1e-12},
    ).x
    controls = jnp.array(
        [[peak - 0.6, 1.0, 1.0], [peak, 1.0, 1.0], [peak + 0.6, 1.0, 1.0]]
    )
    a = make_focus_transmission(specification=aperture_specification())
    b = make_focus_transmission(
        specification=aperture_specification(), order=512, edge_samples=512
    )
    values = np.asarray(jax.vmap(a, in_axes=(None, 0))(p, controls))
    refined = np.asarray(jax.vmap(b, in_axes=(None, 0))(p, controls))
    np.testing.assert_allclose(values, refined, atol=2e-6, rtol=0)
    assert np.all((values >= 0) & (values <= 1.000002))
    assert np.all(values[1] > values[[0, 2]].max(axis=0))
    alternative = equivalent_aperture_model(p, objective_strength=-0.6)
    np.testing.assert_allclose(
        jax.vmap(a, in_axes=(None, 0))(alternative, controls),
        values,
        atol=1e-9,
        rtol=1e-9,
    )
    # Unknown initial ellipse orientation means only the three gains and a
    # common intercept are measured. The old family preserves all of them.
    np.testing.assert_allclose(np.exp(alternative[9:12]), np.exp(p[9:12]))
    direction = np.zeros(22)
    direction[[0, 5, 9, 19]] = [0.1, -0.15, 0.05, 0.2]
    command = controls[0]
    exact = np.asarray(jax.jacfwd(a, argnums=0)(p, command)) @ direction
    step = 1e-6
    difference = np.asarray(
        a(p + step * direction, command) - a(p - step * direction, command)
    ) / (2 * step)
    np.testing.assert_allclose(exact, difference, atol=2e-5, rtol=1e-3)
