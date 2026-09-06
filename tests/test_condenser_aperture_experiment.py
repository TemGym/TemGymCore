"""Physical aperture propagation, derivatives, count covariance and exact ambiguity."""

import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/microscope_models"))
jax.config.update("jax_enable_x64", True)

from condenser_aperture_experiment import (
    aperture_specification,
    equivalent_aperture_model,
    make_transmission_operator,
    measure_aperture_currents,
)
from condenser_calibration_model import generating_parameters, specimen_moments


def parameters():
    return np.r_[generating_parameters(), np.log(0.0025 / (0.056 - 0.0025)), 0.2, -0.15]


def test_actual_gaussian_aperture_component_and_quadrature_convergence(monkeypatch):
    from temgym_core.components import SigmoidAperture

    original = SigmoidAperture._call_gaussian
    calls = []

    def record(self, beam):
        calls.append(len(beam.x))
        return original(self, beam)

    monkeypatch.setattr(SigmoidAperture, "_call_gaussian", record)
    spec = aperture_specification()
    ordinary = make_transmission_operator(specification=spec, side="after")
    refined = make_transmission_operator(
        specification=dict(spec, radial_order=384), side="after"
    )
    commands = np.array([[0.75, 0.72, 1.0], [1.05, 1.02, 1.0], [1.35, 1.32, 1.0]])
    values = np.asarray(jax.vmap(ordinary, in_axes=(None, 0))(parameters(), commands))
    reference = np.asarray(jax.vmap(refined, in_axes=(None, 0))(parameters(), commands))
    assert calls == [192] * 3 + [384] * 3
    assert np.all((values > 0) & (values < 1))
    assert np.all(np.diff(values, axis=1) > 0)
    np.testing.assert_allclose(values, reference, atol=1e-8, rtol=1e-8)


def test_sigmoid_current_derivatives_and_cl3_ordering():
    p = parameters()
    command = np.array([1.05, 1.02, 1.0])
    after = make_transmission_operator(
        specification=aperture_specification(), side="after"
    )
    before = make_transmission_operator(
        specification=aperture_specification(), side="before"
    )
    derivative = np.asarray(jax.jacfwd(after, argnums=0)(p, command))
    for index in (0, 10, 18, 19, 20):
        step = np.zeros(22)
        step[index] = 1e-5
        numerical = np.asarray(after(p + step, command) - after(p - step, command)) / (
            2e-5
        )
        np.testing.assert_allclose(
            derivative[:, index], numerical, atol=1e-8, rtol=2e-5
        )
    assert (
        np.linalg.norm(np.asarray(jax.jacfwd(after, argnums=1)(p, command))[:, 1])
        > 1e-3
    )
    np.testing.assert_array_equal(
        np.asarray(jax.jacfwd(before, argnums=1)(p, command))[:, 1], 0.0
    )


def test_unknown_aperture_and_prefield_leave_an_exact_family():
    p = parameters()
    other = equivalent_aperture_model(p, objective_strength=-0.6)
    commands = np.random.default_rng(13).uniform(
        [0.75, 0.72, 0.8], [1.35, 1.32, 1.3], (11, 3)
    )
    transmission = make_transmission_operator(
        specification=aperture_specification(), side="after"
    )
    batch = jax.jit(jax.vmap(transmission, in_axes=(None, 0)))
    np.testing.assert_allclose(
        batch(p, commands), batch(other, commands), atol=1e-12, rtol=1e-11
    )
    moments = jax.jit(jax.vmap(specimen_moments, in_axes=(None, 0)))
    mean, cov = moments(p[:19], commands)
    other_mean, other_cov = moments(other[:19], commands)
    np.testing.assert_allclose(other_mean, mean, rtol=1e-10, atol=1e-14)
    scale = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    scale = scale[:, :, None] * scale[:, None, :]
    np.testing.assert_allclose(other_cov / scale, cov / scale, atol=1e-12, rtol=1e-10)
    assert abs(np.exp(other[8] - p[8]) - 1) > 0.4


def test_high_count_reference_covariance_does_not_overflow(tmp_path):
    counts = np.array([[2_000_000, 8_000_000, 16_000_000]], dtype=np.int64)
    reference = np.array([20_000_000], dtype=np.int64)
    np.savez(
        tmp_path / "calibration_counts.npz",
        controls=[[1.0, 1.0, 1.0]],
        transmitted_electrons=counts,
        reference_electrons=reference,
    )
    path = measure_aperture_currents(input_directory=tmp_path)
    data = np.load(path)
    ratios = counts[0].astype(float) / reference[0]
    expected = (np.diag(ratios) + np.outer(ratios, ratios)) / float(reference[0])
    np.testing.assert_allclose(data["covariance"][0], expected)
    assert np.all(np.linalg.eigvalsh(data["covariance"][0]) > 0)
