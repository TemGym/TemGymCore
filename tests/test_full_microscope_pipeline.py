"""Aperture-pair dose elimination must preserve the full measurement likelihood."""

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.calibration.condenser_model import condenser_starting_guesses
from temgym_core.calibration.inference import profiled_residual
from temgym_core.calibration.workflow import PRESETS


def test_profiled_dose_matches_full_correlated_linear_least_squares():
    rng = np.random.default_rng(81)
    prediction = rng.uniform(0.2, 2.0, (3, 2, 6))
    covariance = rng.normal(size=(3, 2, 6, 6))
    covariance = covariance @ np.swapaxes(covariance, -1, -2) + np.eye(6)
    white = np.linalg.inv(np.linalg.cholesky(covariance))
    measured = prediction.copy()
    measured[..., 5] *= np.array([10.0, 20.0, 30.0])[:, None]
    measured += rng.normal(0.0, 0.1, measured.shape)
    actual = np.asarray(
        profiled_residual(
            jnp.asarray(prediction), jnp.asarray(measured), jnp.asarray(white)
        )
    )
    for index, (p, y, w) in enumerate(zip(prediction, measured, white)):
        unit = np.zeros_like(p)
        unit[:, 5] = p[:, 5]
        zero = p.copy()
        zero[:, 5] = 0.0
        design = np.einsum("nij,nj->ni", w, unit).ravel()[:, None]
        error = np.einsum("nij,nj->ni", w, zero - y).ravel()
        dose = np.linalg.lstsq(design, -error, rcond=None)[0]
        np.testing.assert_allclose(
            actual[index].ravel(), error + design @ dose, atol=1e-13
        )


def test_profiled_measurement_derivative_matches_finite_difference():
    p = jnp.ones((2, 2, 6))
    values = p.at[..., 5].set(20.0)
    white = jnp.tile(jnp.eye(6), (2, 2, 1, 1))

    def residual(x):
        predicted = p.at[..., 0].set(x[0]).at[:, 0, 5].set(jnp.exp(x[1]))
        return profiled_residual(predicted, values, white).ravel()

    x = jnp.array([0.7, 0.3])
    direction = jnp.array([0.2, -0.1])
    step = 1e-5
    expected = jax.jacfwd(residual)(x) @ direction
    difference = (residual(x + step * direction) - residual(x - step * direction)) / (
        2 * step
    )
    np.testing.assert_allclose(expected, difference, atol=1e-9, rtol=1e-7)


def test_engineering_starts_are_independent_and_inside_hard_bounds():
    starts, lo, hi = condenser_starting_guesses(
        starting_guess_path=PRESETS / "microscope_condenser_starting_guess.json",
        number_of_broad_starts=4,
        number_of_apertures=2,
    )
    assert len(starts) == 8
    assert np.shape(starts) == (8, 31)
    assert all(np.all((p > lo) & (p < hi)) for p in starts)
