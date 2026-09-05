from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_general_source_contract import (
    GeneralSourceStigmator, source_covariance, source_coordinates, quantum_margin,
    initial_parameters, fixed_upstream_source_family, drift, SOURCE_SCALES, INVISIBLE,
    information,
)

jax.config.update('jax_enable_x64', True)


def test_source_chart_spans_ten_moments_and_is_quantum_interior():
    p = initial_parameters(); c = np.asarray(source_covariance(p[12:22]))
    np.testing.assert_allclose(source_coordinates(c), p[12:22], atol=1e-14)
    assert quantum_margin(c) > 0
    derivative = np.asarray(jax.jacfwd(lambda q: source_covariance(q)/jnp.outer(SOURCE_SCALES, SOURCE_SCALES))(p[12:22]))
    assert np.linalg.matrix_rank(derivative.reshape(16, 10)) == 10
    with pytest.raises(ValueError):
        source_covariance(np.zeros(9))


@pytest.mark.parametrize('location', ['after_cl1', 'after_cl3'])
def test_unknown_Q_absorbs_exactly_only_when_upstream_controls_fixed(location):
    model = GeneralSourceStigmator(location); p = initial_parameters()
    optical = np.zeros(12); optical[:4*model.insert_before] = np.tile([.02, .05, .1, -.15], model.insert_before)
    candidate, _ = fixed_upstream_source_family(model, p, p[22:]+[.2, .1, -.05], optical_offsets=optical)
    assert quantum_margin(model.decode(candidate)[1]) > 0
    full = jax.jit(model.observables)
    assert np.max(abs(np.asarray(full(candidate)-full(p)))) > 1e-4
    restricted = GeneralSourceStigmator(location)
    controls = np.asarray(restricted.controls).copy(); controls[:, :restricted.insert_before] = 0
    restricted.controls = jnp.asarray(np.unique(controls, axis=0))
    fixed = jax.jit(restricted.observables)
    np.testing.assert_allclose(fixed(candidate), fixed(p), atol=2e-9, rtol=0)
    # Allowing10 unknown moments does not remove the exact discrete ambiguity
    # when the actual source is rotationally invariant.
    round_p = initial_parameters(rotationally_invariant=True)
    np.testing.assert_allclose(full(model.polarity_reversal(round_p, 0)), full(round_p), atol=2e-10, rtol=0)


def test_source_plane_drift_and_zero_Q_families_are_exact():
    model = GeneralSourceStigmator('after_cl3'); p = initial_parameters()
    covariance = np.asarray(model.decode(p)[1]); moved = GeneralSourceStigmator('after_cl3')
    moved.source_z_m += .002; d = np.asarray(drift(.002)); changed = p.copy()
    changed[12:22] = source_coordinates(d@covariance@d.T)
    np.testing.assert_allclose(jax.jit(moved.observables)(changed), jax.jit(model.observables)(p), atol=2e-10, rtol=0)
    p[23:] = 0; changed = p.copy(); changed[22] += .3
    c = covariance+.04*INVISIBLE*np.outer(SOURCE_SCALES, SOURCE_SCALES)
    assert quantum_margin(c) > 0
    changed[12:22] = source_coordinates(c)
    np.testing.assert_allclose(jax.jit(model.observables)(changed), jax.jit(model.observables)(p), atol=2e-10, rtol=0)


@pytest.mark.parametrize('location', ['after_cl1', 'after_cl3'])
def test_full_ideal_scan_profiles_all_ten_source_moments_and_Q(location):
    model = GeneralSourceStigmator(location)
    result = information(jax.jit(jax.jacfwd(model.observables))(initial_parameters()))
    assert result['full_rank'] == 25
    assert result['nuisance_rank'] == 13
    assert result['profiled_optical_rank'] == 12
