import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_cl1_profile import (
    FREE_ORDER, pack_free, unpack_free, conditional_reference_seed,
)


def test_all22_remaining_coordinates_are_free_and_fixed_centre_has_no_derivative():
    full = jnp.arange(23.)
    free = pack_free(full)
    expanded = jax.jit(lambda q: unpack_free(q, 7.))(free)
    assert len(FREE_ORDER) == len(set(FREE_ORDER)) == 22
    assert set(FREE_ORDER) == set(range(1, 23))
    assert expanded[0] == 7.
    np.testing.assert_array_equal(pack_free(expanded), free)
    derivative = np.asarray(jax.jacfwd(lambda q: unpack_free(q, 7.))(free))
    np.testing.assert_array_equal(derivative[0], np.zeros(22))
    np.testing.assert_array_equal(derivative[FREE_ORDER], np.eye(22))


def test_conditional_seed_matches_gaussian_covariance_formula():
    p = np.zeros(23)
    factor = np.eye(23); factor[4, 0] = .6
    covariance = factor@factor.T
    result, trials = conditional_reference_seed(p, covariance, 2., lambda q: True)
    np.testing.assert_allclose(result, covariance[:, 0]*2/covariance[0, 0])
    assert len(trials) == 1 and result[0] == 2.


def test_seed_backtracking_never_changes_prescribed_centre():
    p = np.zeros(23)
    factor = np.eye(23); factor[4, 0] = 2.
    result, trials = conditional_reference_seed(p, factor@factor.T, 1., lambda q: abs(q[4]) <= .6)
    assert len(trials) == 3
    assert all(row['fixed_centre'] == 1. for row in trials)
    assert result[0] == 1. and result[4] == .5


def test_impossible_target_is_not_replaced_with_feasible_nearby_centre():
    result, trials = conditional_reference_seed(np.zeros(23), np.eye(23), 2.,
        lambda q: q[0] <= 1., max_backtracking=4)
    assert result is None and len(trials) == 4
    assert all(row['fixed_centre'] == 2. for row in trials)


def test_incorrect_free_dimension_rejected():
    with pytest.raises(ValueError):
        unpack_free(jnp.zeros(21), 0.)
