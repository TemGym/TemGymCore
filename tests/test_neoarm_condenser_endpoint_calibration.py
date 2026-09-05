import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
from neoarm_condenser_endpoint_calibration import (
    EndpointRows, acquisition_inputs, check_responses, STEPS,
)


def test_endpoint_round_trip_and_jit():
    model = EndpointRows(-873.)
    p = jnp.array([3.5, .17, 2.8, -3.4, -.15, -.7])
    response = jax.jit(model.response)(p)
    recovered = model.encode_response(response)
    np.testing.assert_allclose(model.response(recovered), response, rtol=1e-11, atol=1e-11)
    assert np.linalg.matrix_rank(np.asarray(jax.jacfwd(model.response)(p)).reshape(16, 6)) == 6


def test_fixed_shear_not_double_counted():
    model = EndpointRows(800.)
    p = np.array([3.5, .17, 2.8, 3.4, -.15, -.7])
    canonical = np.asarray(model.canonical_rows(p))
    mechanical = np.asarray(model.response(p))*model.pixel_m/STEPS
    np.testing.assert_allclose(mechanical@np.linalg.inv(np.asarray(model.shear)), canonical,
                               rtol=1e-12, atol=1e-12)
    assert not np.allclose(mechanical, canonical, rtol=1e-6, atol=1e-12)


def test_frozen_training_and_check_inputs():
    states, directions = acquisition_inputs()
    assert states.shape == (13, 4)
    np.testing.assert_array_equal(states[0], np.zeros(4))
    np.testing.assert_array_equal(states[1:9:2], np.diag(STEPS))
    np.testing.assert_array_equal(states[2:9:2], -np.diag(STEPS))
    np.testing.assert_array_equal(states[9::2], directions)
    np.testing.assert_array_equal(states[10::2], -directions)


def test_heldout_signed_pairs_do_not_share_reference_noise():
    p = np.arange(24.).reshape(4, 6)
    c = np.tile(np.eye(6), (4, 1, 1))
    value, covariance = check_responses(p, c)
    np.testing.assert_array_equal(value, np.full((2, 2), -3.))
    np.testing.assert_array_equal(covariance, np.tile(.5*np.eye(2), (2, 1, 1)))
    with pytest.raises(ValueError):
        check_responses(p[:2], c[:2])


def test_endpoint_input_validation():
    with pytest.raises(ValueError):
        EndpointRows(np.nan)
    with pytest.raises(ValueError):
        EndpointRows(0., steps=[1., 2., 1., 1.])
    with pytest.raises(ValueError):
        EndpointRows(0.).encode_response(np.zeros((2, 4)))
