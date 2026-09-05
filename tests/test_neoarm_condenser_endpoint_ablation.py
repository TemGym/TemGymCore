"""Five-input role selection and correct marginal likelihood checks."""
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from neoarm_condenser_endpoint_ablation import (
    TRAIN_RESPONSE, CHECK_RESPONSE, TRAIN_FRAMES, CHECK_FRAMES,
    select_response, ablated_residual,
)


def test_five_input_indices_include_both_detector_axes_for_x_and_tilt_x():
    data = np.arange(16).reshape(2, 2, 4)
    covariance = np.tile(np.eye(8), (2, 1, 1))
    selected, _ = select_response(data, covariance)
    np.testing.assert_array_equal(selected, [[0, 2, 4, 6], [8, 10, 12, 14]])
    assert set(TRAIN_FRAMES).isdisjoint(CHECK_FRAMES)
    assert set(TRAIN_FRAMES)|set(CHECK_FRAMES) == set(range(9))
    assert set(TRAIN_RESPONSE).isdisjoint(CHECK_RESPONSE)


def test_selection_uses_marginal_not_conditional_covariance():
    rng = np.random.default_rng(726)
    a = rng.normal(size=(8, 8))
    full = a@a.T+np.eye(8)
    _, selected = select_response(np.zeros((2, 2, 4)), np.tile(full, (2, 1, 1)))
    marginal = full[np.ix_(TRAIN_RESPONSE, TRAIN_RESPONSE)]
    np.testing.assert_array_equal(selected[0], marginal)
    conditional = np.linalg.inv(np.linalg.inv(full)[np.ix_(TRAIN_RESPONSE, TRAIN_RESPONSE)])
    assert np.linalg.norm(marginal-conditional) > .1


def test_omitted_calibration_rows_are_removed_and_single_prior_preserved():
    class Endpoint:
        def response(self, parameters):
            return jnp.zeros((2, 2, 4))
    calibration = (np.zeros((2, 2, 4)), np.tile(np.eye(8), (2, 1, 1)))
    full = np.arange(119, dtype=float)
    result = ablated_residual(lambda p: jnp.asarray(full), Endpoint(), calibration)(jnp.zeros(29))
    np.testing.assert_array_equal(result[:102], full[:102])
    np.testing.assert_array_equal(result[102:110], np.zeros(8))
    assert len(result) == 111 and result[-1] == full[-1]
    full[102:118] = -1e9
    changed = ablated_residual(lambda p: jnp.asarray(full), Endpoint(), calibration)(jnp.zeros(29))
    np.testing.assert_array_equal(changed, result)


@pytest.mark.parametrize("indices", [[0, 0, 2, 4], [-1, 1, 2, 4], [0, 2, 4], [0, 2, 4, 8]])
def test_invalid_response_indices_rejected(indices):
    with pytest.raises(ValueError):
        select_response(np.zeros((2, 2, 4)), np.tile(np.eye(8), (2, 1, 1)), indices)
