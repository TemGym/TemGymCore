from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_projector_branch_experiment import (
    approximate_response_covariance, distinct_partition, local_profile_separation, signed_inputs,
)


def test_signed_covariance_retains_xy_and_input_order():
    source = np.array([[[4., 1.2], [1.2, 9.]]])
    covariance = approximate_response_covariance(source, 2., 100.)[0]
    expected = (source[0]/4 + np.eye(2)*(.65**2+1/12))/200
    np.testing.assert_allclose(covariance, np.kron(expected, np.eye(4)))
    assert np.linalg.eigvalsh(covariance).min() > 0
    states = signed_inputs([2., 3., 4., 5.])
    np.testing.assert_array_equal(states[1::2], np.diag([2., 3., 4., 5.]))
    np.testing.assert_array_equal(states[2::2], -states[1::2])


def test_no_challenge_state_reuses_previous_control():
    training = np.array([True, False, False, True])
    a, b = distinct_partition([.3, .45, .6, .75], training, [-.25, -.125, 0., .125, .25])
    assert not np.any(a & b) and np.all(a | b)
    with pytest.raises(ValueError):
        distinct_partition([.25, .45, .6, .75], training, [-.25, .25])
    with pytest.raises(ValueError):
        distinct_partition([.3, .3, .6, .75], training, [])


def test_profile_score_removes_only_local_nuisance_directions():
    jac = np.array([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    delta = jac@np.array([2., -3.])+np.array([0., 0., 0., 4.])
    score = local_profile_separation(delta, jac, np.eye(4))
    assert score['linearized_nuisance_rank'] == 2
    assert score['locally_profiled_separation'] == pytest.approx(16.)
    assert score['fixed_hypothesis_separation'] > score['locally_profiled_separation']
