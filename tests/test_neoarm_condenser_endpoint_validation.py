"""Independent response-role and residual-accounting checks, no rendering."""
from pathlib import Path
import sys

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from neoarm_condenser_endpoint_validation import reconstruct_response, block_diagnostics


def example_image_fits():
    p = np.zeros((2, 13, 6))
    c = np.broadcast_to(np.eye(6), (2, 13, 6, 6)).copy()
    response = np.arange(16).reshape(2, 2, 4)*.03-.2
    for i in range(4):
        p[:, 1+2*i, :2] = 100.+response[:, :, i]
        p[:, 2+2*i, :2] = 100.-response[:, :, i]
        c[:, 1+2*i, :2, :2] = np.array([[2., .2], [.2, 3.]])
        c[:, 2+2*i, :2, :2] = np.array([[4., -.1], [-.1, 5.]])
    return p, c, response


def test_response_units_order_and_correlations_are_reconstructed():
    p, c, expected = example_image_fits()
    response, covariance = reconstruct_response(p, c)
    np.testing.assert_allclose(response, expected, atol=2e-14)
    np.testing.assert_allclose(np.diagonal(covariance, axis1=-2, axis2=-1),
        np.tile(np.r_[np.full(4, 1.5), np.full(4, 2.)], (2, 1)))
    for i in range(4):
        np.testing.assert_allclose(covariance[:, i, i+4], .025)
        np.testing.assert_allclose(covariance[:, i+4, i], .025)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_reference_and_heldout_frames_cannot_change_training_response():
    p, c, _ = example_image_fits()
    original = reconstruct_response(p, c)
    p[:, [0, 9, 10, 11, 12]] = np.nan
    c[:, [0, 9, 10, 11, 12]] = np.nan
    result = reconstruct_response(p, c)
    for actual, expected in zip(result, original):
        np.testing.assert_array_equal(actual, expected)


def test_residual_blocks_use_118_independent_data_coordinates_and_one_prior():
    r = np.arange(119, dtype=float)
    blocks = block_diagnostics(r)
    assert [b["count"] for b in blocks.values()] == [96, 6, 16, 1]
    assert sum(b["chi_square"] for b in blocks.values()) == r@r
    assert blocks["radius_prior"]["chi_square"] == r[-1]**2
    with pytest.raises(ValueError):
        block_diagnostics(np.zeros(118))


def test_reconstruction_requires_declared_two_analyser_image_layout():
    with pytest.raises(ValueError):
        reconstruct_response(np.zeros((2, 9, 6)), np.zeros((2, 9, 6, 6)))
