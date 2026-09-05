"""Moment-estimator closure independent of the large saved acquisitions."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/lens_inversion"))
from neoarm_probe_likelihood import signed_raw_moments


def test_signed_moments_preserve_negative_background_and_uncertainty():
    yy, xx = np.indices((24, 28))
    expected = 20000*np.exp(-((xx-19)**2+(yy-9)**2)/8)
    _, covariance = signed_raw_moments(expected, expected, 5.)
    np.linalg.cholesky(covariance)
    rng = np.random.default_rng(17)
    measured = []
    for _ in range(200):
        counts = rng.poisson(expected)+rng.normal(0, 5., expected.shape)
        value, _ = signed_raw_moments(counts, expected, 5.)
        measured.append(value)
    truth, _ = signed_raw_moments(expected, expected, 5.)
    residuals = np.linalg.solve(np.linalg.cholesky(covariance), (np.asarray(measured)-truth).T)
    assert .90 < np.sqrt(np.mean(residuals**2)) < 1.10
    assert np.max(np.abs(residuals.mean(axis=1))) < .2


def test_signed_moments_reject_unresolved_total_count():
    with pytest.raises(ValueError, match="SNR"):
        signed_raw_moments(np.ones((8, 8)), np.ones((8, 8)), 10.)


@pytest.mark.parametrize("noise", [-1., float("nan")])
def test_signed_moments_require_valid_noise(noise):
    with pytest.raises(ValueError, match="noise"):
        signed_raw_moments(np.ones((8, 8))*100, np.ones((8, 8))*100, noise)
