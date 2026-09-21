"""Native solenoid rotation measured through an anisotropic beam covariance."""

import numpy as np

from temgym_core.calibration.condenser_model import generating_parameters, matrix5


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
