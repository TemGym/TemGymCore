"""Cheap map metrics and Gaussian-support arithmetic; no specimen rendering."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from neoarm_shared_material_design import round_canonical_metrics, neighbour_radius_estimate

jax.config.update("jax_enable_x64", True)


def matrix(magnification, defocus, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    rotation = jnp.array([[c, -s], [s, c]])
    radial = jnp.array([[magnification, magnification*defocus], [0., 1/magnification]])
    grouped = jnp.kron(radial, rotation)
    order = jnp.array([0, 2, 1, 3])
    return grouped[jnp.ix_(order, order)]


def test_round_metric_recovers_signed_effective_defocus_and_rotation():
    expected = np.array([40000., -.24e-6, 2.7])
    actual = round_canonical_metrics(matrix(*expected))
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-13)


def test_metric_derivative_keeps_defocus_for_a_two_control_root():
    derivative = jax.jacfwd(lambda p: round_canonical_metrics(matrix(p[0], p[1], .4))[:2])(
        jnp.array([40000., 0.]))
    np.testing.assert_allclose(derivative, np.eye(2), atol=1e-12, rtol=1e-12)


def test_conjugate_support_recovers_nine_neighbours_and_defocus_exceeds_old_cap():
    wavelength, spacing = 2.5079340450548e-12, 1.6978536820061162e-10
    assert neighbour_radius_estimate(0., wavelength, spacing) == 9
    assert neighbour_radius_estimate(-2.414981384e-7, wavelength, spacing) == 35
    assert neighbour_radius_estimate(2.053864264e-7, wavelength, spacing) == 30
