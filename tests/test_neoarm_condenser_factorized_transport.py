from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_factorized_transport import CanonicalFactors, propagate_factors
from temgym_core.gaussian import make_gaussian
from temgym_core.immersed_objective import uniform_field_canonical_matrix
from temgym_core.neoarm_calibration import apply_transfer_matrix_to_gaussians

jax.config.update('jax_enable_x64', True)


def test_factors_retain_guard_and_ordering():
    with pytest.raises(ValueError):
        CanonicalFactors(np.eye(4)[None]*1.01, np.array([1.]), ('bad',))
    with pytest.raises(ValueError):
        CanonicalFactors(np.tile(np.eye(4), (2, 1, 1)), np.array([1., 0.]), ('a', 'b'))


def test_factorized_map_preserves_centres_curvature_and_beam_dependent_phase():
    step = np.asarray(uniform_field_canonical_matrix(12., .005))
    factors = CanonicalFactors(np.tile(step, (8, 1, 1)), .005*np.arange(1, 9), ('body',)*8)
    beam = make_gaussian(x=jnp.array([0., 1e-6, -2e-6]), y=jnp.array([1e-6, -1e-6, 0.]),
        dx=jnp.array([.001, -.002, .0005]), dy=jnp.array([-.0003, .001, .002]),
        voltage=200e3, waist_x=2e-6, waist_y=3e-6)
    actual = propagate_factors(beam, factors)
    direct = apply_transfer_matrix_to_gaussians(beam, factors.product(), z_out_m=.04, axial_path_m=0.)
    np.testing.assert_allclose(actual.r_xy, direct.r_xy, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(actual.Q_inv, direct.Q_inv, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(actual.pathlength, direct.pathlength, rtol=1e-12, atol=1e-20)
    ratio = np.asarray(actual.amplitude/direct.amplitude)
    np.testing.assert_allclose(ratio/ratio[0], 1., rtol=1e-12, atol=1e-12)
    assert abs(abs(ratio[0])-1.) < 1e-12
