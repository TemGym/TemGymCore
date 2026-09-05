"""Algebra-only checks: no detector simulation or synthetic acquisition."""
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from neoarm_column_calibration_compatibility import (
    compose_projectors, excitation_bridge, sample_kappa_invariant,
)
from neoarm_coupled_projector_fit import grouped_stage
from neoarm_last_lens_tomography import J4, mechanical_shear, rotation

jax.config.update("jax_enable_x64", True)


def test_affine_bridge_equates_excitation_including_reference_control():
    old_bias, old_slope, new_bias, new_slope, reference = -.7, -1.2, .4, .3, .6
    offset, gain = excitation_bridge(old_bias, old_slope, new_bias, new_slope, reference)
    control = np.array([-.6, 0., .35])
    np.testing.assert_allclose(old_bias+old_slope*control,
        new_bias+new_slope*(offset+gain*control-reference), atol=5e-16)


@pytest.mark.parametrize("arguments", [(0., 1., 0., 0.), (np.nan, 1., 0., 1.)])
def test_invalid_affine_law_is_rejected(arguments):
    with pytest.raises(ValueError):
        excitation_bridge(*arguments)


def test_four_stages_follow_physical_order_and_cannot_be_reversed():
    stages = np.array([[np.log(.02), np.log(.315), -.8, .5],
        [np.log(.018), np.log(.03), .5, .4],
        [np.log(.018), np.log(.04), .7, .45],
        [np.log(.02), np.log(.06), .9, .5]])
    controls = np.array([.2, -.1, .15, -.25])
    maps = [np.asarray(grouped_stage(stages[i], controls[i])) for i in range(4)]
    expected = maps[0]@maps[1]@maps[2]@maps[3]
    actual = np.asarray(compose_projectors(stages.ravel(), controls))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
    assert np.linalg.norm(actual-maps[3]@maps[2]@maps[1]@maps[0]) > 1.


@pytest.mark.parametrize("kappa", [800., -872.9409371233595])
def test_mechanical_prefix_retains_sample_field_invariant(kappa):
    radial = np.array([[2., .04], [-5., .4]])
    canonical = np.kron(radial, np.asarray(rotation(.3)))
    np.testing.assert_allclose(canonical.T@J4@canonical, J4, atol=1e-14)
    mechanical = canonical@mechanical_shear(kappa)
    recovered, pullback = sample_kappa_invariant(mechanical)
    np.testing.assert_allclose(recovered, kappa, atol=1e-9)
    np.testing.assert_allclose(pullback,
        mechanical_shear(kappa).T@J4@mechanical_shear(kappa), atol=2e-10)


def test_endpoint_mechanical_conversion_cancels_prefield_conversion():
    kappa = -872.9409371233595
    shear = np.asarray(mechanical_shear(kappa))
    canonical_pre = np.kron(np.array([[.7, .001], [-20., 1.4]]), np.asarray(rotation(.2)))
    canonical_row = np.hstack((3.*np.eye(2), .002*np.eye(2)))
    mechanical_endpoint = canonical_row@shear
    mechanical_pre = np.linalg.solve(shear, canonical_pre)
    np.testing.assert_allclose(mechanical_endpoint@mechanical_pre,
        canonical_row@canonical_pre, atol=1e-14)
    inconsistent_pre = np.linalg.solve(np.asarray(mechanical_shear(800.)), canonical_pre)
    assert np.linalg.norm(mechanical_endpoint@inconsistent_pre-canonical_row@canonical_pre) > 1.


def test_pullback_rejects_wrong_coordinates_or_nan_shape():
    for bad in (np.eye(3), np.full((4, 4), np.nan)):
        with pytest.raises(ValueError):
            sample_kappa_invariant(bad)
