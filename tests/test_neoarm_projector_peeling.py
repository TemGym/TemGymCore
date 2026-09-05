from pathlib import Path
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_projector_peeling import choose_disjoint_pairs, standardized_chi_square

jax.config.update('jax_enable_x64',True)


def test_pair_selection_is_disjoint_and_uses_only_control_and_calibration():
    p = np.array([np.log(.02),np.log(.315),-1.5,-1.])
    pairs, candidates = choose_disjoint_pairs(p,np.array([-.75,-.25,.25,.75]))
    assert len(pairs) == 2 and len(candidates) == 6
    assert sorted(i for pair in pairs for i in pair) == [0,1,2,3]
    assert all(row['condition_number'] > 1 for row in candidates)
    single, _ = choose_disjoint_pairs(p,np.array([-.75,-.25,.25]))
    assert len(single) == 1


def test_singular_and_invalid_pairs_rejected():
    p = np.array([np.log(.02),np.log(.315),0.,1.])
    with pytest.raises(ValueError,match='independent'):
        choose_disjoint_pairs(p,np.array([-1.,1.]))
    with pytest.raises(ValueError,match='two to four'):
        choose_disjoint_pairs(p,np.array([0.]))


def test_unit_balancing_preserves_chi_square_across_large_physical_scales():
    rng = np.random.default_rng(18)
    factor = rng.normal(size=(12,16))
    covariance = factor@factor.T
    error = rng.normal(size=12)
    scales = np.logspace(-12,12,12)
    expected = error@np.linalg.solve(covariance,error)
    assert standardized_chi_square(error*scales,covariance*scales[:,None]*scales[None,:]) == pytest.approx(expected,rel=1e-12)
    with pytest.raises(ValueError,match='uncertainty'):
        standardized_chi_square(error,np.zeros_like(covariance))
