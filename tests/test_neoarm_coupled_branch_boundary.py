from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_coupled_branch_boundary import shifted_bounds


def test_only_bias_bounds_change_all_thirteen_remain_free():
    initial=np.zeros(13);initial[6]=np.pi
    lower=np.full(13,-10.);upper=np.full(13,10.)
    lower[6],upper[6]=np.pi,2*np.pi
    row=dict(parameters=initial.tolist(),bounds=[lower.tolist(),upper.tolist()])
    seed,new_lower,new_upper=shifted_bounds(row)
    np.testing.assert_array_equal(seed,initial)
    np.testing.assert_array_equal(new_lower[np.arange(13)!=6],lower[np.arange(13)!=6])
    np.testing.assert_array_equal(new_upper[np.arange(13)!=6],upper[np.arange(13)!=6])
    assert np.all(new_lower<new_upper)
    assert new_lower[6]<np.pi<new_upper[6]
    assert row['bounds'][0][6]==np.pi


def test_incompatible_seeds_are_not_clipped_into_new_interval():
    row=dict(parameters=np.zeros(13),bounds=[np.full(13,-10.),np.full(13,10.)])
    with pytest.raises(ValueError):shifted_bounds(row)
    row['parameters'][6]=np.pi
    with pytest.raises(ValueError):shifted_bounds(row,(2.,1.))
