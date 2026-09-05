from pathlib import Path
import sys

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_internal_beam_design import largest_component, capture_lower_bound, information

jax.config.update('jax_enable_x64', True)


def test_connected_component_does_not_bridge_rejected_nodes():
    mask = np.zeros((4, 4, 3), bool); mask[:2, :2] = True; mask[3, 3, 2] = True
    result, sizes = largest_component(mask)
    assert sizes == [12, 1]
    assert result.sum() == 12 and not result[3, 3, 2]


def test_full_information_does_not_grant_a_nuisance_prior():
    rng = np.random.default_rng(172)
    j = rng.normal(size=(120, 25))
    result, factor = information(j)
    assert result['rank'] == 25 and result['nuisance_rank'] == 13 and result['profiled_optical_rank'] == 12
    np.testing.assert_allclose(factor@factor.T, np.linalg.inv(j.T@j), atol=3e-16)
    j[:, 0] = j[:, 12]
    result, factor = information(j)
    assert result['rank'] == 24 and factor is None


def test_capture_bound_is_conservative_and_detects_clipping():
    capture = np.asarray(capture_lower_bound(np.array([np.diag([20.**2, 20.**2]), np.diag([200.**2, 200.**2])])) )
    assert capture[0] > .999
    assert capture[1] < .1
