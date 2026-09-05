from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_projector_branch_recovery import selected_arrays, nominal_preserving_seed
from neoarm_coupled_projector_fit import CoupledProjectorResponse, stage_coordinates, grouped_stage, encode_stages
from neoarm_last_lens_physical_fit import RoundLensResponse


def test_training_selection_never_uses_poisoned_fresh_holdouts():
    response = np.ones((4, 2, 2, 4)); response[[1,2]] = np.nan
    factor = np.eye(64)+.001*np.tril(np.ones((64,64)), -1)
    covariance = factor@factor.T
    arrays = dict(response=response, joint_response_covariance=covariance,
        pl1_controls=np.array([-.58,.83]), il3_controls=np.array([.4,.45,.6,.75]),
        training_mask=np.array([True,False,False,True]), heldout_mask=np.array([False,True,True,False]))
    controls, observed, selected_cov = selected_arrays(arrays, True)
    np.testing.assert_array_equal(controls[:,1], [.4,.4,.75,.75])
    assert observed.shape == (4,2,4) and np.all(np.isfinite(observed))
    indices = np.r_[np.arange(16),np.arange(48,64)]
    np.testing.assert_array_equal(selected_cov,covariance[np.ix_(indices,indices)])
    with pytest.raises(ValueError): selected_arrays(arrays,False)


def test_new_excitation_seed_preserves_nominal_unknown_mechanical_map():
    model = CoupledProjectorResponse(RoundLensResponse((5e-9,5e-9,1e-3,1e-3),55e-6))
    p = encode_stages([np.log(.02),np.log(.315),-1.57,-1.52],
        [np.log(.018),np.log(.033),.95,.4],[2.3,.2,.015,.4,.005])
    moved = nominal_preserving_seed(model,p,3.2,-.35)
    original = np.asarray(grouped_stage(stage_coordinates(p)[4:],0.))@np.asarray(model.upstream_matrix(p))
    revised = np.asarray(grouped_stage(stage_coordinates(moved)[4:],0.))@np.asarray(model.upstream_matrix(moved))
    np.testing.assert_allclose(original,revised,rtol=5e-11,atol=2e-7)
    np.testing.assert_allclose(model.predict(p,jnp.array([[.3,0.]])),
        model.predict(moved,jnp.array([[.3,0.]])),rtol=5e-11,atol=1e-9)
    assert moved[6] == pytest.approx(3.2) and moved[7] == pytest.approx(-.35)
