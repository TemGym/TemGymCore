import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_nonparaxial_benchmark import gaussian_phase_space_rule,weighted_covariance,propagate_condensers
from neoarm_condenser_general_source_contract import GeneralSourceStigmator,initial_parameters


def test_gaussian_rule_normalization_mean_covariance_and_fourth_moment():
    lower=np.array([[1.,0.,0.,0.],[.2,2.,0.,0.],[.1,.3,1.5,0.],[0.,.2,.1,.8]])
    covariance=lower@lower.T
    points,weights=gaussian_phase_space_rule(covariance,4)
    np.testing.assert_allclose(weights.sum(),1.,atol=1e-15)
    np.testing.assert_allclose(np.asarray(weights)@np.asarray(points),np.zeros(4),atol=1e-15)
    np.testing.assert_allclose(weighted_covariance(points,weights),covariance,atol=1e-14)
    np.testing.assert_allclose(jnp.sum(weights*points[:,0]**4),3*covariance[0,0]**2,atol=1e-14)


def test_paraxial_sequence_closes_existing_model_without_inverse_map():
    model=GeneralSourceStigmator('after_cl1');p=jnp.asarray(initial_parameters())
    points,_=gaussian_phase_space_rule(np.asarray(model.decode(p)[1]),3)
    control=jnp.array([.2,.3,1.8])
    actual,bodies=jax.jit(lambda x:propagate_condensers(model,p,control,x))(points)
    expected=points@model.maps(p,control)[0].T
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-18)
    assert len(bodies)==3
