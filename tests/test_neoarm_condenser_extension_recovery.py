from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_extension_recovery import selected_data,beam_residual,prediction_diagnostic

jax.config.update('jax_enable_x64',True)


def test_fresh_training_never_reads_poisoned_heldout():
    data=np.ones((4,2,3));data[3]=np.nan
    arrays=dict(controls=np.zeros((4,7)),beam_observables=data,
        beam_observable_covariances=np.tile(np.eye(3),(4,2,1,1)),
        training_mask=np.array([True,True,True,False]),heldout_mask=np.array([False,False,False,True]))
    assert selected_data(arrays,True)[1].shape==(3,2,3)
    with pytest.raises(ValueError):selected_data(arrays,False)
    arrays['heldout_mask']=np.ones(4,dtype=bool)
    with pytest.raises(ValueError):selected_data(arrays,True)


def test_beam_block_keeps_both_analysers_and_full_covariance():
    class Quotient:
        def observables(self,p,controls,aperture):
            return jnp.ones((len(controls),2,3))*p[0],jnp.zeros((len(aperture),3))
    c=np.array([[2.,.3,.1],[.3,1.,.2],[.1,.2,3.]])
    selected=(np.zeros((2,7)),np.arange(12).reshape(2,2,3),np.tile(c,(2,2,1,1)))
    residual=np.asarray(jax.jit(beam_residual(Quotient(),selected))(jnp.array([.7])))
    expected=np.linalg.solve(np.linalg.cholesky(c),(.7-selected[1])[...,None]).reshape(-1)
    np.testing.assert_allclose(residual,expected)


def test_predictive_diagnostic_includes_joint_parameter_covariance():
    a=jnp.array([[1.,2.],[2.,-.3],[.5,.7]])
    residual=lambda p:a@p-jnp.array([2.,1.,3.])
    p=np.array([.2,.7]);c=np.array([[.1,.03],[.03,.4]])
    report=prediction_diagnostic(residual,p,c)
    r=np.asarray(residual(p))
    assert report['coordinate_count']==3
    assert report['chi_square']==pytest.approx(r@r)
    assert report['predictive_chi_square']==pytest.approx(r@np.linalg.solve(np.eye(3)+a@c@a.T,r))
