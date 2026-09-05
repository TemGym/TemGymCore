from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_extension_design import (
    approximate_covariance_noise,covariance_factor,greedy_design,training_residual,
)

jax.config.update('jax_enable_x64',True)


def test_moment_noise_positive_and_inverse_dose_scaling():
    c=jnp.array([[3.,.4],[.4,2.]])
    a=np.asarray(jax.jit(approximate_covariance_noise)(c,1e6))
    b=np.asarray(approximate_covariance_noise(c,4e6))
    assert np.linalg.eigvalsh(a).min()>0
    np.testing.assert_allclose(a,4*b,rtol=1e-14)


def test_inverse_information_uses_all_weak_directions():
    j=np.diag([100.,1.,.001])
    factor=covariance_factor(j)
    np.testing.assert_allclose(factor@factor.T,np.diag([1e-4,1.,1e6]))
    with pytest.raises(ValueError):covariance_factor(np.array([[1.,1.],[2.,2.]]))


def test_design_is_distinct_and_maximin_across_hypotheses():
    # The asymmetric huge response of candidate0 is useless in hypothesis1.
    a=np.array([[[[20.,0.]],[[2.,0.]],[[0.,1.]]],
                [[[0.,0.]],[[2.,0.]],[[0.,1.]]]])
    selected,scores,info=greedy_design(a,2)
    np.testing.assert_array_equal(selected,[1,2])
    assert all(s>0 for s in scores)
    np.testing.assert_allclose(info,np.tile(np.diag([5.,2.]),(2,1,1)))
    with pytest.raises(ValueError):greedy_design(a,4)


def test_old_heldout_values_cannot_enter_training_residual():
    class Quotient:
        def observables(self,p,beam,aperture):
            return jnp.ones((len(beam),2,3))*p[0],jnp.ones((len(aperture),3))*p[0]
    beam=np.ones((11,2,3));beam[8:]=np.nan
    aperture=np.ones((3,3));aperture[2]=np.nan
    arrays=dict(controls=np.zeros((11,7)),beam_observables=beam,
        beam_observable_covariances=np.tile(np.eye(3),(11,2,1,1)),
        aperture_observables=aperture,aperture_observable_covariances=np.tile(np.eye(3),(3,1,1)))
    report=dict(records=[dict(controls=np.zeros(7)) for _ in range(3)])
    residual=np.asarray(training_residual(Quotient(),arrays,report)(jnp.zeros(23)))
    assert residual.shape==(55,) and np.isfinite(residual).all()
    np.testing.assert_array_equal(residual,np.r_[-np.ones(54),0.])
