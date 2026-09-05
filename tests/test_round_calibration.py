import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.round_calibration import calibrated_position_response,round_response_gain_family
from temgym_core.microscope_inversion import hard_edge_solenoid_matrix,drift_matrix

jax.config.update("jax_enable_x64",True)


def predict(stages,prefix,controls,ell=5e-6):
    rho,angle,shear,phi,kappa = prefix
    r,c,s = jnp.exp(rho),jnp.cos(angle),jnp.sin(angle)
    radial = jnp.array([[r*c,ell*r*s],[(shear*r*c-s/r)/ell,shear*r*s+c/r]])
    rotation = jnp.array([[jnp.cos(phi),-jnp.sin(phi)],[jnp.sin(phi),jnp.cos(phi)]])
    matrix = jnp.kron(radial,rotation)
    magnetic = jnp.eye(4).at[2:, :2].set(kappa/ell*jnp.array([[0.,-1.],[1.,0.]]))
    matrix = matrix@magnetic
    order = jnp.array([0,2,1,3])
    for p,u in zip(stages[::-1],controls[::-1]):
        native = drift_matrix(jnp.exp(p[1]))@hard_edge_solenoid_matrix(p[2]+p[3]*u,jnp.exp(p[0]))
        matrix = native[jnp.ix_(order,order)]@matrix
    return matrix[:2]*jnp.array([5e-9,5e-9,1e-3,1e-3])/55e-6


@pytest.mark.parametrize("count",[1,2,3,4])
def test_exact_scale_family_for_entire_round_chain(count):
    stages = jnp.array([[np.log(.02+.002*i),np.log(.1+.03*i),.6+.2*i,.35] for i in range(count)])
    prefix = jnp.array([2.3,.24,.001,.31,.004])
    controls = jnp.linspace(-.3,.4,count)
    gains = jnp.log(jnp.array([1.03,.96,1.004]))
    changed,changed_prefix = round_response_gain_family(stages,prefix,gains)
    expected = calibrated_position_response(predict(stages,prefix,controls),gains)
    np.testing.assert_allclose(predict(changed,changed_prefix,controls),expected,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(changed[:,:2]-stages[:,:2],float(gains[0]+gains[1]-2*gains[2]),atol=1e-14)
    np.testing.assert_array_equal(changed[:,2:],stages[:,2:])
    compensated,compensated_prefix = round_response_gain_family(stages,prefix,-gains)
    np.testing.assert_allclose(calibrated_position_response(predict(compensated,compensated_prefix,controls),gains),
        predict(stages,prefix,controls),rtol=2e-12,atol=2e-12)


def test_gain_family_is_differentiable_and_preserves_exact_covariance_floor():
    stages = jnp.array([[np.log(.02),np.log(.3),-1.,-.3]])
    prefix = jnp.array([2.,.2,.001,.3,.004])
    family = lambda g:jnp.r_[round_response_gain_family(stages,prefix,-g)[0].reshape(-1),
        round_response_gain_family(stages,prefix,-g)[1]]
    derivative = np.asarray(jax.jit(jax.jacfwd(family))(jnp.zeros(3)))
    np.testing.assert_allclose(derivative[0],[-1,-1,2],atol=1e-14)
    covariance = np.diag(np.full(3,.002**2))
    assert np.sqrt(derivative[0]@covariance@derivative[0]) == pytest.approx(np.sqrt(6)*.002)
    np.testing.assert_allclose(jax.jit(jax.vmap(family))(jnp.zeros((2,3))),jnp.tile(family(jnp.zeros(3)),(2,1)))


def test_calibration_rejects_wrong_coordinate_shapes():
    with pytest.raises(ValueError):calibrated_position_response(jnp.eye(4),jnp.zeros(3))
    with pytest.raises(ValueError):calibrated_position_response(jnp.zeros((2,4)),jnp.zeros(4))
    with pytest.raises(ValueError):round_response_gain_family(jnp.zeros((0,4)),jnp.zeros(5),jnp.zeros(3))
