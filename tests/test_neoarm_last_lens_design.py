from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_last_lens_design import response_covariance_from_mean_covariance,profiled_geometry_precision,coefficient_parameters
from neoarm_last_lens_tomography import mechanical_shear

jax.config.update('jax_enable_x64',True)


def test_mean_to_response_covariance_keeps_xy_and_independent_basis_order():
    c=np.array([[[4.,1.],[1.,9.]],[[2.,-.3],[-.3,3.]]])
    actual=np.asarray(response_covariance_from_mean_covariance(c))
    for a,b in zip(actual,c):np.testing.assert_array_equal(a,np.kron(b/2,np.eye(4)))


def test_profiled_precision_responds_to_dose_not_a_common_output_coordinate_scale():
    f=jnp.diag(jnp.array([8.,5.,1/8,1/5]))@mechanical_shear(800.)
    inputs=jnp.array([5e-9,5e-9,1e-3,1e-3]);pixel=55e-6
    p=coefficient_parameters(f,.02,.315,inputs,pixel,-np.pi/2,-1.5)
    t=jnp.linspace(-1.,1.,25);cov=jnp.tile(jnp.eye(8)*1e-8,(25,1,1));mask=jnp.ones(25,bool)
    a=profiled_geometry_precision(p,t,cov,mask,inputs,pixel)
    b=profiled_geometry_precision(p,t,cov/4,mask,inputs,pixel)
    assert a[1]==19 and b[1]==19
    np.testing.assert_allclose(b[0],a[0]/2,rtol=1e-9)
    # Pixel-unit rescaling changes coefficients and covariance together.
    p2=coefficient_parameters(f,.02,.315,inputs,pixel/2,-np.pi/2,-1.5)
    c=profiled_geometry_precision(p2,t,cov*4,mask,inputs,pixel/2)
    np.testing.assert_allclose(c[0],a[0],rtol=1e-8)


def test_rejected_states_contribute_zero_information():
    f=jnp.diag(jnp.array([8.,5.,1/8,1/5]));inputs=jnp.array([5e-9,5e-9,1e-3,1e-3])
    p=coefficient_parameters(f,.02,.315,inputs,55e-6,-1.5,-1.4)
    t=jnp.linspace(-1.,1.,25);cov=jnp.tile(jnp.eye(8)*1e-8,(25,1,1));mask=jnp.arange(25)%3!=0
    masked=profiled_geometry_precision(p,t,cov,mask,inputs,55e-6)
    reduced=profiled_geometry_precision(p,t[mask],cov[mask],jnp.ones(int(mask.sum()),bool),inputs,55e-6)
    np.testing.assert_allclose(masked[0],reduced[0],rtol=1e-9)
