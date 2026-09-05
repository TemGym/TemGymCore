from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_last_lens_images import (
    central_response_transform,signed_response_from_image_fits,image_initializer,
    response_residual,covariance_factor,
)
from neoarm_last_lens_tomography import predict_response_parameters
from temgym_core.neoarm_calibration import DetectorGeometry

jax.config.update('jax_enable_x64',True)


def test_symmetric_response_retains_xy_covariance_and_cancels_shared_reference():
    p=np.arange(54,dtype=float).reshape(9,6)
    c=np.tile(np.eye(6),(9,1,1));c[:,:2,:2]=np.array([[4.,1.],[1.,9.]])
    response,cov=signed_response_from_image_fits(p,c)
    np.testing.assert_allclose(response,(p[1::2,:2]-p[2::2,:2]).T/2)
    assert cov[0,4]==pytest.approx(.5)
    assert cov[0,0]==pytest.approx(2.)
    # Construct one-sided differences, including shared-reference covariance.
    one=np.zeros((16,18))
    for frame in range(8):
        one[2*frame:2*frame+2,2*(frame+1):2*(frame+1)+2]=np.eye(2)
        one[2*frame:2*frame+2,:2]=-np.eye(2)
    central=np.zeros((8,16))
    for a in range(2):
        for i in range(4):
            central[4*a+i,4*i+a]=.5;central[4*a+i,4*i+2+a]=-.5
    np.testing.assert_array_equal(central@one,central_response_transform())
    c[0,:2,:2]*=1e9
    np.testing.assert_allclose(signed_response_from_image_fits(p,c)[1],cov)


def test_response_scale_nuisance_has_exact_coefficient_column_gauge():
    p=jnp.r_[jnp.log(15.75),-1.5,-1.4,jnp.arange(16)*.003+.02]
    t=jnp.linspace(-1,1,9);scales=jnp.array([.002,-.001,.0003,-.0006])
    observed=predict_response_parameters(p,t)
    compensated=p.at[3:].set((p[3:].reshape(2,2,4)*jnp.exp(-scales)).ravel())
    inverse=jnp.tile(jnp.eye(8),(9,1,1))
    residual=response_residual(jnp.r_[compensated,scales],t,observed,inverse,.002)
    np.testing.assert_allclose(residual[:-4],0,atol=1e-14)
    np.testing.assert_allclose(residual[-4:],scales/.002)


def test_column_scaled_covariance_matches_well_conditioned_reference():
    base=np.random.default_rng(21).normal(size=(30,6));scales=10.**np.arange(-9,9,3)
    jac=base*scales;factor,rank,singular=covariance_factor(jac)
    assert rank==6 and singular.min()>0
    reference=np.linalg.inv(base.T@base)/scales[:,None]/scales[None,:]
    np.testing.assert_allclose(factor@factor.T,reference,rtol=1e-12)


def test_data_initializer_and_bad_covariance_checks():
    detector=DetectorGeometry(shape_yx=(32,32),mtf_sigma_px=.65)
    yy,xx=np.indices((32,32));image=1e4*np.exp(-((xx-17.2)**2+(yy-13.1)**2)/8)
    p=image_initializer(image,detector)
    np.testing.assert_allclose(p[:2],[1.7,-2.4],atol=.02)
    assert np.all(np.isfinite(p))
    with pytest.raises(ValueError,match='No significant'):
        image_initializer(np.zeros((32,32)),detector)
    with pytest.raises(ValueError,match='positive definite'):
        signed_response_from_image_fits(np.zeros((9,6)),np.zeros((9,6,6)))
