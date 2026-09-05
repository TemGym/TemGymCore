from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_projector_shape_likelihood import (
    QuantumSourceCovariance,ImageSummaryLikelihood,canonical_position_rows,shape_coordinates,initialize_source,center_source_gauge,
)
from neoarm_coupled_projector_fit import CoupledProjectorResponse,encode_stages
from neoarm_last_lens_physical_fit import RoundLensResponse
from neoarm_last_lens_tomography import J2,J4


@pytest.fixture
def fixture():
    model=CoupledProjectorResponse(RoundLensResponse((5e-9,5e-9,1e-3,1e-3),55e-6))
    optics=encode_stages(np.array([np.log(.02),np.log(.31),-1.4,-1.2]),
        np.array([np.log(.025),np.log(.05),1.1,.55]),[2.3,.13,.025,.4,.005])
    source=QuantumSourceCovariance(5e-9,2.507934e-12)
    p=np.array([-.5,-.2,.1,.15,.2,-.1,-.05,-.3,.2,-.1])
    controls=np.array([[v,u] for u in [-.4,0.,.4] for v in [-1.,-.3,.3,1.]])
    inputs=np.zeros((9,4))
    for k,step in enumerate(model.calibration.input_steps_grouped):inputs[1+2*k,k]=step;inputs[2+2*k,k]=-step
    rows=np.asarray(canonical_position_rows(model,optics,jnp.asarray(controls)))
    covariance=rows@np.asarray(source.covariance(p))@rows.swapaxes(-1,-2)/model.calibration.pixel_m**2
    shape=np.asarray(shape_coordinates(covariance))
    means=np.einsum('sij,fj->sfi',model.predict(optics,controls),inputs/np.asarray(model.calibration.input_steps_grouped))
    y=np.concatenate((means,np.broadcast_to(shape[:,None,:],(len(controls),9,3)),np.full((len(controls),9,1),12.)),axis=-1)
    factor=np.eye(6)+.12*np.tril(np.ones((6,6)),-1)
    cov=np.broadcast_to(.01**2*factor@factor.T,(len(controls),9,6,6)).copy()
    return model,source,optics,p,controls,inputs,y,cov


def test_quantum_chart_schur_completion_and_roundtrip(fixture):
    _,source,_,p,*_=fixture
    covariance=np.asarray(source.balanced_covariance(p))
    assert np.linalg.eigvalsh(covariance+1j*np.asarray(J4)).min()>0
    np.testing.assert_allclose(source.encode_balanced(covariance),p,rtol=2e-13,atol=2e-13)
    result=source.diagnostic(p)
    assert result['quantum_admissible']
    assert min(result['emittances_over_quantum_minimum'])>1.


def test_general_cross_covariance_not_artificially_symmetric(fixture):
    _,source,_,p,*_=fixture
    covariance=np.asarray(source.balanced_covariance(p))
    assert not np.allclose(covariance[:2,2:],covariance[:2,2:].T)
    np.testing.assert_allclose(covariance,covariance.T,atol=1e-14)


def test_round_antisymmetric_cross_gauge_is_exact(fixture):
    model,source,optics,p,controls,*_=fixture
    rows=np.asarray(canonical_position_rows(model,optics,jnp.asarray(controls)))*source.scales[None,None,:]
    gauge=np.block([[np.zeros((2,2)),np.asarray(J2)],[-np.asarray(J2),np.zeros((2,2))]])
    induced=rows@gauge@rows.swapaxes(-1,-2)
    scale=max(np.abs(rows).max()**2,1e-30)
    assert np.abs(induced).max()<1e-13*scale
    covariance=source.balanced_covariance(p)
    np.testing.assert_allclose(rows@(covariance+.01*gauge)@rows.swapaxes(-1,-2),
                               rows@covariance@rows.swapaxes(-1,-2),rtol=3e-13,atol=1e-20)


def test_source_jit_vmap_grad_hessian(fixture):
    _,source,_,p,*_=fixture
    jac=jax.jit(jax.jacfwd(source.balanced_covariance))(p)
    hess=jax.jit(jax.jacfwd(jax.jacfwd(source.balanced_covariance)))(p)
    assert np.all(np.isfinite(hess))
    for k in [0,2,4,5,7,8,9]:
        delta=np.eye(10)[k]*1e-6
        difference=(source.balanced_covariance(p+delta)-source.balanced_covariance(p-delta))/(2e-6)
        np.testing.assert_allclose(jac[...,k],difference,rtol=2e-7,atol=2e-8)
    batched=jax.jit(jax.vmap(source.balanced_covariance))(jnp.stack([p,p]))
    np.testing.assert_allclose(batched[0],source.balanced_covariance(p),rtol=3e-13,atol=1e-13)


def test_profiled_summary_matches_full_six_coordinate_nuisance_elimination(fixture):
    model,source,optics,p,controls,inputs,truth,cov=fixture
    noise=np.random.default_rng(18).normal(size=truth.shape)*.002
    observed=truth+noise
    problem=ImageSummaryLikelihood(model,source,controls,observed,cov,inputs)
    residual=np.asarray(problem.source_residual(p,optics))
    exact_cost=0.
    for y,mean,c in zip(observed,truth,cov):
        lower=np.linalg.cholesky(c)
        difference=mean-y
        design=np.zeros((9,6,11))
        design[:,0,0]=1.;design[:,1,1]=1.
        for frame in range(9):design[frame,5,2+frame]=1.
        white=np.linalg.solve(lower,difference[...,None]).reshape(-1)
        columns=np.linalg.solve(lower,design).reshape(54,11)
        nuisance=np.linalg.lstsq(columns,-white,rcond=None)[0]
        profiled=white+columns@nuisance
        exact_cost+=profiled@profiled
    assert residual@residual==pytest.approx(exact_cost,rel=3e-10,abs=1e-9)
    assert problem.independent_summary_coordinate_count==len(controls)*43
    # Existing 8-coordinate signed responses have not been appended.
    assert residual.size==len(controls)*45


def test_shape_information_has_nine_not_ten_source_directions(fixture):
    model,source,optics,p,controls,inputs,y,cov=fixture
    problem=ImageSummaryLikelihood(model,source,controls,y,cov,inputs)
    jac=np.asarray(jax.jacfwd(lambda q:problem.shapes(optics,q))(p)).reshape(-1,10)
    norms=np.linalg.norm(jac,axis=0)
    singular=np.linalg.svd(jac/norms,compute_uv=False)
    assert np.sum(singular>singular[0]*1e-9)==9
    assert singular[-1]<singular[0]*1e-13


def test_source_initializer_uses_shape_tomography_and_is_admissible(fixture):
    model,source,optics,p,controls,inputs,y,cov=fixture
    problem=ImageSummaryLikelihood(model,source,controls,y,cov,inputs)
    initial,diagnostic=initialize_source(problem,optics)
    assert diagnostic['linear_shape_information_rank']==9
    assert diagnostic['generating_source_used'] is False
    assert source.diagnostic(initial)['quantum_admissible']
    assert np.all(np.isfinite(problem.source_residual(initial,optics)))


def test_centroid_conditioning_is_nonnegative(fixture):
    model,source,optics,p,controls,inputs,y,cov=fixture
    observed=y+np.random.default_rng(21).normal(size=y.shape)*.005
    problem=ImageSummaryLikelihood(model,source,controls,observed,cov,inputs)
    full=np.asarray(problem.source_residual(p,optics));means=np.asarray(problem.centroid_residual(optics))
    assert full@full>=means@means-1e-10


def test_gauge_centering_preserves_shapes_and_quantum_admissibility(fixture):
    model,source,optics,p,controls,inputs,y,cov=fixture
    problem=ImageSummaryLikelihood(model,source,controls,y,cov,inputs)
    centered,diagnostic=center_source_gauge(source,p)
    assert source.diagnostic(centered)['quantum_admissible']
    np.testing.assert_allclose(problem.shapes(optics,centered),problem.shapes(optics,p),rtol=2e-13,atol=1e-13)
    if diagnostic['applied']:
        assert diagnostic['centered_minimum_quantum_eigenvalue']>=diagnostic['original_minimum_quantum_eigenvalue']


def test_invalid_shapes_and_quantum_encoding_rejected(fixture):
    model,source,optics,p,controls,inputs,y,cov=fixture
    with pytest.raises(ValueError):QuantumSourceCovariance(0.,1.)
    with pytest.raises(ValueError):source.balanced_covariance(np.zeros(9))
    with pytest.raises(ValueError):source.encode_balanced(np.eye(4)*.5)
    with pytest.raises(ValueError):ImageSummaryLikelihood(model,source,controls,y[:,:8],cov[:,:8],inputs)
