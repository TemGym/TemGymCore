from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_projector_five_input_ablation import (
    CentroidProfileLikelihood, FIVE_FRAMES, select_y_checks, y_check_responses,
)
from neoarm_coupled_projector_fit import CoupledProjectorResponse, encode_stages
from neoarm_last_lens_physical_fit import RoundLensResponse


@pytest.fixture
def experiment():
    steps=np.array([5e-9,5e-9,1e-3,1e-3])
    model=CoupledProjectorResponse(RoundLensResponse(tuple(steps),55e-6))
    p=encode_stages([np.log(.02),np.log(.31),-1.4,-1.2],
        [np.log(.025),np.log(.05),1.1,.55],[2.3,.13,.025,.4,.005])
    controls=np.array([[t,u] for u in [-.4,0.,.4] for t in np.linspace(-1.,1.,7)])
    inputs=np.zeros((9,4))
    for axis in range(4):
        inputs[1+2*axis,axis]=steps[axis]
        inputs[2+2*axis,axis]=-steps[axis]
    prediction=np.einsum('sij,fj->sfi',model.predict(p,controls),inputs/steps)
    rng=np.random.default_rng(74)
    image_parameters=rng.normal(size=(len(controls),9,6))
    image_parameters[:,:,:2]=prediction+rng.normal(size=prediction.shape)*.004
    factors=np.eye(6)+rng.normal(size=(len(controls),9,6,6))*.15
    cov=.01**2*factors@factors.swapaxes(-1,-2)
    return model,p,controls,inputs,image_parameters,cov


def test_five_input_likelihood_ignores_omitted_y_images(experiment):
    model,p,c,inputs,y,cov=experiment
    baseline=CentroidProfileLikelihood(model,c,y,cov,inputs,FIVE_FRAMES)
    y[:,[3,4,7,8]]=np.nan;cov[:,[3,4,7,8]]=np.nan
    ablated=CentroidProfileLikelihood(model,c,y,cov,inputs,FIVE_FRAMES)
    np.testing.assert_array_equal(baseline.residual(p),ablated.residual(p))
    assert ablated.independent_coordinate_count==len(c)*8
    assert len(ablated.residual(p))==len(c)*10  # two projected-out coordinates per state
    with pytest.raises(ValueError,match='finite'):
        CentroidProfileLikelihood(model,c,y,cov,inputs,tuple(range(9)))


def test_centroid_marginal_matches_full_six_parameter_nuisance_profile(experiment):
    model,p,c,inputs,y,cov=experiment
    problem=CentroidProfileLikelihood(model,c[:1],y[:1],cov[:1],inputs,FIVE_FRAMES)
    prediction=np.zeros((5,6));prediction[:,:2]=problem.prediction(p)[0]
    observed=y[0,list(FIVE_FRAMES)]
    selected_cov=cov[0,list(FIVE_FRAMES)]
    # Unknown shape/dose for every image plus a shared two-coordinate alignment.
    nuisance=np.zeros((5,6,22))
    nuisance[:,:2,:2]=np.eye(2)
    for frame in range(5):nuisance[frame,2:,2+4*frame:2+4*(frame+1)]=np.eye(4)
    chol=np.linalg.cholesky(selected_cov)
    white=np.linalg.solve(chol,(prediction-observed)[...,None]).reshape(-1)
    design=np.linalg.solve(chol,nuisance).reshape(30,22)
    coefficient=np.linalg.lstsq(design,white,rcond=None)[0]
    profiled=white-design@coefficient
    actual=np.asarray(problem.residual(p))
    np.testing.assert_allclose(actual@actual,profiled@profiled,rtol=5e-12)


def test_per_setting_alignment_profiles_without_optical_information_loss(experiment):
    model,p,c,inputs,y,cov=experiment
    baseline=CentroidProfileLikelihood(model,c,y,cov,inputs,FIVE_FRAMES)
    shifted=y.copy();shifted[:,:,:2]+=np.arange(len(c))[:,None,None]*np.array([.17,-.23])
    moved=CentroidProfileLikelihood(model,c,shifted,cov,inputs,FIVE_FRAMES)
    np.testing.assert_allclose(baseline.residual(p),moved.residual(p),atol=3e-11)
    np.testing.assert_allclose(baseline.jacobian(p),moved.jacobian(p),rtol=0,atol=0)
    jac=np.asarray(baseline.jacobian(p));sv=np.linalg.svd(jac/np.linalg.norm(jac,axis=0),compute_uv=False)
    assert np.sum(sv>sv[0]*1e-10)==13


def test_jit_derivatives_of_profiled_x_only_objective(experiment):
    model,p,c,inputs,y,cov=experiment
    problem=CentroidProfileLikelihood(model,c[:3],y[:3],cov[:3],inputs,FIVE_FRAMES)
    jac=np.asarray(problem.jacobian(p))
    for index in [0,3,6,7,10,12]:
        step=np.eye(13)[index]*1e-6
        finite=(problem.residual(p+step)-problem.residual(p-step))/(2e-6)
        np.testing.assert_allclose(jac[:,index],finite,rtol=3e-6,atol=1e-4)
    assert np.all(np.isfinite(jax.jit(jax.grad(lambda q:jnp.sum(problem.residual(q)**2)))(p)))


def test_round_y_prediction_not_arbitrary_response_completion(experiment):
    model,p,c,inputs,y,cov=experiment
    # x-only data do not observe arbitrary columns1/3. This is a restriction
    # of experimental information, not permission to fill them with truth.
    h=np.asarray(model.predict(p,c))
    altered=h.copy();altered[:,:,1]+=12.;altered[:,:,3]-=5.
    x=inputs[list(FIVE_FRAMES)]/np.array(model.calibration.input_steps_grouped)
    np.testing.assert_array_equal(np.einsum('sij,fj->sfi',h,x),np.einsum('sij,fj->sfi',altered,x))
    assert not np.array_equal(h,altered)


def test_omitted_y_signed_vectors_keep_cross_detector_covariance(experiment):
    model,p,c,inputs,y,cov=experiment
    response,response_cov=y_check_responses(y,cov)
    np.testing.assert_allclose(response[:,:,0],(y[:,3,:2]-y[:,4,:2])/2)
    np.testing.assert_allclose(response[:,:,1],(y[:,7,:2]-y[:,8,:2])/2)
    np.testing.assert_allclose(response_cov[:,0,2],(cov[:,3,0,1]+cov[:,4,0,1])/4)
    np.testing.assert_allclose(response_cov[:,1,3],(cov[:,7,0,1]+cov[:,8,0,1])/4)
    assert np.all(np.linalg.eigvalsh(response_cov)>0)


def test_y_check_design_uses_only_predeclared_controls_and_roles():
    controls=np.array([[-.5,0.],[.8,0.],[-.5,.4],[.8,.75],
        [-.5,-.25],[.8,.25],[-.5,.45],[.8,.45],[-.5,.6],[.8,.6]])
    indices,targets=select_y_checks(controls,4,2,[-.5,.8],[.4,.75],[.45,.6])
    assert indices==[0,1,2,3,6,9]
    np.testing.assert_allclose(controls[indices],targets)
    assert len(indices)*4==24


def test_invalid_covariance_and_frame_selection_rejected(experiment):
    model,p,c,inputs,y,cov=experiment
    with pytest.raises(ValueError,match='distinct'):
        CentroidProfileLikelihood(model,c,y,cov,inputs,[0,1,1])
    cov[0,0,0,1]+=1
    with pytest.raises(ValueError,match='symmetric'):
        CentroidProfileLikelihood(model,c,y,cov,inputs,FIVE_FRAMES)
