from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_il2_extension_design import (
    ThreeProjectorResponse,extend_nominal_fitted_map,connected_component,fisher_summary,
    downstream_pair_matrix,reconstruct_with_shared_calibration,
)
from neoarm_coupled_projector_fit import CoupledProjectorResponse,encode_stages
from neoarm_last_lens_physical_fit import RoundLensResponse
from temgym_core.microscope_inversion import drift_matrix,hard_edge_solenoid_matrix


def fixture():
    calibration=RoundLensResponse((5e-9,5e-9,1e-3,1e-3),55e-6)
    old=CoupledProjectorResponse(calibration)
    p=encode_stages([np.log(.02),np.log(.315),-1.57,-1.52],
        [np.log(.018),np.log(.033),.95,.4],[2.3,.2,.015,.4,.005])
    extended=extend_nominal_fitted_map(old,p,[np.log(.022),np.log(.038),2.6,.4])
    return old,p,ThreeProjectorResponse(calibration),extended


def test_three_stage_order_matches_independent_core_trace():
    old,base,model,p=fixture();controls=np.array([[.2,-.1,.3],[-.6,.2,-.4]])
    grouped=np.array([0,2,1,3]);prefix=np.asarray(model.upstream_matrix(p))
    expected=[]
    for c in controls:
        matrix=np.eye(4)
        for i in [2,1,0]:
            stage=drift_matrix(np.exp(p[4*i+1]))@hard_edge_solenoid_matrix(p[4*i+2]+p[4*i+3]*c[i],np.exp(p[4*i]))
            matrix=np.asarray(stage)@matrix
        full=matrix[np.ix_(grouped,grouped)]@prefix
        expected.append(full[:2]*np.array(model.calibration.input_steps_grouped)/model.calibration.pixel_m)
    np.testing.assert_allclose(model.predict(p,controls),expected,rtol=2e-11,atol=2e-9)
    old_controls=np.array([[.2,-.1],[-.6,.2]])
    np.testing.assert_allclose(model.predict(p,np.column_stack((old_controls,np.zeros(2)))),
        old.predict(base,old_controls),rtol=2e-11,atol=2e-9)


def test_unexcited_il2_has_four_gauges_then_scan_adds_rank():
    _,_,model,p=fixture()
    c=np.array([[t,u,0.] for t in [-1.,-.3,.5,1.] for u in [-.2,0.,.4]])
    jac=jax.jacfwd(model.predict)(jnp.asarray(p),jnp.asarray(c))
    old=fisher_summary(jac,np.eye(jac.size//17))
    assert old['rank']==13 and old['parameter_covariance'] is None
    varied=np.concatenate((c,np.array([[t,.1,v] for t in [-.7,.8] for v in [-.5,0.,.5]])))
    derivative=jax.jacfwd(model.predict)(jnp.asarray(p),jnp.asarray(varied))
    full=fisher_summary(derivative,np.eye(derivative.size//17))
    assert full['rank']==17
    assert np.linalg.eigvalsh(full['parameter_covariance']).min()>0


def test_connected_scan_cannot_jump_a_failed_crossover():
    assert connected_component([True,False,True,True,False,True],2)==(2,3)
    assert connected_component([True,False,True],1) is None


def test_shared_downstream_calibration_correlates_reconstructed_maps():
    _,_,model,p=fixture();pair=np.array([[-.6,0.],[.8,.2]])
    controls=np.array([[*c,v] for v in [-.2,0.,.3] for c in pair])
    response=np.asarray(model.predict(p,controls)).reshape(3,2,2,4)
    result=reconstruct_with_shared_calibration(response,np.eye(48)*1e-6,p[:8],np.eye(8)*1e-10,
        pair,model.calibration.input_steps_grouped,model.calibration.pixel_m)
    shared=np.asarray(result['shared_calibration_covariance'])
    assert np.linalg.norm(shared[:16,16:32])>0
    a=np.asarray(downstream_pair_matrix(p[:8],pair))
    expected=np.linalg.solve(a,response.reshape(3,4,4)*(model.calibration.pixel_m/np.array(model.calibration.input_steps_grouped))[None,None,:])
    np.testing.assert_allclose(result['maps'],expected,rtol=2e-10,atol=1e-6)
