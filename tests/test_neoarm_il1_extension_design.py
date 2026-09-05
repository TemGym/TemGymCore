from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_il1_extension_design import FourProjectorResponse,extend_prefix,fisher
from neoarm_il2_extension_design import ThreeProjectorResponse
from neoarm_last_lens_physical_fit import RoundLensResponse
from temgym_core.microscope_inversion import drift_matrix,hard_edge_solenoid_matrix


def fixture():
    cal=RoundLensResponse((5e-9,5e-9,1e-3,1e-3),55e-6)
    old=np.r_[np.log(.02),np.log(.31),-1.57,-1.52,np.log(.018),np.log(.033),.95,.4,np.log(.018),np.log(.0445),2.7,.45,2.,.2,.01,.5,.004]
    p=extend_prefix(old,[np.log(.02),np.log(.058),2.3,.5],cal)
    return cal,old,p


def test_four_stage_independent_trace_and_nominal_prefix_closure():
    cal,old,p=fixture();model=FourProjectorResponse(cal);controls=np.array([[.4,.2,-.3,.5],[-.7,.5,.2,-.4]]);order=np.array([0,2,1,3]);expected=[]
    for c in controls:
        matrix=np.eye(4)
        for i in (3,2,1,0):matrix=np.asarray(drift_matrix(np.exp(p[4*i+1]))@hard_edge_solenoid_matrix(p[4*i+2]+p[4*i+3]*c[i],np.exp(p[4*i])))@matrix
        expected.append((matrix[np.ix_(order,order)]@np.asarray(model.upstream_matrix(p)))[:2]*np.array(cal.input_steps_grouped)/cal.pixel_m)
    np.testing.assert_allclose(model.predict(p,controls),expected,rtol=3e-11,atol=1e-8)
    nominal=controls.copy();nominal[:,3]=0
    np.testing.assert_allclose(model.predict(p,nominal),ThreeProjectorResponse(cal).predict(old,nominal[:,:3]),rtol=3e-11,atol=1e-8)


def test_old_rank_seventeen_new_il1_scan_rank_twenty_one():
    cal,_,p=fixture();model=FourProjectorResponse(cal)
    old=np.array([[t,u,v,0.] for t in [-.8,.4,.9] for u in [-.2,.3] for v in [-.5,0.,.6]])
    derivative=jax.jacfwd(model.predict)(jnp.asarray(p),jnp.asarray(old))
    assert fisher(derivative,np.eye(derivative.size//21))['rank']==17
    controls=np.r_[old,np.array([[.7,u,.2,v] for u in [-.2,.5] for v in [-.5,0.,.5]])]
    derivative=jax.jacfwd(model.predict)(jnp.asarray(p),jnp.asarray(controls))
    assert fisher(derivative,np.eye(derivative.size//21))['rank']==21
