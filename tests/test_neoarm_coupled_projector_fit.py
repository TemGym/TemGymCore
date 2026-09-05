from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))

from neoarm_coupled_projector_fit import (
    CoupledProjectorResponse, JointResponseProblem, stage_coordinates, encode_stages,
    grouped_stage, radial_stage, physical_upstream_coordinates, block_diagonal, information,
)
from neoarm_last_lens_physical_fit import RoundLensResponse
from temgym_core.microscope_inversion import drift_matrix, hard_edge_solenoid_matrix
from temgym_core.jax_least_squares import least_squares


@pytest.fixture
def problem_parameters():
    model = CoupledProjectorResponse(RoundLensResponse((5e-9, 5e-9, 1e-3, 1e-3), 55e-6))
    last = np.array([np.log(.02), np.log(.31), -1.4, -1.2])
    previous = np.array([np.log(.025), np.log(.05), 1.1, .55])
    p = encode_stages(last, previous, [2.3, .13, .025, .4, .005])
    controls = np.array([[t, u] for u in [-.4, 0., .4] for t in np.linspace(-1., 1., 9)])
    return model, p, controls


def test_stage_encoding_preserves_geometry_and_total_span(problem_parameters):
    model, p, _ = problem_parameters
    physical = np.asarray(stage_coordinates(p))
    np.testing.assert_allclose(np.exp(physical[[0,1,4,5]]), [.02, .31, .025, .05], rtol=5e-15)
    assert np.exp(p[0]) == pytest.approx(.405)
    assert physical[2] == pytest.approx(-1.4)
    assert physical[6] == pytest.approx(1.1)


def test_independent_two_stage_trace_and_sample_shear(problem_parameters):
    model, p, controls = problem_parameters
    physical = np.asarray(stage_coordinates(p))
    grouped = np.array([0,2,1,3])
    responses = []
    prefix = np.asarray(model.upstream_matrix(p))
    factors = np.array(model.calibration.input_steps_grouped)/model.calibration.pixel_m
    for t, u in controls:
        last = drift_matrix(np.exp(physical[1]))@hard_edge_solenoid_matrix(physical[2]+physical[3]*t, np.exp(physical[0]))
        previous = drift_matrix(np.exp(physical[5]))@hard_edge_solenoid_matrix(physical[6]+physical[7]*u, np.exp(physical[4]))
        matrix = np.asarray(last@previous)[np.ix_(grouped,grouped)]@prefix
        responses.append(matrix[:2]*factors)
    np.testing.assert_allclose(model.predict(p, controls), responses, rtol=2e-12, atol=2e-11)


def test_prefix_round_chart_preserves_physical_map(problem_parameters):
    model, p, _ = problem_parameters
    upstream = np.asarray(model.upstream_matrix(p))
    encoded = physical_upstream_coordinates(upstream, model.calibration)
    recovered = np.r_[p[:8], encoded]
    np.testing.assert_allclose(model.upstream_matrix(recovered), upstream, rtol=2e-12, atol=2e-9)


def test_jit_vmap_and_first_second_derivatives(problem_parameters):
    model, p, controls = problem_parameters
    predict = lambda value:model.predict(value, controls)
    jac = np.asarray(jax.jit(jax.jacfwd(predict))(p))
    assert np.all(np.isfinite(jax.jit(jax.jacfwd(jax.jacfwd(predict)))(p)))
    for index in [0,1,3,4,5,6,7,9,12]:
        step=np.eye(13)[index]*1e-6
        finite=(predict(p+step)-predict(p-step))/(2e-6)
        np.testing.assert_allclose(jac[...,index], finite, rtol=5e-7, atol=4e-5)
    batched=jax.jit(jax.vmap(predict))(jnp.stack([p,p]))
    np.testing.assert_allclose(batched[0],predict(p),rtol=4e-13,atol=1e-11)


def test_zero_excitation_has_finite_first_second_derivatives():
    p=jnp.array([np.log(.02),np.log(.03),0.,.3])
    assert np.all(np.isfinite(jax.jacfwd(jax.jacfwd(grouped_stage))(p,0.)))
    np.testing.assert_allclose(radial_stage(p,0.),[[1.,.05],[0.,1.]],atol=1e-15)


def test_joint_likelihood_has_no_analyser_prior_or_duplicate_data(problem_parameters):
    model,p,controls=problem_parameters
    responses=np.asarray(model.predict(p,controls))
    n=responses.size
    factor=np.eye(n)+.015*np.tril(np.ones((n,n)),-1)
    covariance=.03**2*factor@factor.T
    problem=JointResponseProblem(model,controls,responses+.001,covariance)
    residual=np.asarray(problem.residual(p))
    # Exactly one residual per original response coordinate. There are no four
    # added calibration-posterior residuals and no repeated standalone block.
    assert residual.shape==(n,)
    expected=np.linalg.solve(np.linalg.cholesky(covariance),np.full(n,-.001))
    np.testing.assert_allclose(residual,expected,rtol=1e-7,atol=2e-9)
    info=information(problem,p)
    assert info['numerical_rank']==13
    assert np.linalg.eigvalsh(info['physical_covariance']).min()>0


def test_joint_noiseless_local_recovery(problem_parameters):
    model,p,controls=problem_parameters
    observations=np.asarray(model.predict(p,controls))
    problem=JointResponseProblem(model,controls,observations,np.eye(observations.size)*.01**2)
    seed=p.copy();seed[:8]+=[.005,-.004,.003,.001,-.001,.002,.001,-.002]
    norms=np.linalg.norm(problem.jacobian(seed),axis=0)
    fit=least_squares(problem.residual,seed,bounds=(seed-.5,seed+.5),x_scale=1/norms,
        max_nfev=400,gtol=1e-8,xtol=1e-13,ftol=1e-13)
    assert fit.success
    assert np.max(np.abs(fit.residual))<1e-6
    np.testing.assert_allclose(stage_coordinates(fit.x),stage_coordinates(p),atol=1e-7)


def test_covariance_and_shapes_rejected(problem_parameters):
    model,p,controls=problem_parameters
    with pytest.raises(ValueError):model.predict(p[:12],controls)
    with pytest.raises(ValueError):model.predict(p,np.zeros((2,3)))
    with pytest.raises(ValueError):JointResponseProblem(model,controls,np.zeros((len(controls),2,4)),np.eye(3))
    blocks=block_diagonal([np.eye(2),np.eye(3)*2])
    np.testing.assert_array_equal(blocks,np.diag([1,1,2,2,2]))
