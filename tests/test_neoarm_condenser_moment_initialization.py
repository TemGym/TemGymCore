import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_condenser_moment_initialization import MomentSourceProjection, observable_to_moments
from neoarm_condenser_covariance_fit import BASIS, SOURCE_SCALE
from neoarm_condenser_image_recovery import covariance_observable, rotation


def fixture():
    rng = np.random.default_rng(52)
    maps = np.zeros((4, 2, 4, 4)); aperture = np.zeros((2, 4, 4))
    maps[:, :, [0, 2], :] = rng.normal(size=(4, 2, 2, 4))*55e-6/SOURCE_SCALE
    aperture[:, [0, 2], :] = rng.normal(size=(2, 2, 4))*1e-5/SOURCE_SCALE
    maps, aperture = jnp.asarray(maps), jnp.asarray(aperture)
    model = SimpleNamespace(
        maps=lambda p, u: maps[u[0].astype(int)]*jnp.exp(.03*p[0]*(1+u[0])),
        condenser_matrix=lambda p, u, to_aperture: aperture[u[0].astype(int)]*jnp.exp(.02*p[0]*(1+u[0])))
    quotient = SimpleNamespace(model=model, decode=lambda p: (p[:12],))
    coefficients = np.array([3., 4., 5., 6., .1, .2, .3, .4, .05])
    covariance = np.einsum('k,kij->ij', coefficients, BASIS)*np.outer(SOURCE_SCALE, SOURCE_SCALE)
    bp = maps@covariance@maps.swapaxes(-1, -2)
    beam = np.asarray(jax.vmap(covariance_observable)(bp.reshape(-1,4,4)[:, jnp.array([0,2])][:,:,jnp.array([0,2])]/55e-6**2)).reshape(4,2,3)
    ac = aperture@covariance@aperture.swapaxes(-1, -2)
    ap = np.asarray(jax.vmap(covariance_observable)(ac[:,jnp.array([0,2])][:,:,jnp.array([0,2])]))
    controls = np.zeros((4,7)); controls[:,0] = np.arange(4)
    acontrols = np.zeros((2,7)); acontrols[:,0] = np.arange(2)
    measurement_cov = np.array([[2., .1, -.2], [.1, 1., .05], [-.2, .05, 3.]])*1e-6
    guide = MomentSourceProjection(quotient, controls, beam, np.tile(measurement_cov,(4,2,1,1)),
        acontrols, ap, np.tile(measurement_cov,(2,1,1)))
    return guide, coefficients


def test_second_moment_mapping_and_jacobian_are_analytic():
    p = jnp.array([np.log(2.), np.log(3.), np.arctanh(.4)])
    np.testing.assert_allclose(observable_to_moments(p), [4., 2.4, 9.])
    np.testing.assert_allclose(jax.jacfwd(observable_to_moments)(p),
        [[8.,0.,0.], [2.4,2.4,6*(1-.4**2)], [0.,18.,0.]], atol=1e-14)


def test_weighted_qr_recovers_all_nine_visible_coefficients():
    guide, expected = fixture()
    coefficients, residual, singular = jax.jit(guide.projected)(jnp.zeros(14))
    np.testing.assert_allclose(coefficients, expected, rtol=1e-11, atol=1e-11)
    assert len(singular) == 9 and singular[-1] > singular[0]*1e-12
    np.testing.assert_allclose(residual, 0., atol=1e-11)
    assert len(guide.residual(jnp.zeros(14))) == 31  #30 data coordinates +one prior.


def test_variable_projection_derivative_matches_directional_difference():
    guide, _ = fixture()
    p = jnp.zeros(14).at[0].set(.003).at[12].set(.002)
    direction = jnp.zeros(14).at[0].set(.4).at[12].set(-.3).at[13].set(.02)
    derivative = np.asarray(jax.jvp(guide.residual, (p,), (direction,))[1])
    step = 1e-5
    difference = (np.asarray(guide.residual(p+step*direction))-np.asarray(guide.residual(p-step*direction)))/(2*step)
    np.testing.assert_allclose(derivative, difference, rtol=1e-7, atol=2e-6)
    assert derivative[-1] == .02/.005


def test_source_coefficients_are_unconstrained_only_in_the_guide():
    guide, _ = fixture()
    assert guide.weighted_design(jnp.zeros(14)).shape == (30,9)
    full = guide.full_parameters(jnp.zeros(14))
    assert full.shape == (23,)
    np.testing.assert_allclose(full[:12], 0.)
    np.testing.assert_allclose(full[21:], 0.)


def test_rank_deficiency_is_rejected_instead_of_dropping_source_columns():
    guide, _ = fixture()
    guide.weighted_design = lambda p: jnp.ones((30,9))
    assert not np.isfinite(np.asarray(guide.residual(jnp.zeros(14)))).all()
