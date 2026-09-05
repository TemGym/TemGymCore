import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.jax_least_squares import least_squares


def test_badly_scaled_linear_physical_coordinates():
    scale = np.array([1e-12, 1e6, 1e-3])
    matrix = jnp.array([[2., 1., 0.], [0., 3., 1.], [1., 0., 4.], [1., -1., 1.]])
    target = jnp.array([1.2, -.7, .9])
    residual = lambda x: matrix @ (x / scale - target)
    result = least_squares(residual, np.zeros(3), bounds=(-5 * scale, 5 * scale),
                           x_scale=scale)
    assert result.success
    np.testing.assert_allclose(result.x / scale, target, atol=1e-8)
    np.testing.assert_allclose(result.jacobian, matrix / scale, rtol=1e-12)
    assert result.cost < 1e-16
    assert result.x.device == jax.devices()[0]


def test_binding_bounds_have_projected_gradient_and_no_infeasible_calls():
    def residual(x):
        physical = jnp.array([x[0] - 3., 2. * (x[1] + 4.), x[2] - .3])
        return jnp.where(jnp.all(jnp.abs(x) <= 1.), physical, jnp.nan)
    result = least_squares(residual, jnp.zeros(3), bounds=(-1., 1.))
    assert result.success
    np.testing.assert_allclose(result.x, [1., -1., .3], atol=1e-8)
    assert result.optimality < 1e-8
    assert result.cost == pytest.approx(20.)


def test_rank_deficient_and_underdetermined_directions_are_not_frozen():
    result = least_squares(lambda x: jnp.array([x[0] + x[1] - 2.]),
                           jnp.array([.1, -.1, .7]), bounds=(-10., 10.))
    assert result.success
    np.testing.assert_allclose(result.x[:2], [1.1, .9], atol=1e-8)
    assert result.x[2] == pytest.approx(.7)
    assert result.cost < 1e-16


def test_rosenbrock_nonlinear_residual_and_step_rejection():
    def residual(x):
        return jnp.array([10. * (x[1] - x[0] ** 2), 1. - x[0]])
    result = least_squares(residual, [-1.2, 1.], bounds=(-3., 3.),
                           initial_damping=1e-7, max_nfev=300)
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [1., 1.], atol=2e-7)
    assert any(not row["accepted"] for row in result.history)
    assert result.nfev <= 300


@pytest.mark.parametrize("bounds", [(-np.inf, 1.), (0., np.inf), (1., 1.),
                                     (2., 1.), ([0., 1.], [1., 2.]), (0., 1., 2.)])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="bounds"):
        least_squares(lambda x: x, [.5], bounds=bounds)


def test_out_of_bounds_initial_is_not_silently_clipped():
    with pytest.raises(ValueError, match="inside bounds"):
        least_squares(lambda x: x, [2.], bounds=(-1., 1.))


@pytest.mark.parametrize("scale", [0., -1., np.nan, np.inf, [1., 2.]])
def test_invalid_scales(scale):
    with pytest.raises(ValueError, match="x_scale"):
        least_squares(lambda x: x, [.5], bounds=(-1., 1.), x_scale=scale)


@pytest.mark.parametrize("options,status", [({"max_nfev": 1}, "evaluation_limit"),
                                            ({"max_iter": 0}, "iteration_limit"),
                                            ({"max_nfev": 2}, "evaluation_limit")])
def test_limits_are_not_reported_as_convergence(options, status):
    result = least_squares(lambda x: jnp.exp(x) - 10., [0.], bounds=(-5., 5.),
                           **options)
    assert not result.success
    assert result.status == status
    if "max_nfev" in options:
        assert result.nfev <= options["max_nfev"]
    np.testing.assert_allclose(result.jacobian, jax.jacfwd(lambda x: jnp.exp(x) - 10.)(result.x))


def test_nonfinite_initial_residual_and_jacobian_fail_honestly():
    result = least_squares(lambda x: jnp.log(x), [-1.], bounds=(-2., 2.))
    assert not result.success
    assert result.status == "nonfinite_initial"
    assert result.njev == 0
    result = least_squares(lambda x: jnp.sqrt(x) - 1., [0.], bounds=(-2., 2.))
    assert not result.success
    assert result.status == "nonfinite_jacobian"


def test_nonfinite_trial_is_rejected_and_can_recover():
    # The first Newton step from x=4 crosses log's domain boundary.  Rejection
    # must retain the current finite point and try more damping.
    result = least_squares(lambda x: jnp.log(x), [4.], bounds=(-5., 5.),
                           initial_damping=1e-7)
    assert result.success
    np.testing.assert_allclose(result.x, [1.], atol=1e-8)
    assert any(not np.isfinite(row["actual_reduction"]) for row in result.history)


def test_small_rejected_step_does_not_claim_convergence():
    # Deliberately inconsistent derivative simulates a faulty forward model.
    @jax.custom_jvp
    def residual(x):
        return jnp.ones_like(x)
    @residual.defjvp
    def derivative(primals, tangents):
        return residual(primals[0]), tangents[0]
    result = least_squares(residual, [0.], bounds=(-1., 1.), max_iter=60)
    assert not result.success
    assert result.status == "stalled"
    assert all(not row["accepted"] for row in result.history)


@pytest.mark.parametrize("residual", [lambda x: jnp.array(1.), lambda x: jnp.array([]),
                                      lambda x: x.astype(complex)])
def test_real_vector_residual_contract(residual):
    with pytest.raises(ValueError, match="real vector"):
        least_squares(residual, [.5], bounds=(-1., 1.))
