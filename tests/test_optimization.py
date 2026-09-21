import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.optimization import (
    LocalResidual,
    fit_shared_least_squares,
    least_squares,
)


def test_badly_scaled_linear_physical_coordinates():
    scale = np.array([1e-12, 1e6, 1e-3])
    matrix = jnp.array(
        [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0], [1.0, -1.0, 1.0]]
    )
    target = jnp.array([1.2, -0.7, 0.9])
    residual = lambda x: matrix @ (x / scale - target)
    result = least_squares(
        residual, np.zeros(3), bounds=(-5 * scale, 5 * scale), x_scale=scale
    )
    assert result.success
    np.testing.assert_allclose(result.x / scale, target, atol=1e-8)
    np.testing.assert_allclose(result.jacobian, matrix / scale, rtol=1e-12)
    assert result.cost < 1e-16
    assert result.x.device == jax.devices()[0]


def test_binding_bounds_have_projected_gradient_and_no_infeasible_calls():
    def residual(x):
        physical = jnp.array([x[0] - 3.0, 2.0 * (x[1] + 4.0), x[2] - 0.3])
        return jnp.where(jnp.all(jnp.abs(x) <= 1.0), physical, jnp.nan)

    result = least_squares(residual, jnp.zeros(3), bounds=(-1.0, 1.0))
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, -1.0, 0.3], atol=1e-8)
    assert result.optimality < 1e-8
    assert result.cost == pytest.approx(20.0)


def test_rank_deficient_and_underdetermined_directions_are_not_frozen():
    result = least_squares(
        lambda x: jnp.array([x[0] + x[1] - 2.0]),
        jnp.array([0.1, -0.1, 0.7]),
        bounds=(-10.0, 10.0),
    )
    assert result.success
    np.testing.assert_allclose(result.x[:2], [1.1, 0.9], atol=1e-8)
    assert result.x[2] == pytest.approx(0.7)
    assert result.cost < 1e-16


def test_rosenbrock_nonlinear_residual_and_step_rejection():
    def residual(x):
        return jnp.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

    result = least_squares(
        residual, [-1.2, 1.0], bounds=(-3.0, 3.0), initial_damping=1e-7, max_nfev=300
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=2e-7)
    assert any(not row["accepted"] for row in result.history)
    assert result.nfev <= 300


@pytest.mark.parametrize(
    "bounds",
    [
        (-np.inf, 1.0),
        (0.0, np.inf),
        (1.0, 1.0),
        (2.0, 1.0),
        ([0.0, 1.0], [1.0, 2.0]),
        (0.0, 1.0, 2.0),
    ],
)
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="bounds"):
        least_squares(lambda x: x, [0.5], bounds=bounds)


def test_out_of_bounds_initial_is_not_silently_clipped():
    with pytest.raises(ValueError, match="inside bounds"):
        least_squares(lambda x: x, [2.0], bounds=(-1.0, 1.0))


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf, [1.0, 2.0]])
def test_invalid_scales(scale):
    with pytest.raises(ValueError, match="x_scale"):
        least_squares(lambda x: x, [0.5], bounds=(-1.0, 1.0), x_scale=scale)


@pytest.mark.parametrize(
    "options,status",
    [
        ({"max_nfev": 1}, "evaluation_limit"),
        ({"max_iter": 0}, "iteration_limit"),
        ({"max_nfev": 2}, "evaluation_limit"),
    ],
)
def test_limits_are_not_reported_as_convergence(options, status):
    result = least_squares(
        lambda x: jnp.exp(x) - 10.0, [0.0], bounds=(-5.0, 5.0), **options
    )
    assert not result.success
    assert result.status == status
    if "max_nfev" in options:
        assert result.nfev <= options["max_nfev"]
    np.testing.assert_allclose(
        result.jacobian, jax.jacfwd(lambda x: jnp.exp(x) - 10.0)(result.x)
    )


def test_nonfinite_initial_residual_and_jacobian_fail_honestly():
    result = least_squares(lambda x: jnp.log(x), [-1.0], bounds=(-2.0, 2.0))
    assert not result.success
    assert result.status == "nonfinite_initial"
    assert result.njev == 0
    result = least_squares(lambda x: jnp.sqrt(x) - 1.0, [0.0], bounds=(-2.0, 2.0))
    assert not result.success
    assert result.status == "nonfinite_jacobian"


def test_nonfinite_trial_is_rejected_and_can_recover():
    # The first Newton step from x=4 crosses log's domain boundary.  Rejection
    # must retain the current finite point and try more damping.
    result = least_squares(
        lambda x: jnp.log(x), [4.0], bounds=(-5.0, 5.0), initial_damping=1e-7
    )
    assert result.success
    np.testing.assert_allclose(result.x, [1.0], atol=1e-8)
    assert any(not np.isfinite(row["actual_reduction"]) for row in result.history)


def test_small_rejected_step_does_not_claim_convergence():
    # Deliberately inconsistent derivative simulates a faulty forward model.
    @jax.custom_jvp
    def residual(x):
        return jnp.ones_like(x)

    @residual.defjvp
    def derivative(primals, tangents):
        return residual(primals[0]), tangents[0]

    result = least_squares(residual, [0.0], bounds=(-1.0, 1.0), max_iter=60)
    assert not result.success
    assert result.status == "stalled"
    assert all(not row["accepted"] for row in result.history)


@pytest.mark.parametrize(
    "residual",
    [lambda x: jnp.array(1.0), lambda x: jnp.array([]), lambda x: x.astype(complex)],
)
def test_real_vector_residual_contract(residual):
    with pytest.raises(ValueError, match="real vector"):
        least_squares(residual, [0.5], bounds=(-1.0, 1.0))


def test_jacobian_scaling_and_supplied_derivative_preserve_physical_solution():
    units = jnp.array([1e-12, 1e6, 1e-3])

    def residual(x):
        z = x / units
        return jnp.array([10 * (z[1] - z[0] ** 2), 1 - z[0], z[2] - 0.7])

    result = least_squares(
        residual,
        units * jnp.array([-1.2, 1.0, 0.0]),
        bounds=(-3 * units, 3 * units),
        x_scale="jac",
        jac=jax.jacfwd(residual),
        max_nfev=300,
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x / units, [1.0, 1.0, 0.7], atol=1e-6)
    np.testing.assert_allclose(result.jacobian, jax.jacfwd(residual)(result.x))


def test_jacobian_scaling_retains_unobserved_parameters():
    result = least_squares(
        lambda x: jnp.array([1e6 * (x[0] - 2)]),
        [0.0, 0.7],
        bounds=(-3.0, 3.0),
        x_scale="jac",
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [2.0, 0.7], atol=1e-8)


def test_host_scheduled_blocks_match_compiled_nonlinear_fit():
    def model(x):
        return jnp.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])

    block, block_jac = jax.jit(model), jax.jit(jax.jacfwd(model))

    def scheduled(x):
        # A real host scheduler may inspect shapes/values and synchronize each
        # compiled block to bound memory. It must not be traced by the solver.
        assert np.isfinite(np.asarray(x)).all()
        return block(x)

    result = least_squares(
        scheduled,
        [-1.2, 1.0],
        jac=block_jac,
        jit=False,
        bounds=(-3.0, 3.0),
        x_scale="jac",
        max_nfev=300,
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result.jacobian, jax.jacfwd(model)(result.x))


def test_block_solution_and_covariance_match_dense_joint_problem():
    def common(s):
        return jnp.array([2 * (s[0] - 0.4)])

    def residual(l, s, data):
        return jnp.array([l[0] + s[0] - data[0], 2 * l[0] - s[0] - data[1]])

    blocks = [
        LocalResidual(residual, (jnp.array([1.2, 1.1]),)),
        LocalResidual(residual, (jnp.array([-0.5, -1.9]),)),
    ]
    r = fit_shared_least_squares(
        shared_residual=common,
        blocks=blocks,
        starting_shared=[0.0],
        starting_local=[[0.0], [0.0]],
        shared_bounds=([-10.0], [10.0]),
    )

    def dense(x):
        return jnp.concatenate(
            (
                common(x[:1]),
                residual(x[1:2], x[:1], blocks[0].args[0]),
                residual(x[2:], x[:1], blocks[1].args[0]),
            )
        )

    a = np.asarray(jax.jacfwd(dense)(jnp.zeros(3)))
    b = -np.asarray(dense(jnp.zeros(3)))
    truth = np.linalg.lstsq(a, b, rcond=None)[0]
    cov = np.linalg.inv(a.T @ a)
    assert r.success
    np.testing.assert_allclose(np.r_[r.shared, r.local.ravel()], truth, atol=1e-12)
    np.testing.assert_allclose(r.shared_covariance, cov[:1, :1], atol=1e-12)
    response = np.asarray(r.local_responses).reshape(2, 1)
    recovered = (
        np.diag(np.asarray(r.local_covariances).ravel())
        + response @ r.shared_covariance @ response.T
    )
    np.testing.assert_allclose(recovered, cov[1:, 1:], atol=1e-12)


def test_geodesic_correction_follows_a_weak_curved_valley():
    # The weak second measurement leaves a long, curved valley. A noise-level
    # gradient threshold can stop near its entrance despite a large x error.
    def residual(x, target):
        return jnp.array([20 * (x[1] - x[0] ** 2), 1e-5 * (x[0] - target)])

    loose = least_squares(
        residual,
        [2.0, 4.0],
        args=(0.7,),
        bounds=([-3.0, -10.0], [3.0, 10.0]),
        x_scale="jac",
        gtol=1e-5,
    )
    assert loose.success
    assert abs(float(loose.x[0]) - 0.7) > 0.5
    tight = least_squares(
        residual,
        loose.x,
        args=(0.7,),
        bounds=([-3.0, -10.0], [3.0, 10.0]),
        x_scale="jac",
        geodesic=True,
        gtol=1e-14,
        ftol=1e-14,
        xtol=1e-13,
        max_nfev=1000,
        max_iter=2000,
    )
    assert tight.success, tight.message
    np.testing.assert_allclose(tight.x, [0.7, 0.49], atol=2e-6)
    assert any(h["accepted"] and h["accelerated"] for h in tight.history)
    np.testing.assert_allclose(
        tight.jacobian, jax.jacfwd(residual)(tight.x, 0.7), atol=1e-10
    )


def test_geodesic_correction_respects_binding_bounds():
    def residual(x):
        valid = jnp.all((x >= 0.0) & (x <= 1.0))
        return jnp.where(
            valid, jnp.array([jnp.exp(x[0]) - 4, x[1] ** 2 - 0.25]), jnp.nan
        )

    result = least_squares(
        residual,
        [0.2, 0.8],
        bounds=(0.0, 1.0),
        geodesic=True,
        gtol=1e-10,
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [1.0, 0.5], atol=1e-7)
    assert all(np.isfinite(h["actual_reduction"]) for h in result.history)


def test_exact_residual_target_is_distinct_from_a_small_gradient():
    result = least_squares(
        lambda x: jnp.array([1e-7 * (x[0] - 1.0)]),
        [0.0],
        bounds=(-2.0, 2.0),
        gtol=0.0,
        ftol=0.0,
        xtol=0.0,
        residual_tol=1e-12,
        x_scale="jac",
    )
    assert result.success
    assert result.status == "residual_tol"
    assert np.max(np.abs(result.residual)) <= 1e-12
    np.testing.assert_allclose(result.x, [1.0], atol=1e-5)
