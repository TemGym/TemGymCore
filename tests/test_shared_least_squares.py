import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.shared_least_squares import LocalResidual, fit_shared_least_squares


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
