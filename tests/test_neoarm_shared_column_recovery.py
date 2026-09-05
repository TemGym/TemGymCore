from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
import neoarm_shared_column_recovery as recovery

jax.config.update("jax_enable_x64", True)


class DiagonalProjector:
    pixel_m = 55e-6

    def matrix(self, p, u):
        scale = jnp.exp(p[0]+u[3])
        return jnp.diag(jnp.array([scale, 1/scale, scale, 1/scale]))

    def predict(self, p, controls):
        matrices = jax.vmap(lambda u: self.matrix(p, u))(controls)
        return matrices[:, [0, 2]][:, :, [0, 2, 1, 3]]/self.pixel_m


def test_physical_endpoint_rows_keep_coordinate_order_and_free_projectors():
    model = DiagonalProjector()
    controls = jnp.zeros((2, 7)).at[:, 3].set(jnp.array([-.2, .2]))
    endpoint = recovery.PhysicalEndpointRows(model, controls)
    p = jnp.zeros(16)
    actual = np.asarray(endpoint.canonical_rows(p))
    np.testing.assert_allclose(actual[:, :, :2], np.exp(np.array([-.2, .2]))[:, None, None]*np.eye(2))
    np.testing.assert_array_equal(actual[:, :, 2:], np.zeros((2, 2, 2)))
    np.testing.assert_allclose(endpoint.response(p), model.predict(p, controls))
    derivative = np.asarray(jax.jacfwd(endpoint.canonical_rows)(p))
    np.testing.assert_allclose(derivative[..., 0], actual)
    assert not np.any(derivative[..., 1:])
    with pytest.raises(ValueError, match="two seven-control"):
        recovery.PhysicalEndpointRows(model, jnp.zeros((3, 7)))


def test_reference_coordinate_permutation_releases_all_39_coordinates():
    chart = SimpleNamespace(encode=lambda p: p*2, decode=lambda p: p/2)
    encode, decode, outer = recovery.source_coordinates(chart)
    p = jnp.arange(39.)
    assert outer == 14
    np.testing.assert_array_equal(decode(encode(p)), p)
    np.testing.assert_allclose(jax.jacfwd(lambda x: decode(encode(x)))(p), np.eye(39), atol=1e-14)


def test_original_observations_and_one_prior_are_concatenated_once(monkeypatch):
    monkeypatch.setattr(recovery, "joint_training_residual",
        lambda *args: lambda p: jnp.r_[jnp.arange(118.), p[22]/.005])
    monkeypatch.setattr(recovery, "make_residual",
        lambda *args: lambda p: jnp.arange(96.)+1000.+p[0])
    joint = SimpleNamespace(endpoints=SimpleNamespace(projectors=object()))
    fun = recovery.shared_residual(joint, None, None, None, None, None)
    p = jnp.zeros(39).at[22].set(.005).at[23].set(2.)
    result = np.asarray(fun(p))
    assert result.shape == (215,)
    np.testing.assert_array_equal(result[:118], np.arange(118.))
    np.testing.assert_array_equal(result[118:-1], np.arange(96.)+1002.)
    assert result[-1] == 1.
    derivative = np.asarray(jax.jacfwd(fun)(p))
    assert np.count_nonzero(derivative[:, 22]) == 1
    assert np.count_nonzero(derivative[:, 23]) == 96
