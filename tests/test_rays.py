import pytest
import numpy as np
import math
import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
from jax import jacobian
import jax.numpy as jnp
from jax import value_and_grad
from temgym_core.run import run_to_end
from temgym_core.components import Detector, Deflector, Lens
from temgym_core.ray import Ray
from temgym_core.propagator import FreeSpaceParaxial, FreeSpaceDirCosine
from temgym_core.utils import custom_jacobian_matrix


def test_propagate_zero_distance():
    ray = Ray(x=1.0, y=-1.0, dx=0.5, dy=0.5, z=2.0, pathlength=1.0)
    new = FreeSpaceParaxial()(ray, 0.)
    np.testing.assert_allclose(new.x, ray.x, atol=1e-6)
    np.testing.assert_allclose(new.y, ray.y, atol=1e-6)
    np.testing.assert_allclose(new.dx, ray.dx, atol=1e-6)
    np.testing.assert_allclose(new.dy, ray.dy, atol=1e-6)
    np.testing.assert_allclose(new.z, ray.z, atol=1e-6)
    np.testing.assert_allclose(new.pathlength, ray.pathlength, atol=1e-6)


@pytest.mark.parametrize("runs", range(5))
def test_propagate_paraxial(runs):
    random_camera_length = np.random.uniform(0.1, 10.0)
    x0, y0, dx0, dy0, z0, pl0 = np.random.uniform(-10, 10, size=6)
    d = random_camera_length

    ray = Ray(x=x0, y=y0, dx=dx0, dy=dy0, z=z0, pathlength=pl0)
    new = FreeSpaceParaxial()(ray, d)
    np.testing.assert_allclose(new.x, x0 + dx0 * d, atol=1e-6)
    np.testing.assert_allclose(new.y, y0 + dy0 * d, atol=1e-6)
    np.testing.assert_allclose(new.dx, dx0, atol=1e-6)
    np.testing.assert_allclose(new.dy, dy0, atol=1e-6)
    np.testing.assert_allclose(new.z, z0 + d, atol=1e-6)
    np.testing.assert_allclose(new.pathlength, pl0 + d, atol=1e-6)


def test_propagate_gradient_no_ray_one():
    # This test ensures that tracing/differentiation through a propagation to
    # a detector does not introduce an unwanted dependency on ray._one.

    # We are checking that gradient of new_ray.z with respect to ray._one is zero.
    # This is important because ray._one is a constant carrier for adding offsets
    # into the system, and should not give gradients with respect to propagation distance.
    def z_deriv_wrapper(ray_one):
        ray = Ray(x=0.5, y=-0.5, dx=0.1, dy=-0.2, z=0.0, pathlength=0.0, _one=ray_one)
        new_ray = FreeSpaceParaxial().propagate(ray, 0.1)
        return new_ray.z

    val, grad = value_and_grad(z_deriv_wrapper)(1.0)

    # gradient should be exactly zero (no dependency on ray._one)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


def test_propagate_dir_cosine():
    x0, y0, dx0, dy0, z0, pl0 = 1.0, -1.0, 2.0, 3.0, 0.5, 10.0
    ray = Ray(x=x0, y=y0, dx=dx0, dy=dy0, z=z0, pathlength=pl0)
    d = np.random.uniform(-10, 10.0)
    # Compute expected using direction cosines
    N = math.sqrt(1.0 + dx0**2 + dy0**2)
    L = dx0 / N
    M = dy0 / N
    expected_x = x0 + (L / N) * d
    expected_y = y0 + (M / N) * d
    expected_z = z0 + d
    expected_pl = pl0 + d * N

    new = FreeSpaceDirCosine()(ray, d)
    np.testing.assert_allclose(new.x, expected_x, atol=1e-6)
    np.testing.assert_allclose(new.y, expected_y, atol=1e-6)
    np.testing.assert_allclose(new.dx, dx0, atol=1e-6)
    np.testing.assert_allclose(new.dy, dy0, atol=1e-6)
    np.testing.assert_allclose(new.z, expected_z, atol=1e-6)
    np.testing.assert_allclose(new.pathlength, expected_pl, atol=1e-6)


def test_propagate_jacobian_matrix():
    # test that gradient of the propagate function with respect to the ray input
    # is a homogeneous 5x5 matrix, where the first two rows are the translation
    # of the ray position by the distance d, and the last three rows are identity
    ray = Ray(x=0.5, y=-0.5, dx=0.1, dy=-0.2, z=1.0, pathlength=0.0)
    d = np.random.uniform(-10.0, 10.0)

    # Compute jacobian of propagate wrt ray input
    jac = jacobian(FreeSpaceParaxial(), argnums=0)(ray, d)
    J = custom_jacobian_matrix(jac)

    # Expected homogeneous 5x5 propagation matrix
    T = np.array(
        [
            [1.0, 0.0, d, 0.0, 0.0],
            [0.0, 1.0, 0.0, d, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(J, T, atol=1e-6)


def test_to_vector():
    import dataclasses
    # Check that if a ray is created with float values, and to_vector is called,
    # the output is an jnp.ndarray
    ray = Ray(x=1.0, y=2.0, dx=0.1, dy=0.2, z=3.0, pathlength=4.0)
    v = ray.to_vector()

    for k, v in dataclasses.asdict(v).items():
        assert isinstance(v, jnp.ndarray)
