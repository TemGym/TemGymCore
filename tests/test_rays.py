import pytest
import numpy as np
import math
from jax import jacobian

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
