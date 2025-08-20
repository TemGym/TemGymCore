import jax.numpy as jnp
import numpy as np
from temgym_core.transfer import transfer_rays, transfer_rays_pt_src, accumulate_matrices


def test_transfer_pt_src_free_space():
    # Define a point source position and random ray angle
    x0, y0 = 1.0, -2.0
    angle = np.pi / 4
    dx0 = np.cos(angle)
    dy0 = np.sin(angle)
    slopes_x = jnp.array([dx0])
    slopes_y = jnp.array([dy0])
    d = 5.0

    # Free-space transfer matrix: only transfer distance d
    transfer_matrix = jnp.array(
        [
            [1.0, 0.0, d, 0.0, 0.0],
            [0.0, 1.0, 0.0, d, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    coords = transfer_rays_pt_src((x0, y0), (slopes_x, slopes_y), transfer_matrix)

    # Expected: x = x0 + d*dx0, y = y0 + d*dy0, slopes unchanged
    x_exp = x0 + d * dx0
    y_exp = y0 + d * dy0
    np.testing.assert_allclose(coords[0, 0], x_exp, atol=1e-6)
    np.testing.assert_allclose(coords[1, 0], y_exp, atol=1e-6)
    np.testing.assert_allclose(coords[2, 0], dx0, atol=1e-6)
    np.testing.assert_allclose(coords[3, 0], dy0, atol=1e-6)


def test_transfer_pt_src_random_matrix():
    # Create a reproducible random 5x5 matrix with integer entries
    rng = np.random.RandomState(np.random.randint(0, 1000))
    T_vals = rng.randint(-5, 5, size=(5, 5))
    # Ensure homogeneous coordinate row
    T_vals[4, :] = [0, 0, 0, 0, 1]

    # Define a sample ray
    x0, y0, dx0, dy0 = 0.7, -1.2, 0.3, -0.4
    slopes_x = jnp.array([dx0])
    slopes_y = jnp.array([dy0])
    T = jnp.array(T_vals, dtype=float)

    # Compute expected output using numpy instead of sympy
    vec = np.array([x0, y0, dx0, dy0, 1.0], dtype=float)
    result_np = T_vals.dot(vec)
    x_exp, y_exp, dx_exp, dy_exp = result_np[:4]

    coords = transfer_rays_pt_src((x0, y0), (slopes_x, slopes_y), T)

    # Verify against symbolic result
    np.testing.assert_allclose(coords[0, 0], x_exp, atol=1e-6)
    np.testing.assert_allclose(coords[1, 0], y_exp, atol=1e-6)
    np.testing.assert_allclose(coords[2, 0], dx_exp, atol=1e-6)
    np.testing.assert_allclose(coords[3, 0], dy_exp, atol=1e-6)


def test_transfer_pt_src_identity():
    # Batch of rays from same source through identity matrix
    x0, y0 = 0.5, -0.5
    N = 4
    dx_vals = np.random.randn(N)
    dy_vals = np.random.randn(N)
    slopes_x = jnp.array(dx_vals)
    slopes_y = jnp.array(dy_vals)
    T = jnp.eye(5)

    coords = transfer_rays_pt_src((x0, y0), (slopes_x, slopes_y), T)

    assert coords.shape == (4, N)
    np.testing.assert_allclose(coords[0], x0)
    np.testing.assert_allclose(coords[1], y0)
    np.testing.assert_allclose(coords[2], dx_vals)
    np.testing.assert_allclose(coords[3], dy_vals)


def test_transfer_pt_src_empty_input():
    # No rays
    slopes_x = jnp.array([], dtype=float)
    slopes_y = jnp.array([], dtype=float)
    T = jnp.eye(5)

    coords = transfer_rays_pt_src((1.0, 2.0), (slopes_x, slopes_y), T)
    assert coords.shape == (4, 0)


# add more exotic transfer tests for different distances
# and zero as a parametrisation
def test_negative_distance_transfer():
    x0, y0 = 1.0, 2.0
    dx0, dy0 = 0.1, -0.2
    slopes_x = jnp.array([dx0])
    slopes_y = jnp.array([dy0])
    d = -3.0
    transfer_matrix = jnp.array(
        [
            [1.0, 0.0, d, 0.0, 0.0],
            [0.0, 1.0, 0.0, d, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    coords = transfer_rays_pt_src((x0, y0), (slopes_x, slopes_y), transfer_matrix)
    x_exp = x0 + d * dx0
    y_exp = y0 + d * dy0
    np.testing.assert_allclose(coords[0, 0], x_exp, atol=1e-6)
    np.testing.assert_allclose(coords[1, 0], y_exp, atol=1e-6)


def test_transfer_rays_shape():
    # input rays of N x 5 shape
    N = 10
    M = 7
    rays = np.zeros((N, 5))  # shape (N, 5)

    # 7 input ray transfer matrices of shape (7, 5, 5)
    transfer_matrices = np.zeros((M, 5, 5))

    output_coords = transfer_rays(rays, transfer_matrices)

    # Output shape should be N, M, 5

    assert output_coords.shape == (N, M, 5), f"Expected shape (N, M, 5), got {output_coords.shape}"


def test_accumulate_transfer_matrices():
    # Define three simple homogeneous matrices
    A = np.eye(5)
    B = np.eye(5) * 2
    C = np.eye(5) * 3
    for M in (A, B, C):
        M[4, :] = [0, 0, 0, 0, 1]
    mats = [jnp.array(A), jnp.array(B), jnp.array(C)]

    total = accumulate_matrices(mats)
    expected = C @ B @ A
    np.testing.assert_allclose(np.array(total), expected)

    # Sub-range from index 1 to 2: only C
    # (since i_start=2, i_end=4 → mats[2:5] == [C])
    sub = accumulate_matrices(mats[2:3])
    expected_sub = C
    np.testing.assert_allclose(np.array(sub), expected_sub)
