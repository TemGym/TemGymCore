import jax.numpy as jnp
import numpy as np
import jax
import pytest
import sympy as sp

from temgym_core.transfer_matrices import (
    accumulate_matrices,
    calculate_z1_and_z2_from_M_and_f,
    imaging_matrix_5x5,
    lens_matrix_5x5,
    propagation_matrix_5x5,
    transfer_rays,
    transfer_rays_pt_src,
)


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


def test_transfer_visits_planes_in_physical_order_and_has_jax_derivatives():
    # A lens and drift do not commute: check physical coordinates at every plane.
    ray = jnp.array([[2., -1., .1, -.2, 1.]])

    def trace(f):
        matrices = jnp.stack((
            propagation_matrix_5x5(3., xp=jnp),
            lens_matrix_5x5(f, xp=jnp),
            propagation_matrix_5x5(2., xp=jnp),
        ))
        return transfer_rays(ray, matrices)

    f = 4.
    first_position = np.array([2.3, -1.6])
    first_slopes = np.array([.1, -.2])
    exit_slopes = first_slopes - first_position / f
    expected = np.array([
        [*first_position, *first_slopes, 1.],
        [*first_position, *exit_slopes, 1.],
        [*(first_position + 2 * exit_slopes), *exit_slopes, 1.],
    ])
    np.testing.assert_allclose(trace(f)[0], expected, atol=1e-14)
    np.testing.assert_allclose(jax.jit(trace)(f)[0], expected, atol=1e-14)
    derivative = np.asarray(jax.jacfwd(trace)(f))[0, -1]
    expected_derivative = np.r_[2 * first_position / f**2, first_position / f**2, 0.]
    np.testing.assert_allclose(derivative, expected_derivative, atol=1e-14)


@pytest.mark.parametrize('xp', [sp, np, jnp])
def test_imaging_matrix_matches_a_lens_between_its_object_and_image_planes(xp):
    magnification, focal_length = -3, sp.Rational(1, 5) if xp is sp else .2
    object_z, image_z = calculate_z1_and_z2_from_M_and_f(magnification, focal_length)
    actual = (
        propagation_matrix_5x5(image_z, xp=xp)
        @ lens_matrix_5x5(focal_length, xp=xp)
        @ propagation_matrix_5x5(-object_z, xp=xp)
    )
    expected = imaging_matrix_5x5(magnification, focal_length, xp=xp)
    np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float), atol=1e-14)


def test_default_matrices_are_numpy_and_need_no_sympy():
    """A clean install without SymPy imports the module and builds numerical matrices."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        sys.modules["sympy"] = None  # any import of sympy now raises ImportError
        import numpy as np
        from temgym_core import transfer_matrices as tm

        matrix = tm.lens_matrix_5x5(2.0)
        assert isinstance(matrix, np.ndarray) and matrix.shape == (5, 5)
        try:
            tm.full_abcd_2lens((1, 2, 3), 1, 1, symbolic=True)
        except ImportError as error:
            assert "symbolic" in str(error)
        else:
            raise SystemExit("expected an ImportError explaining the symbolic extra")
    """)
    subprocess.run([sys.executable, "-c", code], check=True)


def test_sympy_module_still_gives_symbolic_matrices():
    sympy = pytest.importorskip("sympy")
    f = sympy.Symbol("f")
    matrix = lens_matrix_5x5(f, xp=sympy)
    assert isinstance(matrix, sympy.Matrix)
    assert matrix[2, 0] == -1 / f
