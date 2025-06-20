import jax
import jax.numpy as jnp
import numpy as np
from microscope_calibration.components import ScanGrid, Detector, Descanner
from jaxgym.ray import Ray
from microscope_calibration.stemoverfocus import find_input_slopes
from microscope_calibration.components import Detector
from jaxgym.propagate import propagate_rays
import sympy as sp

jax.config.update('jax_platform_name', 'cpu')

def test_find_input_slopes():

    pos = jnp.array([-0.0053, 0.00515])
    shift = -pos
    camera_length = 1.0
    transfer_matrix = jnp.array([
        [ 1.0,  0.0,  camera_length,  0.0, shift[0]],
        [ 0.0,  1.0,  0.0,  camera_length, shift[1]],
        [ 0.0,  0.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  1.0],
    ])

    #Set up the parameters for the simulation
    semi_conv = 0.001
    det_shape = (1, 1)
    det_px_size= (0.0, 0.0)

    detector = Detector(z=1.0, det_shape=det_shape, det_pixel_size=det_px_size, flip_y=False)
    detector_coords = detector.coords

    input_slopes_xy, mask = find_input_slopes(semi_conv, pos, detector_coords, transfer_matrix)

    # Check that the output slopes are 0.0
    np.testing.assert_allclose(input_slopes_xy[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(input_slopes_xy[1], 0.0, atol=1e-5)

def test_propagate_free_space():
    # Define a point source position and random ray angle
    x0, y0 = 1.0, -2.0
    angle = np.pi / 4
    dx0 = np.cos(angle)
    dy0 = np.sin(angle)
    slopes_x = jnp.array([dx0])
    slopes_y = jnp.array([dy0])
    d = 5.0

    # Free-space transfer matrix: only propagation distance d
    transfer_matrix = jnp.array([
        [1.0, 0.0, d,   0.0, 0.0],
        [0.0, 1.0, 0.0, d,   0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    coords = propagate_rays((x0, y0), (slopes_x, slopes_y), transfer_matrix)

    # Expected: x = x0 + d*dx0, y = y0 + d*dy0, slopes unchanged
    x_exp = x0 + d * dx0
    y_exp = y0 + d * dy0
    np.testing.assert_allclose(coords[0, 0], x_exp, atol=1e-6)
    np.testing.assert_allclose(coords[1, 0], y_exp, atol=1e-6)
    np.testing.assert_allclose(coords[2, 0], dx0, atol=1e-6)
    np.testing.assert_allclose(coords[3, 0], dy0, atol=1e-6)


def test_propagate_random_matrix_with_sympy():


    # Create a reproducible random 5x5 matrix with integer entries
    rng = np.random.RandomState(2)
    T_vals = rng.randint(-5, 5, size=(5, 5))
    # Ensure homogeneous coordinate row
    T_vals[4, :] = [0, 0, 0, 0, 1]

    # Define a sample ray
    x0, y0, dx0, dy0 = 0.7, -1.2, 0.3, -0.4
    slopes_x = jnp.array([dx0])
    slopes_y = jnp.array([dy0])
    T = jnp.array(T_vals, dtype=float)

    # Symbolically compute expected output
    T_sym = sp.Matrix(T_vals)
    M_sym = sp.Matrix([x0, y0, dx0, dy0, 1])
    result_sym = T_sym * M_sym
    x_exp, y_exp, dx_exp, dy_exp = [float(result_sym[i]) for i in range(4)]

    coords = propagate_rays((x0, y0), (slopes_x, slopes_y), T)

    # Verify against symbolic result
    np.testing.assert_allclose(coords[0, 0], x_exp, atol=1e-6)
    np.testing.assert_allclose(coords[1, 0], y_exp, atol=1e-6)
    np.testing.assert_allclose(coords[2, 0], dx_exp, atol=1e-6)
    np.testing.assert_allclose(coords[3, 0], dy_exp, atol=1e-6)


