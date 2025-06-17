import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jaxgym.components import ScanGrid, Detector, Descanner
from jaxgym.ray import Ray
from jaxgym.propagate import find_input_slopes
from jaxgym.components import Detector

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

    #Create ray input z plane
    crossover_z = jnp.zeros((1))

    detector = Detector(z=1.0, det_shape=det_shape, det_pixel_size=det_px_size, flip_y=False)
    detector_coords = detector.coords

    input_slopes_xy, mask = find_input_slopes(semi_conv, pos, detector_coords, transfer_matrix)

    # Check that the output slopes are 0.0
    np.testing.assert_allclose(input_slopes_xy[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(input_slopes_xy[1], 0.0, atol=1e-5)