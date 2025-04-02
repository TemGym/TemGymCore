import pytest
import numpy as np
import jax.numpy as jnp
from jaxgym import stemoverfocus as stem
import jaxgym.components as comp
from jaxgym.ray import Ray, ray_matrix
import pytest
from jax.scipy.interpolate import RegularGridInterpolator

from scipy.ndimage import rotate
from scipy.ndimage import zoom

from jaxgym.stemoverfocus import compute_fourdstem_dataset, solve_model_fourdstem_wrapper
from jaxgym.propagate import find_input_slopes_that_hit_detpx_from_pt_src, accumulate_transfer_matrices

import matplotlib.pyplot as plt
import sympy as sp



def make_params_dict(semi_conv, defocus, camera_length, scan_shape, det_shape, scan_step, det_px_size, scan_rotation, descan_error):
    return {
        'semi_conv': semi_conv,
        'defocus': defocus,  # Distance from the crossover to the sample
        'camera_length': camera_length,  # Distance from the sample to the detector
        'scan_shape': scan_shape,  # YX!
        'det_shape': det_shape,  # YX!
        'scan_step': scan_step,  # YX!
        'det_px_size': det_px_size,  # YX!
        'scan_rotation': scan_rotation,
        'descan_error': descan_error,
    }


@pytest.fixture
def test_params_dict():
    return make_params_dict(
        semi_conv=0.001,
        defocus=.1,
        camera_length=0.456,
        scan_shape=(10, 10),
        det_shape=(5, 5),
        scan_step=(0.001, 0.001),
        det_px_size=(0.01, 0.01),
        scan_rotation=0.0,
        descan_error = jnp.array([0., 0., 0., 0., 0., 0., 0., 0.])
    )


@pytest.fixture
def test_image():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[0, 0] = 1.0
    img[0, -1] = 1.0
    img[-1, 0] = 1.0
    img[-1, -1] = 1.0
    img[4, 4] = 1.0
    return img

# Fixture that creates a STEM model with [PointSource, ScanGrid, Descanner, Detector]
@pytest.fixture
def stem_model(test_params_dict):
    params_dict = test_params_dict

    #Create ray input z plane
    crossover_z = jnp.zeros((1))

    PointSource = comp.PointSource(z=crossover_z, semi_conv=params_dict['semi_conv'])

    ScanGrid = comp.ScanGrid(z=jnp.array([params_dict['defocus']]), 
                            scan_step=params_dict['scan_step'], 
                            scan_shape=params_dict['scan_shape'], 
                            scan_rotation=params_dict['scan_rotation'])

    Descanner = comp.Descanner(z=jnp.array([params_dict['defocus']]), 
                                            descan_error=params_dict['descan_error'], 
                                            offset_x=0., 
                                            offset_y=0.)

    Detector = comp.Detector(z=jnp.array([params_dict['camera_length']]), 
                            det_shape=params_dict['det_shape'], 
                            det_pixel_size=params_dict['det_px_size'])

    model = [PointSource, ScanGrid, Descanner, Detector]

    return model


def test_find_input_slopes_that_hit_detpx_from_pt_src():
    detector_coords = np.array([[0, 0]])
    pos = np.array([-0.024, 0.075])

    A_xx, A_xy, A_yx, A_yy = np.array([1, 0, 0.5, 0])
    B_xx, B_xy, B_yx, B_yy = np.array([0, 1, 0, 0.5])
    delta_x, delta_y = pos

    transfer_matrix = np.eye(5)
    transfer_matrix[0, :] = A_xx, A_xy, A_yx, A_yy, delta_x
    transfer_matrix[1, :] = B_xx, B_xy, B_yx, B_yy, delta_y
    transfer_matrix[2, :] = 0, 0, 1, 0, 0
    transfer_matrix[3, :] = 0, 0, 0, 1, 0
    transfer_matrix[4, :] = 0, 0, 0, 0, 1

    # Define symbols for the unknowns and known quantities.
    x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4', real=True)
    y1, y2 = sp.symbols('y1 y2', real=True)  # Known value from the first row of y
    # For the last row, we know the value is 1.

    # Define the symbols for the elements of the 5x5 matrix M.
    M11, M12, M13, M14, M15 = sp.symbols('M11 M12 M13 M14 M15', real=True)
    M21, M22, M23, M24, M25 = sp.symbols('M21 M22 M23 M24 M25', real=True)

    # Define the equations corresponding to row 1 and row 5.
    eq1 = sp.Eq(M11*x1 + M12*x2 + M13*x3 + M14*x4 + M15, y1)
    eq2 = sp.Eq(M21*x1 + M22*x2 + M23*x3 + M24*x4 + M25, y2)

    M11_val, M12_val, M13_val, M14_val, M15_val = transfer_matrix[0, 0], transfer_matrix[0, 1], transfer_matrix[0, 2], transfer_matrix[0, 3], transfer_matrix[0, 4]
    M21_val, M22_val, M23_val, M24_val, M25_val = transfer_matrix[1, 0], transfer_matrix[1, 1], transfer_matrix[1, 2], transfer_matrix[1, 3], transfer_matrix[1, 4]
    x1_val, x2_val = pos
    y1_val, y2_val = detector_coords[0, 0], detector_coords[0, 1]

    # Substitute the known values into the equations.
    eq1 = eq1.subs({M11: M11_val, M12: M12_val, M13: M13_val, M14: M14_val, M15: M15_val,
                    x1: x1_val, x2: x2_val, y1: y1_val})
    eq2 = eq2.subs({M21: M21_val, M22: M22_val, M23: M23_val, M24: M24_val, M25: M25_val,
                    x1: x1_val, x2: x2_val, y2: y2_val})
    
    # Solve the system for x3 and x4.
    solution = sp.solve((eq1, eq2), (x3, x4))

    # Extract the solutions
    x3_solution = float(solution[x3])
    x4_solution = float(solution[x4])

    input_slopes, mask = find_input_slopes_that_hit_detpx_from_pt_src(detector_coords, pos, 0.001, transfer_matrix)

    np.testing.assert_allclose(input_slopes[0], np.array([x3_solution]), rtol=1e-5)
    np.testing.assert_allclose(input_slopes[1], np.array([x4_solution]), rtol=1e-5)


def test_solve_model_fourdstem_wrapper(stem_model, test_params_dict):
    test_params = test_params_dict

    scan_pos = [-0.1, -0.1]


    transfer_matrices, total_transfer_matrix, _ = solve_model_fourdstem_wrapper(stem_model, scan_pos)

    point_source_tm = np.eye(5)
    prop_to_scan_tm = np.eye(5)
    prop_to_scan_tm[0, 2] = test_params['defocus']
    prop_to_scan_tm[1, 3] = test_params['defocus']
    scan_tm = np.eye(5)
    prop_scan_to_descanner_tm = np.eye(5)
    descan_tm = np.eye(5)
    descan_tm[0, -1] = -scan_pos[0]
    descan_tm[1, -1] = -scan_pos[1]
    prop_descan_to_det_tm = np.eye(5)
    prop_descan_to_det_tm[0, 2] = test_params['camera_length'] - test_params['defocus']
    prop_descan_to_det_tm[1, 3] = test_params['camera_length'] - test_params['defocus']
    detector_tm = np.eye(5)

    manual_transfer_matrices = [point_source_tm, 
                                prop_to_scan_tm, 
                                scan_tm, 
                                prop_scan_to_descanner_tm, 
                                descan_tm, 
                                prop_descan_to_det_tm, 
                                detector_tm]

    total_manual_tm = accumulate_transfer_matrices(manual_transfer_matrices, 0, 3)

    np.testing.assert_allclose(total_transfer_matrix, total_manual_tm, rtol=1e-5)
    np.testing.assert_allclose(transfer_matrices, np.array(manual_transfer_matrices), rtol=1e-5)