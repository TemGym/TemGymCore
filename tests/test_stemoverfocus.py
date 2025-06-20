import pytest
import numpy as np
import jax.numpy as jnp
import sympy as sp

from jaxgym.propagate import accumulate_transfer_matrices

from microscope_calibration.stemoverfocus import (
    solve_model_fourdstem_wrapper, 
    compute_fourdstem_dataset,
    find_input_slopes, 
    ray_coords_at_plane,
    compute_scan_grid_rays_and_intensities,
    do_shifted_sum
)
from microscope_calibration import components as comp
from microscope_calibration.parameters import ModelParameters

from scipy.ndimage import rotate
from scipy.ndimage import zoom

from jax.scipy.interpolate import RegularGridInterpolator

@pytest.fixture
def test_params_dict():
    return ModelParameters(
        semi_conv=0.0001,
        defocus=.001,
        camera_length=0.1,
        scan_shape=(10, 10),
        det_shape=(100, 100),
        scan_step=(0.001, 0.001),
        det_px_size=(0.001, 0.001),
        scan_rotation=0.0,
        descan_error = jnp.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
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


@pytest.fixture
def test_sample_interpolant(test_image, test_params_dict):
    
    sample_image = test_image
    model_params = test_params_dict

    scan_step = model_params['scan_step']

    # Downsample the sample_image by a factor of 2
    downsample_factor = 1.0
    sample_image = zoom(sample_image, downsample_factor)

    #This is something we can never access in the reverse model, 
    # but we can use it to make rotation of an image on the detector possible in the forward model
    sample_rotation = model_params['scan_rotation']

    sample_image = np.array(sample_image, dtype=np.complex64)
    sample_px_size = scan_step
    sample_rotated = rotate(sample_image, sample_rotation, reshape=True, cval=1.0)
    sample_rotated_edge_length_x = sample_rotated.shape[1] * sample_px_size[0]
    sample_rotated_edge_length_y = sample_rotated.shape[0] * sample_px_size[1]

    # Set up grid coordinates corresponding to the physical centers of the pixels.
    # Note: We use the rotated image’s physical edge lengths (sample_rotated_edge_length_x/y)
    # to generate coordinates that match each pixel center.
    sample_coords_x = np.linspace(-sample_rotated_edge_length_x/2,
                                +sample_rotated_edge_length_x/2,
                                sample_rotated.shape[1])

    sample_coords_y = np.linspace(-sample_rotated_edge_length_y/2,
                                +sample_rotated_edge_length_y/2,
                                sample_rotated.shape[0])

    # Flip the y-axis of the sample_rotated image because regular grid interpolator from jax cannot 
    # handle a positive to negative grid coordinate
    sample_rotated_flipped = np.flipud(sample_rotated)

    # Build the RegularGridInterpolator
    sample_interpolant = RegularGridInterpolator(
        (sample_coords_y, sample_coords_x), sample_rotated_flipped,
        method='nearest', bounds_error=False, fill_value=0.0
    )

    return sample_interpolant

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
                                            scan_pos_x=0., 
                                            scan_pos_y=0.)

    Detector = comp.Detector(z=jnp.array([params_dict['camera_length'] + params_dict['defocus']]), 
                            det_shape=params_dict['det_shape'], 
                            det_pixel_size=params_dict['det_px_size'])

    model = [PointSource, ScanGrid, Descanner, Detector]

    return model


def test_find_input_slopes_single_on_axis_pixel():
    # Test that for a single pixel on the optical axis and not descan error, the back calculted input slope
    # is zero
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

    detector = comp.Detector(z=1.0, det_shape=det_shape, det_pixel_size=det_px_size, flip_y=False)
    detector_coords = detector.coords

    input_slopes_xy, mask = find_input_slopes(semi_conv, pos, detector_coords, transfer_matrix)

    np.testing.assert_allclose(input_slopes_xy[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(input_slopes_xy[1], 0.0, atol=1e-5)


def test_find_input_slopes_sympy():

    # Test using sympy matries that the inversion of the linear function to back calculate the input slopes
    # works correctly.

    detector_coords = np.array([[0, 0]])
    pos = np.array([-0.024, 0.075])

    # proper convention: first row is [Axx, Axy, Bxx, Bxy, Δx]
    #                 second row is [Ayx, Ayy, Byx, Byy, Δy]
    A_xx, A_xy, B_xx, B_xy = 1, 0, 0.5, 0
    A_yx, A_yy, B_yx, B_yy= 0, 1, 0, 0.5
    delta_x, delta_y = pos

    transfer_matrix = np.eye(5)
    transfer_matrix[0, :] = A_xx, A_xy, B_xx, B_xy, delta_x
    transfer_matrix[1, :] = A_yx, A_yy, B_yx, B_yy, delta_y
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

    input_slopes, mask = find_input_slopes(0.001, pos, detector_coords, transfer_matrix)

    np.testing.assert_allclose(input_slopes[0], np.array([x3_solution]), rtol=1e-5)
    np.testing.assert_allclose(input_slopes[1], np.array([x4_solution]), rtol=1e-5)


def test_ray_coords_at_plane_many_coords_at_source():
    # Test many detector coordinates all map back to the same source point

    det_fwd = np.array([
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    det_back = np.linalg.inv(det_fwd)

    semi_conv = 0.02
    pt_src = jnp.array([1.0, 2.0])

    # Generate a grid of 20x20 = 400 detector coordinates
    xs = np.linspace(-1e-3, 1e-3, 20)
    ys = np.linspace(-1e-3, 1e-3, 20)
    detector_coords = jnp.array([[x, y] for x in xs for y in ys])

    x_plane, y_plane, mask = ray_coords_at_plane(
        semi_conv, pt_src, detector_coords, det_fwd, det_back
    )

    expected_x = np.full(detector_coords.shape[0], pt_src[0])
    expected_y = np.full(detector_coords.shape[0], pt_src[1])
    np.testing.assert_allclose(x_plane, expected_x, atol=1e-6)
    np.testing.assert_allclose(y_plane, expected_y, atol=1e-6)


def test_solve_model_fourdstem_wrapper(stem_model, test_params_dict):
    # Test that the transfer matrices returned by the fourdstem wrapper match the manually constructed ones

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
    prop_descan_to_det_tm[0, 2] = test_params['camera_length'] # - test_params['defocus']
    prop_descan_to_det_tm[1, 3] = test_params['camera_length'] # - test_params['defocus']
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
    np.testing.assert_allclose(transfer_matrices, manual_transfer_matrices, rtol=1e-5)


def test_same_z_components():
    # Test that if one places components at the same z position, and try to run a ray through it, 
    # it does not raise an error and returns the expected number of transfer matrices.
    semi_conv = 0.001
    ps = comp.PointSource(z=1.0, semi_conv=semi_conv)
    sg = comp.ScanGrid(z=jnp.array([1.0]), scan_step=(0.1,0.1), scan_shape=(2,2), scan_rotation=0.0)
    ds = comp.Descanner(z=jnp.array([1.0]), descan_error=jnp.zeros(12), scan_pos_x=0.0, scan_pos_y=0.0)
    dt = comp.Detector(z=jnp.array([1.0]), det_shape=(2,2), det_pixel_size=(0.1,0.1))
    model = [ps, sg, ds, dt]

    tmats, total_tm, inv_tm = solve_model_fourdstem_wrapper(model, [0.0,0.0])

    assert len(tmats) == 7
    assert total_tm.shape == (5,5)
    assert inv_tm.shape == (5,5)


def test_out_of_order_z():
    # Test that if one places components at out of order z positions, and try to run a ray through it, 
    # it does not raise an error and returns the expected number of transfer matrices.
    semi_conv = 0.001
    ps = comp.PointSource(z=3.0, semi_conv=semi_conv)
    sg = comp.ScanGrid(z=jnp.array([2.0]), scan_step=(0.1,0.1), scan_shape=(2,2), scan_rotation=0.0)
    ds = comp.Descanner(z=jnp.array([1.0]), descan_error=jnp.zeros(12), scan_pos_x=0.0, scan_pos_y=0.0)
    dt = comp.Detector(z=jnp.array([0.0]), det_shape=(2,2), det_pixel_size=(0.1,0.1))
    model = [ps, sg, ds, dt]

    tmats, total_tm, inv_tm = solve_model_fourdstem_wrapper(model, [0.0,0.0])

    assert len(tmats) == 7
    assert total_tm.shape == (5,5)
    assert inv_tm.shape == (5,5)


def test_project_frame_forward_and_backward(stem_model, test_sample_interpolant, test_image):
    import matplotlib.pyplot as plt

    sample_interpolant = test_sample_interpolant

    img = test_image
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.savefig("sample_interpolant.png")

    PointSource, ScanGrid, Descanner, Detector = stem_model

    fourdstem_array = jnp.zeros((ScanGrid.scan_shape[0]*ScanGrid.scan_shape[1], *Detector.det_shape), dtype=jnp.complex64)

    fourdstem_array = compute_fourdstem_dataset(
        stem_model, fourdstem_array, sample_interpolant
    )

    sample_px_ys, sample_px_xs, detector_intensities = compute_scan_grid_rays_and_intensities(stem_model, fourdstem_array)

    sample_px_ys = np.array(sample_px_ys, dtype=np.int32).flatten()
    sample_px_xs = np.array(sample_px_xs, dtype=np.int32).flatten()
    detector_intensities = np.array(detector_intensities, dtype=np.float32).flatten()
    shifted_sum_image = np.zeros(ScanGrid.scan_shape, dtype=np.float32)

    shifted_sum_image = do_shifted_sum(shifted_sum_image, sample_px_ys, sample_px_xs, detector_intensities)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(shifted_sum_image, cmap='gray', origin='lower')
    plt.savefig('shifted_sum_image.png')

    

