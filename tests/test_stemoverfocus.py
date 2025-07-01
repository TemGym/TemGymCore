import pytest
import numpy as np
import jax.numpy as jnp
import sympy as sp
from jaxgym.transfer import accumulate_transfer_matrices
from skimage.util import view_as_blocks

from microscope_calibration.stemoverfocus import (
    solve_model_fourdstem_wrapper,
    find_input_slopes,
    ray_coords_at_plane,
    mask_rays,
    project_coordinates_backward,
    inplace_sum
)
from microscope_calibration import components as comp
from microscope_calibration.generate import (
    compute_scan_grid_rays_and_intensities,
    do_shifted_sum,
    generate_dataset_from_image,
)
from microscope_calibration.model import (
    ModelParameters,
    DescanErrorParameters,
    create_stem_model
)


def base_model():
    return ModelParameters(
        semi_conv=1e-12,
        defocus=0.001,
        camera_length=0.1,
        scan_shape=(11, 11),
        det_shape=(11, 11),
        scan_step=(0.001, 0.001),
        det_px_size=(0.001, 0.001),
        scan_rotation=0.0,
        flip_y=False,
        descan_error=DescanErrorParameters(),
        )


def test_find_input_slopes_single_on_axis_pixel():
    # Test that for a single pixel on the optical axis
    # and no descan error, the back calculted input slope
    # is zero
    pos = jnp.array([-0.0053, 0.00515])
    shift = -pos
    camera_length = 1.0
    transfer_matrix = jnp.array(
        [
            [1.0, 0.0, camera_length, 0.0, shift[0]],
            [0.0, 1.0, 0.0, camera_length, shift[1]],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    det_shape = (1, 1)
    det_px_size = (0.0, 0.0)

    detector = comp.Detector(
        z=1.0, det_shape=det_shape, det_pixel_size=det_px_size, flip_y=False
    )
    detector_coords = detector.coords

    input_slopes_xy = find_input_slopes(pos, detector_coords, transfer_matrix)

    np.testing.assert_allclose(input_slopes_xy[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(input_slopes_xy[1], 0.0, atol=1e-5)


def test_find_input_slopes_sympy():
    # Test using sympy matries that the inversion of the linear
    # function to back calculate the input slopes
    # works correctly.

    detector_coords = np.array([[0, 0]])
    pos = np.array([-0.024, 0.075])

    # proper convention: first row is [Axx, Axy, Bxx, Bxy, Δx]
    #                 second row is [Ayx, Ayy, Byx, Byy, Δy]
    A_xx, A_xy, B_xx, B_xy = 1, 0, 0.5, 0
    A_yx, A_yy, B_yx, B_yy = 0, 1, 0, 0.5
    delta_x, delta_y = pos

    transfer_matrix = np.eye(5)
    transfer_matrix[0, :] = A_xx, A_xy, B_xx, B_xy, delta_x
    transfer_matrix[1, :] = A_yx, A_yy, B_yx, B_yy, delta_y
    transfer_matrix[2, :] = 0, 0, 1, 0, 0
    transfer_matrix[3, :] = 0, 0, 0, 1, 0
    transfer_matrix[4, :] = 0, 0, 0, 0, 1

    # Define symbols for the unknowns and known quantities.
    x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", real=True)
    y1, y2 = sp.symbols("y1 y2", real=True)  # Known value from the first row of y
    # For the last row, we know the value is 1.

    # Define the symbols for the elements of the 5x5 matrix M.
    M11, M12, M13, M14, M15 = sp.symbols("M11 M12 M13 M14 M15", real=True)
    M21, M22, M23, M24, M25 = sp.symbols("M21 M22 M23 M24 M25", real=True)

    # Define the equations corresponding to row 1 and row 5.
    eq1 = sp.Eq(M11 * x1 + M12 * x2 + M13 * x3 + M14 * x4 + M15, y1)
    eq2 = sp.Eq(M21 * x1 + M22 * x2 + M23 * x3 + M24 * x4 + M25, y2)

    M11_val, M12_val, M13_val, M14_val, M15_val = (
        transfer_matrix[0, 0],
        transfer_matrix[0, 1],
        transfer_matrix[0, 2],
        transfer_matrix[0, 3],
        transfer_matrix[0, 4],
    )
    M21_val, M22_val, M23_val, M24_val, M25_val = (
        transfer_matrix[1, 0],
        transfer_matrix[1, 1],
        transfer_matrix[1, 2],
        transfer_matrix[1, 3],
        transfer_matrix[1, 4],
    )
    x1_val, x2_val = pos
    y1_val, y2_val = detector_coords[0, 0], detector_coords[0, 1]

    # Substitute the known values into the equations.
    eq1 = eq1.subs(
        {
            M11: M11_val,
            M12: M12_val,
            M13: M13_val,
            M14: M14_val,
            M15: M15_val,
            x1: x1_val,
            x2: x2_val,
            y1: y1_val,
        }
    )
    eq2 = eq2.subs(
        {
            M21: M21_val,
            M22: M22_val,
            M23: M23_val,
            M24: M24_val,
            M25: M25_val,
            x1: x1_val,
            x2: x2_val,
            y2: y2_val,
        }
    )

    # Solve the system for x3 and x4.
    solution = sp.solve((eq1, eq2), (x3, x4))

    # Extract the solutions
    x3_solution = float(solution[x3])
    x4_solution = float(solution[x4])

    input_slopes = find_input_slopes(pos, detector_coords, transfer_matrix)

    np.testing.assert_allclose(input_slopes[0], np.array([x3_solution]), rtol=1e-5)
    np.testing.assert_allclose(input_slopes[1], np.array([x4_solution]), rtol=1e-5)


def test_ray_coords_at_plane_many_coords_at_source():
    # Test many detector coordinates all map back to the same source point

    det_fwd = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    det_back = np.linalg.inv(det_fwd)

    semi_conv = 0.02
    pt_src = jnp.array([1.0, 2.0])

    # Generate a grid of 20x20 = 400 detector coordinates
    xs, dx = np.linspace(-1e-3, 1e-3, 20, retstep=True)
    ys, dy = np.linspace(-1e-3, 1e-3, 20, retstep=True)
    detector_coords = jnp.array([[x, y] for x in xs for y in ys])

    det_px_size = (dy, dx)
    x_plane, y_plane, mask = ray_coords_at_plane(
        semi_conv, pt_src, detector_coords, det_fwd, det_back, det_px_size
    )

    expected_x = np.full(detector_coords.shape[0], pt_src[0])
    expected_y = np.full(detector_coords.shape[0], pt_src[1])
    np.testing.assert_allclose(x_plane, expected_x, atol=1e-6)
    np.testing.assert_allclose(y_plane, expected_y, atol=1e-6)


def test_mask_rays_all_valid_for_large_semi_conv():
    # Test that for a large semi_conv, all rays are valid and could be back projected.
    slopes_x = jnp.array([0.0, 1.0, 2.0])
    slopes_y = jnp.array([0.0, 0.0, 0.0])
    input_slopes = (slopes_x, slopes_y)
    # semi_conv large: all rays valid
    mask = mask_rays(
        input_slopes,
        det_px_size=(1.0, 1.0),
        camera_length=1.0,
        semi_conv=10.0,
    )
    assert mask.tolist() == [True, True, True]


def test_mask_rays_selects_only_last_true_when_semi_conv_small():
    # Test that for a small semi_conv, only the middle ray is valid.
    # This is because
    # with this very large pixel size,
    # and a semi-convergence * camera length smaller than the pixel size,
    # All the rays on the detector are inside the pixel size,
    # except the last one. Thus the first two
    # rays are valid, however we want our model to only choose
    # the last valid ray so one detector pixel is lit
    # up. This is a test for that behavior.
    slopes_x = jnp.array([0.0, 1.0, 2.0])
    slopes_y = jnp.array([0.0, 0.0, 0.0])
    input_slopes = (slopes_x, slopes_y)
    det_px_size = (2.0, 2.0)
    camera_length = 1.0
    semi_conv = 0.1
    mask = mask_rays(
        input_slopes,
        det_px_size=det_px_size,
        camera_length=camera_length,
        semi_conv=semi_conv,
    )
    # only the last valid slope remains
    assert mask.tolist() == [False, True, False]


def test_solve_model_fourdstem_wrapper():
    # Test that the transfer matrices returned by the
    # fourdstem wrapper match the manually constructed ones
    model_params = base_model()
    stem_model = create_stem_model(model_params)

    scan_pos = [-0.1, -0.1]

    transfer_matrices, total_transfer_matrix, _ = solve_model_fourdstem_wrapper(
        stem_model, scan_pos
    )

    point_source_tm = np.eye(5)
    prop_to_scan_tm = np.eye(5)
    prop_to_scan_tm[0, 2] = model_params["defocus"]
    prop_to_scan_tm[1, 3] = model_params["defocus"]
    scan_tm = np.eye(5)
    prop_scan_to_descanner_tm = np.eye(5)
    descan_tm = np.eye(5)
    descan_tm[0, -1] = -scan_pos[0]
    descan_tm[1, -1] = -scan_pos[1]
    prop_descan_to_det_tm = np.eye(5)
    prop_descan_to_det_tm[0, 2] = model_params["camera_length"]
    prop_descan_to_det_tm[1, 3] = model_params["camera_length"]
    detector_tm = np.eye(5)

    manual_transfer_matrices = [
        point_source_tm,
        prop_to_scan_tm,
        scan_tm,
        prop_scan_to_descanner_tm,
        descan_tm,
        prop_descan_to_det_tm,
        detector_tm,
    ]

    total_manual_tm = accumulate_transfer_matrices(manual_transfer_matrices, 0, 3)

    np.testing.assert_allclose(total_transfer_matrix, total_manual_tm, rtol=1e-5)
    np.testing.assert_allclose(transfer_matrices, manual_transfer_matrices, rtol=1e-5)


def test_same_z_components():
    # Test that if one places components at the same z position (zero defocus, zero camera length),
    # and try to run a ray through it,
    # it does not raise an error and returns the expected number of transfer matrices.
    err = jnp.zeros(12)
    descan_error = DescanErrorParameters(*err)

    model_params = ModelParameters(
        semi_conv=0.001,
        defocus=0.0,
        camera_length=0.0,
        scan_shape=(2, 2),
        det_shape=(2, 2),
        scan_step=(0.1, 0.1),
        det_px_size=(0.1, 0.1),
        scan_rotation=0.0,
        flip_y=False,
        descan_error=descan_error,
    )
    model = create_stem_model(model_params)
    tmats, total_tm, inv_tm = solve_model_fourdstem_wrapper(model, [0.0, 0.0])

    assert len(tmats) == 7
    assert total_tm.shape == (5, 5)
    assert inv_tm.shape == (5, 5)


def test_out_of_order_z():
    # Test that if one places components at out of order z positions,
    # and try to run a ray through it,
    # it does not raise an error and returns the expected number of transfer matrices.
    descan_error_params = DescanErrorParameters(*jnp.zeros(12))
    model_params = ModelParameters(
        semi_conv=0.001,
        defocus=0.1,
        camera_length=-0.5,
        scan_shape=(2, 2),
        det_shape=(2, 2),
        scan_step=(0.1, 0.1),
        det_px_size=(0.1, 0.1),
        scan_rotation=0.0,
        flip_y=False,
        descan_error=descan_error_params,
    )
    model = create_stem_model(model_params)
    tmats, total_tm, inv_tm = solve_model_fourdstem_wrapper(model, [0.0, 0.0])

    assert len(tmats) == 7
    assert total_tm.shape == (5, 5)
    assert inv_tm.shape == (5, 5)


@pytest.mark.parametrize("runs", range(3))
def test_project_frame_forward_and_backward_simple_sample(runs):
    # Test that the forward and backward projection of a simple sample
    scan_rotation = np.random.uniform(-180, 180)
    grid_shape = np.random.randint(8, 20, size=2)

    descan_error_params = DescanErrorParameters(*jnp.zeros(12))
    test_image = np.zeros(grid_shape, dtype=np.uint8)
    test_image[0, 0] = 1.0
    test_image[4, 4] = 1.0
    test_image[3, 4] = 1.0
    test_image[4, 3] = 1.0
    test_image[5, 4] = 1.0
    test_image[4, 5] = 1.0

    params_dict = ModelParameters(
        semi_conv=1e-4,
        defocus=0.0,
        camera_length=0.5,
        scan_shape=grid_shape,
        det_shape=grid_shape,
        scan_step=(0.01, 0.01),
        det_px_size=(0.01, 0.01),
        scan_rotation=scan_rotation,
        descan_error=descan_error_params,
        flip_y=False,
    )

    model = create_stem_model(params_dict)

    fourdstem_array = generate_dataset_from_image(params_dict, test_image, sample_scale=1.)

    sample_px_ys, sample_px_xs, detector_intensities = (
        compute_scan_grid_rays_and_intensities(model, fourdstem_array)
    )

    sample_px_ys = np.array(sample_px_ys, dtype=np.int32).flatten()
    sample_px_xs = np.array(sample_px_xs, dtype=np.int32).flatten()
    detector_intensities = np.array(detector_intensities, dtype=np.float32).flatten()

    shifted_sum_image = np.zeros(model.scan_grid.scan_shape, dtype=np.float32)

    shifted_sum_image = do_shifted_sum(
        shifted_sum_image, sample_px_ys, sample_px_xs, detector_intensities
    )

    np.testing.assert_allclose(shifted_sum_image, test_image, atol=1e-6)


@pytest.mark.parametrize("runs", range(3))
def test_project_frame_forward_and_backward_with_descan_random(runs):
    # Test that we get the same image after projecting forward and backward
    # with a random descan error matrix.
    scan_rotation = np.random.uniform(-180, 180)
    grid_shape = np.random.randint(8, 20, size=2)

    descan_error = DescanErrorParameters(*np.random.uniform(-0.01, 0.01, size=12))

    test_image = np.zeros(grid_shape, dtype=np.uint8)
    test_image[0, 0] = 1.0
    test_image[4, 4] = 1.0
    test_image[3, 4] = 1.0
    test_image[4, 3] = 1.0
    test_image[5, 4] = 1.0
    test_image[4, 5] = 1.0

    params_dict = ModelParameters(
        semi_conv=1e-4,
        defocus=0.0,
        camera_length=0.5,
        scan_shape=grid_shape,
        det_shape=grid_shape,
        scan_step=(0.01, 0.01),
        det_px_size=(0.01, 0.01),
        scan_rotation=scan_rotation,
        descan_error=descan_error,
        flip_y=False,
    )

    model = create_stem_model(params_dict)
    fourdstem_array = generate_dataset_from_image(params_dict, test_image, sample_scale=1.)

    sample_px_ys, sample_px_xs, detector_intensities = (
        compute_scan_grid_rays_and_intensities(model, fourdstem_array)
    )

    sample_px_ys = np.array(sample_px_ys, dtype=np.int32).flatten()
    sample_px_xs = np.array(sample_px_xs, dtype=np.int32).flatten()
    detector_intensities = np.array(detector_intensities, dtype=np.float32).flatten()

    shifted_sum_image = np.zeros(model.scan_grid.scan_shape, dtype=np.float32)

    shifted_sum_image = do_shifted_sum(
        shifted_sum_image, sample_px_ys, sample_px_xs, detector_intensities
    )

    np.testing.assert_allclose(shifted_sum_image, test_image, atol=1e-6)


@pytest.mark.parametrize(
    "pxo_pxi, pxo_pyi, pyo_pxi, pyo_pyi, expected_px_output", [(1.0, 0.0, 0.0, 1.0, (0, 0)),
                                                               (0.0, 1.0, 1.0, 0.0, (-1, -1)),
                                                               (0.5, 0.0, 0.0, 0.5, (3, 3))],
)
def test_project_frame_forward_and_backward_with_descan_scale(pxo_pxi,
                                                              pxo_pyi,
                                                              pyo_pxi,
                                                              pyo_pyi,
                                                              expected_px_output):
    # Test that we can predict where a single pixel will end up after the descanner with scale error
    grid_shape = (12, 12)
    scan_step = (0.01, 0.01)
    det_px_size = (0.01, 0.01)

    test_image = np.zeros(grid_shape, dtype=np.uint8)
    test_image[0, 0] = 1

    descan_error = DescanErrorParameters(pxo_pxi=pxo_pxi,
                                         pxo_pyi=pxo_pyi,
                                         pyo_pxi=pyo_pxi,
                                         pyo_pyi=pyo_pyi)

    params = ModelParameters(
        semi_conv=1e-4,
        defocus=0.0,
        camera_length=0.5,
        scan_shape=grid_shape,
        det_shape=grid_shape,
        scan_step=scan_step,
        det_px_size=det_px_size,
        scan_rotation=0.0,
        descan_error=descan_error,
        flip_y=False
    )

    fourdstem_array = generate_dataset_from_image(params, test_image, sample_scale=1.)

    expected_px_output = np.array(expected_px_output, dtype=np.int32)
    result = np.array(fourdstem_array[0, 0, expected_px_output[0], expected_px_output[1]],
                      dtype=np.uint8)

    np.testing.assert_array_equal(result, 1.0)


@pytest.mark.parametrize(
    "sxo_pxi, sxo_pyi, syo_pxi, syo_pyi, expected_px_output", [(0.0, 0.0, 0.0, 0.0, (6, 6)),
                                                               (0.2, 0.0, 0.0, 0.2, (5, 5)),
                                                               (0.0, 0.2, 0.2, 0.0, (6, 6))],
)
def test_project_frame_forward_and_backward_with_descan_slope(sxo_pxi,
                                                              sxo_pyi,
                                                              syo_pxi,
                                                              syo_pyi,
                                                              expected_px_output):
    # Test that we can predict where a single pixel will end up after the descanner with scale error
    grid_shape = (12, 12)
    scan_step = (0.01, 0.01)
    det_px_size = (0.01, 0.01)

    test_image = np.zeros(grid_shape, dtype=np.uint8)
    test_image[0, 0] = 1

    descan_error = DescanErrorParameters(sxo_pxi=sxo_pxi,
                                         sxo_pyi=sxo_pyi,
                                         syo_pxi=syo_pxi,
                                         syo_pyi=syo_pyi)

    params = ModelParameters(
        semi_conv=1e-4,
        defocus=0.0,
        camera_length=0.5,
        scan_shape=grid_shape,
        det_shape=grid_shape,
        scan_step=scan_step,
        det_px_size=det_px_size,
        scan_rotation=0.0,
        flip_y=False,
        descan_error=descan_error,
    )

    fourdstem_array = generate_dataset_from_image(params, test_image, sample_scale=1.)

    expected_px_output = np.array(expected_px_output, dtype=np.int32)
    result = np.array(fourdstem_array[0, 0, expected_px_output[0], expected_px_output[1]],
                      dtype=np.uint8)

    np.testing.assert_array_equal(result, 1.0)


@pytest.mark.parametrize(
    "offpxi, offpyi, offsxi, offsyi, expected_px_output", [(0.0, 0.0, 0.0, 0.0, (6, 6)),
                                                           (0.05, 0.01, 0.0, 0.0, (5, 11)),
                                                           (0.01, 0.05, 0.0, 0.0, (1, 6)),
                                                           (0.0, 0.0, 0.1, 0.1, (1, 11))],
)
def test_project_frame_forward_and_backward_with_descan_offset_single_pixel(offpxi,
                                                                            offpyi,
                                                                            offsxi,
                                                                            offsyi,
                                                                            expected_px_output):
    # Test that we can predict where a single pixel will end up after the descanner
    grid_shape = (12, 12)
    scan_step = (0.01, 0.01)
    det_px_size = (0.01, 0.01)

    test_image = np.zeros(grid_shape, dtype=np.uint8)
    test_image[0, 0] = 1

    descan_error = DescanErrorParameters(offpxi=offpxi,
                                         offpyi=offpyi,
                                         offsxi=offsxi,
                                         offsyi=offsyi)

    params = ModelParameters(
        semi_conv=1e-4,
        defocus=0.0,
        camera_length=0.5,
        scan_shape=grid_shape,
        det_shape=grid_shape,
        scan_step=scan_step,
        det_px_size=det_px_size,
        scan_rotation=0.0,
        descan_error=descan_error,
        flip_y=False,
    )

    fourdstem_array = generate_dataset_from_image(params, test_image, sample_scale=1.)

    expected_px_output = np.array(expected_px_output, dtype=np.int32)
    result = np.array(fourdstem_array[0, 0, expected_px_output[0], expected_px_output[1]],
                      dtype=np.uint8)

    np.testing.assert_array_equal(result, 1.0)


@pytest.mark.parametrize("scan_rotation", [0, 90, -90])
@pytest.mark.parametrize("flip_y", [False, True])
def test_scan_rotation_90_flip(scan_rotation, flip_y):

    params = ModelParameters(
        semi_conv=1,
        defocus=0.5,  # Distance from the crossover to the sample
        camera_length=0.5,  # Distance from the point source to the detector
        scan_shape=(16, 16),  # YX!
        det_shape=(32, 32),  # YX!
        scan_step=(1e-2, 1e-2),  # YX!
        det_px_size=(1e-2, 1e-2),  # YX!
        scan_rotation=scan_rotation,
        descan_error=DescanErrorParameters(),
        flip_y=flip_y,
    )

    test_image = np.random.randint(0, 8, size=params['det_shape'])

    model = create_stem_model(params)
    det_coords = model.detector.get_coords()

    scan_pos = [0.0, 0.0]  # No scan position offset

    # back-projection with wrong descan error
    px_y, px_x, mask = project_coordinates_backward(model, det_coords, scan_pos)
    scan_image = np.zeros(params['scan_shape'])

    inplace_sum(np.array(px_y), np.array(px_x), np.array(mask), test_image.ravel(), scan_image)

    if flip_y:
        test_image = np.flipud(test_image).copy()

    # Crop to even dimensions then reshape and sum over 2×2 blocks
    manual_bin_of_test_image = view_as_blocks(test_image, block_shape=(2, 2))
    manual_bin_of_test_image = np.sum(manual_bin_of_test_image, axis=(-2, -1))

    if scan_rotation == 90:
        manual_bin_of_test_image = np.rot90(manual_bin_of_test_image, k=1)
    elif scan_rotation == -90:
        manual_bin_of_test_image = np.rot90(manual_bin_of_test_image, k=-1)

    np.testing.assert_allclose(scan_image, manual_bin_of_test_image, rtol=1e-5)


@pytest.mark.parametrize("scan_rotation", [0, 45, -45])
def test_scan_rotation_45(scan_rotation):

    params = ModelParameters(
        semi_conv=1,
        defocus=0.5,  # Distance from the crossover to the sample
        camera_length=0.5,  # Distance from the point source to the detector
        scan_shape=(17, 17),  # YX!
        det_shape=(33, 33),  # YX!
        scan_step=(1e-3, 1e-3),  # YX!
        det_px_size=(1e-2, 1e-2),  # YX!
        scan_rotation=scan_rotation,
        descan_error=DescanErrorParameters(),
        flip_y=False,
    )

    test_image = np.zeros(params['det_shape'], dtype=np.uint8)
    test_image[params['det_shape'][0] // 2, :] = 1

    model = create_stem_model(params)
    det_coords = model.detector.get_coords()

    scan_pos = [0.0, 0.0]  # No scan position offset

    # back-projection with wrong descan error
    px_y, px_x, mask = project_coordinates_backward(model, det_coords, scan_pos)
    scan_image = np.zeros(params['scan_shape'])

    inplace_sum(np.array(px_y), np.array(px_x), np.array(mask), test_image.ravel(), scan_image)

    if scan_rotation == -45:
        diagonal_image = np.eye(params['scan_shape'][0], dtype=np.uint8)
    elif scan_rotation == 45:
        diagonal_image = np.fliplr(np.eye(params['scan_shape'][0], dtype=np.uint8))
    else:
        diagonal_image = scan_image.copy()

    scan_image_masked = scan_image[diagonal_image.astype(bool)].copy()
    scan_image_inv_masked = scan_image[~diagonal_image.astype(bool)].copy()

    assert np.sum(scan_image_masked) > 0.0
    assert np.sum(scan_image_inv_masked) == 0.0
