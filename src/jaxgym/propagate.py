import numpy as np


def get_ray_coords_between_planes_from_pt_src(model, 
                                              pt_src_pos_yx, 
                                              total_transfer_matrix, 
                                              all_transfer_matrices, 
                                              det_transfer_matrix_to_specific_plane):
    
    # Note the first model component must be a pt source!

    PointSource = model[0]
    Detector = model[3]
    detector_coords_yx = Detector.coords
    semi_conv = PointSource.semi_conv 

    # Find all input slopes for a max semiconvergence angle that will hit the detector pixels
    input_slopes_yx, mask = find_input_slopes_that_hit_detpx_from_pt_src(
        detector_coords_yx, pt_src_pos_yx, semi_conv, total_transfer_matrix
    )
    
    input_slopes_yx_masked = input_slopes_yx[mask]

    # Run the model to obtain the ray coordinates at each component in the model
    coords = use_transfer_matrices_to_propagate_rays_from_pt_src(all_transfer_matrices, pt_src_pos_yx, input_slopes_yx_masked)

    # Stack coordinates and perform the inverse matrix multiplication to get the sample coordinates
    xs, ys, dxs, dys = coords
    detector_rays = np.stack([xs[-1], ys[-1], dxs[-1], dys[-1], np.ones_like(xs[-1])])
    specified_plane = np.dot(det_transfer_matrix_to_specific_plane, detector_rays)

    # Unpack the sample and detector ray coordinates
    specified_plane_x = specified_plane[0]
    specified_plane_y = specified_plane[1]
    det_rays_x = detector_rays[0]
    det_rays_y = detector_rays[1]

    return specified_plane_x, specified_plane_y, det_rays_x, det_rays_y


def use_transfer_matrices_to_propagate_rays_from_pt_src(transfer_matrices, input_pos_yx, input_slopes_yx):
    # Given an input pt_source position and slopes, propagate the rays through the system
    input_pos_y, input_pos_x = input_pos_yx
    input_slopes_y, input_slopes_x = input_slopes_yx

    # Make the input rays we can run through one last time in the model to find positions at sample and detector
    rays_at_source_with_semi_conv = np.vstack([
        np.full((input_slopes_x.shape[0]), input_pos_x),
        np.full((input_slopes_y.shape[0]), input_pos_y),
        input_slopes_x,
        input_slopes_y,
        np.ones_like(input_slopes_x)
    ])

    # Propagate the point source coordinates through the forward ABCD matrices
    coord_list = [rays_at_source_with_semi_conv]
    for ABCD in transfer_matrices:
        new_coord = np.dot(ABCD, coord_list[-1])
        coord_list.append(new_coord)
        
    # Stack the propagated coordinates into an array for easier indexing
    coords_array = np.stack(coord_list, axis=0)
    
    xs = coords_array[:, 0, :]
    ys = coords_array[:, 1, :]
    dxs = coords_array[:, 2, :]
    dys = coords_array[:, 3, :]

    coords = np.array([xs, ys, dxs, dys])
    
    return coords


def find_input_slopes_that_hit_detpx_from_pt_src(
    detector_coords_yx, pos_yx, semi_conv, transformation_matrix
):
    """
    Given a set of detector pixel coordinates, a semi-convergence angle from a source, and a transformation matrix,
    find a mask that tells us what slopes will hit the detector pixels from the point source.
    """
    pos_y, pos_x = pos_yx

    A_xx, A_xy, B_xx, B_xy = transformation_matrix[0, :4]  # Select first row excluding the last column
    A_yx, A_yy, B_yx, B_yy = transformation_matrix[1, :4]  # Select second row excluding the last column

    delta_x, delta_y = transformation_matrix[0, 4], transformation_matrix[1, 4]

    y_out, x_out = detector_coords_yx[:, 0], detector_coords_yx[:, 1]

    denom = B_xx * B_yy - B_xy * B_yx
    theta_x_in = (
        -A_xx * B_yy * pos_x
        - A_xy * B_yy * pos_y
        + A_yx * B_xy * pos_x
        + A_yy * B_xy * pos_y
        + B_xy * delta_y
        - B_xy * y_out
        - B_yy * delta_x
        + B_yy * x_out
    ) / denom

    theta_y_in = (
        A_xx * B_yx * pos_x
        + A_xy * B_yx * pos_y
        - A_yx * B_xx * pos_x
        - A_yy * B_xx * pos_y
        - B_xx * delta_y
        + B_xx * y_out
        + B_yx * delta_x
        - B_yx * x_out
    ) / denom

    F = (theta_x_in**2 + theta_y_in**2) - semi_conv**2
    mask = F <= 0
    input_slopes_yx = np.stack([theta_y_in, theta_x_in])

    return input_slopes_yx, mask


def accumulate_transfer_matrices(transfer_matrices, start: int, end: int):
    """Compute the total transfer matrix between component indices [start, end] 
    by multiplying in right-to-left order.

    Given that the transfer_matrices list contains both component and intermediate propagation matrices,
    where each component appears at even indices (0, 2, 4, ...),
    this function multiplies the matrices from index 2*start up to index 2*end using reversed order.
    """
    i_start = 2 * start
    i_end = 2 * end
    matrices = transfer_matrices[i_start: i_end + 1]
    total = matrices[-1]
    for tm in reversed(matrices[:-1]):
        total = total @ tm
    return total