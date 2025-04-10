import numpy as np
import jax.numpy as jnp
from . import Coords_XY
from functools import partial
import numpy as np
import jax.numpy as jnp


def ray_coords_at_plane(
    semi_conv: float, 
    pt_src: Coords_XY, 
    detector_coords: Coords_XY,
    total_transfer_matrix: NDArray, 
    det_transfer_matrix_to_specific_plane: NDArray,
    xp
):
    """
    For all rays from a point source within a given semi-convergence angle, that hit the detector pixels,
    find their positions at slopes any any specified plane in the system.

    Parameters:
        semi_conv (float): The maximum semiconvergence angle defining the range of input slopes.
        pt_src (Coords_XY): The (x, y) coordinates of the source point.
        detector_coords (Coords_XY): The (x, y) coordinates defining the detector pixel layout.
        total_transfer_matrix (xp.ndarray): The overall transfer matrix used to propagate rays from the source to the detector.
        det_transfer_matrix_to_specific_plane (xp.ndarray): The transfer matrix used to map detector coordinates 
                                                            to a specific plane.
        xp: Module, either numpy or jax.numpy.
    Returns:
        tuple:
            specified_plane_x (xp.ndarray): The x-coordinates of the rays at the specific plane.
            specified_plane_y (xp.ndarray): The y-coordinates of the rays at the specific plane.
            mask (xp.ndarray): A boolean array indicating which input slopes resulted in valid ray intersections 
                               with the detector.
    """
    
    # Find input slopes and mask (assumes find_input_slopes has been adapted similarly to handle xp)
    input_slopes, mask = find_input_slopes(semi_conv, pt_src, detector_coords, total_transfer_matrix)
    
    # Propagate rays (assumes propagate_rays has been adapted similarly to handle xp)
    coords = propagate_rays(pt_src, input_slopes, total_transfer_matrix)

    xs, ys, dxs, dys = coords

    # Create homogeneous coordinate array using xp equivalents
    detector_rays = xp.stack([xs, ys, dxs, dys, xp.ones_like(xs)])
    specified_plane = xp.dot(det_transfer_matrix_to_specific_plane, detector_rays)

    specified_plane_x = specified_plane[0]
    specified_plane_y = specified_plane[1]

    return specified_plane_x, specified_plane_y, mask

# Create partial function versions to choose between numpy and jax.numpy easily:
ray_coords_at_plane_np = partial(ray_coords_at_plane, xp=np)
ray_coords_at_plane_jnp = partial(ray_coords_at_plane, xp=jnp)


def propagate_rays(input_pos_xy, input_slopes_xy, transfer_matrix):
    """
    Propagate rays through an optical system using the provided transfer matrix.
    This function takes an initial point source position (x, y) and their corresponding slopes, constructs a ray vector, 
    and propagates these rays through the system by applying the transfer matrix. The output is a set of propagated ray coordinates.
    Parameters
    ----------
    input_pos_xy : tuple
        A tuple (input_pos_x, input_pos_y) representing the x and y coordinates of the source position.
    input_slopes_xy : tuple
        A tuple (input_slopes_x, input_slopes_y) representing the slopes of the rays in the x and y directions.
    transfer_matrix : numpy.ndarray
        A matrix used to propagate the rays. It should be compatible with the constructed ray vector so that the dot product results in a propagated coordinate array.
    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (4, N) where N is the number of rays. The rows correspond to:
            - x positions
            - y positions
            - x slopes (dxs)
            - y slopes (dys)
    """


    # Given an input pt_source position and slopes, propagate the rays through the system
    input_pos_x, input_pos_y = input_pos_xy
    input_slopes_x, input_slopes_y = input_slopes_xy

    # Make the input rays we can run through one last time in the model to find positions at sample and detector
    rays_at_source_with_semi_conv = np.vstack([
        np.full(input_slopes_x.shape[0], input_pos_x),
        np.full(input_slopes_y.shape[0], input_pos_y),
        input_slopes_x,
        input_slopes_y,
        np.ones_like(input_slopes_x)
    ])

    # Propagate the point source coordinates through the forward ABCD matrices
    coord_list = [rays_at_source_with_semi_conv]
    end_coords = np.dot(transfer_matrix, coord_list[-1])
        
    xs = end_coords[0, :]
    ys = end_coords[1, :]
    dxs = end_coords[2, :]
    dys = end_coords[3, :]

    coords = np.array([xs, ys, dxs, dys])
    
    return coords


def find_input_slopes(
    semi_conv: float,
    pos: Coords_XY,
    detector_coords: Coords_XY,
    transformation_matrix: np.ndarray
):
    """
    Given a set of detector pixel coordinates, a semi-convergence angle from a source, and a transformation matrix,
    find the slopes and mask that tells us what slopes will hit the detector pixels from the point source.
    """
    pos_x, pos_y = pos

    A_xx, A_xy, B_xx, B_xy = transformation_matrix[0, :4]  # Select first row excluding the last column
    A_yx, A_yy, B_yx, B_yy = transformation_matrix[1, :4]  # Select second row excluding the last column

    delta_x, delta_y = transformation_matrix[0, 4], transformation_matrix[1, 4]

    x_out, y_out = detector_coords[:, 0], detector_coords[:, 1]

    denom = B_xx * B_yy - B_xy * B_yx
    theta_x_in = (
        - A_xx * B_yy * pos_x
        - A_xy * B_yy * pos_y
        + A_yx * B_xy * pos_x
        + A_yy * B_xy * pos_y
        + B_xy * delta_y
        - B_xy * y_out
        - B_yy * delta_x
        + B_yy * x_out
    ) / denom

    theta_y_in = (
        + A_xx * B_yx * pos_x
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

    input_slopes_xy = np.stack([theta_x_in, theta_y_in])

    return input_slopes_xy, mask


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