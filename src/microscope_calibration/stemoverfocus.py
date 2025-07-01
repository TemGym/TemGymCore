import numpy as np
import jax
import jax.numpy as jnp
from numba import njit

from jaxgym.ray import Ray
from jaxgym.run import solve_model
from jaxgym.transfer import accumulate_transfer_matrices, transfer_rays
from jaxgym import Coords_XY, Scale_YX

from . import components as comp
from .model import Model
from jax import lax


def find_input_slopes(
    pos: Coords_XY,
    detector_coords: Coords_XY,
    transformation_matrix: np.ndarray,
):
    """
    Given a set of detector pixel coordinates, a semi-convergence angle from a source,
    and a transformation matrix,
    find the slopes that tells us what
    rays will hit the detector pixels from the point source.
    """
    pos_x, pos_y = pos

    A_xx, A_xy, B_xx, B_xy = transformation_matrix[
        0, :4
    ]  # Select first row excluding the last column
    A_yx, A_yy, B_yx, B_yy = transformation_matrix[
        1, :4
    ]  # Select second row excluding the last column

    delta_x, delta_y = transformation_matrix[0, 4], transformation_matrix[1, 4]

    x_out, y_out = detector_coords[:, 0], detector_coords[:, 1]

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
        +A_xx * B_yx * pos_x
        + A_xy * B_yx * pos_y
        - A_yx * B_xx * pos_x
        - A_yy * B_xx * pos_y
        - B_xx * delta_y
        + B_xx * y_out
        + B_yx * delta_x
        - B_yx * x_out
    ) / denom

    input_slopes_xy = jnp.stack([theta_x_in, theta_y_in])

    return input_slopes_xy


def ray_coords_at_plane(
    semi_conv: float,
    pt_src: Coords_XY,
    detector_coords: Coords_XY,
    total_transfer_matrix: np.ndarray,
    det_transfer_matrix_to_specific_plane: np.ndarray,
    det_px_size: Scale_YX,
):
    """
    For all rays from a point source within a given semi-convergence angle,
    that hit the detector pixels, find their positions and slopes at any
    specified plane in the system.

    Parameters:
        semi_conv (float): The maximum semiconvergence angle defining the range of input slopes.
        pt_src (Coords_XY): The (x, y) coordinates of the source point.
        detector_coords (Coords_XY): The (x, y) coordinates defining the detector pixel layout.
        total_transfer_matrix (xp.ndarray): The overall transfer matrix used to propagate rays
        from the source to the detector.
        det_transfer_matrix_to_specific_plane (xp.ndarray): The transfer matrix used to
        map detector coordinates to a specific plane.
        xp: Module, either numpy or jax.numpy.
    Returns:
        tuple:
            specified_plane_x (xp.ndarray): The x-coordinates of the rays at the specific plane.
            specified_plane_y (xp.ndarray): The y-coordinates of the rays at the specific plane.
            mask (xp.ndarray): A boolean array indicating which input slopes resulted in valid ray
                               intersections with the detector.
    """

    input_slopes = find_input_slopes(pt_src, detector_coords, total_transfer_matrix)

    coords = transfer_rays(pt_src, input_slopes, total_transfer_matrix)

    xs, ys, dxs, dys = coords

    detector_rays = jnp.stack([xs, ys, dxs, dys, jnp.ones_like(xs)])
    specified_plane = jnp.dot(det_transfer_matrix_to_specific_plane, detector_rays)

    specified_plane_x = specified_plane[0]
    specified_plane_y = specified_plane[1]

    camera_length_and_defocus_distance = (
        total_transfer_matrix[0, 2] + total_transfer_matrix[1, 3]
    ) / 2

    detector_semi_conv_mask = mask_rays(
        input_slopes, det_px_size, camera_length_and_defocus_distance, semi_conv
    )

    return specified_plane_x, specified_plane_y, detector_semi_conv_mask


def _no_op_arg(mask, idx):
    return mask


def _fill_value(mask, idx):
    return mask.at[idx].set(True)


def _select_last_ray(mask):
    rev = mask[::-1]
    idx = jnp.argmax(rev)
    last_idx = (mask.size - idx) - 1
    new_mask = jnp.zeros_like(mask)
    return lax.cond(
        mask[last_idx],
        _fill_value,
        _no_op_arg,
        new_mask,
        last_idx,
    )


def _no_op(mask):
    return mask


def mask_rays(input_slopes, det_px_size, camera_length, semi_conv):
    """
    Mask rays that have a slope which means they won't hit the detector pixels.
    Except for the case where the semi-convergence is smaller than a so-called minimum alpha,
    which is the angle between the radial distance between two detector pixels and the
    distance from the point source. If the semi-convergence is smaller than this minimum
    alpha, in theory no rays would hit the detector unless the central pixel is under
    the beam, which happens when there is an odd amount of detector pixels. In the even
    case, the beam is going inbetween pixels. This case is realised when the semi-convergence
    angle is smaller than min-alpha, and thus 4 pixels could be selected. If four pixels are
    selected, we then select only the last one in the array
    """

    det_px_dy, det_px_dx = det_px_size

    # minimum beam radius between two pixels
    min_radius = jnp.hypot(det_px_dx / 2, det_px_dy / 2) - 1e-12

    # Minimum alpha is the angle between the radial distance between
    # two detector pixels and the distance from detector to the point source.
    min_alpha = min_radius / camera_length

    theta_x, theta_y = input_slopes
    r2 = theta_x**2 + theta_y**2

    # include rays up to the larger of semi_conv or min_alpha
    mask = r2 <= jnp.maximum(semi_conv**2, min_alpha**2)
    return lax.cond(
        semi_conv > min_alpha,
        _no_op,  # true branch
        _select_last_ray,  # false branch
        mask,
    )


def solve_model_fourdstem_wrapper(model: Model, scan_pos_m: Coords_XY) -> tuple:
    # Unpack model components.
    PointSource = model.source
    ScanGrid = model.scan_grid
    Descanner = model.descanner
    Detector = model.detector

    scan_x, scan_y = scan_pos_m[0], scan_pos_m[1]

    ray = Ray(
        x=scan_x,
        y=scan_y,
        dx=0.0,
        dy=0.0,
        _one=1.0,
        z=PointSource.z,
        pathlength=jnp.zeros(1),
    )

    # Create a new Descanner with the current scan offsets.
    new_Descanner = comp.Descanner(
        z=ScanGrid.z,
        descan_error=Descanner.descan_error,
        scan_pos_x=scan_x,
        scan_pos_y=scan_y,
    )

    # Make a new model each time:
    current_model = Model(PointSource, ScanGrid, new_Descanner, Detector)

    # Index and name the model components
    PointSource_idx, ScanGrid_idx, _, Detector_idx = 0, 1, 2, 3

    # via a single ray and it's jacobian, get the transfer matrices for the model
    transfer_matrices = solve_model(ray, current_model)

    total_transfer_matrix = accumulate_transfer_matrices(
        transfer_matrices, PointSource_idx, Detector_idx
    )

    scan_grid_to_detector = accumulate_transfer_matrices(
        transfer_matrices, ScanGrid_idx, Detector_idx
    )

    detector_to_scan_grid = jnp.linalg.inv(scan_grid_to_detector)

    return transfer_matrices, total_transfer_matrix, detector_to_scan_grid


@jax.jit
def project_coordinates_backward(
    model: Model, det_coords: np.ndarray, scan_pos: Coords_XY
) -> np.ndarray:
    PointSource = model.source
    ScanGrid = model.scan_grid
    Detector = model.detector
    semi_conv = PointSource.semi_conv

    # Return all the transfer matrices necessary for us to propagate rays through the system
    # We do this by propagating a single ray through the system, and finding it's gradients
    _, total_transfer_matrix, det_to_scan = solve_model_fourdstem_wrapper(
        model, scan_pos
    )

    # Get ray coordinates at the scan from the det
    scan_rays_x, scan_rays_y, detector_mask = ray_coords_at_plane(
        semi_conv,
        scan_pos,
        det_coords,
        total_transfer_matrix,
        det_to_scan,
        Detector.det_pixel_size,
    )

    # Convert the ray coordinates to pixel indices.
    scan_y_px, scan_x_px = ScanGrid.metres_to_pixels([scan_rays_x, scan_rays_y])

    return scan_y_px, scan_x_px, detector_mask


@njit
def inplace_sum(px_y, px_x, mask, frame, buffer):
    h, w = buffer.shape
    n = px_y.size
    for i in range(n):
        py = px_y[i]
        px = px_x[i]
        if mask[i] and (0 <= px_y[i] < h) and (0 <= px_x[i] < w):
            buffer[py, px] += frame[i]


def check_diameter_on_scan_and_det(params):
    semi_conv = params["semi_conv"]
    defocus = params["defocus"]
    scan_step = params["scan_step"][0]  # Assuming square scan step
    det_px_size = params["det_px_size"][0]  # Assuming square detector pixel
    camera_length = params["camera_length"]

    scan_disk_diameter = defocus * 2 * semi_conv  # Diameter at the scan plane
    detector_disk_diameter = (
        (defocus + camera_length) * 2 * semi_conv
    )  # Radius at the detector plane

    if scan_disk_diameter < scan_step:
        Warning(
            f"Scan disk radius {scan_disk_diameter} is smaller than scan step {scan_step}. "
        )
    if detector_disk_diameter < det_px_size:
        Warning(
            f"Detector disk radius {detector_disk_diameter} "
            f"is smaller than detector pixel size {det_px_size}."
        )

    return scan_disk_diameter, detector_disk_diameter
