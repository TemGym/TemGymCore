from ast import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from numba import njit

from jaxgym.ray import Ray
from jaxgym.run import solve_model
from jaxgym.transfer import accumulate_transfer_matrices, transfer_rays_pt_src
from jaxgym import CoordsXY, ScaleYX
import jaxgym.components as comp

from .model import Model
from jax import lax


def find_input_slopes(
    pos: CoordsXY,
    detector_coords: CoordsXY,
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
    pt_src: CoordsXY,
    detector_coords: CoordsXY,
    total_transfer_matrix: np.ndarray,
    det_transfer_matrix_to_specific_plane: np.ndarray,
    det_px_size: ScaleYX,
):
    """
    Propagate rays from a point source through the detector and back‐project them
    onto an arbitrary plane.

    For each detector pixel coordinate, this function:
      1. Computes the input slopes from the point source that hit the detector
         within a given semi-convergence angle.
      2. Propagates those rays to the detector plane using total_transfer_matrix.
      3. Maps the resulting detector‐plane rays onto a specified plane via
         det_transfer_matrix_to_specific_plane.
      4. Computes a boolean mask of rays that both hit the detector pixels and
         satisfy the semi-convergence and pixel‐size constraints.

    Parameters
    ----------
    semi_conv : float
        Maximum semi-convergence angle (in radians) defining the allowable input slopes.
    pt_src : CoordsXY
        The (x, y) position of the point source.
    detector_coords : CoordsXY
        An (N, 2) array of (x, y) coordinates for each detector pixel.
    total_transfer_matrix : np.ndarray
        The 5×5 transfer matrix mapping rays from the source plane to the detector plane.
    det_transfer_matrix_to_specific_plane : np.ndarray
        The transfer matrix mapping detector-plane ray coordinates onto the target plane.
    det_px_size : ScaleYX
        A (pixel_y, pixel_x) tuple specifying the detector pixel dimensions.

    Returns
    -------
    specified_plane_x : jax.Array
        The x-coordinates of each ray intersection on the specified plane.
    specified_plane_y : jax.Array
        The y-coordinates of each ray intersection on the specified plane.
    detector_semi_conv_mask : jax.Array[bool]
        A boolean mask indicating which rays hit the detector pixels and
        satisfy the semi-convergence/pixel‐size criteria.
    """

    input_slopes = find_input_slopes(pt_src, detector_coords, total_transfer_matrix)

    coords = transfer_rays_pt_src(pt_src, input_slopes, total_transfer_matrix)

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
    Filter rays by their input slope so that only those intersecting detector pixels remain.

    Rays outside the semi-convergence cone (defined by semi_conv) cannot reach the detector
    and are discarded. We also compute a minimum acceptance angle

        min_alpha = (pixel_diagonal / 2) / camera_length

    where pixel_diagonal = hypot(det_px_dx, det_px_dy). If semi_conv < min_alpha, the beam
    footprint is smaller than the pixel spacing and would not fully cover any pixel
    centers except possibly one. In this edge case, multiple candidate rays may satisfy
    the slope test; we resolve this by keeping only the last valid ray to ensure a single
    pixel hit and avoid ambiguous multi-pixel selection.
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


def create_scan_pos_transfer_matrix(sp_x, sp_y, sp_tilt_x, sp_tilt_y, tm, tm_grad):
    return (
        sp_x * tm_grad[0]
        + sp_y * tm_grad[1]
        + sp_tilt_x * tm_grad[2]
        + sp_tilt_y * tm_grad[3]
        + tm
    )


def solve_model_fourdstem_wrapper(model: Model) -> tuple:
    # Unpack model components.
    pointsource = model.source
    scangrid = model.scan_grid
    descanner = model.descanner
    detector = model.detector

    ray = Ray(
        x=0.0,
        y=0.0,
        dx=0.0,
        dy=0.0,
        _one=1.0,
        z=pointsource.z,
        pathlength=jnp.zeros(1),
    )

    scan_coords = (0.0, 0.0, 0.0, 0.0)  # (scan_pos_x, scan_pos_y, scan_tilt_x, scan_tilt_y)

    def _solve_model(scan_pos, descanner, idx_one, idx_two):
        # Create a new Descanner with the current scan offsets.
        wrapped_descanner = comp.Descanner(
            z=scangrid.z,
            descan_error=descanner.descan_error,
            scan_pos_x=scan_pos[0],
            scan_pos_y=scan_pos[1],
            scan_tilt_x=scan_pos[2],
            scan_tilt_y=scan_pos[3],
        )

        # Make a new model each time:
        current_model = Model(pointsource, scangrid, wrapped_descanner, detector)

        # via a single ray and it's jacobian, get the transfer matrices for the model
        transfer_matrices = solve_model(ray, current_model)

        total_tm = accumulate_transfer_matrices(
            transfer_matrices, idx_one, idx_two
        )
        return total_tm, total_tm

    model_jac_fn = jax.jacobian(_solve_model, has_aux=True)

    total_grad_tm, total_tm = model_jac_fn(scan_coords, descanner, 0, 3)
    scangrid_to_det_grad_tm, scangrid_to_det_tm = model_jac_fn(scan_coords, descanner, 1, 3)

    return (total_tm, total_grad_tm), (scangrid_to_det_tm, scangrid_to_det_grad_tm)


@jax.jit
def project_coordinates_backward(
    model: Model,
    total_matrix_and_grad: Tuple,
    scangrid_to_det_matrix_and_grad: Tuple,
    det_coords: np.ndarray,
    scan_pos: CoordsXY
) -> np.ndarray:
    PointSource = model.source
    ScanGrid = model.scan_grid
    Detector = model.detector
    semi_conv = PointSource.semi_conv

    total_transfer_matrix = create_scan_pos_transfer_matrix(scan_pos[0],
                                                            scan_pos[1],
                                                            0,
                                                            0,
                                                            *total_matrix_and_grad)

    scan_to_det_matrix = create_scan_pos_transfer_matrix(scan_pos[0],
                                                         scan_pos[1],
                                                         0,
                                                         0,
                                                         *scangrid_to_det_matrix_and_grad)

    # Compute the inverse of the scan to detector matrix - avoiding the use of np.linalg.inv
    # which can be less accurate.
    det_to_scan_matrix = jnp.linalg.solve(
        scan_to_det_matrix,
        jnp.eye(scan_to_det_matrix.shape[0], dtype=scan_to_det_matrix.dtype)
    )

    # Get ray coordinates at the scan from the det
    scan_rays_x, scan_rays_y, detector_mask = ray_coords_at_plane(
        semi_conv,
        scan_pos,
        det_coords,
        total_transfer_matrix,
        det_to_scan_matrix,
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
