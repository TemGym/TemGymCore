import numpy as np
import numba
import jax
import jax.numpy as jnp
import tqdm.auto as tqdm

from jaxgym.ray import Ray
from jaxgym.run import solve_model
from jaxgym.propagate import accumulate_transfer_matrices, propagate_rays
from jaxgym import Coords_XY

from . import components as comp


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

    input_slopes_xy = jnp.stack([theta_x_in, theta_y_in])

    return input_slopes_xy, mask


def ray_coords_at_plane(
    semi_conv: float, 
    pt_src: Coords_XY, 
    detector_coords: Coords_XY,
    total_transfer_matrix: np.ndarray, 
    det_transfer_matrix_to_specific_plane: np.ndarray,
    xp: jnp.ndarray = jnp
):
    """
    For all rays from a point source within a given semi-convergence angle, that hit the detector pixels,
    find their positions and slopes at any specified plane in the system.

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
    
    input_slopes, mask = find_input_slopes(semi_conv, pt_src, detector_coords, total_transfer_matrix)
    
    coords = propagate_rays(pt_src, input_slopes, total_transfer_matrix)

    xs, ys, dxs, dys = coords

    detector_rays = xp.stack([xs, ys, dxs, dys, xp.ones_like(xs)])
    specified_plane = xp.dot(det_transfer_matrix_to_specific_plane, detector_rays)

    specified_plane_x = specified_plane[0]
    specified_plane_y = specified_plane[1]

    return specified_plane_x, specified_plane_y, mask


def solve_model_fourdstem_wrapper(model: list, scan_pos_m: Coords_XY) -> tuple:
    # Unpack model components.
    PointSource = model[0]
    ScanGrid = model[1]
    Descanner = model[2]
    Detector = model[3]

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
    current_model = [PointSource, ScanGrid, new_Descanner, Detector]

    # Index and name the model components
    PointSource_idx, ScanGrid_idx, Descanner_idx, Detector_idx = 0, 1, 2, 3

    "VIA A SINGLE RAY AND IT'S JACOBIAN, GET THE TRANSFER MATRICES FOR THE MODEL"
    transfer_matrices = solve_model(ray, current_model)

    total_transfer_matrix = accumulate_transfer_matrices(
        transfer_matrices, PointSource_idx, Detector_idx
    )
    
    scan_grid_to_detector = accumulate_transfer_matrices(
        transfer_matrices, ScanGrid_idx, Detector_idx
    )

    detector_to_scan_grid = jnp.linalg.inv(scan_grid_to_detector)

    try:
        detector_to_scan_grid = jnp.linalg.inv(scan_grid_to_detector)
    except jnp.linalg.LinAlgError:
        print(
            "scan_grid to Detector Matrix is singular, cannot invert. Returning identity matrix."
        )
        detector_to_scan_grid = jnp.eye(5)

    return transfer_matrices, total_transfer_matrix, detector_to_scan_grid


@jax.jit
def project_frame_backward(
    model: list, 
    det_coords: np.ndarray, 
    det_frame: np.ndarray, 
    scan_pos: Coords_XY
) -> np.ndarray:
    PointSource = model[0]
    ScanGrid = model[1]
    semi_conv = PointSource.semi_conv

    # Return all the transfer matrices necessary for us to propagate rays through the system
    # We do this by propagating a single ray through the system, and finding it's gradients
    _, total_transfer_matrix, det_to_scan = solve_model_fourdstem_wrapper(
        model, scan_pos
    )

    # Get ray coordinates at the scan from the det
    scan_rays_x, scan_rays_y, semi_conv_mask = ray_coords_at_plane(
        semi_conv, scan_pos, det_coords, total_transfer_matrix, det_to_scan
    )

    det_flat = det_frame.flatten()

    # substitute -1 wherever mask is False
    scan_rays_x = jnp.where(semi_conv_mask, scan_rays_x, -1.0)
    scan_rays_y = jnp.where(semi_conv_mask, scan_rays_y, -1.0)
    det_values = jnp.where(semi_conv_mask, det_flat, -1.0)

    # Convert the ray coordinates to pixel indices.
    scan_y_px, scan_x_px = ScanGrid.metres_to_pixels([scan_rays_x, scan_rays_y])

    return scan_y_px, scan_x_px, det_values


def project_frame_forward(
    model: list,
    det_coords: np.ndarray,
    sample_interpolant: callable,
    scan_pos: Coords_XY,
) -> np.ndarray:
    PointSource = model[0]
    Detector = model[3]
    semi_conv = PointSource.semi_conv

    # Return all the transfer matrices necessary for us to propagate rays through the system
    # We do this by propagating a single ray through the system, and finding it's gradients
    _, total_transfer_matrix, detector_to_scan = solve_model_fourdstem_wrapper(
        model, scan_pos
    )

    # Get ray coordinates at the scan from the detector
    scan_rays_x, scan_rays_y, mask = ray_coords_at_plane(
        semi_conv, scan_pos, det_coords, total_transfer_matrix, detector_to_scan
    )

    # ensure mask is a JAX array of booleans
    mask = jnp.asarray(mask, dtype=bool)

    scan_pts = jnp.stack([scan_rays_y, scan_rays_x], axis=-1)  # (n_rays, 2)

    # interpolate and add 1 to avoid zero artefacts in the point image, then zero‐out invalid rays
    sample_vals = sample_interpolant(scan_pts)# + 1.0
    sample_vals = jnp.where(mask, sample_vals, 0.0)

    # compute detector pixel indices for all rays
    det_rays_x = det_coords[:, 0]
    det_rays_y = det_coords[:, 1]

    det_pixels_y, det_pixels_x = Detector.metres_to_pixels([det_rays_x, det_rays_y])

    return det_pixels_y, det_pixels_x, sample_vals


def compute_fourdstem_dataset_vmap(
    model: list, fourdstem_array: jnp.ndarray, sample_interpolant: callable
) -> jnp.ndarray:
    Detector = model[-1]
    ScanGrid = model[1]
    scan_coords = ScanGrid.coords  # shape (n_scan, 2)
    det_coords = Detector.coords  # shape (n_rays, 2)

    det_y, det_x, vals = jax.vmap(
        lambda sp: project_frame_forward(model, det_coords, sample_interpolant, sp),
        in_axes=0,
        out_axes=0,
    )(scan_coords)

    scan_idx = jnp.arange(scan_coords.shape[0])[:, None]

    fourdstem_array = fourdstem_array.at[scan_idx, det_y, det_x].set(vals)

    return fourdstem_array


def compute_fourdstem_dataset(
    model: list, fourdstem_array: np.ndarray, sample_interpolant: callable
) -> np.ndarray:
    Detector = model[-1]
    ScanGrid = model[1]
    scan_coords = ScanGrid.coords
    det_coords = Detector.coords

    for idx in tqdm.trange(fourdstem_array.shape[0], desc="Scan Y", leave=True):
        scan_pos = scan_coords[idx]
        det_pixels_y, det_pixels_x, sample_vals = project_frame_forward(
            model, det_coords, sample_interpolant, scan_pos
        )
        fourdstem_array = fourdstem_array.at[idx, det_pixels_y, det_pixels_x].set(
            sample_vals
        )

    return fourdstem_array


@numba.njit
def do_shifted_sum(
    shifted_sum_image: np.ndarray,
    flat_sample_y_px: np.ndarray,
    flat_sample_x_px: np.ndarray,
    flat_detector_intensity: np.ndarray,
) -> np.ndarray:
    height = shifted_sum_image.shape[0]
    width = shifted_sum_image.shape[1]
    n = flat_sample_y_px.shape[0]
    for i in range(n):
        y = flat_sample_y_px[i]
        x = flat_sample_x_px[i]
        if y >= 0 and y < height and x >= 0 and x < width:
            shifted_sum_image[y, x] += flat_detector_intensity[i]
    return shifted_sum_image


# def compute_scan_grid_rays_and_intensities(
#     model: list, fourdstem_array: np.ndarray
# ) -> np.ndarray:
#     ScanGrid = model[1]
#     Detector = model[-1]
#     det_coords = Detector.coords
#     scan_coords = ScanGrid.coords

#     sample_px_ys = []
#     sample_px_xs = []
#     detector_intensities = []

#     for idx in tqdm.trange(fourdstem_array.shape[0], desc="Scan Y"):
#         scan_pos = scan_coords[idx]

#         # Compute the backward projection for this scan position.
#         sample_px_y, sample_px_x, detector_intensity = project_frame_backward(
#             model, det_coords, fourdstem_array[idx], scan_pos
#         )
#         sample_px_ys.append(sample_px_y)
#         sample_px_xs.append(sample_px_x)
#         detector_intensities.append(detector_intensity)

#     return sample_px_ys, sample_px_xs, detector_intensities

