import numpy as np
import jax
import jax.numpy as jnp
from .ray import Ray
from . import components as comp
from .run import solve_model
from .propagate import (find_input_slopes_that_hit_detpx_from_pt_src, 
                        use_transfer_matrices_to_propagate_rays_from_pt_src,
                        get_ray_coords_between_planes_from_pt_src,
                        accumulate_transfer_matrices)

from . import Coords_XY

@jax.jit
def solve_model_fourdstem_wrapper(model: list, 
                                  scan_pos_m: Coords_XY) -> tuple:

    # Unpack model components.
    PointSource = model[0]
    ScanGrid = model[1]
    Descanner = model[2]
    Detector = model[3]

    scan_x, scan_y = scan_pos_m[0], scan_pos_m[1]

    # Prepare input ray position for this scan point.
    input_ray_positions = jnp.array([scan_x, scan_y, 0.0, 0.0, 1.0])

    ray = Ray(
        z=PointSource.z,
        matrix=input_ray_positions,
        pathlength=jnp.zeros(1),
    )

    # Create a new Descanner with the current scan offsets.
    new_Descanner = comp.Descanner(
        z=ScanGrid.z,
        descan_error=Descanner.descan_error,
        offset_x=scan_x,
        offset_y=scan_y
    )

    # Make a new model each time:
    current_model = [PointSource, ScanGrid, new_Descanner, Detector]

    #Index and name the model components
    PointSource_idx, ScanGrid_idx, Descanner_idx, Detector_idx = 0, 1, 2, 3

    "VIA A SINGLE RAY AND IT'S JACOBIAN, GET THE TRANSFER MATRICES FOR THE MODEL"
    transfer_matrices = solve_model(ray, current_model)

    total_transfer_matrix = accumulate_transfer_matrices(transfer_matrices, PointSource_idx, Detector_idx)
    scan_grid_to_detector = accumulate_transfer_matrices(transfer_matrices, ScanGrid_idx, Detector_idx)

    try:
        detector_to_scan_grid = jnp.linalg.inv(scan_grid_to_detector)
    except jnp.linalg.LinAlgError:
        print("scan_grid to Detector Matrix is singular, cannot invert. Returning identity matrix.")
        detector_to_scan_grid = jnp.eye(5)

    return transfer_matrices, total_transfer_matrix, detector_to_scan_grid


def map_px_on_detector_to_scan(ScanGrid, Detector, detector_image,
                               ray_scan_coords_x, ray_scan_coords_y,
                               ray_det_coords_x, ray_det_coords_y):
    
    scan_shape = ScanGrid.scan_shape_yx

    scan_pixel_ys, scan_pixel_xs = ScanGrid.metres_to_pixels([ray_scan_coords_x, ray_scan_coords_y])

    # Create a mask of indices within bounds.
    mask = (scan_pixel_ys >= 0) & (scan_pixel_ys < scan_shape[0]) & \
           (scan_pixel_xs >= 0) & (scan_pixel_xs < scan_shape[1])
    
    det_pixels_ys, det_pixels_xs = Detector.metres_to_pixels([ray_det_coords_x, ray_det_coords_y])

    detector_vals = detector_image[det_pixels_ys, det_pixels_xs]
    
    return scan_pixel_ys[mask], scan_pixel_xs[mask], detector_vals[mask]


def map_px_on_scan_to_detector(Detector, sample_interpolant, 
                               ray_scan_coords_x, ray_scan_coords_y,
                               ray_det_coords_x, ray_det_coords_y):
    
    # Stack the scan coordinates.
    scan_pts = np.stack([ray_scan_coords_x, ray_scan_coords_y], axis=-1)

    # Interpolate the sample intensity at the scan coordinates.
    sample_vals = sample_interpolant(scan_pts)
    
    # Convert the ray detector coordinates to pixel indices.
    ray_det_pixel_ys, ray_det_pixel_xs = Detector.metres_to_pixels([ray_det_coords_x, ray_det_coords_y])

    return ray_det_pixel_ys, ray_det_pixel_xs, sample_vals


def project_frame_forward(model: list, 
                          detector_frame: np.ndarray, 
                          sample_interpolant: callable, 
                          scan_pos: Coords_XY) -> np.ndarray:

    # Return all the transfer matrices necessary for us to propagate rays through the system
    transfer_matrices, total_transfer_matrix, detector_to_sample = solve_model_fourdstem_wrapper(model, scan_pos)

    # Get ray coordinates at the scan and detector
    sample_rays_x, sample_rays_y, det_rays_x, det_rays_y = get_ray_coords_between_planes_from_pt_src(
        model, scan_pos, total_transfer_matrix, transfer_matrices, detector_to_sample
    )

    # Unpack model components.
    Detector = model[-1]

    # Map the detector pixel coordinates from scan grid to the detector
    det_y_px, det_x_px, sample_intensity = map_px_on_scan_to_detector(
        Detector, sample_interpolant, sample_rays_x, sample_rays_y, det_rays_x, det_rays_y
    )
    
    detector_frame[det_y_px, det_x_px] = sample_intensity

    return detector_frame


def project_frame_backward(model: list, 
                           detector_frame: np.ndarray, 
                           scan_pos: Coords_XY) -> np.ndarray:

    # Return all the transfer matrices necessary for us to propagate rays through the system
    # We do this by propagating a single ray through the system, and finding it's gradients
    transfer_matrices, total_transfer_matrix, detector_to_sample = solve_model_fourdstem_wrapper(model, scan_pos)

    # Get ray coordinates at the scan and detector
    sample_rays_x, sample_rays_y, det_rays_x, det_rays_y = get_ray_coords_between_planes_from_pt_src(
        model, scan_pos, total_transfer_matrix, transfer_matrices, detector_to_sample
    )

    # Map the detector pixel coordinates to the scan grid
    ScanGrid = model[1]
    Detector = model[-1]

    sample_y_px, sample_x_px, detector_intensity = map_px_on_detector_to_scan(
        ScanGrid, Detector, detector_frame, sample_rays_x, sample_rays_y, det_rays_x, det_rays_y
    )

    shifted_sum = np.zeros(ScanGrid.scan_shape_yx, dtype=np.complex64)
    np.add.at(shifted_sum, (sample_y_px, sample_x_px), detector_intensity)

    return shifted_sum
