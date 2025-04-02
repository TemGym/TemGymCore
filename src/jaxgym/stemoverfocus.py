import numpy as np
import jax
import jax.numpy as jnp
from .ray import Ray
from . import components as comp
from .run import solve_model
from .propagate import (get_ray_coords_between_planes_from_pt_src,
                        get_ray_coords_between_planes_from_pt_src_jit,
                        accumulate_transfer_matrices)

import tqdm.auto as tqdm
from functools import partial
import numba

from . import Coords_XY

# @jax.jit
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

    detector_to_scan_grid = jnp.linalg.inv(scan_grid_to_detector)
    
    try:
        detector_to_scan_grid = jnp.linalg.inv(scan_grid_to_detector)
    except jnp.linalg.LinAlgError:
        print("scan_grid to Detector Matrix is singular, cannot invert. Returning identity matrix.")
        detector_to_scan_grid = jnp.eye(5)

    return transfer_matrices, total_transfer_matrix, detector_to_scan_grid


def map_px_on_detector_to_scan_jit(ScanGrid, Detector, detector_image,
                               ray_scan_coords_x, ray_scan_coords_y,
                               ray_det_coords_x, ray_det_coords_y):
    
    scan_pixel_ys, scan_pixel_xs = ScanGrid.metres_to_pixels([ray_scan_coords_x, ray_scan_coords_y])
    det_pixels_ys, det_pixels_xs = Detector.metres_to_pixels([ray_det_coords_x, ray_det_coords_y])

    detector_vals = detector_image[det_pixels_ys, det_pixels_xs]
    
    return scan_pixel_ys, scan_pixel_xs, detector_vals


def map_px_on_scan_to_detector_jit(Detector, sample_interpolant, 
                               ray_scan_coords_x, ray_scan_coords_y,
                               ray_det_coords_x, ray_det_coords_y):
    

    scan_pts = jnp.stack([ray_scan_coords_y, ray_scan_coords_x], axis=-1)

    # Interpolate the sample intensity at the scan coordinates.
    sample_vals = sample_interpolant(scan_pts)
    
    # Convert the ray detector coordinates to pixel indices.
    ray_det_pixel_ys, ray_det_pixel_xs = Detector.metres_to_pixels([ray_det_coords_x, ray_det_coords_y])

    return ray_det_pixel_ys, ray_det_pixel_xs, sample_vals


def map_px_on_detector_to_scan(ScanGrid, Detector, detector_image,
                               ray_scan_coords_x, ray_scan_coords_y,
                               ray_det_coords_x, ray_det_coords_y):
    
    scan_pixel_ys, scan_pixel_xs = ScanGrid.metres_to_pixels([ray_scan_coords_x, ray_scan_coords_y])
    det_pixels_ys, det_pixels_xs = Detector.metres_to_pixels([ray_det_coords_x, ray_det_coords_y])

    detector_vals = detector_image[det_pixels_ys, det_pixels_xs]
    
    return scan_pixel_ys, scan_pixel_xs, detector_vals


def map_px_on_scan_to_detector(Detector, sample_interpolant, 
                               ray_scan_coords_x, ray_scan_coords_y,
                               ray_det_coords_x, ray_det_coords_y):
    

    scan_pts = np.stack([ray_scan_coords_y, ray_scan_coords_x], axis=-1)

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
    sample_rays_x, sample_rays_y, det_rays_x, det_rays_y, mask = get_ray_coords_between_planes_from_pt_src(
        model, scan_pos, total_transfer_matrix, transfer_matrices, detector_to_sample
    )

    sample_rays_x = sample_rays_x[mask]
    sample_rays_y = sample_rays_y[mask]
    det_rays_x = det_rays_x[mask]
    det_rays_y = det_rays_y[mask]

    # Unpack model components.
    Detector = model[-1]

    # Map the detector pixel coordinates from scan grid to the detector
    det_y_px, det_x_px, sample_intensity = map_px_on_scan_to_detector(
        Detector, sample_interpolant, sample_rays_x, sample_rays_y, det_rays_x, det_rays_y
    )
    
    detector_frame[det_y_px, det_x_px] = sample_intensity

    return detector_frame


def project_frame_forward_jit(model: list, 
                              sample_interpolant: callable, 
                              detector_frame: np.ndarray, 
                              scan_pos: Coords_XY) -> np.ndarray:

    # Return all the transfer matrices necessary for us to propagate rays through the system
    transfer_matrices, total_transfer_matrix, detector_to_sample = solve_model_fourdstem_wrapper(model, scan_pos)

    # Get ray coordinates at the scan and detector
    sample_rays_x, sample_rays_y, det_rays_x, det_rays_y, mask = get_ray_coords_between_planes_from_pt_src_jit(
        model, scan_pos, total_transfer_matrix, transfer_matrices, detector_to_sample
    )

    # Unpack model components.
    Detector = model[-1]

    # Map the detector pixel coordinates from scan grid to the detector
    det_y_px, det_x_px, sample_intensity = map_px_on_scan_to_detector_jit(
        Detector, sample_interpolant, sample_rays_x, sample_rays_y, det_rays_x, det_rays_y
    )
    
    sample_intensity = jnp.where(mask, sample_intensity, 0)

    detector_frame = detector_frame.at[det_y_px, det_x_px].set(sample_intensity)

    return detector_frame

@numba.njit
def do_shifted_sum(shifted_sum_image: np.ndarray,
                   flat_sample_y_px: np.ndarray, 
                   flat_sample_x_px: np.ndarray, 
                   flat_detector_intensity: np.ndarray) -> np.ndarray:
    height = shifted_sum_image.shape[0]
    width = shifted_sum_image.shape[1]
    n = flat_sample_y_px.shape[0]
    for i in range(n):
        y = flat_sample_y_px[i]
        x = flat_sample_x_px[i]
        if y >= 0 and y < height and x >= 0 and x < width:
            shifted_sum_image[y, x] += flat_detector_intensity[i]
    return shifted_sum_image

@jax.jit
def do_shifted_sum_jit(shifted_sum_image: jnp.ndarray,
                       flat_sample_y_px: jnp.ndarray,
                       flat_sample_x_px: jnp.ndarray,
                       flat_detector_intensity: jnp.ndarray) -> jnp.ndarray:
    
    # Accumulate the detector intensities into shifted_sum_image
    return shifted_sum_image.at[flat_sample_y_px, flat_sample_x_px].add(flat_detector_intensity)


def compute_fourdstem_dataset(model: list,
                              fourdstem_array: np.ndarray,
                              sample_interpolant: callable) -> np.ndarray:
    
    Detector = model[-1]
    ScanGrid = model[1]
    scan_coords = ScanGrid.coords

    for idx in tqdm.trange(fourdstem_array.shape[0], desc='Scan Y', leave=True):
            scan_pos = scan_coords[idx]
            det_frame = np.zeros(Detector.det_shape, dtype=np.complex64)
            fourdstem_array[idx] = project_frame_forward(model, det_frame, sample_interpolant, scan_pos)

    return fourdstem_array


@jax.jit
def compute_fourdstem_dataset_jit(model: list, 
                                  fourdstem_array: jnp.ndarray, 
                                  sample_interpolant: callable) -> jnp.ndarray:

    ScanGrid = model[1]
    scan_coords = ScanGrid.coords

    def _project_frame_forward(scan_pos, det_frame):
        return project_frame_forward_jit(model, sample_interpolant, det_frame, scan_pos)
    
    # Vectorize over both scan grid axes.
    vmapped_process = jax.vmap(_project_frame_forward, in_axes=(0, 0))

    # Return a 4D stem array
    return vmapped_process(scan_coords, fourdstem_array)


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

    sample_y_px, sample_x_px, detector_intensity, mask = map_px_on_detector_to_scan(
        ScanGrid, Detector, detector_frame, sample_rays_x, sample_rays_y, det_rays_x, det_rays_y
    )
    detector_intensity_masked = np.where(mask, detector_intensity, 0)

    return sample_y_px, sample_x_px, detector_intensity_masked


def project_frame_backward_jit(model: list, 
                               detector_frame: np.ndarray, 
                               scan_pos: Coords_XY) -> np.ndarray:

    Detector = model[-1]
    ScanGrid = model[1]

    # Return all the transfer matrices necessary for us to propagate rays through the system
    # We do this by propagating a single ray through the system, and finding it's gradients
    transfer_matrices, total_transfer_matrix, detector_to_sample = solve_model_fourdstem_wrapper(model, scan_pos)

    # Get ray coordinates at the scan and detector
    sample_rays_x, sample_rays_y, det_rays_x, det_rays_y = get_ray_coords_between_planes_from_pt_src_jit(
        model, scan_pos, total_transfer_matrix, transfer_matrices, detector_to_sample
    )

    # Map the detector pixel coordinates to the scan grid
    ScanGrid = model[1]
    Detector = model[-1]

    sample_y_px, sample_x_px, detector_intensity = map_px_on_detector_to_scan_jit(
        ScanGrid, Detector, detector_frame, sample_rays_x, sample_rays_y, det_rays_x, det_rays_y
    )

    return sample_y_px, sample_x_px, detector_intensity


def collect_backprojection_coords_and_intensities(model: list,
                                                  fourdstem_array: np.ndarray) -> np.ndarray:
    
    ScanGrid = model[1]
    scan_coords = ScanGrid.coords

    sample_px_ys = []
    sample_px_xs = []
    detector_intensities = []

    for idx in tqdm.trange(fourdstem_array.shape[0], desc='Scan Y'):
            scan_pos = scan_coords[idx]
        
             # Compute the backward projection for this scan position.
            sample_px_y, sample_px_x, detector_intensity = project_frame_backward(model, fourdstem_array[idx], scan_pos)
            sample_px_ys.append(sample_px_y)
            sample_px_xs.append(sample_px_x)
            detector_intensities.append(detector_intensity)

    return sample_px_ys, sample_px_xs, detector_intensities


@jax.jit
def collect_backprojection_coords_and_intensities_jit(model: list, 
                                                      fourdstem_array: jnp.ndarray) -> np.ndarray:

    ScanGrid = model[1]
    scan_coords = ScanGrid.coords

    def _project_frame_backward(scan_pos, det_frame):
        return project_frame_backward_jit(model, det_frame, scan_pos)
    
    # Vectorize over both scan grid axes.
    vmapped_process = jax.vmap(_project_frame_backward, in_axes=(0, 0))

    # Sum over all scan points to get the total shifted sum image.
    return vmapped_process(scan_coords, fourdstem_array) # This returns sample_px_ys, sample_px_xs, detector_intensities



