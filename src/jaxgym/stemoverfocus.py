import numpy as np
import jax
import jax.numpy as jnp
from .ray import Ray
from . import components as comp
from .run import solve_model
from .propagate import (ray_coords_at_plane,
                        propagate_rays,
                        accumulate_transfer_matrices)

import jax.numpy as jnp

import tqdm.auto as tqdm
from functools import partial
import numba

from . import Coords_XY
import jax
import line_profiler

# @jax.jit
def solve_model_fourdstem_wrapper(model: list, 
                                  scan_pos_m: Coords_XY) -> tuple:

    # Unpack model components.
    PointSource = model[0]
    ScanGrid = model[1]
    Descanner = model[2]
    Detector = model[3]

    scan_x, scan_y = scan_pos_m[0], scan_pos_m[1]

    # # Prepare input ray position for this scan point.
    # input_ray_positions = jnp.array([scan_x, scan_y, 0.0, 0.0, 1.0])
    
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

@jax.jit
def project_frame_backward(model: list, 
                           det_coords: np.ndarray,
                           det_frame: np.ndarray,
                           scan_pos: Coords_XY) -> np.ndarray:

    PointSource = model[0]
    ScanGrid = model[1]
    semi_conv = PointSource.semi_conv

    # Return all the transfer matrices necessary for us to propagate rays through the system
    # We do this by propagating a single ray through the system, and finding it's gradients
    _, total_transfer_matrix, det_to_scan = solve_model_fourdstem_wrapper(model, scan_pos)

    # Get ray coordinates at the scan from the det
    scan_rays_x, scan_rays_y, semi_conv_mask = ray_coords_at_plane(
        semi_conv, scan_pos, det_coords, total_transfer_matrix, det_to_scan
    )

    det_flat = det_frame.flatten()

    # substitute -1 wherever mask is False - remember that these values must be removed later!
    scan_rays_x = jnp.where(semi_conv_mask, scan_rays_x, -1)
    scan_rays_y = jnp.where(semi_conv_mask, scan_rays_y, -1)
    det_values = jnp.where(semi_conv_mask, det_flat, -1)

    # Convert the ray coordinates to pixel indices.
    scan_y_px, scan_x_px = ScanGrid.metres_to_pixels([scan_rays_x, scan_rays_y])

    return scan_y_px, scan_x_px, det_values


def project_frame_forward(model: list,
                          det_coords: np.ndarray,
                          sample_interpolant: callable,
                          scan_pos: Coords_XY) -> np.ndarray:
     
    PointSource = model[0]
    Detector = model[3]
    semi_conv = PointSource.semi_conv

    # Return all the transfer matrices necessary for us to propagate rays through the system
    # We do this by propagating a single ray through the system, and finding it's gradients
    _, total_transfer_matrix, detector_to_scan = solve_model_fourdstem_wrapper(model, scan_pos)

    # Get ray coordinates at the scan from the detector
    scan_rays_x, scan_rays_y, mask = ray_coords_at_plane(
        semi_conv, scan_pos, det_coords, total_transfer_matrix, detector_to_scan
    )

    # ensure mask is a JAX array of booleans
    mask = jnp.asarray(mask, dtype=bool)

    scan_pts = jnp.stack([scan_rays_y, scan_rays_x], axis=-1)

    # interpolate and add 1 to avoid zero artefacts in the point image, then zero‐out invalid rays
    sample_vals = sample_interpolant(scan_pts) + 1.0
    sample_vals = jnp.where(mask, sample_vals, 0.0)

    # compute detector pixel indices for all rays
    det_rays_x = det_coords[:, 0]
    det_rays_y = det_coords[:, 1]

    # det_pixels_y, det_pixels_x = Detector.metres_to_pixels([det_rays_x, det_rays_y])

    return det_pixels_y, det_pixels_x, sample_vals


def compute_fourdstem_dataset_vmap(model: list,
                                fourdstem_array: jnp.ndarray,
                                sample_interpolant: callable) -> jnp.ndarray:
    
    Detector  = model[-1]
    ScanGrid  = model[1]
    scan_coords = ScanGrid.coords        # shape (n_scan, 2)
    det_coords  = Detector.coords        # shape (n_rays, 2)

    det_y, det_x, vals = jax.vmap(
        lambda sp: project_frame_forward(model, det_coords, sample_interpolant, sp),
        in_axes=0, out_axes=0
    )(scan_coords)

    # build a (n_scan,1)-shaped index to broadcast into the first axis
    scan_idx = jnp.arange(scan_coords.shape[0])[:, None]  # (n_scan,1)

    # scatter each (scan_idx, det_y, det_x) := vals
    fourdstem_array = fourdstem_array.at[scan_idx, det_y, det_x].set(vals)

    return fourdstem_array


def compute_fourdstem_dataset(model: list,
                              fourdstem_array: np.ndarray,
                              sample_interpolant: callable) -> np.ndarray:
    
    Detector = model[-1]
    ScanGrid = model[1]
    scan_coords = ScanGrid.coords
    det_coords = Detector.coords

    for idx in tqdm.trange(fourdstem_array.shape[0], desc='Scan Y', leave=True):
        scan_pos = scan_coords[idx]
        det_pixels_y, det_pixels_x, sample_vals = project_frame_forward(model, det_coords, sample_interpolant, scan_pos)
        fourdstem_array = fourdstem_array.at[idx, det_pixels_y, det_pixels_x].set(sample_vals)
        #fourdstem_array[idx, det_pixels_y, det_pixels_x] = sample_vals
        break
    return fourdstem_array



def compute_scan_grid_rays_and_intensities(model: list,
                                           fourdstem_array: np.ndarray) -> np.ndarray:
    
    ScanGrid = model[1]
    Detector = model[-1]
    det_coords = Detector.coords
    scan_coords = ScanGrid.coords

    sample_px_ys = []
    sample_px_xs = []
    detector_intensities = []

    for idx in tqdm.trange(fourdstem_array.shape[0], desc='Scan Y'):
        scan_pos = scan_coords[idx]
    
        # Compute the backward projection for this scan position.
        sample_px_y, sample_px_x, detector_intensity = project_frame_backward(model, det_coords, fourdstem_array[idx], scan_pos)
        sample_px_ys.append(sample_px_y)
        sample_px_xs.append(sample_px_x)
        detector_intensities.append(detector_intensity)

    return sample_px_ys, sample_px_xs, detector_intensities


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