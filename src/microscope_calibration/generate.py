from typing_extensions import Literal
import numpy as np
import jax
from functools import partial
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import jax.numpy as jnp
import tqdm.auto as tqdm
import numba

from jaxgym import CoordsXY
from jaxgym.components import ScanGrid
from .model import Model
from .stemoverfocus import (
    ray_coords_at_plane,
    solve_model_fourdstem_wrapper,
    project_coordinates_backward,
)
from .model import ModelParameters, create_stem_model


def project_frame_forward(
    model: Model,
    det_coords: np.ndarray,
    sample_interpolant: callable,
    scan_pos: CoordsXY,
) -> np.ndarray:
    PointSource = model.source
    Detector = model.detector
    semi_conv = PointSource.semi_conv

    # Return all the transfer matrices necessary for us to propagate rays through the system
    # We do this by propagating a single ray through the system, and finding it's gradients
    _, total_transfer_matrix, detector_to_scan = solve_model_fourdstem_wrapper(
        model, scan_pos
    )

    # Get ray coordinates at the scan from the detector
    scan_rays_x, scan_rays_y, mask = ray_coords_at_plane(
        semi_conv,
        scan_pos,
        det_coords,
        total_transfer_matrix,
        detector_to_scan,
        Detector.det_pixel_size,
    )

    mask = np.asarray(mask, dtype=bool)

    sample_vals = sample_interpolant((scan_rays_y, scan_rays_x))
    sample_vals = np.where(mask, sample_vals, 0.0)

    # compute detector pixel indices for all rays
    det_rays_x = det_coords[:, 0]
    det_rays_y = det_coords[:, 1]

    det_pixels_y, det_pixels_x = Detector.metres_to_pixels([det_rays_x, det_rays_y])

    return det_pixels_y, det_pixels_x, sample_vals


def compute_fourdstem_dataset_vmap(
    model: Model, fourdstem_array: jnp.ndarray, sample_interpolant: callable
) -> jnp.ndarray:
    Detector = model.detector
    ScanGrid = model.scan_grid
    scan_coords = ScanGrid.coords  # shape (n_scan, 2)
    det_coords = Detector.coords  # shape (n_rays, 2)

    det_y, det_x, vals = jax.vmap(
        lambda sp: project_frame_forward(model, det_coords, sample_interpolant, sp),
        in_axes=0,
        out_axes=0,
    )(scan_coords)

    scan_idx = jnp.arange(scan_coords.shape[0])[:, None]

    fourdstem_array = fourdstem_array.at[scan_idx, det_y, det_x].set(vals)

    fourdstem_array = fourdstem_array.reshape(
        ScanGrid.scan_shape[0], ScanGrid.scan_shape[1], *Detector.det_shape
    )

    return fourdstem_array


def compute_fourdstem_dataset(
    model: Model, fourdstem_array: np.ndarray,
    sample_interpolant: callable, progress: bool = False,
) -> np.ndarray:
    Detector = model.detector
    ScanGrid = model.scan_grid
    scan_coords = ScanGrid.coords
    det_coords = Detector.coords

    idxs = range(np.prod(ScanGrid.scan_shape).astype(int))
    pbar = tqdm.tqdm if progress else lambda it, **kw: it

    for idx in pbar(idxs):
        iy, ix = np.unravel_index(idx, ScanGrid.scan_shape)
        scan_pos = scan_coords[idx]
        det_pixels_y, det_pixels_x, sample_vals = project_frame_forward(
            model, det_coords, sample_interpolant, scan_pos
        )
        fourdstem_array[iy, ix, det_pixels_y, det_pixels_x] = sample_vals

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


def compute_scan_grid_rays_and_intensities(
    model: Model, fourdstem_array: np.ndarray
) -> np.ndarray:
    ScanGrid = model.scan_grid
    Detector = model.detector
    det_coords = Detector.coords
    scan_coords = ScanGrid.coords

    sample_px_ys = []
    sample_px_xs = []
    detector_intensities = []

    for iy in tqdm.trange(fourdstem_array.shape[0], desc="Scan Y"):
        for ix in tqdm.trange(fourdstem_array.shape[1], desc="Scan X", leave=False):
            idx = iy * fourdstem_array.shape[1] + ix
            scan_pos = scan_coords[idx]

            # Compute the backward projection for this scan position.
            sample_px_y, sample_px_x, mask = project_coordinates_backward(
                model, det_coords, scan_pos
            )
            sample_px_ys.append(sample_px_y)
            sample_px_xs.append(sample_px_x)
            detector_intensities.append(fourdstem_array[iy, ix].ravel() * mask)

    return sample_px_ys, sample_px_xs, detector_intensities


def generate_dataset_from_image(
    params: ModelParameters,
    image: np.ndarray,
    method: Literal["nearest", "linear"] = "nearest",
    sample_scale: float = 2,
    progress: bool = False,
):
    assert method in ("nearest", "linear")
    model = create_stem_model(params)
    scan_shape = model.scan_grid.shape
    scan_step = model.scan_grid.scan_step
    grid_extent = tuple(s * scale for s, scale in zip(scan_shape, scan_step))
    image_shape = image.shape
    image_scale = tuple(
        extent / size
        for extent, size
        in zip(grid_extent, image_shape)
    )

    interpolant_grid = ScanGrid(
        z=model.scan_grid.z,
        scan_shape=image_shape,
        scan_step=image_scale,
        scan_rotation=model.scan_grid.scan_rotation,
    )
    x, y = interpolant_grid.get_coords().T
    interp_t = (
        NearestNDInterpolator
        if method == "nearest"
        else partial(LinearNDInterpolator, fill_value=1.0)
    )
    interpolant = interp_t(
        (y * sample_scale, x * sample_scale), image.flatten(),
    )

    fourdstem_array = np.zeros(
        (*model.scan_grid.scan_shape, *model.detector.det_shape),
        dtype=jnp.float32,
    )

    return compute_fourdstem_dataset(
        model,
        fourdstem_array,
        interpolant,
        progress=progress,
    )
