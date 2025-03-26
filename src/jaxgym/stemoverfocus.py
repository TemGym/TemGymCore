import numpy as np
import jax
import jax.numpy as jnp
from .ray import Ray
from . import components as comp
from .run import run_model_for_jacobians

# This is the only function where it's unavoidable to use jax, as we are getting the jacobian of the model
@jax.jit
def get_transfer_matrices(model, scan_pos_m):

    # Unpack model components.
    PointSource = model[0]
    ScanGrid = model[1]
    Descanner = model[2]
    Detector = model[3]

    scan_y, scan_x = scan_pos_m[0], scan_pos_m[1]

    # Prepare input ray position for this scan point.
    input_ray_positions = jnp.array([scan_x, scan_y, 0.0, 0.0, 1.0])

    ray = Ray(
        z=PointSource.z,
        matrix=input_ray_positions,
        amplitude=jnp.ones(1),
        pathlength=jnp.zeros(1),
        wavelength=jnp.ones(1),
        blocked=jnp.zeros(1, dtype=float)
    )

    # Create a new Descanner with the current scan offsets.
    new_Descanner = comp.Descanner(
        z=ScanGrid.z,
        descan_error=Descanner.descan_error,
        offset_x=scan_x,
        offset_y=scan_y
    )

    current_model = [PointSource, ScanGrid, new_Descanner, Detector]

    transfer_matrices = run_model_for_jacobians(ray, current_model)

    total_transfer_matrix = transfer_matrices[-1]
    
    for tm in reversed(transfer_matrices[:-1]):
        total_transfer_matrix = total_transfer_matrix @ tm
    
    sample_to_detector = transfer_matrices[2] @ transfer_matrices[1]

    try:
        detector_to_sample = jnp.linalg.inv(sample_to_detector)
    except jnp.linalg.LinAlgError:
        print("Sample to Detector Matrix is singular, cannot invert. Returning identity matrix.")
        detector_to_sample = jnp.eye(5)

    return transfer_matrices, total_transfer_matrix, detector_to_sample


def map_px_on_detector_to_scan(ScanGrid, Detector, detector_image,
                               ray_scan_coords_x, ray_scan_coords_y,
                               ray_det_coords_x, ray_det_coords_y):
    
    scan_shape = ScanGrid.scan_shape_yx

    scan_pixel_ys, scan_pixel_xs = ScanGrid.metres_to_pixels([ray_scan_coords_y, ray_scan_coords_x])

    # Create a mask of indices within bounds.
    mask = (scan_pixel_ys >= 0) & (scan_pixel_ys < scan_shape[0]) & \
           (scan_pixel_xs >= 0) & (scan_pixel_xs < scan_shape[1])
    
    det_pixels_ys, det_pixels_xs = Detector.metres_to_pixels([ray_det_coords_y, ray_det_coords_x])

    detector_vals = detector_image[det_pixels_ys, det_pixels_xs]
    
    return scan_pixel_ys[mask], scan_pixel_xs[mask], detector_vals[mask]


def map_px_on_scan_to_detector(Detector, sample_interpolant, 
                               ray_scan_coords_x, ray_scan_coords_y,
                               ray_det_coords_x, ray_det_coords_y):
    
    # Stack the scan coordinates.
    scan_pts = np.stack([ray_scan_coords_y, ray_scan_coords_x], axis=-1)

    # Interpolate the sample intensity at the scan coordinates.
    sample_vals = sample_interpolant(scan_pts)
    
    # Convert the ray detector coordinates to pixel indices.
    ray_det_pixel_ys, ray_det_pixel_xs = Detector.metres_to_pixels([ray_det_coords_y, ray_det_coords_x])

    return ray_det_pixel_ys, ray_det_pixel_xs, sample_vals


def run_model_for_rays_and_slopes(transfer_matrices, input_slopes, scan_position):
    # Given an input fourdstem model and its transfer matrix, run the model to find the positions of the rays at the sample and detector
    # from a starting point source with a certain scan position.
    scan_pos_y, scan_pos_x = scan_position

    input_slopes_x = input_slopes[0]
    input_slopes_y = input_slopes[1]

    # Make the input rays we can run through one last time in the model to find positions at sample and detector
    rays_at_source_with_semi_conv = np.vstack([
        np.full((input_slopes_x.shape[0],), scan_pos_x),
        np.full((input_slopes_y.shape[0],), scan_pos_y),
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


def find_input_slopes_that_hit_detpx_from_pt_source_with_semiconv(
    pixel_coords, scan_pos, semi_conv, transformation_matrix
):
    """
    Given a set of detector pixel coordinates, a semi-convergence angle from a source, and a transformation matrix,
    find a mask that tells us what slopes will hit the detector pixels from the point source.
    """
    scan_pos_y, scan_pos_x = scan_pos

    A_xx, A_xy, B_xx, B_xy = transformation_matrix[0, :4]  # Select first row excluding the last column
    A_yx, A_yy, B_yx, B_yy = transformation_matrix[1, :4]  # Select second row excluding the last column

    delta_x, delta_y = transformation_matrix[0, 4], transformation_matrix[1, 4]

    y_out, x_out = pixel_coords[:, 0], pixel_coords[:, 1]

    denom = B_xx * B_yy - B_xy * B_yx
    theta_x_in = (
        -A_xx * B_yy * scan_pos_x
        - A_xy * B_yy * scan_pos_y
        + A_yx * B_xy * scan_pos_x
        + A_yy * B_xy * scan_pos_y
        + B_xy * delta_y
        - B_xy * y_out
        - B_yy * delta_x
        + B_yy * x_out
    ) / denom

    theta_y_in = (
        A_xx * B_yx * scan_pos_x
        + A_xy * B_yx * scan_pos_y
        - A_yx * B_xx * scan_pos_x
        - A_yy * B_xx * scan_pos_y
        - B_xx * delta_y
        + B_xx * y_out
        + B_yx * delta_x
        - B_yx * x_out
    ) / denom

    F = (theta_x_in**2 + theta_y_in**2) - semi_conv**2
    mask = F <= 0
    input_slopes_masked = np.stack([theta_x_in, theta_y_in]) * mask
    
    return input_slopes_masked


def get_ray_coords_at_scan_and_det(model, scan_pos, total_transfer_matrix, transfer_matrices, detector_to_sample):
    
    PointSource = model[0]
    Detector = model[3]
    detector_coords_yx = Detector.coords
    semi_conv = PointSource.semi_conv 

    # Find all input slopes for a max semiconvergence angle that will hit the detector pixels
    input_slopes = find_input_slopes_that_hit_detpx_from_pt_source_with_semiconv(
        detector_coords_yx, scan_pos, semi_conv, total_transfer_matrix
    )
    
    # Run the model to obtain the ray coordinates at each component in the model
    coords = run_model_for_rays_and_slopes(transfer_matrices, input_slopes, scan_pos)

    # Stack coordinates and perform the inverse matrix multiplication to get the sample coordinates
    xs, ys, dxs, dys = coords
    detector_rays = np.stack([xs[-1], ys[-1], dxs[-1], dys[-1], np.ones_like(xs[-1])])
    sample_rays = np.dot(detector_to_sample, detector_rays)

    # Unpack the sample and detector ray coordinates
    sample_rays_x = sample_rays[0]
    sample_rays_y = sample_rays[1]
    det_rays_x = detector_rays[0]
    det_rays_y = detector_rays[1]

    return sample_rays_x, sample_rays_y, det_rays_x, det_rays_y


def project_frame_forward(model, detector_frame, sample_interpolant, scan_pos):
    # Return all the transfer matrices necessary for us to propagate rays through the system
    transfer_matrices, total_transfer_matrix, detector_to_sample = get_transfer_matrices(model, scan_pos)

    # Get ray coordinates at the scan and detector
    sample_rays_x, sample_rays_y, det_rays_x, det_rays_y = get_ray_coords_at_scan_and_det(
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


def project_frame_backward(model, detector_frame, scan_pos):

    # Return all the transfer matrices necessary for us to propagate rays through the system
    transfer_matrices, total_transfer_matrix, detector_to_sample = get_transfer_matrices(model, scan_pos)

    # Get ray coordinates at the scan and detector
    sample_rays_x, sample_rays_y, det_rays_x, det_rays_y = get_ray_coords_at_scan_and_det(
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
