import pytest
import jax
import numpy as np
from jaxgym.components import ScanGrid, Detector, Descanner
from jaxgym.ray import Ray

jax.config.update('jax_platform_name', 'cpu')

# Test cases for ScanGrid:
@pytest.mark.parametrize(
    "xy, rotation, expected_pixel_coords",
    [
        # No rotation cases
        ((0.0, 0.0), 0.0, (5, 5)),
        ((-0.5, 0.5), 0.0, (0, 0)),
        ((0.5, -0.5), 0.0, (10, 10)),
        ((0.0, 0.5), 0.0, (0, 5)),
        ((-0.5, 0.0), 0.0, (5, 0)),
        # With rotation cases
        ((0.0, 0.0), 90.0, (5, 5)),
        ((-0.5, 0.5), 90.0, (10, 0)),
        ((0.5, -0.5), 90.0, (0, 10)),
        ((0.0, 0.5), 90.0, (5, 0)),
        ((-0.5, 0.0), 90.0, (10, 5)),
    ],
)
def test_scan_grid_metres_to_pixels(xy, rotation, expected_pixel_coords):
    scan_grid = ScanGrid(
        z=0.0,
        scan_rotation=rotation,
        scan_step=(0.1, 0.1),
        scan_shape=(10, 10),
        scan_centre=(0.0, 0.0)
    )
    pixel_coords_y, pixel_coords_x = scan_grid.metres_to_pixels(xy)
    np.testing.assert_allclose(pixel_coords_y, expected_pixel_coords[0], atol=1e-6)
    np.testing.assert_allclose(pixel_coords_x, expected_pixel_coords[1], atol=1e-6)


# Test cases for ScanGrid:
@pytest.mark.parametrize(
    "pixel_coords, rotation, expected_xy",
    [
        # No rotation cases
        ((5, 5), 0.0, (0.0, 0.0)),
        ((0, 0), 0.0, (-0.5, 0.5)),
        ((10, 10), 0.0, (0.5, -0.5)),
        ((0, 5), 0.0, (0.0, 0.5)),
        ((5, 0), 0.0, (-0.5, 0.0)),
        # With rotation cases
        ((5, 5), 90.0, (0.0, 0.0)),
        ((10, 0), 90.0, (-0.5, 0.5)),
        ((0, 10), 90.0, (0.5, -0.5)),
        ((5, 0), 90.0, (0.0, 0.5)),
        ((10, 5), 90.0, (-0.5, 0.0)),
    ],
)
def test_scan_grid_pixels_to_metres(pixel_coords, rotation, expected_xy):
    scan_grid = ScanGrid(
        z=0.0,
        scan_rotation=rotation,
        scan_step=(0.1, 0.1),
        scan_shape=(10, 10),
        scan_centre=(0.0, 0.0)
    )
    metres_coords_x, metres_coords_y = scan_grid.pixels_to_metres(pixel_coords)
    np.testing.assert_allclose(metres_coords_x, expected_xy[0], atol=1e-6)
    np.testing.assert_allclose(metres_coords_y, expected_xy[1], atol=1e-6)


# Test cases for Detector:
@pytest.mark.parametrize(
    "xy, rotation, expected_pixel_coords",
    [
        # No rotation cases
        ((0.0, 0.0), 0.0, (5, 5)),
        ((-0.5, 0.5), 0.0, (0, 0)),
        ((0.5, -0.5), 0.0, (10, 10)),
        ((0.0, 0.5), 0.0, (0, 5)),
        ((-0.5, 0.0), 0.0, (5, 0)),
        # With rotation cases
        ((0.0, 0.0), 90.0, (5, 5)),
        ((-0.5, 0.5), 90.0, (10, 0)),
        ((0.5, -0.5), 90.0, (0, 10)),
        ((0.0, 0.5), 90.0, (5, 0)),
        ((-0.5, 0.0), 90.0, (10, 5)),
    ],
)
def test_detector_metres_to_pixels(xy, rotation, expected_pixel_coords):
    detector = Detector(
        z=0.0,
        det_pixel_size=(0.1, 0.1),
        det_shape=(10, 10),
        det_centre=(0.0, 0.0),
        det_rotation=rotation,
        flip_y=False,
    )
    pixel_coords_y, pixel_coords_x = detector.metres_to_pixels(xy)
    np.testing.assert_allclose(pixel_coords_y, expected_pixel_coords[0], atol=1e-6)
    np.testing.assert_allclose(pixel_coords_x, expected_pixel_coords[1], atol=1e-6)


# Test cases for Detector:
@pytest.mark.parametrize(
    "pixel_coords, rotation, expected_xy",
    [
        # No rotation cases
        ((5, 5), 0.0, (0.0, 0.0)),
        ((0, 0), 0.0, (-0.5, 0.5)),
        ((10, 10), 0.0, (0.5, -0.5)),
        ((0, 5), 0.0, (0.0, 0.5)),
        ((5, 0), 0.0, (-0.5, 0.0)),
        # With rotation cases
        ((5, 5), 90.0, (0.0, 0.0)),
        ((10, 0), 90.0, (-0.5, 0.5)),
        ((0, 10), 90.0, (0.5, -0.5)),
        ((5, 0), 90.0, (0.0, 0.5)),
        ((10, 5), 90.0, (-0.5, 0.0)),
    ],
)
def test_detector_pixels_to_metres(pixel_coords, rotation, expected_xy):
    detector = Detector(
        z=0.0,
        det_rotation=rotation,
        det_pixel_size=(0.1, 0.1),
        det_shape=(10, 10),
        det_centre=(0.0, 0.0),
        flip_y=False,
    )
    metres_coords_x, metres_coords_y = detector.pixels_to_metres(pixel_coords)
    np.testing.assert_allclose(metres_coords_x, expected_xy[0], atol=1e-6)
    np.testing.assert_allclose(metres_coords_y, expected_xy[1], atol=1e-6)

@pytest.mark.parametrize(
    "scan_pos_xy, input_ray_xy, expected_output_xy",
    [
        ((0.0, 0.0), (1.0, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
        ((-1.0, -1.0), (2.0, 2.0), (1.0, 1.0)),
        ((2.5, -2.5), (-3.5, 3.5), (-1.0, 1.0)),
    ],
)
def test_descanner_offset(scan_pos_xy, input_ray_xy, expected_output_xy):

    input_ray = Ray(
        x=input_ray_xy[0],
        y=input_ray_xy[1],
        dx=0.0,
        dy=0.0,
        z=0.0,
        pathlength=1.0,
        
    )

    descanner = Descanner(
        z=0.0,
        scan_pos_x=scan_pos_xy[0],
        scan_pos_y=scan_pos_xy[1],
        descan_error=[0,0,0,0,0,0,0,0]
    )

    output_ray = descanner.step(input_ray)

    np.testing.assert_allclose(output_ray.x, expected_output_xy[0], atol=1e-6)
    np.testing.assert_allclose(output_ray.y, expected_output_xy[1], atol=1e-6)

    

