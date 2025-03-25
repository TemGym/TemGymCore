import pytest
import jax.numpy as jnp
from jaxgym.components import ScanGrid, Detector
import pytest


# Test cases for ScanGrid:
@pytest.mark.parametrize(
    "yx, rotation, expected_pixel_coords",
    [
        # No rotation cases (same as original)
        ((0.2, 0.2), 0.0, (7, 7)),
        ((0.0, 0.0), 0.0, (5, 5)),
        ((-0.2, -0.2), 0.0, (3, 3)),
        ((0.4, 0.4), 0.0, (9, 9)),
        # Rotation of 45° (0.785398 radians)
        ((0.2, 0.2), 0.785398, (5, 8)),
        ((0.0, 0.0), 0.785398, (5, 5)),
        ((-0.2, -0.2), 0.785398, (5, 2)),
        ((0.4, 0.4), 0.785398, (5, 11)),
    ],
)
def test_scan_grid_metres_to_pixels(yx, rotation, expected_pixel_coords):
    scan_grid = ScanGrid(
        z=0.0,
        scan_rotation=rotation,
        scan_step=(0.1, 0.1),
        scan_shape=(10, 10),
        center=(0.0, 0.0)
    )
    pixel_coords_y, pixel_coords_x = scan_grid.metres_to_pixels(yx)
    assert pixel_coords_y == expected_pixel_coords[0]
    assert pixel_coords_x == expected_pixel_coords[1]

# Test cases for Detector:
@pytest.mark.parametrize(
    "yx, rotation, expected_pixel_coords",
    [
        # No rotation cases (same as original)
        ((0.2, 0.2), 0.0, (7, 7)),
        ((0.0, 0.0), 0.0, (5, 5)),
        ((-0.2, -0.2), 0.0, (3, 3)),
        ((0.4, 0.4), 0.0, (9, 9)),
        # Rotation of 45° (0.785398 radians)
        ((0.2, 0.2), 0.785398, (5, 8)),
        ((0.0, 0.0), 0.785398, (5, 5)),
        ((-0.2, -0.2), 0.785398, (5, 2)),
        ((0.4, 0.4), 0.785398, (5, 11)),
    ],
)
def test_detector_metres_to_pixels(yx, rotation, expected_pixel_coords):
    detector = Detector(
        z=0.0,
        pixel_size=0.1,
        shape=(10, 10),
        centre=(0.0, 0.0),
        rotation=-rotation,
        flip_y=False
    )
    pixel_coords_y, pixel_coords_x = detector.metres_to_pixels(yx)
    assert pixel_coords_y == expected_pixel_coords[0]
    assert pixel_coords_x == expected_pixel_coords[1]
