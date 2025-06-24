import pytest
import numpy as np

from microscope_calibration.components import ScanGrid, Detector, Descanner

from jaxgym.ray import Ray
from jax import jacobian
from jaxgym.utils import custom_jacobian_matrix


def test_scan_grid_zero_in_metres_coords_for_odd_length():
    # For an odd number of pixels (scan_shape=(5,5) → 6 points each axis),
    # the metres grid should include 0.0 exactly at the centre.
    scan_grid = ScanGrid(
        z=0.0,
        scan_rotation=0.0,
        scan_step=(0.1, 0.1),
        scan_shape=(5, 7),
        scan_centre=(0.0, 0.0),
    )
    xs, ys = [], []
    sy, sx = scan_grid.scan_shape
    for py in range(sy):
        for px in range(sx):
            mx, my = scan_grid.pixels_to_metres((py, px))
            xs.append(mx)
            ys.append(my)
    assert any(np.isclose(v, 0.0) for v in xs), "Expected 0.0 in X coordinates for odd grid length"
    assert any(np.isclose(v, 0.0) for v in ys), "Expected 0.0 in Y coordinates for odd grid length"


def test_scan_grid_zero_not_in_metres_coords_for_even_length():
    # For an even number of pixels (scan_shape=(4,4) → 5 points each axis),
    # the metres grid should be centred around 0 but not include it exactly.
    scan_grid = ScanGrid(
        z=0.0,
        scan_rotation=0.0,
        scan_step=(0.1, 0.1),
        scan_shape=(4, 6),
        scan_centre=(0.0, 0.0),
    )
    xs, ys = [], []
    sy, sx = scan_grid.scan_shape
    for py in range(sy):
        for px in range(sx):
            mx, my = scan_grid.pixels_to_metres((py, px))
            xs.append(mx)
            ys.append(my)
    assert not any(np.isclose(v, 0.0) for v in xs), "Did not expect 0.0 in X coordinates for even grid length"
    assert not any(np.isclose(v, 0.0) for v in ys), "Did not expect 0.0 in Y coordinates for even grid length"


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
        scan_shape=(11, 11),
        scan_centre=(0.0, 0.0),
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
        scan_shape=(11, 11),
        scan_centre=(0.0, 0.0),
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
        det_shape=(11, 11),
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
        det_shape=(11, 11),
        det_centre=(0.0, 0.0),
        flip_y=False,
    )
    metres_coords_x, metres_coords_y = detector.pixels_to_metres(pixel_coords)
    np.testing.assert_allclose(metres_coords_x, expected_xy[0], atol=1e-6)
    np.testing.assert_allclose(metres_coords_y, expected_xy[1], atol=1e-6)


def test_descanner_random_descan_error():
    # Randomly chosen scan position and ray parameters
    sp_x, sp_y = np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0)
    x, y, dx, dy = np.random.uniform(-5.0, 5.0, size=4)

    # Randomly chosen non-zero descan error (length 12)
    err = np.random.rand(12)

    desc = Descanner(z=0.0, scan_pos_x=sp_x, scan_pos_y=sp_y, descan_error=err)
    ray = Ray(x=x, y=y, dx=dx, dy=dy, _one=1.0, z=0.0, pathlength=0.0)
    out = desc.step(ray)

    # Expected values computed using the same formula as in the implementation
    exp_x = x + (sp_x * err[0] + sp_y * err[1] + err[8] - sp_x)
    exp_y = y + (sp_y * err[3] + sp_y * err[2] + err[9] - sp_y)
    exp_dx = dx + (sp_x * err[4] + sp_y * err[5] + err[10])
    exp_dy = dy + (sp_y * err[7] + sp_y * err[6] + err[11])

    np.testing.assert_allclose(out.x, exp_x, atol=1e-8)
    np.testing.assert_allclose(out.y, exp_y, atol=1e-8)
    np.testing.assert_allclose(out.dx, exp_dx, atol=1e-8)
    np.testing.assert_allclose(out.dy, exp_dy, atol=1e-8)


def test_descanner_offset_consistency():
    # random scan position and descan error
    scan_pos_x = np.random.uniform(-5.0, 5.0)
    scan_pos_y = np.random.uniform(-5.0, 5.0)
    err = np.random.randn(12)
    desc = Descanner(
        z=0.0, scan_pos_x=scan_pos_x, scan_pos_y=scan_pos_y, descan_error=err
    )

    # generate a batch of random rays
    num_rays = 10
    xs = np.random.randn(num_rays)
    ys = np.random.randn(num_rays)
    dxs = np.random.randn(num_rays)
    dys = np.random.randn(num_rays)
    rays = [
        Ray(x=xs[i], y=ys[i], dx=dxs[i], dy=dys[i], _one=1.0, z=0.0, pathlength=0.0)
        for i in range(num_rays)
    ]

    # pass all rays through the descanner
    outputs = [desc.step(r) for r in rays]

    # compute per-ray offsets [Δx, Δy, Δdx, Δdy]
    offsets = np.array(
        [
            [out.x - r.x, out.y - r.y, out.dx - r.dx, out.dy - r.dy]
            for out, r in zip(outputs, rays)
        ]
    )

    # assert that all rays have received the same offset
    first = offsets[0]
    for off in offsets:
        np.testing.assert_allclose(off, first, atol=1e-8)


def test_descanner_jacobian_matrix():
    # Test that Jacobian of descanner.step yields correct 5x5 matrix when
    # jax.jacobian is called on it.
    sp_x, sp_y = 1.5, -2.0
    err = np.arange(12, dtype=float)
    desc = Descanner(z=0.0, scan_pos_x=sp_x, scan_pos_y=sp_y, descan_error=err)
    ray = Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, _one=1.0, z=0.0, pathlength=0.0)

    # Compute Jacobian wrt input ray
    jac = jacobian(desc.step)(ray)
    J = custom_jacobian_matrix(jac)

    # Compute expected coefficients
    K1 = sp_x * err[0] + sp_y * err[1] + err[8] - sp_x
    K2 = sp_y * err[3] + sp_y * err[2] + err[9] - sp_y
    K3 = sp_x * err[4] + sp_y * err[5] + err[10]
    K4 = sp_y * err[7] + sp_y * err[6] + err[11]
    T = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, K1],
            [0.0, 1.0, 0.0, 0.0, K2],
            [0.0, 0.0, 1.0, 0.0, K3],
            [0.0, 0.0, 0.0, 1.0, K4],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(J, T, atol=1e-6)


def test_scan_grid_rotation_random():

    step = (0.1, 0.1)
    shape = (11, 11)
    centre_pt = (0.0, 0.0)
    centre_pix = (shape[0] // 2, shape[1] // 2)

    # test several random rotations
    for scan_rot in np.random.uniform(-180.0, 180.0, size=5):
        scan_grid = ScanGrid(
            z=0.0,
            scan_rotation=scan_rot,
            scan_step=step,
            scan_shape=shape,
            scan_centre=centre_pt,
        )
        # world‐space vector for one pixel step in scan‐grid x
        mx0, my0 = scan_grid.pixels_to_metres(centre_pix)
        mx1, my1 = scan_grid.pixels_to_metres((centre_pix[0], centre_pix[1] + 1))
        vec_scan = np.array([mx1 - mx0, my1 - my0])

        # expected rotated step vector = R(scan_rot) @ [step_x, 0]
        theta = np.deg2rad(scan_rot)
        exp_scan = np.array([np.cos(theta) * step[0], -np.sin(theta) * step[0]])
        np.testing.assert_allclose(vec_scan, exp_scan, atol=1e-6)
