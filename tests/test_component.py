import pytest
import numpy as np
from jax import jacobian
import jax.numpy as jnp

from temgym_core.components import ScanGrid, Detector, Descanner, DescanError, Component
from temgym_core.ray import Ray
from temgym_core.utils import custom_jacobian_matrix


@jdc.pytree_dataclass
# A component that should give a singular jacobian used for testing
class SingularComponent(Component):
    def __call__(self, ray: Ray):
        new_x = ray.x
        new_y = ray.x
        return Ray(
            x=new_x,
            y=new_y,
            dx=ray.dx,
            dy=ray.dy,
            _one=ray._one,
            pathlength=ray.pathlength,
            z=ray.z,
        )


@pytest.mark.parametrize(
    "scan_shape",
    [(5, 5), (3, 7), (4, 4), (5, 8)],
)
def test_scan_grid_coords_symmetry(scan_shape):
    # coordinates should be symmetric around zero
    # with zero in the middle for odd dimensions
    # and no zero value for even dimensions
    h, w = scan_shape
    scan_grid = ScanGrid(
        z=0.0,
        rotation=0.0,
        pixel_size=(0.1, 0.1),
        shape=scan_shape,
    )
    ycoords, xcoords = np.arange(h), np.arange(w)
    _, yvals = scan_grid.pixels_to_metres((ycoords, np.zeros_like(ycoords)))
    xvals, _ = scan_grid.pixels_to_metres((np.zeros_like(xcoords), xcoords))

    def check_symmetry(size, vals):
        if (size % 2) == 0:  # even
            assert abs(vals[size // 2]) > 0.
            assert vals[size // 2] == pytest.approx(-1 * vals[size // 2 - 1])
            assert np.count_nonzero(vals) == vals.size
        else:  # odd
            assert np.count_nonzero(vals) == vals.size - 1
            assert vals[size // 2] == pytest.approx(0.)
            assert vals[size // 2 - 1] == pytest.approx(
                -1 * vals[size // 2 + 1]
            )

    check_symmetry(h, yvals)
    check_symmetry(w, xvals)


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
        rotation=rotation,
        pixel_size=(0.1, 0.1),
        shape=(11, 11),
    )
    pixel_coords_y, pixel_coords_x = scan_grid.metres_to_pixels(xy)
    np.testing.assert_allclose(pixel_coords_y, expected_pixel_coords[0], atol=1e-6)
    np.testing.assert_allclose(pixel_coords_x, expected_pixel_coords[1], atol=1e-6)


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
        rotation=rotation,
        pixel_size=(0.1, 0.1),
        shape=(11, 11),
    )
    metres_coords_x, metres_coords_y = scan_grid.pixels_to_metres(pixel_coords)
    np.testing.assert_allclose(metres_coords_x, expected_xy[0], atol=1e-6)
    np.testing.assert_allclose(metres_coords_y, expected_xy[1], atol=1e-6)


@pytest.mark.parametrize(
    "xy, expected_pixel_coords",
    [
        ((0.0, 0.0), (5, 5)),
        ((-0.5, 0.5), (0, 0)),
        ((0.5, -0.5), (10, 10)),
        ((0.0, 0.5), (0, 5)),
        ((-0.5, 0.0), (5, 0)),
    ],
)
def test_detector_metres_to_pixels(xy, expected_pixel_coords):
    detector = Detector(
        z=0.0,
        pixel_size=(0.1, 0.1),
        shape=(11, 11),
        flip_y=False,
    )
    pixel_coords_y, pixel_coords_x = detector.metres_to_pixels(xy)
    np.testing.assert_allclose(pixel_coords_y, expected_pixel_coords[0], atol=1e-6)
    np.testing.assert_allclose(pixel_coords_x, expected_pixel_coords[1], atol=1e-6)


# Test cases for Detector:
@pytest.mark.parametrize(
    "pixel_coords, expected_xy",
    [
        # No rotation cases
        ((5, 5), (0.0, 0.0)),
        ((0, 0), (-0.5, 0.5)),
        ((10, 10), (0.5, -0.5)),
        ((0, 5), (0.0, 0.5)),
        ((5, 0), (-0.5, 0.0)),
    ],
)
def test_detector_pixels_to_metres(pixel_coords, expected_xy):
    detector = Detector(
        z=0.0,
        pixel_size=(0.1, 0.1),
        shape=(11, 11),
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

    err = DescanError(
        pxo_pxi=err[0],
        pxo_pyi=err[1],
        pyo_pxi=err[2],
        pyo_pyi=err[3],
        sxo_pxi=err[4],
        sxo_pyi=err[5],
        syo_pxi=err[6],
        syo_pyi=err[7],
        offpxi=err[8],
        offpyi=err[9],
        offsxi=err[10],
        offsyi=err[11],
    )
    desc = Descanner(z=0.0, scan_pos_x=sp_x, scan_pos_y=sp_y, descan_error=err)
    ray = Ray(x=x, y=y, dx=dx, dy=dy, _one=1.0, z=0.0, pathlength=0.0)
    out = desc(ray)

    # Expected values computed using the same formula as in the implementation
    exp_x = x + sp_x * err[0] + sp_y * err[1] + err[8] - sp_x
    exp_y = y + sp_x * err[2] + sp_y * err[3] + err[9] - sp_y
    exp_dx = dx + sp_x * err[4] + sp_y * err[5] + err[10]
    exp_dy = dy + sp_x * err[6] + sp_y * err[7] + err[11]

    np.testing.assert_allclose(out.x, exp_x, atol=1e-6)
    np.testing.assert_allclose(out.y, exp_y, atol=1e-6)
    np.testing.assert_allclose(out.dx, exp_dx, atol=1e-6)
    np.testing.assert_allclose(out.dy, exp_dy, atol=1e-6)


def test_descanner_offset_consistency():
    # random scan position and descan error
    scan_pos_x = np.random.uniform(-5.0, 5.0)
    scan_pos_y = np.random.uniform(-5.0, 5.0)
    err = np.random.rand(12)
    err = DescanError(
        pxo_pxi=err[0],
        pxo_pyi=err[1],
        pyo_pxi=err[2],
        pyo_pyi=err[3],
        sxo_pxi=err[4],
        sxo_pyi=err[5],
        syo_pxi=err[6],
        syo_pyi=err[7],
        offpxi=err[8],
        offpyi=err[9],
        offsxi=err[10],
        offsyi=err[11],
    )
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
    outputs = [desc(r) for r in rays]

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
        np.testing.assert_allclose(off, first, atol=1e-6)


def test_descanner_jacobian_matrix():
    # Test that Jacobian of descanner yields correct 5x5 matrix when
    # jax.jacobian is called on it.
    sp_x, sp_y = 1.5, -2.0
    err = np.random.rand(12)
    err = DescanError(
        pxo_pxi=err[0],
        pxo_pyi=err[1],
        pyo_pxi=err[2],
        pyo_pyi=err[3],
        sxo_pxi=err[4],
        sxo_pyi=err[5],
        syo_pxi=err[6],
        syo_pyi=err[7],
        offpxi=err[8],
        offpyi=err[9],
        offsxi=err[10],
        offsyi=err[11],
    )
    desc = Descanner(z=0.0, scan_pos_x=sp_x, scan_pos_y=sp_y, descan_error=err)
    ray = Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, _one=1.0, z=0.0, pathlength=0.0)

    # Compute Jacobian wrt input ray
    jac = jacobian(desc)(ray)
    J = custom_jacobian_matrix(jac)

    # Compute expected coefficients
    K1 = sp_x * err[0] + sp_y * err[1] + err[8] - sp_x
    K2 = sp_x * err[2] + sp_y * err[3] + err[9] - sp_y
    K3 = sp_x * err[4] + sp_y * err[5] + err[10]
    K4 = sp_x * err[6] + sp_y * err[7] + err[11]
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


@pytest.mark.parametrize("repeat", tuple(range(5)))
def test_scan_grid_rotation_random(repeat):
    step = (0.1, 0.1)
    shape = (11, 11)
    centre_pix = (shape[0] // 2, shape[1] // 2)

    # test several random rotations
    scan_rot = np.random.uniform(-180.0, 180.0)
    scan_grid = ScanGrid(
        z=0.0,
        rotation=scan_rot,
        pixel_size=step,
        shape=shape,
    )
    # world‐space vector for one pixel step in scan‐grid x
    mx0, my0 = scan_grid.pixels_to_metres(centre_pix)
    mx1, my1 = scan_grid.pixels_to_metres((centre_pix[0], centre_pix[1] + 1))
    vec_scan = np.array([mx1 - mx0, my1 - my0])

    # expected rotated step vector = R(scan_rot) @ [step_x, 0]
    theta = np.deg2rad(scan_rot)
    exp_scan = np.array([np.cos(theta) * step[0], -np.sin(theta) * step[0]])
    np.testing.assert_allclose(vec_scan, exp_scan, atol=1e-6)


def test_singular_component_jacobian():
    # Test that the Jacobian of a singular component is a zero matrix
    singular_component = SingularComponent()
    ray = Ray(x=0.0, y=0.0, dx=1.0, dy=1.0, _one=1.0, z=0.0, pathlength=0.0)

    # Compute Jacobian wrt input ray
    jac = jacobian(singular_component)(ray)
    J = custom_jacobian_matrix(jac)

    inv = jnp.linalg.inv(J)

    # Check that jax.jacobian called on a singular component
    # and used with our custom_jacobian_matrix
    # returns a matrix that is singular (i.e., has NaN or Inf values)
    assert np.isnan(inv).any() or np.isinf(inv).any()
