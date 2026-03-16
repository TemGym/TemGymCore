import pytest
import numpy as np
import jax
from jax import jacobian
import jax.numpy as jnp
import jax_dataclasses as jdc

from temgym_core.source import ParallelBeam
from temgym_core.components import (
    ScanGrid,
    Detector,
    Descanner,
    DescanError,
    Component,
    Biprism,
    Deflector,
    DoubleDeflector,
    Lens,
    ElectromagneticLens,
    Rotator,
)
from temgym_core.gaussian import make_gaussian
from temgym_core.ray import Ray
from temgym_core.utils import custom_jacobian_matrix
from temgym_core.run import run_to_end
from temgym_core.transfer_matrices import (
    propagation_matrix_5x5,
    lens_matrix_5x5,
    biprism_matrix_5x5,
    double_deflector_matrix_5x5,
)
from temgym_core.constants import compute_Kv_from_voltage
jax.config.update("jax_enable_x64", True)


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


def test_biprism():
    deflection = 1e-3
    biprism = Biprism(def_x=deflection, z=0.0)
    ray = Ray(x=1e-15, y=0.0, dx=0.0, dy=0.0, _one=1.0, z=0.0, pathlength=0.0)

    out_jac = jacobian(biprism)(ray)
    J = custom_jacobian_matrix(out_jac)
    jac_def = J[2, -1]
    np.testing.assert_allclose(jac_def, deflection, atol=1e-6)


def test_biprism_with_prop():
    deflection = 1e-3
    z_biprism = 0.0
    z_det = 0.234
    biprism = Biprism(def_x=deflection, z=z_biprism)
    detector = Detector(z=z_det, pixel_size=(1e-4, 1e-4), shape=(512, 512))
    ray = Ray(x=1e-15, y=0.0, dx=0.0, dy=0.0, _one=1.0, z=0.0, pathlength=0.0)
    model = [biprism, detector]
    ABCD = jacobian(lambda r: run_to_end(r, model))(ray)
    ABCD = custom_jacobian_matrix(ABCD)
    d_x_d_one = ABCD[0, -1]
    d_dx_d_one = ABCD[2, -1]
    analytic_def = deflection * (z_det - z_biprism)
    np.testing.assert_allclose(d_x_d_one, analytic_def, atol=1e-6)
    np.testing.assert_allclose(d_dx_d_one, deflection, atol=1e-6)


def test_biprism_with_lens_and_prop():

    M1 = -10
    F1 = 0.0002

    defocus = 1e-4
    L1_z1 = F1 * (1/M1 - 1)
    L1_z2 = F1 * (1 - M1)

    deflection = 1e-4
    input_beam = ParallelBeam(z=0.0 + defocus, radius=0.0)
    lens = Lens(focal_length=F1, z=abs(L1_z1))
    biprism = Biprism(z=abs(L1_z1) + abs(L1_z2) / 2, rotation=0.0, def_x=deflection)
    detector = Detector(z=abs(L1_z1) + abs(L1_z2), pixel_size=(0.01, 0.01), shape=(128, 128))

    z1 = lens.z - input_beam.z
    z2 = biprism.z - lens.z
    z3 = detector.z - biprism.z

    analytic_ABCD = (
        propagation_matrix_5x5(z3, xp=jnp)
        @ biprism_matrix_5x5(deflection, xp=jnp)
        @ propagation_matrix_5x5(z2, xp=jnp)
        @ lens_matrix_5x5(F1, xp=jnp)
        @ propagation_matrix_5x5(z1, xp=jnp)
    )

    model = [
        input_beam,
        lens,
        biprism,
        detector,
    ]

    central_ray = Ray(x=-1e-15, y=0.0, dx=0.0, dy=0.0, z=input_beam.z, pathlength=0.0, _one=1.0)

    ABCD = jacobian(lambda r: run_to_end(r, model))(central_ray)
    ABCD = custom_jacobian_matrix(ABCD)

    np.testing.assert_allclose(ABCD, analytic_ABCD, atol=1e-12)


def test_double_deflector_matches_explicit_pair_ray():
    dd = DoubleDeflector(
        z=0.1,
        spacing=0.03,
        shift_x=1.3e-4,
        shift_y=-0.9e-4,
        shift_balance_x=1.15,
        shift_balance_y=0.85,
    )
    detector = Detector(z=0.4, pixel_size=(1e-6, 1e-6), shape=(16, 16))

    ray = Ray(
        x=2.1e-4,
        y=-1.3e-4,
        dx=1.2e-3,
        dy=-0.8e-3,
        z=0.0,
        pathlength=0.0,
        _one=1.0,
    )

    model_dd = (dd, detector)
    model_pair = (
        Deflector(z=dd.z, def_x=dd.def1_x, def_y=dd.def1_y),
        Deflector(z=dd.z_second, def_x=dd.def2_x, def_y=dd.def2_y),
        detector,
    )

    out_dd = run_to_end(ray, model_dd)
    out_pair = run_to_end(ray, model_pair)

    np.testing.assert_allclose(out_dd.x, out_pair.x, atol=1e-12)
    np.testing.assert_allclose(out_dd.y, out_pair.y, atol=1e-12)
    np.testing.assert_allclose(out_dd.dx, out_pair.dx, atol=1e-12)
    np.testing.assert_allclose(out_dd.dy, out_pair.dy, atol=1e-12)
    np.testing.assert_allclose(out_dd.z, out_pair.z, atol=1e-12)


def test_double_deflector_matches_explicit_pair_gaussian():
    dd = DoubleDeflector(
        z=0.05,
        spacing=0.015,
        shift_x=2.0e-4,
        shift_y=-1.0e-4,
        shift_balance_x=1.2,
        shift_balance_y=0.9,
    )
    detector = Detector(z=0.3, pixel_size=(1e-6, 1e-6), shape=(16, 16))

    beam = make_gaussian(
        x=0.0,
        y=0.0,
        dx=1.1e-3,
        dy=-0.7e-3,
        z=0.0,
        voltage=200e3,
        waist_x=1.6e-6,
        waist_y=1.9e-6,
    )

    model_dd = (dd, detector)
    model_pair = (
        Deflector(z=dd.z, def_x=dd.def1_x, def_y=dd.def1_y),
        Deflector(z=dd.z_second, def_x=dd.def2_x, def_y=dd.def2_y),
        detector,
    )

    out_dd = run_to_end(beam, model_dd)
    out_pair = run_to_end(beam, model_pair)

    np.testing.assert_allclose(np.asarray(out_dd.x), np.asarray(out_pair.x), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out_dd.y), np.asarray(out_pair.y), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out_dd.dx), np.asarray(out_pair.dx), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out_dd.dy), np.asarray(out_pair.dy), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out_dd.z), np.asarray(out_pair.z), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out_dd.Q_inv), np.asarray(out_pair.Q_inv), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(out_dd.amplitude), np.asarray(out_pair.amplitude), atol=1e-12
    )


def test_double_deflector_jacobian_matches_analytic_matrix():
    dd = DoubleDeflector(
        z=0.0,
        spacing=0.02,
        shift_x=1.4e-4,
        shift_y=-1.1e-4,
        shift_balance_x=1.3,
        shift_balance_y=0.7,
    )
    ray = Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=dd.z, pathlength=0.0, _one=1.0)

    out_jac = jacobian(dd)(ray)
    J = custom_jacobian_matrix(out_jac)

    T = double_deflector_matrix_5x5(
        dd.spacing,
        shift_x=dd.shift_x,
        shift_y=dd.shift_y,
        tilt_x=dd.tilt_x,
        tilt_y=dd.tilt_y,
        shift_balance_x=dd.shift_balance_x,
        shift_balance_y=dd.shift_balance_y,
        tilt_balance_x=dd.tilt_balance_x,
        tilt_balance_y=dd.tilt_balance_y,
        xp=jnp,
    )

    np.testing.assert_allclose(J, T, atol=1e-12)


def test_double_deflector_advances_to_second_plane():
    dd = DoubleDeflector(
        z=0.12,
        spacing=0.025,
        shift_x=1e-4,
        shift_y=-2e-4,
    )
    ray = Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=dd.z, pathlength=0.0, _one=1.0)

    out = dd(ray)
    np.testing.assert_allclose(out.z, dd.z_second, atol=1e-12)


def test_double_deflector_balance_one_gives_zero_net_slope():
    dd = DoubleDeflector(
        z=0.1,
        spacing=0.03,
        shift_x=2.0e-4,
        shift_y=-3.0e-4,
        shift_balance_x=1.0,
        shift_balance_y=1.0,
        tilt_balance_x=1.0,
        tilt_balance_y=1.0,
    )

    ray = Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=0.0, pathlength=0.0, _one=1.0)
    out = run_to_end(ray, (dd,))

    np.testing.assert_allclose(out.dx, 0.0, atol=1e-12)
    np.testing.assert_allclose(out.dy, 0.0, atol=1e-12)


def test_double_deflector_two_lens_optimization_smoke():
    ray0 = Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=0.0, pathlength=0.0, _one=1.0)
    target_xy = jnp.array([2.5e-4, -1.5e-4], dtype=jnp.float64)

    spacing = 0.02
    z_def = 0.1
    lens1 = Lens(z=0.27, focal_length=0.18)
    lens2 = Lens(z=0.42, focal_length=0.22)
    sample = Detector(z=0.57, pixel_size=(1e-6, 1e-6), shape=(32, 32))

    def loss_fn(params):
        shift_x, shift_y, shift_balance_x, shift_balance_y = params
        model = (
            DoubleDeflector(
                z=z_def,
                spacing=spacing,
                shift_x=shift_x,
                shift_y=shift_y,
                shift_balance_x=shift_balance_x,
                shift_balance_y=shift_balance_y,
            ),
            lens1,
            lens2,
            sample,
        )
        out = run_to_end(ray0, model)
        pos_term = jnp.sum(((jnp.array([out.x, out.y]) - target_xy) / 1e-4) ** 2)
        ang_term = jnp.sum((jnp.array([out.dx, out.dy]) / 5e-5) ** 2)
        return pos_term + 0.2 * ang_term

    grad_fn = jax.value_and_grad(loss_fn)
    params = jnp.array([1e-4, -1e-4, 1.0, 1.0], dtype=jnp.float64)

    init_loss, init_grads = grad_fn(params)
    assert np.isfinite(np.asarray(init_loss)).all()
    assert np.isfinite(np.asarray(init_grads)).all()

    lr = 0.005
    for _ in range(120):
        loss, grads = grad_fn(params)
        assert np.isfinite(np.asarray(loss)).all()
        assert np.isfinite(np.asarray(grads)).all()
        grads = jnp.clip(grads, -1e3, 1e3)
        params = params - lr * grads
        params = params.at[:2].set(jnp.clip(params[:2], -2e-3, 2e-3))
        params = params.at[2:].set(jnp.clip(params[2:], 0.2, 3.0))

    final_loss = loss_fn(params)
    assert float(final_loss) < float(init_loss)


def test_electromagnetic_lens():
    """Test ElectromagneticLens matches Lens + Rotator composition."""
    # Setup electromagnetic lens parameters
    Rc = compute_Kv_from_voltage(200e3)  # 200 kV
    turns = 100.0
    current = 50.0  # turns * current = 5000 ampere-turns
    Gc = 5e-6    # 1/(AT²·m)

    # Create ElectromagneticLens
    em_lens = ElectromagneticLens(z=0.0, turns=turns, current=current, Gc=Gc, Rc=Rc)

    # Create equivalent manual Lens + Rotator
    focal_length = em_lens.focal_length
    rotation_rad = em_lens.rotation_angle
    manual_lens = Lens(z=0.0, focal_length=focal_length)
    rotator = Rotator(z=0.0, angle=np.rad2deg(rotation_rad))

    # Test ray with non-zero position and slopes
    test_ray = Ray(
        x=jnp.array(1e-3),
        y=jnp.array(0.5e-3),
        dx=jnp.array(0.01),
        dy=jnp.array(-0.005),
        _one=jnp.array(1.0),
        pathlength=jnp.array(0.0),
        z=jnp.array(0.0)
    )

    # Apply transformations
    ray_em = em_lens(test_ray)
    ray_manual = rotator(manual_lens(test_ray))

    # Compare results (numerical precision ~1e-5)
    np.testing.assert_allclose(ray_em.x, ray_manual.x, rtol=1e-5)
    np.testing.assert_allclose(ray_em.y, ray_manual.y, rtol=1e-5)
    np.testing.assert_allclose(ray_em.dx, ray_manual.dx, rtol=1e-5)
    np.testing.assert_allclose(ray_em.dy, ray_manual.dy, rtol=1e-5)
    np.testing.assert_allclose(ray_em.pathlength, ray_manual.pathlength, rtol=1e-5)


def test_electromagnetic_lens_properties():
    """Test ElectromagneticLens focal_length and rotation_angle properties."""
    Rc = 5.3e-4  # rad/AT
    turns = 200.0
    current = 25.0  # turns * current = 5000 AT
    excitation = turns * current
    Gc = 5e-6    # 1/(AT²·m)

    lens = ElectromagneticLens(z=0.0, turns=turns, current=current, Gc=Gc, Rc=Rc)

    # Test focal length: f = 1/(Gc·(turns·current)²)
    expected_f = 1.0 / (Gc * excitation**2)
    np.testing.assert_allclose(lens.focal_length, expected_f, rtol=1e-10)

    # Test rotation angle: ψ = Rc·(turns·current)
    expected_psi = Rc * excitation
    np.testing.assert_allclose(lens.rotation_angle, expected_psi, rtol=1e-10)
    np.testing.assert_allclose(lens.I0, excitation, rtol=1e-10)


def test_electromagnetic_lens_zero_rotation():
    """Test ElectromagneticLens with zero rotation (Rc=0) behaves like pure Lens."""
    turns = 100.0
    current = 50.0
    Gc = 5e-6
    Rc = 0.0  # No rotation

    em_lens = ElectromagneticLens(z=0.0, turns=turns, current=current, Gc=Gc, Rc=Rc)
    pure_lens = Lens(z=0.0, focal_length=em_lens.focal_length)

    test_ray = Ray(
        x=jnp.array(1e-3),
        y=jnp.array(0.5e-3),
        dx=jnp.array(0.01),
        dy=jnp.array(-0.005),
        _one=jnp.array(1.0),
        pathlength=jnp.array(0.0),
        z=jnp.array(0.0)
    )

    ray_em = em_lens(test_ray)
    ray_pure = pure_lens(test_ray)

    # Should match exactly when no rotation
    np.testing.assert_allclose(ray_em.x, ray_pure.x, rtol=1e-10)
    np.testing.assert_allclose(ray_em.y, ray_pure.y, rtol=1e-10)
    np.testing.assert_allclose(ray_em.dx, ray_pure.dx, rtol=1e-10)
    np.testing.assert_allclose(ray_em.dy, ray_pure.dy, rtol=1e-10)
    np.testing.assert_allclose(ray_em.pathlength, ray_pure.pathlength, rtol=1e-10)


def test_electromagnetic_lens_thick_advances_ray_when_tc_positive():
    turns = 100.0
    current = 50.0
    Gc = 5e-6
    Rc = 2.0e-4

    lens_thin = ElectromagneticLens(
        z=0.0,
        turns=turns,
        current=current,
        Gc=Gc,
        Rc=Rc,
        Tc=0.0,
    )
    lens_with_tc = ElectromagneticLens(
        z=0.0,
        turns=turns,
        current=current,
        Gc=Gc,
        Rc=Rc,
        Tc=1.0e-3,
    )

    ray = Ray(
        x=jnp.array(1e-3),
        y=jnp.array(-0.5e-3),
        dx=jnp.array(0.005),
        dy=jnp.array(-0.002),
        _one=jnp.array(1.0),
        pathlength=jnp.array(0.0),
        z=jnp.array(0.0),
    )

    out_thin = lens_thin(ray)
    out_tc = lens_with_tc(ray)

    assert float(out_tc.z) > float(out_thin.z)
    assert float(out_tc.pathlength) > float(out_thin.pathlength)
