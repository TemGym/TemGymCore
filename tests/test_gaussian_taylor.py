import jax
import numpy as np
import jax.numpy as jnp
import pytest

from temgym_core.gaussian import GaussianRayBeta, TaylorExpofAction
from temgym_core.gaussian_taylor import Lens, run_to_end
from temgym_core.components import Detector
from temgym_core.evaluate import evaluate_gaussians_for
from temgym_core.utils import fibonacci_spiral
from abtem.core.energy import energy2wavelength

jax.config.update("jax_enable_x64", True)


def _make_initial_rays(
    num_rays=1000,
    w0=1e-9,
    aperture_radius=50e-9,
    voltage=200e3,
    x_shift=0.0,
):
    wavelength = energy2wavelength(voltage) / 1e10  # Å -> m
    k0 = 2 * np.pi / wavelength
    rx, ry = fibonacci_spiral(nb_samples=num_rays, radius=aperture_radius, alpha=0)

    # Complex beam parameter inverse at waist: 1/q =  -i λ / (π w0^2)
    q_inv_waist = -1j * (wavelength / (np.pi * w0**2))
    Q_inv = jnp.array([[q_inv_waist, 0.0], [0.0, q_inv_waist]])
    Q_inv = jnp.tile(Q_inv, (num_rays, 1, 1))

    C0 = jnp.ones(num_rays) * (1.0 + 0.0j)
    voltage_arr = jnp.full((num_rays,), voltage)

    S = TaylorExpofAction(
        const=jnp.zeros(num_rays, dtype=jnp.complex128),
        lin=jnp.zeros((num_rays, 2), dtype=jnp.complex128),
        quad=Q_inv,
    )

    rays_in = GaussianRayBeta(
        x=rx + x_shift,
        y=ry,
        dx=jnp.zeros(num_rays),
        dy=jnp.zeros(num_rays),
        z=jnp.zeros(num_rays),
        pathlength=jnp.zeros(num_rays),
        _one=jnp.ones(num_rays),
        S=S,
        C=C0,
        voltage=voltage_arr,
    )
    return rays_in, wavelength, k0


def _waist_from_Q_inv(Q_inv_elem, wavelength):
    # Im(1/q) = - λ / (π w^2)
    im_part = Q_inv_elem.imag
    return np.sqrt(-wavelength / (np.pi * im_part))


def _radius_from_Q_inv(Q_inv_elem):
    # Re(1/q) = 1/R
    re_part = Q_inv_elem.real
    if abs(re_part) < 1e-30:
        return np.inf
    return 1.0 / re_part


def _expected_q_inv_free_space(w0, wavelength, L):
    z_R = np.pi * w0**2 / wavelength
    if L == 0:
        return 1j / z_R
    return 1.0 / (L + 1j * z_R)


def _expected_waist_R(w0, wavelength, L):
    z_R = np.pi * w0**2 / wavelength
    if L == 0:
        return w0, np.inf
    w = w0 * np.sqrt(1.0 + (L / z_R) ** 2)
    R = L * (1.0 + (z_R**2 / L**2))
    return w, R


@pytest.mark.parametrize("M1,F1", [(-10, 2.5)])
def test_lens_magnification_and_beam_waist(M1, F1):
    """
    Verifies:
    1. Transverse coordinate magnification matches target M1.
    2. Beam waist scales as |M1| * w0 at image plane.
    3. Radius of curvature ~ 1/M*F at image plane.
    """
    w0 = 1.0
    rays_in, wavelength, _ = _make_initial_rays(w0=w0, aperture_radius=1e-2, voltage=2000, num_rays=10)

    L1_z1 = F1 * (1.0 / M1 - 1.0)
    L1_z2 = F1 * (1.0 - M1)
    L1_z1, L1_z2 = map(abs, (L1_z1, L1_z2))

    lens = Lens(focal_length=F1, z=L1_z1)
    detector = Detector(z=L1_z1 + L1_z2, pixel_size=(1e-6, 1e-6), shape=(100, 100))
    model = [lens, detector]

    rays_out = jax.vmap(run_to_end, in_axes=(0, None))(rays_in, model)

    mask = np.abs(np.array(rays_in.x)) > 1e-15
    r_in = np.sqrt(np.array(rays_in.x) ** 2 + np.array(rays_in.y) ** 2)
    r_out = np.sqrt(np.array(rays_out.x) ** 2 + np.array(rays_out.y) ** 2)
    mask = r_in > 1e-15
    measured_M = np.mean(r_out[mask] / r_in[mask])
    assert np.isclose(measured_M, np.abs(M1), rtol=5e-3, atol=5e-3), f"Magnification mismatch: got {measured_M}, expected {M1}"

    # Beam waist from Q_inv
    q_inv_elem = np.array(rays_out.S.quad[0, 0, 0])
    waist_measured = _waist_from_Q_inv(q_inv_elem, wavelength)
    waist_expected = w0 * abs(M1)
    assert np.isclose(waist_measured, waist_expected, rtol=5e-3), (
        f"Waist mismatch: got {waist_measured} vs {waist_expected}"
    )

    # Radius of curvature of Q_inv can be calculated as -(C/A) (When A != 0)
    # In the case of a lens system, this is -1/(M*F)
    R = _radius_from_Q_inv(q_inv_elem)
    np.allclose(R, -1/(M1*F1))


def test_defocused_plane_radius_and_waist():
    """
    Introduce a deliberate defocus after the image plane and test:
    1. Waist grows according to Gaussian beam propagation.
    2. Radius of curvature matches analytic value R(z) = z (1 + z_R^2 / z^2).
    Implementation: compute expected using z_R = π w0^2 / λ and free-space formulas,
    by reinterpreting Q_inv after adding a defocus drift distance dz.
    """
    M1, F1 = -200.0, 3e-3
    w0 = 1.2e-6
    rays_in, wavelength, _ = _make_initial_rays(w0=w0)

    L1_z1 = F1 * (1.0 / M1 - 1.0)
    L1_z2 = F1 * (1.0 - M1)
    L1_z1, L1_z2 = map(abs, (L1_z1, L1_z2))
    lens = Lens(focal_length=F1, z=L1_z1)
    detector = Detector(z=L1_z1 + L1_z2, pixel_size=(1e-6, 1e-6), shape=(100, 100))
    model = [lens, detector]

    rays_out = jax.vmap(run_to_end, in_axes=(0, None))(rays_in, model)
    q_inv_image = np.array(rays_out.S.quad[0, 0, 0])

    # At image plane (new waist)
    w_image = _waist_from_Q_inv(q_inv_image, wavelength)
    assert np.isclose(w_image, w0 * abs(M1), rtol=1e-2)

    # Now emulate defocus: propagate a distance dz past the waist analytically,
    # compare against constructed analytic Q_inv.
    dz = 0.01  # 1 cm defocus
    zR_new = np.pi * (w_image**2) / wavelength
    # q(z) = z + i zR  => 1/q(z) = (z - i zR)/(z^2 + zR^2)
    denom = dz**2 + zR_new**2
    q_inv_defocus = (dz / denom) - 1j * (zR_new / denom)

    # Extract waist, R from analytic defocus
    waist_expected = w_image * np.sqrt(1.0 + (dz / zR_new) ** 2)
    R_expected = dz * (1.0 + (zR_new**2 / dz**2))

    waist_from_q = _waist_from_Q_inv(q_inv_defocus, wavelength)
    R_from_q = _radius_from_Q_inv(q_inv_defocus)

    assert np.isclose(waist_from_q, waist_expected, rtol=1e-10), "Defocus waist formula mismatch"
    assert np.isclose(R_from_q, R_expected, rtol=1e-10), "Defocus radius formula mismatch"

    # This test anchors correctness of interpreting Q_inv into physical beam parameters.


@pytest.mark.parametrize("L_factor", [1.0, 2.0, 10.0])
def test_free_space_propagation_Q_inv_waist_radius(L_factor):
    """
    Free-space propagation test:
    Start at a waist (z=0) and propagate to z = L_factor * z_R.
    Validate:
    - Q_inv matches analytic 1/(z + i z_R)
    - Waist evolution w(z) = w0 * sqrt(1 + (z / z_R)^2)
    - Radius of curvature R(z) = z * (1 + z_R^2 / z^2); infinite at waist
    - Off-diagonal elements remain zero; x/y symmetry.
    """
    w0 = 1e-9
    rays_in, wavelength, _ = _make_initial_rays(w0=w0)
    z_R = np.pi * w0**2 / wavelength
    L = L_factor * z_R

    # Use a detector to define the propagation end plane (free space segment)
    detector = Detector(z=L, pixel_size=(1e-6, 1e-6), shape=(8, 8))
    model = [detector]

    rays_out = jax.vmap(run_to_end, in_axes=(0, None))(rays_in, model)

    q_inv_expected = _expected_q_inv_free_space(w0, wavelength, L)
    w_expected, R_expected = _expected_waist_R(w0, wavelength, L)

    q_inv_out = np.array(rays_out.S.quad[0, 0, 0])

    # Check diagonal equality (x/y symmetry)
    q_inv_y = np.array(rays_out.S.quad[0, 1, 1])
    assert np.allclose(q_inv_out, q_inv_y, rtol=1e-6, atol=1e-9), "Anisotropy detected in Q_inv"

    # Off-diagonals should remain (near) zero
    off_diag = np.array(rays_out.S.quad[0, 0, 1])
    assert abs(off_diag) < 1e-12, f"Off-diagonal coupling appeared: {off_diag}"

    # Real / Imag parts vs analytic
    assert np.allclose(q_inv_out.real, q_inv_expected.real, rtol=2e-3, atol=1e-4), \
        f"Re(1/q) mismatch at L={L} (factor {L_factor})"
    assert np.allclose(q_inv_out.imag, q_inv_expected.imag, rtol=2e-3, atol=1e-4), \
        f"Im(1/q) mismatch at L={L} (factor {L_factor})"

    # Waist from Q_inv
    w_out = _waist_from_Q_inv(q_inv_out, wavelength)
    assert np.allclose(w_out, w_expected, rtol=2e-3), \
        f"Waist mismatch at L={L}: got {w_out}, expected {w_expected}"

    # Radius from Q_inv
    R_out = _radius_from_Q_inv(q_inv_out)
    if L_factor == 0.0:
        assert R_out > 1e6, f"Radius at waist should be ~infinite, got {R_out}"
    else:
        assert np.allclose(R_out, R_expected, rtol=3e-3), \
            f"Radius mismatch at L={L}: got {R_out}, expected {R_expected}"


def test_beam_field_evaluation():
    w0 = 1
    rays_in, wavelength, k0 = _make_initial_rays(w0=w0, num_rays=1)
    detector = Detector(z=10, pixel_size=(1e-1, 1e-1), shape=(200, 200))
    model = [detector]

    ray_out = run_to_end(rays_in, model)

    field = evaluate_gaussians_for(ray_out, detector)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(np.abs(field))
    plt.savefig("test_gaussian_field.png")


@pytest.mark.parametrize("f", [1e-3, 3e-3, 1e-2])
def test_parallel_rays_focus_at_back_focal_plane_center(f):
    """
    Rays entering parallel to the optical axis (dx=dy=0) must intersect the
    optical axis (x=y=0) at the back focal plane z=f of a thin lens.
    """
    num_rays = 2000
    aperture_radius = 2e-4  # sufficiently wide to sample various x,y
    rays_in, _, _ = _make_initial_rays(num_rays=num_rays, aperture_radius=aperture_radius)

    # Sanity: directions are parallel to the axis
    assert np.allclose(np.array(rays_in.dx), 0.0)
    assert np.allclose(np.array(rays_in.dy), 0.0)

    lens = Lens(focal_length=f, z=0.0)
    detector = Detector(z=f, pixel_size=(1e-6, 1e-6), shape=(8, 8))
    rays_out = jax.vmap(run_to_end, in_axes=(0, None))(rays_in, [lens, detector])

    x = np.array(rays_out.x)
    y = np.array(rays_out.y)

    tol = 1e-9
    assert np.max(np.abs(x)) < tol and np.max(np.abs(y)) < tol, (
        f"Rays did not converge to (0,0) at z=f. "
        f"max|x|={np.max(np.abs(x))}, max|y|={np.max(np.abs(y))}, f={f}"
    )
