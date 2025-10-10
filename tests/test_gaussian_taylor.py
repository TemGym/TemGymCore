import jax
import numpy as np
import jax.numpy as jnp
import pytest

from temgym_core.gaussian import GaussianRayBeta, TaylorExpofAction
from temgym_core.gaussian_taylor import Lens, run_to_end, SigmoidAperture
from temgym_core.components import Detector
from temgym_core.evaluate import evaluate_gaussians_for
from temgym_core.utils import fibonacci_spiral, zero_phase, energy2wavelength
from temgym_core.gaussian import q_inv, gaussian_beam
from temgym_core.utils import make_aperture, fresnel_lens_imaging_solution

from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


def plot_cross_sections(
    x,
    amplitudes,
    phases,
    labels=None,
    xlabel="x (m)",
    suffix="",
    linestyle="-",
    fig=None,
):
    """
    Plot n amplitude/phase cross sections side-by-side.

    Parameters
    ----------
    x : 1D array
        Common x-axis for all series.
    amplitudes : Sequence[1D array]
        List/tuple of amplitude arrays, one per series.
    phases : Sequence[1D array]
        List/tuple of phase arrays, one per series (same length as amplitudes).
    labels : Sequence[str] | None
        Labels for each series; if None, uses "Input i".
    xlabel : str
        Label for x-axis.
    suffix : str
        Suffix appended to labels in legends.
    """
    if len(amplitudes) != len(phases):
        raise ValueError("amplitudes and phases must have the same length")
    n = len(amplitudes)
    if labels is None:
        labels = [f"Input {i+1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must match number of series")

    if fig is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    else:
        axs = fig.axes

    # Amplitude cross sections
    for ampl, lab in zip(amplitudes, labels):
        axs[0].plot(x, ampl, label=f"{lab} {suffix} Amplitude", linestyle=linestyle)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Central Row Amplitude Cross Section")
    axs[0].legend()
    axs[0].grid(True)

    # Phase cross sections
    for ph, lab in zip(phases, labels):
        axs[1].plot(x, ph, label=f"{lab} {suffix} Phase", linestyle=linestyle)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("Phase (rad)")
    axs[1].set_title("Central Row Phase Cross Section")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    return fig, axs


def plot_overview(field1, field2, det_size_x, det_size_y,
                  label1='Input 1', label2='Input 2',
                  suffix='', unwrap=True):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    cbar_kwargs = dict(fraction=0.046, pad=0.04)

    im0 = axs[0, 0].imshow(
        np.abs(field1),
        extent=(-det_size_x/2, det_size_x/2, -det_size_y/2, det_size_y/2),
        cmap="gray",
    )
    axs[0, 0].set_title(f"{label1} Amplitude {suffix}")
    fig.colorbar(im0, ax=axs[0, 0], **cbar_kwargs)

    im1 = axs[0, 1].imshow(
        np.angle(field1) if unwrap else np.angle(field1),
        extent=(-det_size_x/2, det_size_x/2, -det_size_y/2, det_size_y/2),
        cmap="viridis",
    )
    axs[0, 1].set_title(f"{label1} Phase {suffix}")
    fig.colorbar(im1, ax=axs[0, 1], **cbar_kwargs)

    im2 = axs[1, 0].imshow(
        np.abs(field2),
        extent=(-det_size_x/2, det_size_x/2, -det_size_y/2, det_size_y/2),
        cmap="gray",
    )
    axs[1, 0].set_title(f"{label2} Amplitude {suffix}")
    fig.colorbar(im2, ax=axs[1, 0], **cbar_kwargs)

    im3 = axs[1, 1].imshow(
        np.angle(field2) if unwrap else np.angle(field2),
        extent=(-det_size_x/2, det_size_x/2, -det_size_y/2, det_size_y/2),
        cmap="viridis",
    )
    axs[1, 1].set_title(f"{label2} Phase {suffix}")
    fig.colorbar(im3, ax=axs[1, 1], **cbar_kwargs)

    plt.tight_layout()

    return fig, axs


def _make_initial_rays(
    num_rays=1000,
    w0=1e-9,
    aperture_radius=50e-9,
    voltage=200e3,
    x_shift=0.0,
):
    wavelength = energy2wavelength(voltage)
    k0 = 2 * np.pi / wavelength
    rx, ry = fibonacci_spiral(nb_samples=num_rays, radius=aperture_radius, alpha=0)

    # Complex beam parameter inverse at waist: 1/q =  -i λ / (π w0^2)
    q_inv_waist = 1j * (wavelength / (np.pi * w0**2))
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
    return np.sqrt(wavelength / (np.pi * im_part))


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
def test_lens_magnification_and_beam_waist_output_variables(M1, F1):
    """
    Verifies:
    1. Transverse coordinate magnification matches target M1.
    2. Beam waist scales as |M1| * w0 at image plane.
    3. Radius of curvature ~ 1/M*F at image plane.
    """
    w0 = 1.0
    voltage = 200e3
    rays_in, wavelength, k0 = _make_initial_rays(w0=w0, aperture_radius=1e-2, voltage=voltage, num_rays=1000)

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


@pytest.mark.parametrize("L, description, plotfile", [
    (1e-2, "free-space propagation", "test_evaluate_gaussians_for_vs_analytic.png"),
    (0.0, "waist (z=0)", "test_evaluate_gaussians_for_vs_analytic_z0.png"),
])
def test_evaluate_gaussians_for_matches_analytic_beam_param(L, description, plotfile):
    """
    Test that evaluate_gaussians_for produces a field matching the analytic Gaussian beam
    at the detector plane for a simple free-space propagation (no lens), or at the waist (z=0).
    Checks both amplitude and phase, and plots both for visual inspection.
    """
    w0 = 2e-6
    voltage = 200e3
    num_rays = 1
    aperture_radius = 1e-2
    rays_in, wavelength, k0 = _make_initial_rays(num_rays=num_rays, w0=w0, aperture_radius=aperture_radius, voltage=voltage)
    pixel_size = (2e-7, 2e-7)
    shape = (256, 256)
    detector = Detector(z=L, pixel_size=pixel_size, shape=shape)
    model = [detector]
    rays_out = jax.vmap(run_to_end, in_axes=(0, None))(rays_in, model)
    # Evaluate field using the code under test
    field = evaluate_gaussians_for(rays_out, detector)
    field = zero_phase(field, field.shape[0] // 2, field.shape[1] // 2)
    # Build analytic solution
    det_edge_x, det_edge_y = detector.coords_1d
    Y, X = np.meshgrid(det_edge_y, det_edge_x, indexing="ij")
    z_R = np.pi * w0**2 / wavelength

    # Analytic Gaussian beam field at plane z=L, using your sign conventions:
    # q(z) = L + 1j * z_R
    # 1/q(z) = 1/R(z) - 1j * λ/(π w(z)^2)
    # The field: E(x, y, z) = (w0/wz) * exp(-r^2/wz^2) * exp[+i k0 L + i k0 r^2/(2 Rz) - i psi]
    # Note: The sign of the quadratic phase (r^2) and Gouy phase (-psi) matches the convention
    # where the Taylor expansion is S = const + lin - quad.

    if L == 0.0:
        wz = w0
        Rz = np.inf
        psi = 0.0
    else:
        wz = w0 * np.sqrt(1 + (L / z_R) ** 2)
        Rz = L * (1 + (z_R / L) ** 2)
        psi = np.arctan(L / z_R)
    r2 = X**2 + Y**2

    analytic = (w0 / wz) * np.exp(-r2 / wz**2) * np.exp(
        -1j * (k0 * L + k0 * r2 / (2 * Rz) - psi)   # note overall minus; Gouy inside becomes +i*psi
    )
    analytic = zero_phase(analytic, analytic.shape[0] // 2, analytic.shape[1] // 2)
    # Normalize both fields for fair comparison
    field /= np.max(np.abs(field))
    analytic /= np.max(np.abs(analytic))
    # Compare amplitude and phase
    np.testing.assert_allclose(
        np.abs(field), np.abs(analytic), rtol=1e-2, atol=1e-2,
        err_msg=f"Amplitude mismatch between evaluate_gaussians_for and analytic ({description})"
    )
    np.testing.assert_allclose(
        unwrap_phase(np.angle(field)),
        unwrap_phase(np.angle(analytic)),
        rtol=1e-2, atol=1e-2,
        err_msg=f"Phase mismatch between evaluate_gaussians_for and analytic ({description})"
    )

    # Plot amplitude and phase of both images
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    im0 = axs[0, 0].imshow(np.abs(field), cmap='gray')
    axs[0, 0].set_title(f"evaluate_gaussians_for Amplitude ({description})")
    fig.colorbar(im0, ax=axs[0, 0])
    im1 = axs[0, 1].imshow(unwrap_phase(np.angle(field)), cmap='twilight')
    axs[0, 1].set_title(f"evaluate_gaussians_for Phase ({description})")
    fig.colorbar(im1, ax=axs[0, 1])
    im2 = axs[1, 0].imshow(np.abs(analytic), cmap='gray')
    axs[1, 0].set_title(f"Analytic Amplitude ({description})")
    fig.colorbar(im2, ax=axs[1, 0])
    im3 = axs[1, 1].imshow(unwrap_phase(np.angle(analytic)), cmap='twilight')
    axs[1, 1].set_title(f"Analytic Phase ({description})")
    fig.colorbar(im3, ax=axs[1, 1])
    plt.tight_layout()
    plt.savefig(plotfile)
    plt.close(fig)

# ...existing code...


@pytest.mark.parametrize("M1,F1", [(-10, 2.5)])
def test_lens_magnification_and_beam_waist_output_image(M1, F1):
    """
    Verifies:
    1. Transverse coordinate magnification matches target M1.
    2. Beam waist scales as |M1| * w0 at image plane.
    3. Radius of curvature ~ 1/M*F at image plane.
    """
    w0 = 1e-6
    rays_in, wavelength, k0 = _make_initial_rays(num_rays=1, w0=w0, aperture_radius=1e-2, voltage=200e3)

    L1_z1 = F1 * (1.0 / M1 - 1.0)
    L1_z2 = F1 * (1.0 - M1)
    L1_z1, L1_z2 = map(abs, (L1_z1, L1_z2))

    lens = Lens(focal_length=F1, z=L1_z1)
    pixel_size = (1e-7, 1e-7)
    shape = (2048, 2048)
    detector = Detector(z=L1_z1 + L1_z2, pixel_size=pixel_size, shape=shape)
    model = [lens, detector]

    rays_out = jax.vmap(run_to_end, in_axes=(0, None))(rays_in, model)

    det_edge_x, det_edge_y = detector.coords_1d

    Y, X = np.meshgrid(det_edge_y, det_edge_x, indexing="ij")

    analytic_gauss_image = evaluate_gaussians_for(rays_out, detector)
    analytic_gauss_image = zero_phase(
        analytic_gauss_image,
        analytic_gauss_image.shape[0] // 2,
        analytic_gauss_image.shape[1] // 2,
    )
    # Fresnel Version
    q1_inv = q_inv(0.0, w0, wavelength)
    gauss_input = gaussian_beam(X, Y, q1_inv, 2 * np.pi / wavelength)
    gauss_input = zero_phase(
        gauss_input,
        gauss_input.shape[0] // 2,
        gauss_input.shape[1] // 2,
    )

    fresnel_gauss_image = fresnel_lens_imaging_solution(gauss_input, Y, X, pixel_size[0], wavelength,
                                                       L1_z1, F1, L1_z2)

    fresnel_gauss_image = zero_phase(
        fresnel_gauss_image,
        fresnel_gauss_image.shape[0] // 2,
        fresnel_gauss_image.shape[1] // 2,
    )

    # Normalize amplitude so the maximum magnitude is 1
    analytic_gauss_image /= np.max(np.abs(analytic_gauss_image))
    fresnel_gauss_image /= np.max(np.abs(fresnel_gauss_image))

    # replace placeholder with
    det_circular_mask = make_aperture(X, Y, aperture_ratio=0.4)

    # mask amplitude
    analytic_gauss_image *= det_circular_mask
    fresnel_gauss_image *= det_circular_mask

    # force zero phase outside the aperture by replacing
    # the field there with its absolute‐value (i.e. exp(0j))
    analytic_gauss_image = np.where(det_circular_mask,
                                analytic_gauss_image,
                                np.abs(analytic_gauss_image))
    fresnel_gauss_image = np.where(det_circular_mask,
                                fresnel_gauss_image,
                                np.abs(fresnel_gauss_image))

    # Uncomment to plot cross-sections and overview plots
    central_index = analytic_gauss_image.shape[0] // 2
    analytic_phase_cross_section = np.angle(analytic_gauss_image[central_index, :])
    fresnel_phase_cross_section = np.angle(fresnel_gauss_image[central_index, :])

    analytic_amplitude_cross_section = np.abs(analytic_gauss_image[central_index, :])
    fresnel_amplitude_cross_section = np.abs(fresnel_gauss_image[central_index, :])

    # Plot cross-sections using helper
    fig, _ = plot_cross_sections(
        det_edge_x,
        [analytic_amplitude_cross_section, fresnel_amplitude_cross_section],
        [analytic_phase_cross_section, fresnel_phase_cross_section],
    )

    plt.savefig("test_gaussian_lens_vs_fresnel_cross_sections.png")
    # Overview plots using helper
    fig, _ = plot_overview(
        analytic_gauss_image,
        fresnel_gauss_image,
        pixel_size[0]*shape[0],
        pixel_size[1]*shape[1],
        suffix="",
        label1="Analytic Gaussian",
        label2="FFT"
    )
    plt.savefig("test_gaussian_lens_vs_fresnel_overview.png")

    # Assertions remain unchanged
    np.testing.assert_allclose(
        np.abs(analytic_gauss_image),
        np.abs(fresnel_gauss_image),
        rtol=5e-1,
        atol=5e-1,
        err_msg="Amplitude mismatch between analytic and fresnel",
    )

    np.testing.assert_allclose(
        unwrap_phase(np.angle(analytic_gauss_image)),
        unwrap_phase(np.angle(fresnel_gauss_image)),
        rtol=2,
        atol=2,
        err_msg="Phase mismatch between analytic and fresnel",
    )


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
    assert np.isclose(w_image, w0 * abs(M1), rtol=1e-8)

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
    assert np.allclose(w_out, w_expected, rtol=1e-12), \
        f"Waist mismatch at L={L}: got {w_out}, expected {w_expected}"

    # Radius from Q_inv
    R_out = _radius_from_Q_inv(q_inv_out)
    if L_factor == 0.0:
        assert R_out > 1e6, f"Radius at waist should be ~infinite, got {R_out}"
    else:
        assert np.allclose(R_out, R_expected, rtol=3e-3), \
            f"Radius mismatch at L={L}: got {R_out}, expected {R_expected}"


def test_beam_field_evaluation_smoke():
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


@pytest.mark.parametrize("voltage_ev", [200e3, 80e3])
def test_sigmoid_aperture_outside(voltage_ev):
    """
    If a Gaussian packet impinges on a SigmoidAperture well outside its radius,
    and the aperture's outside attenuation length L_outside is chosen so that
    exp(-k * L_outside) = 0.5, then the packet amplitude should be halved.

    We verify by inspecting the change in the imaginary part of the action S.const
    produced by the aperture: ΔA = exp(k * Im(ΔS)) = exp(-k * L_outside) = 0.5.
    """
    # Central ray placed far outside the aperture edge so the sigmoid -> 1
    radius = 1e-6  # m
    x_outside = 1e-4  # m (>> radius)

    # Build a single Gaussian packet with unit initial amplitude (C=1)
    S = TaylorExpofAction(
        const=jnp.array(0.0 + 0.0j),
        lin=jnp.zeros((2,), dtype=jnp.complex128),
        quad=jnp.zeros((2, 2), dtype=jnp.complex128),
    )

    ray_in = GaussianRayBeta(
        x=x_outside,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        pathlength=0.0,
        _one=1.0,
        S=S,
        C=1.0 + 0.0j,
        voltage=voltage_ev,
    )

    # Choose L_outside so attenuation outside is exactly 0.5
    k = float(2 * np.pi / ray_in.wavelength)  # use scalar for readability
    t_outside = 0.5

    # Make the aperture extremely sharp so sigmoid(r - radius) ~ 1 outside
    aperture = SigmoidAperture(
        z=0.0,
        radius=radius,
        sharpness=1e8,   # large for near-step behaviour
        t_outside=t_outside,
    )

    ray_out = aperture(ray_in)

    # Imaginary action increment at the packet centre
    d_im = float(jnp.imag(ray_out.S.const - ray_in.S.const))

    # Attenuation factor applied to the packet's amplitude
    attenuation = np.exp(-k * d_im)

    assert attenuation == pytest.approx(t_outside, rel=1e-6, abs=1e-9), (
        f"Expected amplitude halved; got attenuation={attenuation} (k={k}, ΔImS={d_im})"
    )


def test_beam_field_evaluation_attenuation():
    # Build a single Gaussian packet placed well outside the aperture so the
    # sigmoid -> 1 outside and the amplitude should be multiplied by t_outside.
    radius = 1e-6  # m
    x_outside = 1e-4  # m (>> radius)
    t_outside = 0.5
    sharpness = 1e8
    voltage_ev = 200e3

    S = TaylorExpofAction(
        const=jnp.array(0.0 + 0.0j),
        lin=jnp.zeros((2,), dtype=jnp.complex128),
        quad=jnp.zeros((2, 2), dtype=jnp.complex128),
    )

    ray_in = GaussianRayBeta(
        x=x_outside,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        pathlength=0.0,
        _one=1.0,
        S=S,
        C=1.0 + 0.0j,
        voltage=voltage_ev,
    )
    ray_in = ray_in.to_vector()
    detector = Detector(z=1e-15, pixel_size=(1e-1, 1e-1), shape=(200, 200))
    aperture = SigmoidAperture(
        z=0.0,
        radius=radius,
        sharpness=sharpness,
        t_outside=t_outside,
    )

    # # Propagate without aperture
    ray_no_ap = run_to_end(ray_in, [detector])
    field_no_ap = evaluate_gaussians_for(ray_no_ap, detector)

    # Propagate with aperture in front of the detector
    ray_with_ap = run_to_end(ray_in, [aperture, detector])
    field_with_ap = evaluate_gaussians_for(ray_with_ap, detector)

    # Compare peak amplitudes (peak should correspond to beam centre)
    max_no = np.max(np.abs(field_no_ap))
    max_with = np.max(np.abs(field_with_ap))

    # Attenuation factor applied by the aperture
    attenuation = float(max_with / max_no)

    assert attenuation == pytest.approx(t_outside, rel=1e-2, abs=1e-3), (
        f"Field peak attenuation mismatch: got {attenuation}, expected {t_outside}"
    )
