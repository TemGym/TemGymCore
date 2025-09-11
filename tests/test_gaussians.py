from temgym_core.gaussian import (
    zR,
    w_z,
    R,
    q_inv,
    gaussian_beam,
    Qinv_ABCD,
    propagate_misaligned_gaussian_jax_scan,
    make_gaussian_image,
    evaluate_gaussian_input_image,
    GaussianRay,
    decompose_Q_inv
)

from temgym_core.source import ParallelBeam
from temgym_core.components import Detector, Lens

from transfer_matrices import (
    calculate_z1_and_z2_from_M_and_f,
)
from skimage.restoration import unwrap_phase
from temgym_core.utils import make_aperture, zero_phase, FresnelPropagator, fresnel_lens_imaging_solution
import numpy as np
import jax.numpy as jnp
import jax
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


def _make_single_gaussian_ray_at(x0, y0, dx, dy, wavelength=500e-9, wo=1e-4):
    xs = jnp.array([x0])
    ys = jnp.array([y0])
    dxs = jnp.array([dx])
    dys = jnp.array([dy])
    zs = jnp.array([0.0])
    pathlengths = jnp.array([0.0])
    ones = jnp.array([1.0])
    amplitudes = jnp.array([1.0])
    radii_of_curv = jnp.array([[jnp.inf, jnp.inf]])
    theta = jnp.array([0.0])
    wavelengths = jnp.array([wavelength])
    waist_xy = jnp.array([[wo, wo]])
    return GaussianRay(
        x=xs,
        y=ys,
        dx=dxs,
        dy=dys,
        z=zs,
        pathlength=pathlengths,
        _one=ones,
        amplitude=amplitudes,
        waist_xy=waist_xy,
        radii_of_curv=radii_of_curv,
        wavelength=wavelengths,
        theta=theta,
    )


def test_zR_wz_R_basic():
    w0 = 1e-3  # 1 mm
    wl = 500e-9  # 500 nm
    zr = zR(w0, wl)
    # Known formula: zR = pi * w0^2 / wl
    assert np.isclose(
        float(zr), np.pi * w0 * w0 / wl, rtol=1e-9, atol=0.0
    )

    # w(z) at z=0 is w0
    assert np.isclose(w_z(w0, 0.0, zr), w0, rtol=1e-9, atol=0.0)

    # w(zR) = w0 * sqrt(2)
    assert np.isclose(
        float(w_z(w0, float(zr), zr)), w0 * np.sqrt(2.0), rtol=1e-9, atol=0.0
    )

    # R(zR) = 2 * zR
    assert np.isclose(float(R(float(zr), zr)), 2.0 * float(zr), rtol=1e-9, atol=0.0)


def test_q_inv_focus_and_away():
    w0 = 1e-3
    wl = 500e-9
    zr = zR(w0, wl)

    # At focus (z=0): q_inv = i * wl / (pi * w0^2)
    q0 = q_inv(0.0, w0, wl)
    expected_q0 = 1j * wl / (np.pi * w0 * w0)
    assert np.allclose(np.array(q0), np.array(expected_q0), rtol=1e-12, atol=0.0)

    # At z=zR: q_inv = 1/R - i * wl / (pi * w(z)^2)
    qz = q_inv(float(zr), w0, wl)
    R_at_zr = float(R(float(zr), zr))
    w_at_zr = float(w_z(w0, float(zr), zr))
    expected = (1.0 / R_at_zr) - 1j * wl / (np.pi * (w_at_zr ** 2))
    assert np.allclose(np.array(qz), np.array(expected), rtol=1e-12, atol=0.0)


def test_gaussian_beam_center_is_one():
    # When (x+offset_x, y+offset_y) == (0,0) the phase is 0 and exp(0)=1
    q = 1j * 1e-6  # arbitrary purely imaginary q^{-1}
    k = 2 * np.pi / 500e-9
    val = gaussian_beam(0.0, 0.0, q, k)
    assert np.allclose(np.array(val), np.array(1.0 + 0j), rtol=1e-12, atol=0.0)

    # With offsets cancelling coordinates
    val2 = gaussian_beam(1e-3, -2e-3, q, k, offset_x=-1e-3, offset_y=2e-3)
    assert np.allclose(np.array(val2), np.array(1.0 + 0j), rtol=1e-12, atol=0.0)


def test_Qinv_ABCD_identity_returns_same_Qinv():
    # For identity ABCD, Q2^{-1} should equal Q1^{-1}
    qx = 1j * 2.0
    qy = 1j * 3.0
    Q1_inv = jnp.array([[qx, 0.0], [0.0, qy]], dtype=jnp.complex128)
    eye2 = jnp.eye(2)
    Z = jnp.zeros((2, 2))
    Q2_inv = Qinv_ABCD(Q1_inv, eye2, Z, Z, eye2)
    assert np.allclose(np.array(Q2_inv), np.array(Q1_inv), rtol=1e-12, atol=0.0)


def test_propagate_free_space_matches_expected_formula():
    # Single beam, free-space ABCD: A=I, B=L I, C=0, D=I (B must be invertible here)
    w0 = 1e-3
    wl = 500e-9
    k = 2 * np.pi / wl
    L = 0.25  # metres

    # Input Gaussian at focus (z=0): q_inv = i * wl / (pi w0^2)
    q_scalar = 1j * wl / (np.pi * w0 * w0)
    Q1_inv = jnp.array([[[q_scalar, 0.0], [0.0, q_scalar]]], dtype=jnp.complex128)  # (1,2,2)

    A = jnp.array([np.eye(2)])
    B = jnp.array([L * np.eye(2)])
    C = jnp.array([np.zeros((2, 2))])
    D = jnp.array([np.eye(2)])
    e = jnp.array([[0.0, 0.0]])
    f = jnp.array([[0.0, 0.0]])

    # Observation points r2: simple small grid (N,2)
    xs = jnp.array([-1e-4, 0.0, 2e-4])
    ys = jnp.array([0.0, 1e-4, -2e-4])
    r2 = jnp.stack(jnp.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)

    # Zero misalignment at input
    r1m = jnp.array([[0.0, 0.0]])
    theta1m = jnp.array([[0.0, 0.0]])

    amp = jnp.array([1.0 + 0j])
    kval = jnp.array([k])

    # Compute field with implementation
    field = propagate_misaligned_gaussian_jax_scan(
        amp, Q1_inv, A, B, C, D, e, f, r1m, theta1m, kval, r2
    )  # (N,)

    # Expected: pref * exp(i*k/2 * r^T Q2_inv r)
    Q2_inv = Qinv_ABCD(Q1_inv[0], A[0], B[0], C[0], D[0])
    denom = A[0] + B[0] @ Q1_inv[0]
    pref = amp[0] / jnp.sqrt(jnp.linalg.det(denom))
    quad = jnp.einsum("ni,ij,nj->n", r2, Q2_inv, r2)
    expected = pref * jnp.exp(1j * (k / 2.0) * quad)

    assert np.allclose(np.array(field), np.array(expected), rtol=1e-9, atol=1e-12)


def test_gaussian_free_space_vs_fresnel():

    num_rays = 1
    propagation_distance = 0.001
    pixel_size = (0.0005, 0.0005)
    wavelength = 0.001
    wo = 0.1
    shape = (2000, 2000)
    beam = ParallelBeam(z=0.0, radius=0.0, offset_xy=(0.0, 0.0))
    detector = Detector(z=propagation_distance, pixel_size=pixel_size, shape=shape)
    model = [beam, detector]
    base_rays = beam.make_rays(num_rays, random=False)
    num_rays = int(base_rays.x.size)

    xs = jnp.array(np.asarray([base_rays.x]))
    ys = jnp.array(np.asarray([base_rays.y]))
    dxs = jnp.array(np.asarray([base_rays.dx]))
    dys = jnp.array(np.asarray([base_rays.dy]))
    zs = jnp.array(np.zeros(num_rays))
    pathlengths = jnp.array(np.zeros(num_rays))
    ones = jnp.array(np.ones(num_rays))
    amplitudes = jnp.array(np.ones(num_rays))
    radii_of_curv = jnp.array(np.full((num_rays, 2), np.inf))
    theta = jnp.array(np.zeros(num_rays))
    wavelengths = jnp.array(np.asarray([wavelength]))
    waist_xy = jnp.array(np.full((num_rays, 2), wo))

    rays = GaussianRay(
        x=xs,
        y=ys,
        dx=dxs,
        dy=dys,
        z=zs,
        pathlength=pathlengths,
        _one=ones,
        amplitude=amplitudes,
        waist_xy=waist_xy,  # 1x2 per Gaussian Ray
        radii_of_curv=radii_of_curv,  # 1x2 per Gaussian Ray
        wavelength=wavelengths,
        theta=theta,
    )

    det_edge_x, det_edge_y = detector.coords_1d
    Y, X = np.meshgrid(det_edge_y, det_edge_x, indexing="ij")

    analytic_gauss_image = make_gaussian_image(rays, model)

    # Fresnel Version
    q1_inv = q_inv(0.0, wo, wavelength)
    gauss_input = gaussian_beam(X, Y, q1_inv, 2 * np.pi / wavelength)
    gauss_input = zero_phase(
        gauss_input,
        gauss_input.shape[0] // 2,
        gauss_input.shape[1] // 2,
    )

    fresnel_gauss_image = FresnelPropagator(gauss_input, det_edge_x, wavelength, propagation_distance)

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

    # Assertions remain unchanged
    np.testing.assert_allclose(
        np.abs(analytic_gauss_image),
        np.abs(fresnel_gauss_image),
        rtol=1e-1,
        atol=1e-1,
        err_msg="Amplitude mismatch between analytic and fresnel",
    )

    np.testing.assert_allclose(
        unwrap_phase(np.angle(analytic_gauss_image)),
        unwrap_phase(np.angle(fresnel_gauss_image)),
        rtol=1e-1,
        atol=1e-1,
        err_msg="Phase mismatch between analytic and fresnel",
    )


def test_gaussian_lens_vs_fresnel():

    M = -2
    f = 5e-3
    defocus = 1e-4
    z1, z2 = calculate_z1_and_z2_from_M_and_f(M, f)

    beam = ParallelBeam(z=0, radius=0.0, offset_xy=(0.0, 0.0))
    lens = Lens(z=abs(z1) + defocus, focal_length=f)

    # setup params
    pixel_size = (5e-6, 5e-6)
    shape = (2048, 2048)
    detector = Detector(z=abs(z1) + z2, pixel_size=pixel_size, shape=shape)

    num_rays = 1

    model = [beam, lens, detector]
    base_rays = beam.make_rays(num_rays, random=False)
    num_rays = int(base_rays.x.size)

    xs = jnp.array(np.asarray([base_rays.x]))
    ys = jnp.array(np.asarray([base_rays.y]))
    dxs = jnp.array(np.asarray([base_rays.dx]))
    dys = jnp.array(np.asarray([base_rays.dy]))
    zs = jnp.array(np.zeros(num_rays))
    pathlengths = jnp.array(np.zeros(num_rays))
    ones = jnp.array(np.ones(num_rays))
    amplitudes = jnp.array(np.ones(num_rays))
    radii_of_curv = jnp.array(np.full((num_rays, 2), np.inf))
    theta = jnp.array(np.zeros(num_rays))
    wavelength = 1e-5
    wavelengths = jnp.array(np.full((num_rays,), wavelength))
    wo = 5e-4
    waist_xy = jnp.array(np.full((num_rays, 2), wo))

    rays = GaussianRay(
        x=xs,
        y=ys,
        dx=dxs,
        dy=dys,
        z=zs,
        pathlength=pathlengths,
        _one=ones,
        amplitude=amplitudes,
        waist_xy=waist_xy,  # 1x2 per Gaussian Ray
        radii_of_curv=radii_of_curv,  # 1x2 per Gaussian Ray
        wavelength=wavelengths,
        theta=theta,
    )

    det_edge_x, det_edge_y = detector.coords_1d
    Y, X = np.meshgrid(det_edge_y, det_edge_x, indexing="ij")

    analytic_gauss_image = make_gaussian_image(rays, model)
    analytic_gauss_image = zero_phase(
        analytic_gauss_image,
        analytic_gauss_image.shape[0] // 2,
        analytic_gauss_image.shape[1] // 2,
    )
    # Fresnel Version
    q1_inv = q_inv(0.0, wo, wavelength)
    gauss_input = gaussian_beam(X, Y, q1_inv, 2 * np.pi / wavelength)
    gauss_input = zero_phase(
        gauss_input,
        gauss_input.shape[0] // 2,
        gauss_input.shape[1] // 2,
    )

    fresnel_gauss_image = fresnel_lens_imaging_solution(gauss_input, Y, X, pixel_size[0], wavelength,
                                                       defocus+np.abs(z1), f, z2)

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
    # central_index = analytic_gauss_image.shape[0] // 2
    # analytic_phase_cross_section = np.angle(analytic_gauss_image[central_index, :])
    # fresnel_phase_cross_section = np.angle(fresnel_gauss_image[central_index, :])

    # analytic_amplitude_cross_section = np.abs(analytic_gauss_image[central_index, :])
    # fresnel_amplitude_cross_section = np.abs(fresnel_gauss_image[central_index, :])

    # # Plot cross-sections using helper
    # fig, _ = plot_cross_sections(
    #     det_edge_x,
    #     [analytic_amplitude_cross_section, fresnel_amplitude_cross_section],
    #     [unwrap_phase(analytic_phase_cross_section), unwrap_phase(fresnel_phase_cross_section)],
    # )

    # plt.savefig("test_gaussian_lens_vs_fresnel_cross_sections.png")
    # # Overview plots using helper
    # fig, _ = plot_overview(
    #     analytic_gauss_image,
    #     fresnel_gauss_image,
    #     pixel_size[0]*shape[0],
    #     pixel_size[1]*shape[1],
    #     suffix="",
    #     label1="Analytic Gaussian",
    #     label2="Collins FFT"
    # )
    # plt.savefig("test_gaussian_lens_vs_fresnel_overview.png")

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


def test_gaussian_two_beam_interference_vs_fresnel():
    # Detector setup
    pixel_size = (5e-6, 5e-6)
    shape = (1024, 1024)
    # Model Creation
    f = 5e-3
    defocus = 2e-3
    z2 = (1 / f) ** -1 + defocus

    lens = Lens(z=0.0, focal_length=f)
    detector = Detector(z=z2, pixel_size=pixel_size, shape=shape)
    det_edge_x, det_edge_y = detector.coords_1d
    Y, X = np.meshgrid(det_edge_y, det_edge_x, indexing="ij")
    model = [lens, detector]

    num_rays = 2
    wavelength = 1e-4
    wo = 0.5e-3

    x01 = 0.
    y01 = 0.
    x02 = 0.
    y02 = 0.
    dx01 = 0.
    dy01 = 1e-1
    dx02 = 1e-1
    dy02 = 0.

    input_x = jnp.array([x01, x02])
    input_y = jnp.array([y01, y02])
    input_dx = jnp.array([dx01, dx02])
    input_dy = jnp.array([dy01, dy02])

    num_rays = len(input_x)

    # Gaussian Beam Input
    xs = input_x
    ys = input_y
    dxs = input_dx
    dys = input_dy
    zs = jnp.array(np.zeros(num_rays))
    pathlengths = jnp.array(np.zeros(num_rays))
    ones = jnp.array(np.ones(num_rays))
    amplitudes = jnp.array(np.ones(num_rays))
    radii_of_curv = jnp.array(np.full((num_rays, 2), np.inf))
    theta = jnp.array(np.zeros(num_rays))
    wavelengths = jnp.array(np.full((num_rays,), wavelength))
    waist_xy = jnp.array(np.full((num_rays, 2), wo))

    rays = GaussianRay(
        x=xs,
        y=ys,
        dx=dxs,
        dy=dys,
        z=zs,
        pathlength=pathlengths,
        _one=ones,
        amplitude=amplitudes,
        waist_xy=waist_xy,  # 1x2 per Gaussian Ray
        radii_of_curv=radii_of_curv,  # 1x2 per Gaussian Ray
        wavelength=wavelengths,
        theta=theta,
    )
    analytic_gauss_image = make_gaussian_image(rays, model)
    analytic_gauss_image = zero_phase(analytic_gauss_image, shape[0]//2, shape[1]//2)

    # Fresnel Version
    q1_inv = q_inv(1e-13, wo, wavelength)

    k = 2 * np.pi / wavelength
    gaussian_shifted_1 = gaussian_beam((Y - y01), (X - x01), q1_inv, k).ravel()
    gaussian_shifted_2 = gaussian_beam((Y - y02), (X - x02), q1_inv, k).ravel()

    # Ensure r1 is (n_pix, 2)
    r1 = np.stack([Y.ravel(), X.ravel()], axis=0).T

    # Build per-beam offsets and tilts as (n_gaussians, 2): [y, x]
    r1m = np.array([[y01, x01],
                    [y02, x02]])
    theta1m = np.column_stack((input_dy, input_dx))

    # For each beam g and pixel n: phase = k * (r1[n] - r1m[g]) · theta1m[g]
    dot = np.einsum('gni,gi->gn', r1[None, :, :] - r1m[:, None, :], theta1m)
    tilted_shifted_plane_wave1 = np.exp(1j * k * dot[0])
    tilted_shifted_plane_wave2 = np.exp(1j * k * dot[1])

    gaussian_misaligned1 = (gaussian_shifted_1 * tilted_shifted_plane_wave1).reshape(shape[0], shape[1])
    gaussian_misaligned2 = (gaussian_shifted_2 * tilted_shifted_plane_wave2).reshape(shape[0], shape[1])
    gaussian_misaligned = gaussian_misaligned1 + gaussian_misaligned2

    fresnel_gauss_image = fresnel_lens_imaging_solution(gaussian_misaligned, Y, X, pixel_size[0], wavelength, 0.0, f, z2)
    fresnel_gauss_image = zero_phase(fresnel_gauss_image, shape[0]//2, shape[1]//2)

    # Normalize amplitude so the maximum magnitude is 1
    analytic_gauss_image /= np.max(np.abs(analytic_gauss_image))
    fresnel_gauss_image /= np.max(np.abs(fresnel_gauss_image))

    det_circular_mask = make_aperture(X, Y, aperture_ratio=0.8)

    # mask amplitude
    analytic_gauss_image *= det_circular_mask
    fresnel_gauss_image *= det_circular_mask

    # force zero phase outside the aperture by replacing
    # the field there with its absolute‐value (i.e. exp(0j))
    analytic_gauss_image = np.where(det_circular_mask,
                                analytic_gauss_image,
                                np.abs(analytic_gauss_image) * 0.0)
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
        [unwrap_phase(analytic_phase_cross_section), unwrap_phase(fresnel_phase_cross_section)],
    )

    plt.savefig("test_two_beam_interference_cross_section.png")
    # Overview plots using helper
    fig, _ = plot_overview(
        analytic_gauss_image,
        fresnel_gauss_image,
        pixel_size[0]*shape[0],
        pixel_size[1]*shape[1],
        suffix="",
        label1="Analytic Gaussian",
        label2="Fresnel Gaussian"
    )
    plt.savefig("test_two_beam_interference_overview.png")

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


def test_gaussian_propagates_to_correct_quadrant():
    # Setup detector
    z_prop = 1.0
    detector = Detector(z=z_prop, pixel_size=(1e-4, 1e-4), shape=(256, 256))
    det_edge_x, det_edge_y = detector.coords_1d
    Y, X = np.meshgrid(det_edge_y, det_edge_x, indexing="ij")

    # Use the max of the zero-tilt (central) beam to define the reference center
    central_ray = _make_single_gaussian_ray_at(0.0, 0.0, 0.0, 0.0)
    central_img = evaluate_gaussian_input_image(central_ray, detector)
    cy, cx = np.unravel_index(np.argmax(np.abs(central_img)), central_img.shape)
    x_center = X[cy, cx]
    y_center = Y[cy, cx]

    # Small helper for robust sign with tolerance
    def sign_with_tol(v, eps=1e-12):
        if v > eps:
            return 1
        if v < -eps:
            return -1
        return 0

    # Test four tilt combinations: (++), (+-), (-+), (--)
    tilt_mag = 1e-3
    combos = [
        (+tilt_mag, +tilt_mag),  # top-right
        (+tilt_mag, -tilt_mag),  # bottom-right
        (-tilt_mag, +tilt_mag),  # top-left
        (-tilt_mag, -tilt_mag),  # bottom-left
    ]

    for dx, dy in combos:
        ray = _make_single_gaussian_ray_at(0.0, 0.0, dx, dy)
        output_image = make_gaussian_image(ray, [detector])
        my, mx = np.unravel_index(np.argmax(np.abs(output_image)), output_image.shape)
        x_max = X[my, mx]
        y_max = Y[my, mx]

        # Compare quadrant via sign of (max - center) in physical coordinates
        sx = sign_with_tol(float(x_max - x_center))
        sy = sign_with_tol(float(y_max - y_center))
        exp_sx = sign_with_tol(dx)
        exp_sy = sign_with_tol(dy)

        assert sx == exp_sx, f"dx={dx:+} expected {'right' if exp_sx > 0 else 'left'} shift; got x={x_max:.3e} (center {x_center:.3e})"  # noqa
        assert sy == exp_sy, f"dy={dy:+} expected {'top' if exp_sy > 0 else 'bottom'} shift; got y={y_max:.3e} (center {y_center:.3e})"  # noqa


def _principal_angle(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def test_input_gaussian_phase_ramp_center_and_slope():
    # Detector and on-grid reference point (ensures exact sampling at r1m)
    detector = Detector(z=0.0, pixel_size=(1e-5, 1e-5), shape=(512, 512))
    det_x, det_y = detector.coords_1d
    ix = len(det_x) // 2
    iy = len(det_y) // 2
    r1m = (float(det_x[ix]), float(det_y[iy]))  # on-grid

    wavelength = 500e-9
    k = 2 * np.pi / wavelength
    delta = 8

    cases = [
        # X-only tilt
        dict(dx=2e-4, dy=0.0, rtol_x=1e-2, rtol_y=None, atol_y=1e-3),
        # Y-only tilt
        dict(dx=0.0, dy=-1e-4, rtol_x=None, atol_x=1e-3, rtol_y=1e-2),
        # Combined tilt
        dict(dx=1.5e-4, dy=-1.0e-4, rtol_x=2e-2, rtol_y=2e-2),
    ]

    for case in cases:
        dx = case["dx"]
        dy = case["dy"]
        ray = _make_single_gaussian_ray_at(r1m[0], r1m[1], dx, dy, wavelength=wavelength, wo=1e-4)
        img = evaluate_gaussian_input_image(ray, detector)

        # Phase at r1m should be ~0 (mod 2π)
        phi0 = np.angle(img[iy, ix])
        assert abs(_principal_angle(phi0)) < 1e-6

        # Local slopes from 1D unwrapped lines through r1m
        row_phase = np.unwrap(np.angle(img[iy, :]))
        row_phase -= row_phase[ix]
        slope_x_est = (row_phase[ix + delta] - row_phase[ix - delta]) / (det_x[ix + delta] - det_x[ix - delta])

        col_phase = np.unwrap(np.angle(img[:, ix]))
        col_phase -= col_phase[iy]
        slope_y_est = (col_phase[iy + delta] - col_phase[iy - delta]) / (det_y[iy + delta] - det_y[iy - delta])

        # Check X slope
        if case.get("rtol_x") is not None:
            np.testing.assert_allclose(slope_x_est, k * dx, rtol=case["rtol_x"], atol=0.0)
        else:
            np.testing.assert_allclose(slope_x_est, 0.0, atol=case["atol_x"])

        # Check Y slope
        if case.get("rtol_y") is not None:
            np.testing.assert_allclose(slope_y_est, k * dy, rtol=case["rtol_y"], atol=0.0)
        else:
            np.testing.assert_allclose(slope_y_est, 0.0, atol=case["atol_y"])

def test_random_gaussian_input_slope_and_angle_against_fresnel():
    execution_number = 5
    for _ in range(execution_number):
        z_prop = 1
        rng = np.random.default_rng()
        r1m = jnp.array(rng.uniform(-1e-3, 1e-3, size=2))  # m, random in [0, 1e-5]
        theta1m = jnp.array(rng.uniform(-1e-4, 1e-4, size=2))  # rad, random in [-2e-6, 2e-6]
        input_ray = GaussianRay(x=r1m[0], y=r1m[1], dx=theta1m[0], dy=theta1m[1], z=0.0, pathlength=0.0,
                                _one=1.0, amplitude=1.0, waist_xy=jnp.array([1e-4, 1e-4]),
                                radii_of_curv=jnp.array([jnp.inf, jnp.inf]),
                                wavelength=500e-9, theta=0.0)

        detector = Detector(z=z_prop, pixel_size=(1e-5, 1e-5), shape=(1024, 1024))
        detector_width = detector.pixel_size[0] * detector.shape[0]
        input_image = evaluate_gaussian_input_image(input_ray, detector)

        model = [detector]
        output_image = make_gaussian_image(input_ray, model)

        fresnel_image = FresnelPropagator(input_image, detector_width, wavelength=input_ray.wavelength, z=z_prop)  # noqa

        max_idx_fresnel = jnp.argmax(jnp.abs(fresnel_image))
        max_idx_output = jnp.argmax(jnp.abs(output_image))

        np.testing.assert_allclose(
            max_idx_fresnel,
            max_idx_output,
            atol=3,
            err_msg="Maximum pixel index mismatch between Fresnel and analytic Gaussian"
        )


def test_astigmatic_rotated_gaussian_beam():
    # Helper to get principal-axis angle (major axis) from intensity image
    def principal_axis_angle(field, X, Y):
        I = np.abs(field) ** 2  # noqa
        wsum = np.sum(I) + 1e-30
        xbar = np.sum(I * X) / wsum
        ybar = np.sum(I * Y) / wsum
        Xc = X - xbar
        Yc = Y - ybar
        Cxx = np.sum(I * Xc * Xc) / wsum
        Cyy = np.sum(I * Yc * Yc) / wsum
        Cxy = np.sum(I * Xc * Yc) / wsum
        C = np.array([[Cxx, Cxy], [Cxy, Cyy]])
        vals, vecs = np.linalg.eigh(C)
        v = vecs[:, np.argmax(vals)]  # major axis
        ang = np.arctan2(v[1], v[0])
        # Map to [-pi/2, pi/2) to remove 180° ambiguity
        return (ang + np.pi / 2) % np.pi - np.pi / 2

    def angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    # Detector and coordinate grids
    detector = Detector(z=0.0, pixel_size=(1e-5, 1e-5), shape=(512, 512))
    det_x, det_y = detector.coords_1d
    Y, X = np.meshgrid(det_y, det_x, indexing="ij")

    # Elliptical (astigmatic) beam: major axis along local x' (w_x > w_y)
    waist_xy = jnp.array([2e-4, 1e-4])

    # Test several rotations (radians)
    thetas = np.array([-np.pi / 3, -np.pi / 6, 0.0, np.pi / 6, np.pi / 4, np.pi / 3])
    tol = 0.1  # ~5.7 degrees

    for theta in thetas:
        rays = GaussianRay(
            x=0.0,
            y=0.0,
            dx=0.0,
            dy=0.0,
            z=0.0,
            pathlength=0.0,
            _one=1.0,
            amplitude=1.0,
            waist_xy=waist_xy,
            radii_of_curv=jnp.array([jnp.inf, jnp.inf]),
            wavelength=500e-9,
            theta=theta,
        )
        img = evaluate_gaussian_input_image(rays, detector)
        phi = principal_axis_angle(img, X, Y)

        # Major-axis angle should match theta (up to pi periodicity handled in principal_axis_angle)
        assert abs(angle_diff(phi, float(theta))) < tol, f"Estimated angle {phi:.3f} rad != theta {theta:.3f} rad"


def _angle_diff_mod_pi(a, b):
    """Smallest angular difference between a and b, modulo pi (axis direction)."""
    d = np.arctan2(np.sin(a - b), np.cos(a - b))  # wrap to [-pi, pi]
    # Equivalent directions under 180° flip
    return min(abs(d), abs(d + np.pi), abs(d - np.pi))


def test_decompose_q_inv_recovers_parameters_single():
    wl = 500e-9
    # Distinct waists so eigenvalue order is well-defined (ascending imag parts)
    w1 = 1.0e-4  # first principal axis (smaller)
    w2 = 2.0e-4  # second principal axis (larger)
    R1 = 0.5     # m
    R2 = np.inf  # planar in second axis
    theta = np.pi / 6

    ray = GaussianRay(x=0.0, y=0.0, dx=0.0, dy=0.0, z=0.0, pathlength=0.0, _one=1.0, amplitude=1.0,
                      waist_xy=jnp.array([w1, w2]), radii_of_curv=jnp.array([R1, R2]),
                      wavelength=wl, theta=theta)

    ray = ray.to_vector()
    Qinv = ray.Q_inv

    ow1, ow2, oR1, oR2, otheta = decompose_Q_inv(Qinv, wl)

    np.testing.assert_allclose(np.array(ow1), w1, rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(np.array(ow2), w2, rtol=1e-9, atol=0.0)
    # Radius: allow inf handling
    assert np.isinf(oR2) and oR2 > 0
    np.testing.assert_allclose(np.array(oR1), R1, rtol=1e-9, atol=0.0)
    assert _angle_diff_mod_pi(float(otheta[0]), theta) < 1e-9


def test_decompose_q_inv_batched():
    wl = 1.3e-6
    params = [
        # (w1, w2, R1, R2, theta)
        (1e-5, 3e-5, np.inf, np.inf, np.pi/6),
        (1.2e-4, 5e-5, 0.0001, 0.1, -np.pi/4),
        (1e-4, 3e-4, 0.01, np.inf, -np.pi/3),
    ]

    for i, (w1, w2, R1, R2, th) in enumerate(params):
        rays_vec = GaussianRay(
            x=0.0,
            y=0.0,
            dx=0.0,
            dy=0.0,
            z=0.0,
            pathlength=0.0,
            _one=1.0,
            amplitude=1.0,
            waist_xy=jnp.array([[w1, w2]]),
            radii_of_curv=jnp.array([[R1, R2]]),
            wavelength=wl,
            theta=th,
        ).to_vector()

        Qs = rays_vec.Q_inv

        detector = Detector(z=0.0, pixel_size=(1e-6, 1e-6), shape=(512, 512))
        original_img = evaluate_gaussian_input_image(rays_vec, detector)

        ow1, ow2, oR1, oR2, otheta = decompose_Q_inv(Qs, wl)

        # Check that re-constructed parameters reproduce the original image
        recon_rays = GaussianRay(
            x=0.0,
            y=0.0,
            dx=0.0,
            dy=0.0,
            z=0.0,
            pathlength=0.0,
            _one=1.0,
            amplitude=1.0,
            waist_xy=jnp.array([[ow1[0], ow2[0]]]),
            radii_of_curv=jnp.array([[oR1[0], oR2[0]]]),
            wavelength=wl,
            theta=otheta,
        ).to_vector()
        recon_img = evaluate_gaussian_input_image(recon_rays, detector)

        np.testing.assert_allclose(
            np.abs(original_img),
            np.abs(recon_img),
            rtol=1e-9,
        )

        # Amplitude and phase comparison (original vs reconstructed)
        orig_amp = np.abs(original_img)
        recon_amp = np.abs(recon_img)

        # Unwrapped phase (optional for clearer visualization)
        orig_phase = np.angle(original_img)
        recon_phase = np.angle(recon_img)

        extent = [
            detector.coords_1d[0][0],
            detector.coords_1d[0][-1],
            detector.coords_1d[1][0],
            detector.coords_1d[1][-1],
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        im0 = axes[0, 0].imshow(orig_amp, cmap="viridis", extent=extent)
        axes[0, 0].set_title("Original Amplitude")
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(recon_amp, cmap="viridis", extent=extent)
        axes[0, 1].set_title("Reconstructed Amplitude")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(orig_phase, cmap="twilight", extent=extent)
        axes[1, 0].set_title("Original Phase")
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im3 = axes[1, 1].imshow(recon_phase, cmap="twilight", extent=extent)
        axes[1, 1].set_title("Reconstructed Phase")
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        for ax_ in axes.ravel():
            ax_.set_xlabel("x (m)")
            ax_.set_ylabel("y (m)")

        plt.tight_layout()
        plt.savefig(f"test_decompose_q_inv_batched_case_{i}_comparison.png")

        # np.testing.assert_allclose(np.array(ow1[i]), w1, rtol=1e-9, atol=0.0)
        # np.testing.assert_allclose(np.array(ow2[i]), w2, rtol=1e-9, atol=0.0)
        # if np.isinf(R1):
        #     assert np.isinf(oR1[i]) and oR1[i] > 0
        # else:
        #     np.testing.assert_allclose(np.array(oR1[i]), R1, rtol=1e-9, atol=0.0)
        # if np.isinf(R2):
        #     assert np.isinf(oR2[i]) and oR2[i] > 0
        # else:
        #     np.testing.assert_allclose(np.array(oR2[i]), R2, rtol=1e-9, atol=0.0)
        # assert _angle_diff_mod_pi(float(otheta[i]), th) < 1e-9
