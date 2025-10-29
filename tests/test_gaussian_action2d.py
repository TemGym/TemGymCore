import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from skimage.restoration import unwrap_phase

from temgym_core.gaussian_action2D import (
    make_gaussian,
    ABCDPropagator2D,
    Lens,
    run_to_end,
)
from temgym_core.components import Detector

from temgym_core.utils import (
    energy2wavelength,
    wavelength2energy,
    zero_phase,
    make_aperture,
    FresnelPropagator,
    fresnel_lens_imaging_solution,
)

from temgym_core.evaluate import evaluate_gaussians_for
from transfer_matrices import calculate_z1_and_z2_from_M_and_f

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

    for ampl, lab in zip(amplitudes, labels):
        axs[0].plot(x, ampl, label=f"{lab} {suffix} Amplitude", linestyle=linestyle)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Central Row Amplitude Cross Section")
    axs[0].legend()
    axs[0].grid(True)

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


def _field_on_grid(ray, grid: Detector) -> np.ndarray:
    field = evaluate_gaussians_for(ray, grid)
    return np.asarray(field)


def _detector_mesh(detector: Detector) -> tuple[np.ndarray, np.ndarray]:
    det_edge_x, det_edge_y = detector.coords_1d
    det_edge_x = np.asarray(det_edge_x)
    det_edge_y = np.asarray(det_edge_y)
    Y, X = np.meshgrid(det_edge_y, det_edge_x, indexing="ij")
    return Y, X


def _window_size(detector: Detector) -> float:
    width_x = float(detector.pixel_size[1]) * detector.shape[1]
    return width_x


def test_free_space_paraxial_updates_q_inv():
    # Analytic solution to free space propagation of complex q_inv parameter
    # of gaussian.
    voltage = 100e3
    r_curv = -0.001
    waist = 1e-6
    wavelength = energy2wavelength(voltage)
    q_inv = 1/r_curv + 1j * (wavelength) / (jnp.pi * waist ** 2)
    Q_inv = jnp.diag(jnp.array([q_inv, q_inv], dtype=jnp.complex128))

    ray_in = make_gaussian(
        x=0.0,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=waist,
        waist_y=waist,
        phase=0.0,
        amp=1.0,
        rcurv_x=r_curv,
        rcurv_y=r_curv,
    )

    dist = 1e-3
    propagator = ABCDPropagator2D.free_space(dist)
    ray_out = propagator(ray_in)

    I2 = jnp.eye(2, dtype=jnp.complex128)
    expected_Q = Q_inv @ jnp.linalg.inv(I2 + dist * Q_inv)

    np.testing.assert_allclose(
        np.asarray(ray_out.S2),
        np.asarray(expected_Q),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_in.C),
        np.asarray(1.0 + 0.0j),
        rtol=0.0,
        atol=1e-12,
    )
    k = 2 * jnp.pi / wavelength
    det_term = jnp.linalg.det(I2 + dist * Q_inv)
    expected_C = ray_in.C / jnp.sqrt(det_term)

    np.testing.assert_allclose(
        np.asarray(ray_out.C),
        np.asarray(expected_C),
        rtol=1e-12,
        atol=1e-12,
    )


def test_thin_lens_updates_q_inv():
    # Analytic solution for a thin lens: q_inv_out = q_inv_in - 1/f
    voltage = 100e3
    r_curv = -0.001
    waist = 1e-6
    wavelength = energy2wavelength(voltage)
    q_inv = 1 / r_curv + 1j * (wavelength) / (jnp.pi * waist ** 2)
    Q_inv = jnp.diag(jnp.array([q_inv, q_inv], dtype=jnp.complex128))

    ray_in = make_gaussian(
        x=0.0,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=waist,
        waist_y=waist,
        phase=0.0,
        amp=1.0,
        rcurv_x=r_curv,
        rcurv_y=r_curv,
    )

    f = 5e-3

    # support either ABCDPropagator2D.thin_lens or ABCDPropagator2D.lens if present
    lens_ctor = getattr(ABCDPropagator2D, "thin_lens", None) or getattr(
        ABCDPropagator2D, "lens", None
    )
    if lens_ctor is None:
        pytest.skip("No thin_lens or lens constructor on ABCDPropagator2D")

    propagator = lens_ctor(f)
    ray_out = propagator(ray_in)

    I2 = jnp.eye(2, dtype=jnp.complex128)
    expected_Q = Q_inv - (1.0 / f) * I2

    np.testing.assert_allclose(
        np.asarray(ray_out.S2),
        np.asarray(expected_Q),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_in.C),
        np.asarray(1.0 + 0.0j),
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out.C),
        np.asarray(ray_in.C),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fourier_transform_ABCD_matrix_updates_q_inv():
    # Test general ABCD transform: A=0, B=f, C=-1/f, D=0
    voltage = 100000
    r_curv = jnp.inf
    waist = 3e-8
    wavelength = energy2wavelength(voltage)
    q_inv = 1 / r_curv + 1j * (wavelength) / (jnp.pi * waist ** 2)
    Q_inv = jnp.diag(jnp.array([q_inv, q_inv], dtype=jnp.complex128))

    ray_in = make_gaussian(
        x=0.0e-6,
        y=0.2e-6,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=waist,
        waist_y=waist,
        phase=0.0,
        amp=1.0,
        rcurv_x=r_curv,
        rcurv_y=r_curv,
    )

    f = 1e-2
    A, B, C, D = 0.0, f, -1.0 / f, 0.0

    propagator = ABCDPropagator2D.fourier_transform(f)

    ray_out = propagator(ray_in)

    I2 = jnp.eye(2, dtype=jnp.complex128)
    expected_Q = (C * I2 + D * Q_inv) @ jnp.linalg.inv(A * I2 + B * Q_inv)

    np.testing.assert_allclose(
        np.asarray(ray_out.S2),
        np.asarray(expected_Q),
        rtol=1e-12,
        atol=1e-12,
    )

    # ---- Prefactor checks ----
    # Theory: C_out / C_in = det(A + B Q_in)^(-1/2) * exp(i k (dS0 + L))
    # For a centered beam with zero slopes, dS0 = 0.
    ABQ = (A * I2 + B * Q_inv)
    det_ABQ = jnp.linalg.det(ABQ)

    C_in = ray_in.C
    C_out = ray_out.C

    # Magnitude: |C_out/C_in| = 1 / sqrt(|det(A + B Q_in)|)
    exp_mag = 1.0 / jnp.sqrt(jnp.abs(det_ABQ))
    got_mag = jnp.abs(C_out) / jnp.abs(C_in)
    np.testing.assert_allclose(np.asarray(got_mag),
                               np.asarray(exp_mag),
                               rtol=1e-12, atol=1e-12)


def test_fourier_transform_ABCD_matrix_updates_against_stepwise():
    # Test general ABCD transform: A=0, B=f, C=-1/f, D=0
    voltage = 100000
    r_curv = jnp.inf
    waist = 3e-8

    ray_in = make_gaussian(
        x=0.0e-6,
        y=0.2e-6,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=waist,
        waist_y=waist,
        phase=0.0,
        amp=1.0,
        rcurv_x=r_curv,
        rcurv_y=r_curv,
    )
    f = 1e-2
    ray_out_abcd = ABCDPropagator2D.fourier_transform(f)(ray_in)

    lens = Lens(z=f, focal_length=f)
    output_probe_grid = Detector(z=2 * f, pixel_size=(1, 1), shape=(1, 1))

    components = (lens, output_probe_grid)
    ray_out_step_wise = run_to_end(ray_in, components)

    propagator = ABCDPropagator2D.free_space(f)
    A = jnp.eye(2)
    B = jnp.zeros((2, 2))
    C = -1.0 / f * jnp.eye(2)
    D = jnp.eye(2)
    lens = ABCDPropagator2D(A=A, B=B, C=C, D=D)
    ray_out_first = propagator(ray_in)
    ray_out_lens = lens(ray_out_first)
    ray_out_step_wise_abcd = propagator(ray_out_lens)

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.S2),
        np.asarray(ray_out_step_wise_abcd.S2),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.C),
        np.asarray(ray_out_step_wise_abcd.C),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.S2),
        np.asarray(ray_out_step_wise.S2),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.C),
        np.asarray(ray_out_step_wise.C),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_step_wise_abcd.S2),
        np.asarray(ray_out_step_wise.S2),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_step_wise_abcd.C),
        np.asarray(ray_out_step_wise.C),
        rtol=1e-7,
        atol=1e-7,
    )


def test_gaussian_free_space_vs_fresnel():
    propagation_distance = 0.001
    pixel_size = (0.0005, 0.0005)
    wavelength = 0.0001
    wo = 0.1
    shape = (2000, 2000)

    voltage = wavelength2energy(wavelength)

    input_grid = Detector(z=0.0, pixel_size=pixel_size, shape=shape)
    detector = Detector(z=propagation_distance, pixel_size=pixel_size, shape=shape)

    gaussian = make_gaussian(
        x=0.0,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=wo,
        waist_y=wo,
        amp=1.0,
        phase=0.0,
        rcurv_x=jnp.inf,
        rcurv_y=jnp.inf,
    )

    propagated = run_to_end(gaussian, (detector,))
    analytic_gauss_image = _field_on_grid(propagated, detector)

    gauss_input = _field_on_grid(gaussian, input_grid)
    gauss_input = zero_phase(
        gauss_input,
        gauss_input.shape[0] // 2,
        gauss_input.shape[1] // 2,
    )

    Y, X = _detector_mesh(detector)
    fresnel_gauss_image = FresnelPropagator(
        gauss_input,
        _window_size(detector),
        wavelength,
        propagation_distance,
    )
    analytic_gauss_image = np.array(analytic_gauss_image)
    fresnel_gauss_image = np.array(fresnel_gauss_image)

    analytic_gauss_image /= np.max(np.abs(analytic_gauss_image))
    fresnel_gauss_image /= np.max(np.abs(fresnel_gauss_image))

    det_circular_mask = make_aperture(X, Y, aperture_ratio=0.4)

    analytic_gauss_image *= det_circular_mask
    fresnel_gauss_image *= det_circular_mask

    analytic_gauss_image = np.where(
        det_circular_mask,
        analytic_gauss_image,
        np.abs(analytic_gauss_image),
    )
    fresnel_gauss_image = np.where(
        det_circular_mask,
        fresnel_gauss_image,
        np.abs(fresnel_gauss_image),
    )

    fig, axs = plot_overview(analytic_gauss_image, fresnel_gauss_image, det_size_x=detector.pixel_size[0] * detector.shape[1], det_size_y=detector.pixel_size[1] * detector.shape[0],
                             label1='Analytic Gaussian', label2='Fresnel Propagation', suffix='', unwrap=True)
    fig.savefig("test_gaussian_free_space_vs_fresnel.png")
    plt.close(fig)

    np.testing.assert_allclose(
        np.abs(analytic_gauss_image),
        np.abs(fresnel_gauss_image),
        rtol=1e-1,
        atol=1e-1,
        err_msg="Amplitude mismatch between analytic and fresnel",
    )

    np.testing.assert_allclose(
        np.angle(analytic_gauss_image),
        np.angle(fresnel_gauss_image),
        rtol=1e-1,
        atol=1e-1,
        err_msg="Phase mismatch between analytic and fresnel",
    )



def test_gaussian_lens_vs_fresnel():
    M = -2
    f = 5e-3
    defocus = 1e-4
    z1, z2 = calculate_z1_and_z2_from_M_and_f(M, f)

    pixel_size = (5e-6, 5e-6)
    shape = (2048, 2048)
    wavelength = 1e-5
    wo = 5e-4

    voltage = wavelength2energy(wavelength)

    input_grid = Detector(z=0.0, pixel_size=pixel_size, shape=shape)
    lens = Lens(z=abs(z1) + defocus, focal_length=f)
    detector = Detector(z=abs(z1) + z2, pixel_size=pixel_size, shape=shape)

    gaussian = make_gaussian(
        x=0.0,
        y=0.0,
        dx=0.0,
        dy=0.0,
        z=0.0,
        voltage=voltage,
        waist_x=wo,
        waist_y=wo,
        amp=1.0,
        phase=0.0,
        rcurv_x=jnp.inf,
        rcurv_y=jnp.inf,
    )

    propagated = run_to_end(gaussian, (lens, detector))
    analytic_gauss_image = _field_on_grid(propagated, detector)
    analytic_gauss_image = zero_phase(
        analytic_gauss_image,
        analytic_gauss_image.shape[0] // 2,
        analytic_gauss_image.shape[1] // 2,
    )

    gauss_input = _field_on_grid(gaussian, input_grid)
    gauss_input = zero_phase(
        gauss_input,
        gauss_input.shape[0] // 2,
        gauss_input.shape[1] // 2,
    )

    Y, X = _detector_mesh(detector)
    fresnel_gauss_image = fresnel_lens_imaging_solution(
        gauss_input,
        Y,
        X,
        pixel_size[0],
        wavelength,
        defocus + abs(z1),
        f,
        z2,
    )
    fresnel_gauss_image = zero_phase(
        fresnel_gauss_image,
        fresnel_gauss_image.shape[0] // 2,
        fresnel_gauss_image.shape[1] // 2,
    )

    analytic_gauss_image /= np.max(np.abs(analytic_gauss_image))
    fresnel_gauss_image /= np.max(np.abs(fresnel_gauss_image))

    det_circular_mask = make_aperture(X, Y, aperture_ratio=0.4)

    analytic_gauss_image *= det_circular_mask
    fresnel_gauss_image *= det_circular_mask

    analytic_gauss_image = np.where(
        det_circular_mask,
        analytic_gauss_image,
        np.abs(analytic_gauss_image),
    )
    fresnel_gauss_image = np.where(
        det_circular_mask,
        fresnel_gauss_image,
        np.abs(fresnel_gauss_image),
    )

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
    pixel_size = (5e-6, 5e-6)
    shape = (1024, 1024)

    f = 5e-3
    defocus = 2e-3
    z2 = (1 / f) ** -1 + defocus

    lens = Lens(z=0.0, focal_length=f)
    detector = Detector(z=z2, pixel_size=pixel_size, shape=shape)
    input_grid = Detector(z=0.0, pixel_size=pixel_size, shape=shape)
    Y, X = _detector_mesh(detector)

    wavelength = 1e-4
    wo = 0.5e-3
    voltage = wavelength2energy(wavelength)

    x01, y01, dx01, dy01 = 0.0, 0.0, 0.0, 1e-1
    x02, y02, dx02, dy02 = 0.0, 0.0, 1e-1, 0.0

    def _make_single_gaussian(x0, y0, dx0, dy0):
        return make_gaussian(
            x=x0,
            y=y0,
            dx=dx0,
            dy=dy0,
            z=0.0,
            voltage=voltage,
            waist_x=wo,
            waist_y=wo,
            amp=1.0,
            phase=0.0,
            rcurv_x=jnp.inf,
            rcurv_y=jnp.inf,
        )

    rays_in = [
        _make_single_gaussian(x01, y01, dx01, dy01),
        _make_single_gaussian(x02, y02, dx02, dy02),
    ]

    analytic_fields = []
    input_fields = []
    for ray in rays_in:
        propagated = run_to_end(ray, (lens, detector))
        analytic_fields.append(_field_on_grid(propagated, detector))
        input_fields.append(_field_on_grid(ray, input_grid))

    analytic_gauss_image = np.sum(analytic_fields, axis=0)
    analytic_gauss_image = zero_phase(
        analytic_gauss_image,
        shape[0] // 2,
        shape[1] // 2,
    )

    gaussian_misaligned = np.sum(input_fields, axis=0)

    fresnel_gauss_image = fresnel_lens_imaging_solution(
        gaussian_misaligned,
        Y,
        X,
        pixel_size[0],
        wavelength,
        0.0,
        f,
        z2,
    )
    fresnel_gauss_image = zero_phase(
        fresnel_gauss_image,
        shape[0] // 2,
        shape[1] // 2,
    )

    analytic_gauss_image /= np.max(np.abs(analytic_gauss_image))
    fresnel_gauss_image /= np.max(np.abs(fresnel_gauss_image))

    det_circular_mask = make_aperture(X, Y, aperture_ratio=0.8)

    analytic_gauss_image *= det_circular_mask
    fresnel_gauss_image *= det_circular_mask

    analytic_gauss_image = np.where(
        det_circular_mask,
        analytic_gauss_image,
        np.abs(analytic_gauss_image) * 0.0,
    )
    fresnel_gauss_image = np.where(
        det_circular_mask,
        fresnel_gauss_image,
        np.abs(fresnel_gauss_image),
    )

    det_edge_x, _ = detector.coords_1d
    det_edge_x = np.asarray(det_edge_x)

    central_index = analytic_gauss_image.shape[0] // 2
    analytic_phase_cross_section = np.angle(analytic_gauss_image[central_index, :])
    fresnel_phase_cross_section = np.angle(fresnel_gauss_image[central_index, :])

    analytic_amplitude_cross_section = np.abs(analytic_gauss_image[central_index, :])
    fresnel_amplitude_cross_section = np.abs(fresnel_gauss_image[central_index, :])

    fig, _ = plot_cross_sections(
        det_edge_x,
        [analytic_amplitude_cross_section, fresnel_amplitude_cross_section],
        [
            unwrap_phase(analytic_phase_cross_section),
            unwrap_phase(fresnel_phase_cross_section),
        ],
    )
    plt.savefig("test_two_beam_interference_cross_section.png")
    plt.close(fig)

    det_size_x = pixel_size[0] * shape[0]
    det_size_y = pixel_size[1] * shape[1]

    fig, _ = plot_overview(
        analytic_gauss_image,
        fresnel_gauss_image,
        det_size_x,
        det_size_y,
        suffix="",
        label1="Analytic Gaussian",
        label2="Fresnel Gaussian",
    )
    plt.savefig("test_two_beam_interference_overview.png")
    plt.close(fig)

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
