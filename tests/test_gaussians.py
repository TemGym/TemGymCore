import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from skimage.restoration import unwrap_phase

from temgym_core.gaussian import (
    make_gaussian,
    FreeSpacePropagator,
    Lens,
    ConstantPhaseShift,
    LinearPhaseShift,
    QuadraticPhaseShift,
    scalar_grad_hess_complex,
    run_to_end,
)
from temgym_core.components import Detector

from temgym_core.utils import (
    fresnel_fft_2d,
    zero_phase,
    make_aperture,
    FresnelPropagator,
    fresnel_lens_imaging_solution,
)

from temgym_core.constants import energy2wavelength
from temgym_core.evaluate import evaluate_gaussians_for
from temgym_core.transfer_matrices import calculate_z1_and_z2_from_M_and_f

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
    propagator = FreeSpacePropagator()
    ray_out = propagator(ray_in, dist)

    I2 = jnp.eye(2, dtype=jnp.complex128)
    expected_Q = Q_inv @ jnp.linalg.inv(I2 + dist * Q_inv)

    np.testing.assert_allclose(
        np.asarray(ray_out.Q_inv),
        np.asarray(expected_Q),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ray_in.amplitude),
        np.asarray(1.0 + 0.0j),
        rtol=0.0,
        atol=1e-12,
    )
    det_term = jnp.linalg.det(I2 + dist * Q_inv)
    expected_amplitude = ray_in.amplitude / jnp.sqrt(det_term)

    np.testing.assert_allclose(
        np.asarray(ray_out.amplitude),
        np.asarray(expected_amplitude),
        rtol=1e-12,
        atol=1e-12,
    )

    expected_pathlength = ray_in.pathlength + dist

    np.testing.assert_allclose(
        np.asarray(ray_out.pathlength),
        np.asarray(expected_pathlength),
        rtol=1e-12,
        atol=1e-12,
    )


def test_constant_phase_component():
    input_phase_shift = 0.5
    input_coord = jnp.array([0.5, -0.5])
    constant_component = ConstantPhaseShift(z=0.0, constant_phase_shift=input_phase_shift)

    val = constant_component.phase_shift(input_coord)
    grad = jax.grad(constant_component.phase_shift, argnums=0)(input_coord)
    grad_grad = jax.jacobian(jax.grad(constant_component.phase_shift), argnums=0)(input_coord)

    np.testing.assert_allclose(np.asarray(val), input_phase_shift, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(grad), 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(grad_grad), 0.0, rtol=1e-12, atol=1e-12)


def test_linear_phase_component():
    input_phase_shift = jnp.array([-0.23, 0.512])
    input_coord = jnp.array([0.5, -0.5])
    linear_component = LinearPhaseShift(z=0.0, linear_phase_shift=input_phase_shift)

    val = linear_component.phase_shift(input_coord)
    grad = jax.grad(linear_component.phase_shift, argnums=0)(input_coord)
    grad_grad = jax.jacobian(jax.grad(linear_component.phase_shift), argnums=0)(input_coord)

    np.testing.assert_allclose(np.asarray(val), jnp.dot(input_phase_shift, input_coord), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(grad), input_phase_shift, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(grad_grad), 0.0, rtol=1e-12, atol=1e-12)


def test_quadratic_phase_component():

    fx, fy = -0.1, 0.23
    input_phase_shift = jnp.array([[fx, 0.0], [0.0, fy]])
    input_coord = jnp.array([0.5, -0.5])
    quadratic_component = QuadraticPhaseShift(z=0.0, quadratic_phase_shift=input_phase_shift)

    val = quadratic_component.phase_shift(input_coord)
    grad = jax.grad(quadratic_component.phase_shift, argnums=0)(input_coord)
    grad_grad = jax.jacobian(jax.grad(quadratic_component.phase_shift), argnums=0)(input_coord)

    np.testing.assert_allclose(np.asarray(val), 0.5 * input_coord @ input_phase_shift @ input_coord, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(grad), input_phase_shift @ input_coord, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(grad_grad), input_phase_shift @ jnp.eye(2), rtol=1e-12, atol=1e-12)


def test_scalar_grad_hessian_function():
    constant_phase_shift = 0.5
    constant_phase_shift_comp = ConstantPhaseShift(z=0.0, constant_phase_shift=constant_phase_shift)
    constant_phase_shift_func = constant_phase_shift_comp.phase_shift
    x = jnp.array([0.000552, -0.000326])

    val, grad, hess = scalar_grad_hess_complex(constant_phase_shift_func, x)

    np.testing.assert_allclose(np.asarray(val), 0.5, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(grad), 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(hess), 0.0, rtol=1e-12, atol=1e-12)

    linear_phase_shift = jnp.array([-0.1e-2, -0.8e-3])
    linear_phase_shift_comp = LinearPhaseShift(z=0.0, linear_phase_shift=linear_phase_shift)

    linear_phase_shift_func = linear_phase_shift_comp.phase_shift

    val, grad, hess = scalar_grad_hess_complex(linear_phase_shift_func, x)

    np.testing.assert_allclose(np.asarray(val), jnp.dot(linear_phase_shift, x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(grad), linear_phase_shift, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(hess), 0.0, rtol=1e-12, atol=1e-12)


def test_gaussian_free_space():
    voltage = 0.6e-5
    w0x, w0y = 0.35e-3, 0.25e-3
    theta_x, theta_y = 1.4e-3, -0.7e-3
    z1 = 0.18
    x0, y0 = 0.3e-3, -0.2e-3
    Nx, Ny = 1024, 1024
    Lx, Ly = 16e-3, 16e-3
    pixel_size = (Lx / Nx, Ly / Ny)

    input_grid = Detector(z=0.0, pixel_size=pixel_size, shape=(Ny, Nx))
    detector = Detector(z=z1, pixel_size=pixel_size, shape=(Ny, Nx))

    wavelength = energy2wavelength(voltage)
    ray_in = make_gaussian(
        x=x0,
        y=y0,
        dx=theta_x,
        dy=theta_y,
        z=0.0,
        voltage=voltage,
        waist_x=w0x,
        waist_y=w0y,
        phase=0.0,
        amp=1.0,
    )
    gauss_input = evaluate_gaussians_for(ray_in, input_grid)
    propagator = FreeSpacePropagator()
    ray_out = propagator(ray_in, z1)
    analytic_gauss_image = evaluate_gaussians_for(ray_out, detector)

    Y, X = _detector_mesh(detector)

    fresnel_gauss_image = fresnel_fft_2d(X, Y, gauss_input, wavelength, z1)
    analytic_gauss_image = np.array(analytic_gauss_image)
    fresnel_gauss_image = np.array(fresnel_gauss_image)

    fig, axs = plot_overview(analytic_gauss_image, fresnel_gauss_image, det_size_x=detector.pixel_size[0] * detector.shape[1], det_size_y=detector.pixel_size[1] * detector.shape[0],
                             label1='Analytic Gaussian', label2='Fresnel Propagation', suffix='', unwrap=True)
    fig.savefig("test_gaussian_free_space_vs_fresnel.png")
    plt.close(fig)


def test_gaussian_constant_phase_shift_vs_fresnel():
    voltage = 6.0165e-6
    w0x, w0y = 0.35e-3, 0.25e-3
    theta_x, theta_y = 1.4e-3, -0.7e-3
    z1, z2 = 0.18, 0.27
    x0, y0 = 0.3e-3, -0.2e-3
    Nx, Ny = 2048, 2048
    Lx, Ly = 16e-3, 16e-3
    pixel_size = (Lx / Nx, Ly / Ny)
    constant_phase_shift = 2.0e-6
    input_grid = Detector(z=0.0, pixel_size=pixel_size, shape=(Ny, Nx))
    constant_phase_shift_comp = ConstantPhaseShift(z=z1, constant_phase_shift=constant_phase_shift)
    detector = Detector(z=z1 + z2, pixel_size=pixel_size, shape=(Ny, Nx))

    wavelength = energy2wavelength(voltage)
    ray_in = make_gaussian(
        x=x0,
        y=y0,
        dx=theta_x,
        dy=theta_y,
        z=0.0,
        voltage=voltage,
        waist_x=w0x,
        waist_y=w0y,
        phase=0.0,
        amp=1.0,
    )
    gauss_input = evaluate_gaussians_for(ray_in, input_grid)
    ray_out = run_to_end(ray_in, (constant_phase_shift_comp, detector))
    analytic_gauss_image = evaluate_gaussians_for(ray_out, detector)

    Y, X = _detector_mesh(detector)

    U1 = fresnel_fft_2d(X, Y, gauss_input, wavelength, z1)
    U1k = U1 * np.exp(1j*(2*np.pi/wavelength)*constant_phase_shift)
    fresnel_gauss_image = fresnel_fft_2d(X, Y, U1k, wavelength, z2)

    analytic_gauss_image = np.array(analytic_gauss_image)
    fresnel_gauss_image = np.array(fresnel_gauss_image)

    fig, axs = plot_overview(analytic_gauss_image, fresnel_gauss_image, det_size_x=detector.pixel_size[0] * detector.shape[1], det_size_y=detector.pixel_size[1] * detector.shape[0],
                             label1='Analytic Gaussian', label2='Fresnel Propagation', suffix='', unwrap=False)
    fig.savefig("test_gaussian_vs_constant_phase_shift.png")
    plt.close(fig)


def test_gaussian_linear_phase_shift_vs_fresnel():
    voltage = 6.0165e-6  # 500 e-9 m wavelength
    w0x, w0y = 0.35e-3, 0.25e-3
    theta_x, theta_y = 1.4e-3, -0.7e-3
    d_theta = jnp.array([-0.1e-2, -0.8e-3])
    z1, z2 = 0.18, 0.27
    x0, y0 = 0.0e-3, 0.0e-3
    Nx, Ny = 2048, 2048
    Lx, Ly = 16e-3, 16e-3
    pixel_size = (Lx / Nx, Ly / Ny)
    input_grid = Detector(z=0.0, pixel_size=pixel_size, shape=(Ny, Nx))
    linear_phase_shift_comp = LinearPhaseShift(z=z1, linear_phase_shift=d_theta)
    detector = Detector(z=z1 + z2, pixel_size=pixel_size, shape=(Ny, Nx))

    wavelength = energy2wavelength(voltage)
    ray_in = make_gaussian(
        x=x0,
        y=y0,
        dx=theta_x,
        dy=theta_y,
        z=0.0,
        voltage=voltage,
        waist_x=w0x,
        waist_y=w0y,
        phase=0.0,
        amp=1.0,
    )
    gauss_input = evaluate_gaussians_for(ray_in, input_grid)
    ray_out = run_to_end(ray_in, (linear_phase_shift_comp, detector))
    analytic_gauss_image = evaluate_gaussians_for(ray_out, detector)

    Y, X = _detector_mesh(detector)

    U1 = fresnel_fft_2d(X, Y, gauss_input, wavelength, z1)
    U1k = U1 * np.exp(1j*(2*np.pi/wavelength)*(d_theta[0]*X + d_theta[1]*Y))
    fresnel_gauss_image = fresnel_fft_2d(X, Y, U1k, wavelength, z2)

    analytic_gauss_image = np.array(analytic_gauss_image)
    fresnel_gauss_image = np.array(fresnel_gauss_image)

    fig, axs = plot_overview(analytic_gauss_image, fresnel_gauss_image, det_size_x=detector.pixel_size[0] * detector.shape[1], det_size_y=detector.pixel_size[1] * detector.shape[0],
                             label1='Analytic Gaussian', label2='Fresnel Propagation', suffix='', unwrap=False)
    fig.savefig("test_gaussian_vs_linear_phase_shift.png")
    plt.close(fig)


def test_gaussian_quadratic_phase_shift_vs_fresnel():
    voltage = 6.0165e-6  # 500 e-9 m wavelength
    w0x, w0y = 0.35e-3, 0.25e-3
    theta_x, theta_y = 1.4e-3, -0.7e-3
    z1, z2 = 0.18, 0.27
    x0, y0 = 0.6e-3, -0.2e-3
    Nx, Ny = 1024, 1024
    Lx, Ly = 16e-3, 16e-3
    pixel_size = (Lx / Nx, Ly / Ny)
    input_grid = Detector(z=0.0, pixel_size=pixel_size, shape=(Ny, Nx))
    f_x, f_y = 0.7, 0.35
    K = jnp.array([[-1/f_x, 0.0], [0.0, -1/f_y]])

    quadratic_phase_shift_comp = QuadraticPhaseShift(z=z1, quadratic_phase_shift=K)
    detector = Detector(z=z1 + z2, pixel_size=pixel_size, shape=(Ny, Nx))

    wavelength = energy2wavelength(voltage)
    ray_in = make_gaussian(
        x=x0,
        y=y0,
        dx=theta_x,
        dy=theta_y,
        z=0.0,
        voltage=voltage,
        waist_x=w0x,
        waist_y=w0y,
        phase=0.0,
        amp=1.0,
    )
    gauss_input = evaluate_gaussians_for(ray_in, input_grid)
    ray_out = run_to_end(ray_in, (quadratic_phase_shift_comp, detector))
    analytic_gauss_image = evaluate_gaussians_for(ray_out, detector)

    Y, X = _detector_mesh(detector)

    U1 = fresnel_fft_2d(X, Y, gauss_input, wavelength, z1)
    U1k = U1 * np.exp(1j*(2*np.pi/wavelength)*0.5*(K[0,0]*X**2 + 2*K[0,1]*X*Y + K[1,1]*Y**2))
    fresnel_gauss_image = fresnel_fft_2d(X, Y, U1k, wavelength, z2)

    analytic_gauss_image = np.array(analytic_gauss_image)
    fresnel_gauss_image = np.array(fresnel_gauss_image)

    fig, axs = plot_overview(analytic_gauss_image, fresnel_gauss_image, det_size_x=detector.pixel_size[0] * detector.shape[1], det_size_y=detector.pixel_size[1] * detector.shape[0],
                             label1='Analytic Gaussian', label2='Fresnel Propagation', suffix='', unwrap=False)
    fig.savefig("test_gaussian_vs_quadratic_phase_shift.png")
    plt.close(fig)


@pytest.mark.skip(reason='ABCD for new propagator not yet implemented')
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

    propagator = ABCDPropagator.fourier_transform(f)

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


@pytest.mark.skip(reason='ABCD for new propagator not yet implemented')
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
    ray_out_abcd = ABCDPropagator.fourier_transform(f)(ray_in)

    lens = Lens(z=f, focal_length=f)
    output_probe_grid = Detector(z=2 * f, pixel_size=(1, 1), shape=(1, 1))

    components = (lens, output_probe_grid)
    ray_out_step_wise = run_to_end(ray_in, components)

    propagator = ABCDPropagator.free_space(f)
    A = jnp.eye(2)
    B = jnp.zeros((2, 2))
    C = -1.0 / f * jnp.eye(2)
    D = jnp.eye(2)
    lens = ABCDPropagator(A=A, B=B, C=C, D=D)
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
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.S2),
        np.asarray(ray_out_step_wise.S2),
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_abcd.C),
        np.asarray(ray_out_step_wise.C),
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_step_wise_abcd.S2),
        np.asarray(ray_out_step_wise.S2),
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        np.asarray(ray_out_step_wise_abcd.C),
        np.asarray(ray_out_step_wise.C),
        rtol=1e-5,
        atol=1e-5,
    )


# @pytest.mark.skip(reason='ABCD for new propagator not yet implemented')
def test_gaussian_free_space_vs_fresnel():
    propagation_distance = 20
    pixel_size = (0.000005, 0.000005)
    voltage = 1e-5
    wavelength = energy2wavelength(voltage)
    wo = 0.001
    shape = (2000, 2000)

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
    analytic_gauss_image = evaluate_gaussians_for(propagated, detector)

    gauss_input = evaluate_gaussians_for(gaussian, input_grid)
    gauss_input = zero_phase(gauss_input, shape[0]//2, shape[1]//2)

    Y, X = _detector_mesh(detector)

    fresnel_gauss_image = FresnelPropagator(
        gauss_input,
        _window_size(detector),
        wavelength,
        propagation_distance,
    )
    analytic_gauss_image = np.array(analytic_gauss_image)
    fresnel_gauss_image = np.array(fresnel_gauss_image)

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
        rtol=1e-4,
        atol=1e-4,
        err_msg="Amplitude mismatch between analytic and fresnel",
    )

    np.testing.assert_allclose(
        np.angle(analytic_gauss_image),
        np.angle(fresnel_gauss_image),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Phase mismatch between analytic and fresnel",
    )


def test_gaussian_lens_vs_fresnel():
    M = -2
    f = 5e-3
    defocus = 0.0
    z1, z2 = calculate_z1_and_z2_from_M_and_f(M, f)

    pixel_size = (1e-5, 1e-5)
    shape = (1024, 1024)
    voltage = 1e-8
    wavelength = energy2wavelength(voltage)
    wo = 1e-3

    input_grid = Detector(z=0.0, pixel_size=pixel_size, shape=shape)
    lens = Lens(z=abs(z1), focal_length=f)
    detector = Detector(z=abs(z1) + z2 + defocus, pixel_size=pixel_size, shape=shape)

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
        z1,
        f,
        z2 + defocus,
    )
    fresnel_gauss_image = zero_phase(
        fresnel_gauss_image,
        fresnel_gauss_image.shape[0] // 2,
        fresnel_gauss_image.shape[1] // 2,
    )

    det_circular_mask = make_aperture(X, Y, aperture_ratio=0.3)

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
                             label1='Analytic Gaussian', label2='Fresnel Propagation', suffix='', unwrap=False)

    fig.savefig("test_gaussian_lens_vs_fresnel.png")
    plt.close(fig)

    central_index = analytic_gauss_image.shape[0] // 2
    analytic_phase_cross_section = np.angle(analytic_gauss_image[central_index, :])
    fresnel_phase_cross_section = np.angle(fresnel_gauss_image[central_index, :])

    analytic_amplitude_cross_section = np.abs(analytic_gauss_image[central_index, :])
    fresnel_amplitude_cross_section = np.abs(fresnel_gauss_image[central_index, :])

    det_edge_x, _ = detector.coords_1d
    det_edge_x = np.asarray(det_edge_x)

    fig, _ = plot_cross_sections(
        det_edge_x,
        [analytic_amplitude_cross_section, fresnel_amplitude_cross_section],
        [
            analytic_phase_cross_section,
            fresnel_phase_cross_section,
        ],
        labels=["Analytic", "Fresnel"],
    )

    fig.savefig("test_gaussian_lens_vs_fresnel_cross_section.png")
    plt.close(fig)

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


def test_gaussian_two_beam_interference_vs_fresnel():
    pixel_size = (1e-5, 1e-5)
    shape = (2048, 2048)

    f = 5e-3
    defocus = 2e-3
    z2 = (1 / f) ** -1 + defocus

    lens = Lens(z=0.0, focal_length=f)
    detector = Detector(z=z2, pixel_size=pixel_size, shape=shape)
    input_grid = Detector(z=0.0, pixel_size=pixel_size, shape=shape)
    Y, X = _detector_mesh(detector)

    wo = 2e-3
    voltage = 1e-10
    wavelength = energy2wavelength(voltage)

    x01, y01, dx01, dy01 = 0.0, 0.0, 0.0, 0.5e-1
    x02, y02, dx02, dy02 = 0.0, 0.0, 0.5e-1, 0.0

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

    det_circular_mask = make_aperture(X, Y, aperture_ratio=0.2)

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
            analytic_phase_cross_section,
            fresnel_phase_cross_section,
        ],
    )
    fig.savefig("test_two_beam_interference_cross_section.png")
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
    fig.savefig("test_two_beam_interference_overview.png")
    plt.close(fig)

    np.testing.assert_allclose(
        np.abs(analytic_gauss_image),
        np.abs(fresnel_gauss_image),
        rtol=1e-2,
        atol=1e-2,
        err_msg="Amplitude mismatch between analytic and fresnel",
    )

    np.testing.assert_allclose(
        np.angle(analytic_gauss_image),
        np.angle(fresnel_gauss_image),
        rtol=1e-2,
        atol=1e-2,
        err_msg="Phase mismatch between analytic and fresnel",
    )
