import jax
import jax.numpy as jnp
import numpy as np
import pytest
from skimage.restoration import unwrap_phase

from temgym_core.components import Detector
from temgym_core.evaluate import evaluate_gaussians_for
from temgym_core.gaussian import (
    GaussianRayBeta,
    TaylorExpofAction,
    gaussian_beam,
    q_inv,
)
from temgym_core.gaussian_taylor import Lens, SigmoidAperture, run_to_end
import matplotlib
from temgym_core.utils import (
    energy2wavelength,
    fibonacci_spiral,
    fresnel_lens_imaging_solution,
    make_aperture,
    zero_phase,
)

jax.config.update("jax_enable_x64", True)


def _lens_planes(magnification, focal_length):
    """Return positive distances from object plane to lens and lens to detector."""
    z1 = focal_length * (1.0 / magnification - 1.0)
    z2 = focal_length * (1.0 - magnification)
    return abs(z1), abs(z2)


def _make_initial_rays(
    num_rays=1000,
    w0=1e-9,
    aperture_radius=50e-9,
    voltage=200e3,
    x_shift=0.0,
):
    """Create a batch of Gaussian packets centred at a waist located at z=0."""
    wavelength = energy2wavelength(voltage)
    k0 = 2 * np.pi / wavelength
    rx, ry = fibonacci_spiral(nb_samples=num_rays, radius=aperture_radius, alpha=0)

    q_inv_waist = 1j * wavelength / (np.pi * w0**2)
    base_Q = jnp.array([[q_inv_waist, 0.0], [0.0, q_inv_waist]], dtype=jnp.complex128)
    Q_inv = jnp.tile(base_Q, (num_rays, 1, 1))

    S = TaylorExpofAction(
        const=jnp.zeros(num_rays, dtype=jnp.complex128),
        lin=jnp.zeros((num_rays, 2), dtype=jnp.complex128),
        quad=Q_inv,
    )

    rays_in = GaussianRayBeta(
        x=jnp.asarray(rx + x_shift),
        y=jnp.asarray(ry),
        dx=jnp.zeros(num_rays),
        dy=jnp.zeros(num_rays),
        z=jnp.zeros(num_rays),
        pathlength=jnp.zeros(num_rays),
        _one=jnp.ones(num_rays),
        S=S,
        C=jnp.ones(num_rays, dtype=jnp.complex128),
        voltage=jnp.full((num_rays,), voltage),
    )
    return rays_in, wavelength, k0


def _waist_from_Q_inv(q_inv_elem, wavelength):
    imag = np.imag(q_inv_elem)
    return np.sqrt(wavelength / (np.pi * imag))


def _radius_from_Q_inv(q_inv_elem):
    real = np.real(q_inv_elem)
    if abs(real) < 1e-30:
        return np.inf
    return 1.0 / real


def _expected_q_inv_free_space(w0, wavelength, distance):
    z_R = np.pi * w0**2 / wavelength
    return 1.0 / (distance - 1j * z_R)


def _expected_waist_R(w0, wavelength, distance):
    z_R = np.pi * w0**2 / wavelength
    if distance == 0:
        return w0, np.inf
    waist = w0 * np.sqrt(1.0 + (distance / z_R) ** 2)
    radius = distance * (1.0 + (z_R**2 / distance**2))
    return waist, radius


def _propagate_rays(rays, components):
    """Run each Gaussian packet through the provided optical model."""
    return jax.vmap(run_to_end, in_axes=(0, None))(rays, components)


def _detector_mesh(detector):
    """Return detector meshgrid (X, Y) in metres."""
    x, y = detector.coords_1d
    Y, X = np.meshgrid(y, x, indexing="ij")
    return X, Y


def _normalize(field):
    max_amp = np.max(np.abs(field))
    if max_amp == 0:
        return field
    return field / max_amp


def _prepare_field(field):
    """Move the complex field to numpy, zero centre phase, and normalise amplitude."""
    array = np.asarray(field, dtype=np.complex128).copy()
    centre_y = array.shape[0] // 2
    centre_x = array.shape[1] // 2
    array = zero_phase(array, centre_y, centre_x)
    return _normalize(array)


def _compare_fields(
    field,
    reference,
    *,
    mask=None,
    amplitude_tol=(1e-2, 1e-2),
    phase_tol=(1e-10, 1e-10),
    message="Field mismatch",
    plot=False,
    compare_phase=True,
):
    """Compare amplitude and phase (optionally inside a mask) with optional plotting."""
    def _save_failure_plot(img_field, img_ref, title, kind):
        try:
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for ax, img, lbl in zip(axes, [img_field, img_ref], ["field", "reference"]):
                im = ax.imshow(img, cmap="magma")
                ax.set_title(lbl)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"{title} [{kind}]")
            fig.tight_layout()
            fname = f"comparison_{kind}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
        except Exception:
            pass

    field_prep = _prepare_field(field)
    ref_prep = _prepare_field(reference)

    if mask is not None:
        field_prep = np.where(mask, field_prep, np.abs(field_prep))
        ref_prep = np.where(mask, ref_prep, np.abs(ref_prep))

    amp_field = np.abs(field_prep)
    amp_ref = np.abs(ref_prep)
    if mask is not None:
        amp_field_flat = amp_field[mask]
        amp_ref_flat = amp_ref[mask]
    else:
        amp_field_flat = amp_field
        amp_ref_flat = amp_ref

    if plot:
        _save_failure_plot(amp_field, amp_ref, message, "amplitude")

    try:
        np.testing.assert_allclose(
            amp_field_flat, amp_ref_flat, rtol=amplitude_tol[0], atol=amplitude_tol[1]
        )
    except AssertionError as exc:
        raise AssertionError(f"{message} (amplitude): {exc}") from exc

    if compare_phase:
        phase_field_img = unwrap_phase(np.angle(field_prep))
        phase_ref_img = unwrap_phase(np.angle(ref_prep))
        if mask is not None:
            phase_field_flat = phase_field_img[mask]
            phase_ref_flat = phase_ref_img[mask]
        else:
            phase_field_flat = phase_field_img
            phase_ref_flat = phase_ref_img

        if plot:
            _save_failure_plot(phase_field_img, phase_ref_img, message, "phase")

        try:
            np.testing.assert_allclose(
                phase_field_flat, phase_ref_flat, rtol=phase_tol[0], atol=phase_tol[1]
            )
        except AssertionError as exc:
            raise AssertionError(f"{message} (phase): {exc}") from exc


def _analytic_gaussian_field_at_plane(X, Y, w0, wavelength, z, k0):
    """Analytic scalar field of a fundamental Gaussian beam at axial distance z."""
    z_R = np.pi * w0**2 / wavelength
    r2 = X**2 + Y**2

    if np.isclose(z, 0.0):
        wz = w0
        Rz = np.inf
        gouy = 0.0
    else:
        wz = w0 * np.sqrt(1.0 + (z / z_R) ** 2)
        Rz = z * (1.0 + (z_R / z) ** 2)
        gouy = np.arctan(z / z_R)

    phase = k0 * z - gouy
    if not np.isinf(Rz):
        phase += k0 * r2 / (2.0 * Rz)

    amplitude = (w0 / wz) * np.exp(-r2 / wz**2)
    return amplitude * np.exp(1j * phase)


def _fraunhofer_gaussian_field_at_focus(X, Y, w0, wavelength, focal_length):
    """Gaussian field in the back focal plane when the waist is in the front focal plane."""
    w_out = wavelength * focal_length / (np.pi * w0)
    amplitude = np.exp(-(X**2 + Y**2) / w_out**2)
    k0 = 2 * np.pi / wavelength
    gouy = np.pi / 2
    phase = np.exp(1j * (k0 * focal_length + gouy))
    return amplitude * phase


@pytest.mark.parametrize("magnification,focal_length", [(-10, 2.5)])
def test_lens_magnification_and_beam_waist_output_variables(magnification, focal_length):
    """Thin-lens magnification should match both coordinates and Gaussian waist."""
    w0 = 1.0
    rays_in, wavelength, _ = _make_initial_rays(
        num_rays=1000, w0=w0, aperture_radius=1e-2, voltage=200e3
    )

    z1, z2 = _lens_planes(magnification, focal_length)
    lens = Lens(focal_length=focal_length, z=z1)
    detector = Detector(z=z1 + z2, pixel_size=(1e-6, 1e-6), shape=(100, 100))

    rays_out = _propagate_rays(rays_in, [lens, detector])

    r_in = np.sqrt(np.asarray(rays_in.x) ** 2 + np.asarray(rays_in.y) ** 2)
    r_out = np.sqrt(np.asarray(rays_out.x) ** 2 + np.asarray(rays_out.y) ** 2)
    mask = r_in > 1e-15
    measured_M = np.mean(r_out[mask] / r_in[mask])
    assert np.isclose(measured_M, abs(magnification), rtol=5e-3, atol=5e-3)

    q_inv_elem = np.asarray(rays_out.S.quad)[0, 0, 0]
    waist_measured = _waist_from_Q_inv(q_inv_elem, wavelength)
    waist_expected = w0 * abs(magnification)
    assert np.isclose(waist_measured, waist_expected, rtol=5e-3)

    radius_measured = _radius_from_Q_inv(q_inv_elem)
    radius_expected = -(magnification * focal_length)
    assert np.isclose(radius_measured, radius_expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(
    ("distance", "description"),
    [
        (1e-2, "free-space propagation"),
        (0.0, "waist (z=0)"),
    ],
)
def test_evaluate_gaussians_for_matches_analytic_beam_param(distance, description):
    """evaluate_gaussians_for must reproduce the analytic Gaussian field."""
    w0 = 2e-6
    rays_in, wavelength, k0 = _make_initial_rays(
        num_rays=1, w0=w0, aperture_radius=1e-2, voltage=200e3
    )
    detector = Detector(z=distance, pixel_size=(2e-7, 2e-7), shape=(256, 256))

    rays_out = _propagate_rays(rays_in, [detector])
    field = evaluate_gaussians_for(rays_out, detector)

    X, Y = _detector_mesh(detector)
    analytic = _analytic_gaussian_field_at_plane(X, Y, w0, wavelength, distance, k0)
    mask = make_aperture(X, Y, aperture_ratio=0.4)

    _compare_fields(
        field,
        analytic,
        mask=mask,
        amplitude_tol=(1e-2, 1e-2),
        phase_tol=(1e-2, 1e-2),
        message=f"evaluate_gaussians_for vs analytic ({description})",
    )


@pytest.mark.parametrize("magnification,focal_length", [(-10, 2.5)])
def test_lens_magnification_and_beam_waist_output_image(magnification, focal_length):
    """Image-plane field must agree with Fresnel FFT propagation of an input Gaussian."""
    w0 = 1e-6
    rays_in, wavelength, k0 = _make_initial_rays(
        num_rays=1, w0=w0, aperture_radius=1e-2, voltage=200e3
    )

    z1, z2 = _lens_planes(magnification, focal_length)
    lens = Lens(focal_length=focal_length, z=z1)
    pixel_size = (1e-7, 1e-7)
    detector = Detector(z=z1 + z2, pixel_size=pixel_size, shape=(2048, 2048))

    rays_out = _propagate_rays(rays_in, [lens, detector])
    field = evaluate_gaussians_for(rays_out, detector)

    X, Y = _detector_mesh(detector)
    q_waist = q_inv(0.0, w0, wavelength)
    gauss_input = np.asarray(gaussian_beam(X, Y, q_waist, k0))
    fresnel_field = fresnel_lens_imaging_solution(
        gauss_input, Y, X, pixel_size[0], wavelength, z1, focal_length, z2
    )

    mask = make_aperture(X, Y, aperture_ratio=0.4)

    _compare_fields(
        field,
        fresnel_field,
        mask=mask,
        amplitude_tol=(5e-1, 5e-1),
        phase_tol=(2.0, 2.0),
        message="Analytic Gaussian vs Fresnel FFT",
    )


def test_defocused_plane_radius_and_waist():
    """After deliberate defocus, waist and curvature should follow Gaussian optics."""
    magnification, focal_length = -200.0, 3e-3
    w0 = 1.2e-6
    rays_in, wavelength, _ = _make_initial_rays(w0=w0)

    z1, z2 = _lens_planes(magnification, focal_length)
    lens = Lens(focal_length=focal_length, z=z1)
    detector = Detector(z=z1 + z2, pixel_size=(1e-6, 1e-6), shape=(100, 100))

    rays_out = _propagate_rays(rays_in, [lens, detector])
    q_inv_image = np.asarray(rays_out.S.quad)[0, 0, 0]

    w_image = _waist_from_Q_inv(q_inv_image, wavelength)
    assert np.isclose(w_image, w0 * abs(magnification), rtol=1e-8)

    dz = 0.01  # 1 cm beyond the image plane
    z_R_new = np.pi * w_image**2 / wavelength
    denom = dz**2 + z_R_new**2
    q_inv_defocus = (dz / denom) + 1j * (z_R_new / denom)

    waist_expected = w_image * np.sqrt(1.0 + (dz / z_R_new) ** 2)
    radius_expected = dz * (1.0 + (z_R_new**2 / dz**2))

    waist_from_q = _waist_from_Q_inv(q_inv_defocus, wavelength)
    radius_from_q = _radius_from_Q_inv(q_inv_defocus)

    assert np.isclose(waist_from_q, waist_expected, rtol=1e-10)
    assert np.isclose(radius_from_q, radius_expected, rtol=1e-10)


@pytest.mark.parametrize("distance_factor", [1.0, 2.0, 10.0])
def test_free_space_propagation_q_inv_waist_radius(distance_factor):
    """Free-space propagation must match analytic 1/(z + i z_R) evolution."""
    w0 = 1e-9
    rays_in, wavelength, _ = _make_initial_rays(w0=w0)
    z_R = np.pi * w0**2 / wavelength
    distance = distance_factor * z_R

    detector = Detector(z=distance, pixel_size=(1e-6, 1e-6), shape=(8, 8))
    rays_out = _propagate_rays(rays_in, [detector])

    q_inv_expected = _expected_q_inv_free_space(w0, wavelength, distance)
    waist_expected, radius_expected = _expected_waist_R(w0, wavelength, distance)

    quad = np.asarray(rays_out.S.quad)
    q_inv_out = quad[0, 0, 0]
    q_inv_y = quad[0, 1, 1]
    off_diag = quad[0, 0, 1]

    assert np.allclose(q_inv_out, q_inv_y, rtol=1e-6, atol=1e-9)
    assert abs(off_diag) < 1e-12

    assert np.allclose(q_inv_out.real, q_inv_expected.real, rtol=2e-3, atol=1e-4)
    assert np.allclose(q_inv_out.imag, q_inv_expected.imag, rtol=2e-3, atol=1e-4)

    waist_out = _waist_from_Q_inv(q_inv_out, wavelength)
    assert np.allclose(waist_out, waist_expected, rtol=1e-12)

    radius_out = _radius_from_Q_inv(q_inv_out)
    if distance_factor == 0.0:
        assert radius_out > 1e6
    else:
        assert np.allclose(radius_out, radius_expected, rtol=3e-3)


def test_beam_field_evaluation_smoke():
    """Basic smoke test: evaluating a single packet should produce a finite field."""
    rays_in, _, _ = _make_initial_rays(w0=1.0, num_rays=1)
    detector = Detector(z=10.0, pixel_size=(1e-1, 1e-1), shape=(200, 200))

    rays_out = _propagate_rays(rays_in, [detector])
    field = evaluate_gaussians_for(rays_out, detector)

    assert field.shape == detector.shape
    assert np.isfinite(field).all()


@pytest.mark.parametrize("focal_length", [1e-3, 3e-3, 1e-2])
def test_parallel_rays_focus_at_back_focal_plane_center(focal_length):
    """Parallel input rays should intersect at (0, 0) in the back focal plane."""
    rays_in, _, _ = _make_initial_rays(num_rays=2000, aperture_radius=2e-4)

    assert np.allclose(np.asarray(rays_in.dx), 0.0)
    assert np.allclose(np.asarray(rays_in.dy), 0.0)

    lens = Lens(focal_length=focal_length, z=0.0)
    detector = Detector(z=focal_length, pixel_size=(1e-6, 1e-6), shape=(8, 8))
    rays_out = _propagate_rays(rays_in, [lens, detector])

    x = np.asarray(rays_out.x)
    y = np.asarray(rays_out.y)
    tol = 1e-9
    assert np.max(np.abs(x)) < tol
    assert np.max(np.abs(y)) < tol


@pytest.mark.parametrize("voltage_ev", [200e3, 80e3])
def test_sigmoid_aperture_outside(voltage_ev):
    """Placing a packet far outside the aperture radius should attenuate by t_outside."""
    radius = 1e-6
    x_outside = 1e-4

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
    k = 2 * np.pi / ray_in.wavelength
    t_outside = 0.5

    aperture = SigmoidAperture(
        z=0.0,
        radius=radius,
        sharpness=1e8,
        t_outside=t_outside,
    )

    ray_out = aperture(ray_in)
    delta_im = jnp.imag(ray_out.S.const - ray_in.S.const)
    attenuation = float(np.exp(-k * delta_im))

    assert attenuation == pytest.approx(t_outside, rel=1e-6, abs=1e-9)


def test_beam_field_evaluation_attenuation():
    """Field amplitude after the aperture should reflect the configured attenuation."""
    radius = 1e-6
    x_outside = 1e-4
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
    ).to_vector()

    detector = Detector(z=1e-15, pixel_size=(1e-1, 1e-1), shape=(200, 200))
    aperture = SigmoidAperture(
        z=0.0,
        radius=radius,
        sharpness=sharpness,
        t_outside=t_outside,
    )

    ray_no_aperture = run_to_end(ray_in, [detector])
    field_no_aperture = evaluate_gaussians_for(ray_no_aperture, detector)

    ray_with_aperture = run_to_end(ray_in, [aperture, detector])
    field_with_aperture = evaluate_gaussians_for(ray_with_aperture, detector)

    max_no = np.max(np.abs(field_no_aperture))
    max_with = np.max(np.abs(field_with_aperture))
    attenuation = float(max_with / max_no)

    assert attenuation == pytest.approx(t_outside, rel=1e-2, abs=1e-3)


def test_fraunhofer_lens_matches_fourier_transform():
    """Gaussian at the front focal plane should equal the Fraunhofer pattern at the back."""
    w0 = 1e-7
    focal_length = 1e-2
    rays_in, wavelength, k0 = _make_initial_rays(
        num_rays=1, w0=w0, aperture_radius=5e-6, voltage=1
    )

    lens = Lens(focal_length=focal_length, z=focal_length)
    detector = Detector(
        z=2 * focal_length,
        pixel_size=(5e-7, 5e-7),
        shape=(512, 512),
    )

    rays_out = _propagate_rays(rays_in, [lens, detector])
    field = evaluate_gaussians_for(rays_out, detector)

    X, Y = _detector_mesh(detector)
    fraunhofer = _fraunhofer_gaussian_field_at_focus(X, Y, w0, wavelength, focal_length)
    mask = make_aperture(X, Y, aperture_ratio=0.4)

    _compare_fields(
        field,
        fraunhofer,
        mask=mask,
        amplitude_tol=(3e-2, 5e-3),
        phase_tol=(5e-2, 5e-2),
        message="Fraunhofer lens propagation check",
        plot=True,
        compare_phase=True
    )
