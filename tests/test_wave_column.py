"""Columns as wave propagation: composition, roles, error paths, gradients, energy spread, consistency."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import (
    Deflector,
    Detector,
    ElectromagneticLens,
    HardEdgeSolenoid,
    Lens,
    Plane,
    Rotator,
    SigmoidAperture,
    Stigmator,
)
from temgym_core.constants import (
    compute_Rc_from_voltage,
    effective_accelerating_potential,
    energy2wavelength,
)
from temgym_core.gaussian import make_gaussian
from temgym_core.propagator import AngularSpectrumPropagator, FresnelPropagator
from temgym_core.ray import Ray
from temgym_core.run import column_matrix, run_to_end
from temgym_core.wave import (
    Lattice,
    LatticeError,
    SamplingError,
    Wavefront,
    adapted_input_lattice,
    apply_lct,
    column_transform,
    gaussian_field,
    input_lattice,
    lattice_chain,
    plan_lct,
    propagate,
)
from temgym_core.wave.reference import chief_from_beam
from wave_validation import complex_q_chain, gaussian_Q, rel_err, rel_err_no_phase

VOLTAGE = 200e3
LAM = float(energy2wavelength(VOLTAGE))


def _source(lattice, w=1e-7, **kw):
    """A round Gaussian source wavefront on a lattice at z = 0."""
    return Wavefront(gaussian_field(lattice.positions(), 1.0, np.zeros(2), np.zeros(2),
                                    gaussian_Q(w, w, LAM), LAM, **kw), lattice, 0.0, VOLTAGE)


# ------------------------------------------------------------------------------------------------
# composition: two spans against one, with the Gouy sign carried across components
# ------------------------------------------------------------------------------------------------
Z1, F1 = 3e-3, 2e-3
Z2, F2 = 8e-3, 2.5e-3


@pytest.mark.parametrize("z_mid, z_end", [(5e-3, 12e-3), (6e-3, 9.5e-3)],
                         ids=["through the back focal plane", "before the image plane"])
def test_two_spans_compose_to_one_including_the_gouy_sign(z_mid, z_end):
    """The same wave through [lens, lens] in one Collins transform and in two (cut at an intermediate
    plane, on the lattice the first leaves) equals the complex-q result *including its phase*: the
    chief Gaussian carries the continuous Gouy branch across the components, which the matrices alone
    cannot (each is defined up to a sign)."""
    first, second = [Lens(z=Z1, focal_length=F1), Plane(z=z_mid)], [Lens(z=Z2, focal_length=F2), Plane(z=z_end)]
    Q0 = gaussian_Q(1e-7, 1e-7, LAM)
    Qo, amp = complex_q_chain(Q0, [("drift", Z1), ("lens", F1), ("drift", Z2 - Z1), ("lens", F2), ("drift", z_end - Z2)])
    source = _source(Lattice.regular((192, 192), 10e-9))
    one = propagate(source, first[:1] + second, tol=1e-14)
    middle = propagate(source, first, tol=1e-14)
    two = propagate(middle, second, tol=1e-14)
    for result in (one, two):
        reference = gaussian_field(result.lattice.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)
        assert rel_err(result.data, reference) < 1e-6
    assert one.z == two.z == z_end
    # the matrices compose exactly: M(one) = M(second) @ M(first), the second starting where the first ends
    m_one = column_transform(first[:1] + second, 0.0, VOLTAGE).matrix
    m_first = column_transform(first, 0.0, VOLTAGE).matrix
    m_second = column_transform(second, z_mid, VOLTAGE).matrix
    assert float(jnp.max(jnp.abs(m_one - m_second @ m_first))) < 1e-9 * float(jnp.max(jnp.abs(m_one)))


def test_the_focal_plane_flips_the_sign_of_the_matrix_lift_but_not_of_the_physical_field():
    """Cutting the column at, before and after the back focal plane changes the sign of the first
    span's Maslov lift; the phases from the chief Gaussian compensate exactly."""
    source = _source(Lattice.regular((192, 192), 10e-9))
    results = []
    for z_mid in (4.5e-3, 5e-3, 5.5e-3):
        middle = propagate(source, [Lens(z=Z1, focal_length=F1), Plane(z=z_mid)], tol=1e-14)
        results.append(propagate(middle, [Lens(z=Z2, focal_length=F2), Plane(z=12e-3)], tol=1e-14))
    Q0 = gaussian_Q(1e-7, 1e-7, LAM)
    Qo, amp = complex_q_chain(Q0, [("drift", Z1), ("lens", F1), ("drift", Z2 - Z1), ("lens", F2), ("drift", 12e-3 - Z2)])
    for result in results:
        reference = gaussian_field(result.lattice.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)
        assert rel_err(result.data, reference) < 1e-6


# ------------------------------------------------------------------------------------------------
# the lattice bookkeeping: input_lattice, lattice_chain, explicit lattices
# ------------------------------------------------------------------------------------------------
def _imaging_case(v_factor=1.0):
    u, f = 3e-3, 2e-3
    v = v_factor * u * f / (u - f)
    return [Lens(z=u, focal_length=f), Plane(z=u + v)]


def test_a_source_on_the_input_lattice_lands_on_the_requested_detector_grid():
    detector = Detector(z=0.0, pixel_size=(8e-9, 8e-9), shape=(192, 192), rotation=15.0, centre=(5e-9, -2e-9))
    column = [Lens(z=3e-3, focal_length=2e-3), Plane(z=3e-3 + 6e-3 * 1.01)]
    lattice = input_lattice(column, detector, source_z=0.0, voltage=VOLTAGE)
    ct = column_transform(column, 0.0, VOLTAGE)
    w0 = 5e-8
    Q0 = gaussian_Q(w0, w0, LAM)
    field = Wavefront(gaussian_field(lattice.positions(), 1.0, np.zeros(2), np.zeros(2), Q0, LAM), lattice, 0.0, VOLTAGE)
    out = propagate(field, column, detector)
    grid = Lattice.from_grid(detector)
    assert np.allclose(np.asarray(out.lattice.pitch), np.asarray(grid.pitch), rtol=1e-12)
    Qo, amp = complex_q_chain(Q0, [("drift", 3e-3), ("lens", 2e-3), ("drift", 6e-3 * 1.01)])
    ref = gaussian_field(grid.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)
    assert rel_err_no_phase(out.data, ref) < 1e-9
    # the near-image branch maps the lattice by the ray matrix: no interpolation
    A = np.asarray(ct.matrix)[:2, :2]
    assert np.allclose(np.asarray(grid.pitch), A @ np.asarray(lattice.pitch), rtol=1e-9)


def test_lattice_chain_is_the_inverse_of_forward_planning():
    """Backward lattices, then forward plans on them, end exactly on the requested lattice."""
    out = Lattice.regular((96, 96), 5e-9, rotation=10.0)
    m1 = np.asarray(column_matrix([Lens(z=3e-3, focal_length=2e-3), Plane(z=3.2e-3)], source_z=0.0, voltage=VOLTAGE))
    m2 = np.asarray(column_matrix([Plane(z=3.4e-3)], source_z=3.2e-3, voltage=VOLTAGE))
    chain = lattice_chain([m1, m2], LAM, out, ["mft", "near_image"])
    assert len(chain) == 3 and chain[-1] is out
    for matrix, method, before, after in ((m1, "mft", chain[0], chain[1]), (m2, "near_image", chain[1], chain[2])):
        plan = plan_lct(matrix, LAM, before, after, method=method, check="off")
        assert np.allclose(np.asarray(plan.out_lattice.pitch), np.asarray(after.pitch), rtol=1e-9)
    with pytest.raises(ValueError, match="one method per span"):
        lattice_chain([m1, m2], LAM, out, ["mft"])
    with pytest.raises(ValueError, match="unknown method"):
        lattice_chain([m1], LAM, out, ["fft"])


def test_gamma_zooms_the_source_lattice_of_a_matrix_fourier_transform():
    detector = Lattice.regular((64, 64), 4e-9)
    column = [Lens(z=3e-3, focal_length=2e-3), Plane(z=5e-3)]
    natural = input_lattice(column, detector, source_z=0.0, voltage=VOLTAGE)
    zoomed = input_lattice(column, detector, source_z=0.0, voltage=VOLTAGE, gamma=(0.5 / 64, 0.5 / 64), in_shape=(128, 128))
    assert zoomed.shape == (128, 128)
    assert np.allclose(np.asarray(zoomed.pitch), 0.5 * np.asarray(natural.pitch), rtol=1e-12)


def test_explicit_lattices_of_one_span_are_used_as_given():
    """With one linear span and an ``out_lattice`` the pair is taken as it is, so the lattice between two
    stages can be chosen freely (one ``propagate`` call per stage)."""
    column = [Lens(z=3e-3, focal_length=2e-3), Plane(z=5.06e-3)]
    detector = Lattice.regular((96, 96), 4e-9)
    B = np.asarray(column_transform(column, 0.0, VOLTAGE).matrix)[:2, 2:4]
    zoomed = adapted_input_lattice(B, LAM, detector, (0.6 / 96, 0.6 / 96), (128, 128))
    field = Wavefront(gaussian_field(zoomed.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(1e-7, 1e-7, LAM), LAM),
                      zoomed, 0.0, VOLTAGE)
    out = propagate(field, column, detector, tol=1e-12)
    Qo, amp = complex_q_chain(gaussian_Q(1e-7, 1e-7, LAM), [("drift", 3e-3), ("lens", 2e-3), ("drift", 2.06e-3)])
    assert rel_err_no_phase(out.data, gaussian_field(detector.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)) < 1e-6


# ------------------------------------------------------------------------------------------------
# error paths and the expert override
# ------------------------------------------------------------------------------------------------
def test_propagate_raises_sampling_error_for_a_span_no_branch_can_hold():
    """y at its focal plane (only the MFT works) while x sits at F ~ 0.7: refused with the inequalities."""
    fx, fy = 2.0e-3, 2.6e-3
    column = [Stigmator(z=3e-3, focal_length_x=fx, focal_length_y=fy), Plane(z=3e-3 + fy)]
    chief = make_gaussian(x=2e-8, y=-1e-8, dx=6e-6, dy=-9e-6, z=0.0, voltage=VOLTAGE, waist_x=1.0e-7, waist_y=0.85e-7)
    amp, centre, slopes, Q, path = chief_from_beam(chief)
    lattice = Lattice.regular((192, 192), 10e-9)
    field = Wavefront(gaussian_field(lattice.positions(), amp, centre, slopes, Q, LAM, path), lattice, 0.0, VOLTAGE)
    with pytest.raises(SamplingError, match="no branch") as info:
        propagate(field, column, tol=1e-22)
    assert "near_image[I]" in str(info.value) and "mft" in str(info.value) and "intermediate plane" in str(info.value)
    wave, reports = propagate(field, column, check="report", tol=1e-22, return_reports=True)   # never raises
    assert not reports[0].ok and wave.data.shape == field.data.shape


def test_field_on_the_wrong_lattice_raises_lattice_error():
    two_spans = [Lens(z=3e-3, focal_length=2e-3), SigmoidAperture(z=3e-3, radius=2e-7, edge_width=1e-7), Plane(z=4e-3),
                 SigmoidAperture(z=4e-3, radius=2e-7, edge_width=1e-7), Plane(z=5e-3)]
    detector = Lattice.regular((64, 64), 5e-9)
    field = _source(Lattice.regular((64, 64), 5e-9))
    with pytest.raises(LatticeError, match="input_lattice"):
        propagate(field, two_spans, detector)
    rotated = [Rotator(z=0.0, angle=20.0), Plane(z=1e-3)]      # B is a rotation: two Cartesian lattices are not adapted
    with pytest.raises(LatticeError, match="adapted"):
        propagate(field, rotated, detector, method="mft")


def test_check_off_runs_the_inverse_imaging_plane_that_the_planner_refuses():
    """The image plane carries the Nyquist-saturated lens chirp; multiplying by the exact conjugate chirp
    sample by sample restores a smooth field. The planner cannot know that, so the inverse propagation
    (a negative drift, the lens with -f, a negative drift) needs the expert override."""
    u, f = 3e-3, 2e-3
    v = 1.01 * u * f / (u - f)
    source = _source(Lattice.regular((192, 192), 10e-9))
    image = propagate(source, [Lens(z=u, focal_length=f), Plane(z=u + v)], tol=1e-14)
    inverse = [Lens(z=u, focal_length=-f), Plane(z=0.0)]
    with pytest.raises(SamplingError, match="Nyquist"):
        propagate(image, inverse, method="near_image_II", tol=1e-14)
    back = propagate(image, inverse, method="near_image_II", check="off")
    assert np.allclose(np.asarray(back.lattice.pitch), np.asarray(source.lattice.pitch), rtol=1e-9)
    assert rel_err_no_phase(back.data, source.data) < 1e-9
    assert back.z == 0.0


def test_solenoid_beyond_the_first_radial_half_period_is_refused():
    long_solenoid = [HardEdgeSolenoid(z=0.0, length=0.02, q=3.3), Plane(z=0.05)]
    with pytest.raises(ValueError, match="half period"):
        column_transform(long_solenoid, 0.0, VOLTAGE)
    field = Wavefront(jnp.ones((8, 8), dtype=complex), Lattice.regular((8, 8), 1e-9), 0.0, VOLTAGE)
    with pytest.raises(ValueError, match="half period"):
        propagate(field, long_solenoid)
    column_transform([HardEdgeSolenoid(z=0.0, length=0.02, q=3.0), Plane(z=0.05)], 0.0, VOLTAGE)   # just inside


def test_a_mask_is_not_part_of_a_single_collins_transform():
    with pytest.raises(ValueError, match="acts as a mask"):
        column_transform([SigmoidAperture(z=0.0, radius=1e-6, edge_width=1e-7), Plane(z=1e-3)], 0.0, VOLTAGE)


def test_inconsistent_ray_and_gaussian_branches_of_a_component_are_detected():
    """ElectromagneticLens ignores x0 and y0 for rays but not for Gaussians: the matrix and the chief
    disagree, so the phase would be meaningless; propagate refuses."""
    lens = ElectromagneticLens(z=3e-3, turns=100.0, current=0.5, Gc=1e3, x0=5e-8)
    ct = column_transform([lens, Plane(z=4e-3)], 0.0, VOLTAGE)
    assert float(ct.mismatch) > 1e-3
    field = _source(Lattice.regular((64, 64), 10e-9))
    with pytest.raises(ValueError, match="disagree"):
        propagate(field, [lens, Plane(z=4e-3)])


def test_invalid_options_are_rejected():
    field = _source(Lattice.regular((16, 16), 1e-8))
    with pytest.raises(ValueError, match="check"):
        propagate(field, [Plane(z=1e-3)], check="sometimes")
    with pytest.raises(ValueError, match="method has 2 entries for 1"):
        propagate(field, [Plane(z=1e-3)], method=["near_image", "mft"])


def test_zero_length_columns_and_masks_only_columns():
    field = _source(Lattice.regular((64, 64), 5e-9))
    same = propagate(field, [Plane(z=0.0)])
    assert rel_err(same.data, field.data) == 0.0 and same.z == 0.0
    aperture = SigmoidAperture(z=0.0, radius=2e-7, edge_width=1e-7)
    masked = propagate(field, [aperture])
    assert masked.lattice is field.lattice and float(jnp.max(jnp.abs(masked.data))) < float(jnp.max(jnp.abs(field.data)))
    with pytest.raises(LatticeError):
        propagate(field, [aperture], Lattice.regular((64, 64), 6e-9))


# ------------------------------------------------------------------------------------------------
# rays, Gaussians and waves see the same chief ray
# ------------------------------------------------------------------------------------------------
def test_ray_chief_gaussian_centre_and_wave_centroid_agree():
    """A lens, a deflector and a solenoid: the ray, the centre of the Gaussian beamlet and the intensity
    centroid of the sampled wave land on the same point (the wave from the same Gaussian, sampled)."""
    column = [Lens(z=3e-3, focal_length=2e-3), Deflector(z=4e-3, def_x=3e-6, def_y=-2e-6),
              HardEdgeSolenoid(z=5e-3, length=2e-3, q=0.3), Plane(z=8e-3)]
    start = dict(x=2e-8, y=-1.5e-8, dx=1.5e-6, dy=-1e-6)
    ray = run_to_end(Ray(z=0.0, pathlength=0.0, voltage=VOLTAGE, **start), column)
    beam = make_gaussian(z=0.0, voltage=VOLTAGE, waist_x=1e-7, waist_y=0.9e-7, **start)
    at_end = run_to_end(beam, column)
    np.testing.assert_allclose(np.asarray(at_end.r_xy), [float(ray.x), float(ray.y)], rtol=1e-11, atol=1e-20)
    np.testing.assert_allclose(np.asarray(at_end.d_xy), [float(ray.dx), float(ray.dy)], rtol=1e-11, atol=1e-20)

    amp, centre, slopes, Q, path = chief_from_beam(beam)
    lattice = Lattice.regular((192, 192), 10e-9)
    field = Wavefront(gaussian_field(lattice.positions(), amp, centre, slopes, Q, LAM, path), lattice, 0.0, VOLTAGE)
    wave = propagate(field, column, tol=1e-16)
    intensity = jnp.abs(wave.data) ** 2
    x, y = wave.lattice.coords()
    centroid = np.array([float(jnp.sum(x * intensity) / jnp.sum(intensity)),
                         float(jnp.sum(y * intensity) / jnp.sum(intensity))])
    width = 1e-7
    assert np.max(np.abs(centroid - [float(ray.x), float(ray.y)])) < 1e-7 * width
    # the phase gradient of the wave is the ray slope: the mean local frequency times lambda / pitch
    u = np.asarray(wave.data)
    freq = np.array([np.angle(np.sum(np.conj(u[:, :-1]) * u[:, 1:])), np.angle(np.sum(np.conj(u[:-1]) * u[1:]))]) / (2 * np.pi)
    slope = LAM * np.linalg.inv(np.asarray(wave.lattice.pitch)).T @ freq
    assert np.max(np.abs(slope - [float(ray.dx), float(ray.dy)])) < 1e-3 * abs(float(ray.dx)) + 1e-9


# ------------------------------------------------------------------------------------------------
# free space against the reference propagators, carrier phase removed
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("F", [0.1, 0.5, 0.9])
def test_free_space_column_matches_the_fresnel_propagator(F):
    N, p = 128, 25e-9
    dz = F * N * p * p / LAM
    lattice = Lattice.regular((N, N), p)
    field = _source(lattice, w=8 * p)
    out = propagate(field, [Plane(z=dz)])
    carrier = jnp.exp(-1j * 2 * jnp.pi * dz / LAM)                   # FresnelPropagator includes exp(ikz)
    assert rel_err(out.data, FresnelPropagator(field.data, N * p, LAM, dz) * carrier) < 1e-10


def test_free_space_column_matches_the_angular_spectrum_up_to_its_float64_phase_floor():
    """At electron scale the angular-spectrum reference forms exp(i kz z) with k z ~ 1e10 rad, so it
    is only good to ~1e-16 k z (the engine excludes the carrier and has no such floor)."""
    N, p = 128, 25e-9
    dz = 0.3 * N * p * p / LAM
    field = _source(Lattice.regular((N, N), p), w=8 * p)
    out = propagate(field, [Plane(z=dz)])
    reference = AngularSpectrumPropagator(field.data, N * p, LAM, dz, xp=jnp) * jnp.exp(-1j * 2 * jnp.pi * dz / LAM)
    floor = 1e-16 * 2 * np.pi / LAM * dz
    assert rel_err_no_phase(out.data, reference) < 100 * floor
    assert rel_err(out.data, FresnelPropagator(field.data, N * p, LAM, dz) * jnp.exp(-1j * 2 * jnp.pi * dz / LAM)) < 1e-10


# ------------------------------------------------------------------------------------------------
# differentiability through column_transform -> plan -> apply
# ------------------------------------------------------------------------------------------------
def _loss_setup(n=48):
    """A fixed output lattice, a decentred lens (so the constant path depends on f) and a smooth target."""
    out_lattice = Lattice.regular((n, n), 12e-9)
    target = jnp.exp(-((out_lattice.positions() ** 2).sum(-1)) / (2 * (0.1e-6) ** 2)) * (1 + 0.3j)
    return out_lattice, target


def test_gradient_with_respect_to_a_focal_length_matches_finite_differences():
    out_lattice, target = _loss_setup()
    n, u, d, w0 = 48, 3e-3, 4.6e-3, 1e-7

    def loss(f):
        components = [Lens(z=u, focal_length=f, x0=6e-8), Plane(z=u + d)]
        ct = column_transform(components, 0.0, VOLTAGE)
        source = adapted_input_lattice(ct.matrix[:2, 2:4], LAM, out_lattice, (1.0 / n, 1.0 / n), (n, n))
        field = gaussian_field(source.positions(), 1.0, jnp.zeros(2), jnp.zeros(2), gaussian_Q(w0, w0, LAM), LAM)
        plan = plan_lct(ct.matrix, LAM, source, out_lattice, method="mft", check="off")
        out = apply_lct(plan, field, ct.phase0)
        return jnp.real(jnp.sum(jnp.conj(target) * out)) / jnp.max(jnp.abs(target))

    f0 = 2e-3
    grad = jax.grad(loss)(f0)
    h = 1e-6 * f0
    fd = (loss(f0 + h) - loss(f0 - h)) / (2 * h)
    assert abs(float(fd)) > 0 and abs(float(grad) - float(fd)) < 1e-6 * abs(float(fd)), (float(grad), float(fd))


def test_gradient_with_respect_to_the_wavelength_matches_finite_differences():
    """The voltage sets the wavelength of the transform, the chief Gaussian, the adapted lattice and the
    source; the derivative goes through all of them."""
    out_lattice, target = _loss_setup()
    n, u, d, w0 = 48, 3e-3, 4.6e-3, 1e-7

    def loss(voltage):
        lam = energy2wavelength(voltage)
        components = [Lens(z=u, focal_length=2e-3, x0=6e-8), Plane(z=u + d)]
        ct = column_transform(components, 0.0, voltage)
        source = adapted_input_lattice(ct.matrix[:2, 2:4], lam, out_lattice, (1.0 / n, 1.0 / n), (n, n))
        Q = 1j * lam / (jnp.pi * w0**2) * jnp.eye(2)
        field = gaussian_field(source.positions(), 1.0, jnp.zeros(2), jnp.zeros(2), Q, lam)
        plan = plan_lct(ct.matrix, lam, source, out_lattice, method="mft", check="off")
        out = apply_lct(plan, field, ct.phase0)
        return jnp.real(jnp.sum(jnp.conj(target) * out)) / jnp.max(jnp.abs(target))

    v0 = 200e3
    grad = jax.grad(loss)(v0)
    h = 1e-6 * v0
    fd = (loss(v0 + h) - loss(v0 - h)) / (2 * h)
    assert abs(float(fd)) > 0 and abs(float(grad) - float(fd)) < 1e-6 * abs(float(fd)), (float(grad), float(fd))


def test_propagate_is_differentiable_in_the_field():
    lattice = Lattice.regular((64, 64), 10e-9)
    column = [Lens(z=3e-3, focal_length=2e-3), Plane(z=4.6e-3)]
    weights = jnp.asarray(np.random.default_rng(0).normal(size=(64, 64)))
    base = _source(lattice, w=5e-8)
    plan_out = propagate(base, column, tol=1e-12)          # fixes the branches and the lattices

    def loss(scale):
        out = propagate(Wavefront(scale * base.data, lattice, 0.0, VOLTAGE), column, check="off")
        return jnp.sum(weights * jnp.abs(out.data) ** 2)

    assert abs(float(loss(1.0)) - float(jnp.sum(weights * jnp.abs(plan_out.data) ** 2))) < 1e-12 * abs(float(loss(1.0)))
    g = jax.grad(loss)(1.0)
    assert abs(float(g) - 2 * float(loss(1.0))) < 1e-9 * abs(float(g))


# ------------------------------------------------------------------------------------------------
# energy spread: vmap over the voltage, a matrix and a phase per wavelength
# ------------------------------------------------------------------------------------------------
def _em_lens_of(z, rotation_rad=0.4, focal=2.0e-3):
    Rc = float(compute_Rc_from_voltage(VOLTAGE))
    NI = rotation_rad / Rc
    return ElectromagneticLens(z=z, turns=100.0, current=NI / 100.0, Gc=float(effective_accelerating_potential(VOLTAGE)) / (focal * NI**2))


def test_energy_spread_batches_over_the_voltage_with_a_phase_per_wavelength():
    """A chromatic magnetic lens: each voltage has its own focal length, Larmor angle, adapted lattice, chief
    phase and source. vmap over the voltage reproduces the loop, and every member matches the chief
    Gaussian of its own voltage, sign included."""
    n, p = 96, 3e-9
    detector = Lattice.regular((n, n), p)
    components = [_em_lens_of(3e-3), Plane(z=3e-3 + 1.02 * 2e-3)]
    w0 = 1e-7
    voltages = VOLTAGE * (1.0 + jnp.array([-3e-4, -1e-4, 0.0, 1e-4, 3e-4]))

    def one(voltage):
        lam = energy2wavelength(voltage)
        ct = column_transform(components, 0.0, voltage)
        source = adapted_input_lattice(ct.matrix[:2, 2:4], lam, detector, (1.0 / n, 1.0 / n), (n, n))
        Q = 1j * lam / (jnp.pi * w0**2) * jnp.eye(2)
        field = gaussian_field(source.positions(), 1.0, jnp.zeros(2), jnp.zeros(2), Q, lam)
        plan = plan_lct(ct.matrix, lam, source, detector, method="mft", check="off")
        return apply_lct(plan, field, ct.phase0), ct.phase0, ct.matrix

    batched, phases, matrices = jax.jit(jax.vmap(one))(voltages)
    assert batched.shape == (5, n, n) and phases.shape == (5,) and matrices.shape == (5, 5, 5)
    looped = jnp.stack([one(v)[0] for v in voltages])
    assert rel_err(batched, looped) < 1e-10
    # a different focal length and Larmor angle per member, and a chief phase that is 0 or pi for every one
    assert float(jnp.ptp(matrices[:, 2, 0])) > 1e-4 * float(jnp.abs(matrices[2, 2, 0]))
    assert float(jnp.max(jnp.abs(jnp.sin(phases)))) < 1e-9
    for j, voltage in enumerate(voltages):
        lam = float(energy2wavelength(voltage))
        chief = make_gaussian(z=0.0, voltage=voltage, waist_x=w0, waist_y=w0)
        ct = column_transform(components, 0.0, voltage, chief=chief)
        amp, centre, slopes, Q, path = chief_from_beam(ct.chief_out)
        reference = gaussian_field(detector.positions(), amp, centre, slopes, Q, lam, path)
        assert rel_err(batched[j], reference) < 1e-9, j
    incoherent = jnp.mean(jnp.abs(batched) ** 2, axis=0)
    assert float(jnp.max(incoherent)) < float(jnp.max(jnp.abs(batched[2]) ** 2))     # chromatic blur lowers the peak


def test_batched_fields_share_one_plan():
    lattice = Lattice.regular((64, 64), 10e-9)
    column = [Lens(z=3e-3, focal_length=2e-3), Plane(z=4.6e-3)]
    single = _source(lattice, w=5e-8)
    batch = Wavefront(jnp.stack([single.data, 2 * single.data, 1j * single.data]), lattice, 0.0, VOLTAGE)
    out = propagate(batch, column, tol=1e-12)
    ref = propagate(single, column, tol=1e-12)
    assert out.data.shape == (3, 64, 64)
    assert rel_err(out.data[1], 2 * ref.data) < 1e-12 and rel_err(out.data[2], 1j * ref.data) < 1e-12


def test_rotator_column_needs_the_output_lattice_to_be_rotated():
    """A rotator plus a drift has C = 0 and no chirp: the engine's output lattice is the rotated source lattice."""
    angle = 25.0
    column = [Rotator(z=0.0, angle=angle), Plane(z=2e-3)]
    lattice = Lattice.regular((128, 128), 10e-9)
    field = _source(lattice, w=8 * 10e-9)
    out = propagate(field, column)
    R = np.asarray(column_transform(column, 0.0, VOLTAGE).matrix)[:2, :2]
    assert np.allclose(np.asarray(out.lattice.pitch), R @ np.asarray(lattice.pitch), rtol=1e-9)


# ------------------------------------------------------------------------------------------------
# thick elements work through the ray matrix
# ------------------------------------------------------------------------------------------------
def _thick_magnetic_lens(z, stigmator=(0.0, 0.0), rotation_rad=0.6, focal=2e-3, thickness=4e-4):
    """A magnetic lens with a body: two half-strength lenses around a drift, then the Larmor rotation."""
    Rc = float(compute_Rc_from_voltage(VOLTAGE))
    NI = rotation_rad / Rc
    return ElectromagneticLens(z=z, turns=100.0, current=NI / 100.0,
                               Gc=float(effective_accelerating_potential(VOLTAGE)) / (focal * NI**2),
                               Tc=thickness / NI, stigmator_strength_x=stigmator[0], stigmator_strength_y=stigmator[1])


THICK = {
    "magnetic lens with a body": [_thick_magnetic_lens(3e-3), Plane(z=3e-3 + 4e-4 + 2.1e-3)],
    "magnetic lens with a body and a stigmator": [_thick_magnetic_lens(3e-3, stigmator=(0.05, -0.03)),
                                                   Plane(z=3e-3 + 4e-4 + 2.1e-3)],
    "hard-edge solenoid": [HardEdgeSolenoid(z=3e-3, length=2e-3, q=0.9), Plane(z=8e-3)],
}


@pytest.mark.parametrize("name", THICK)
def test_thick_elements_match_the_chief_gaussian_including_the_phase(name):
    """The matrix of a finite body comes from differentiating the ray code, the phase from the Gaussian
    branch (which continues the Gouy phase through the body): an offset, tilted, asymmetric beam agrees
    with the Gaussian field through the same components, sign included."""
    components = THICK[name]
    beam = make_gaussian(x=2e-8, y=-1e-8, dx=4e-6, dy=-3e-6, z=0.0, voltage=VOLTAGE, waist_x=1e-7, waist_y=0.9e-7)
    amp, centre, slopes, Q, path = chief_from_beam(beam)
    lattice = Lattice.regular((192, 192), 10e-9)
    field = Wavefront(gaussian_field(lattice.positions(), amp, centre, slopes, Q, LAM, path), lattice, 0.0, VOLTAGE)
    assert float(column_transform(components, 0.0, VOLTAGE).mismatch) < 1e-9
    wave = propagate(field, components, tol=1e-16)
    amp, centre, slopes, Q, path = chief_from_beam(run_to_end(beam, components))
    assert rel_err(wave.data, gaussian_field(wave.lattice.positions(), amp, centre, slopes, Q, LAM, path)) < 1e-9
