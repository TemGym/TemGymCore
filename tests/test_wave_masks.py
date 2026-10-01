"""Thin components as masks, the split of a column into roles, and the wave against the beamlet sum."""

import functools
import types

import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.aberrations import KrivanekCoeffs
from temgym_core.components import (
    ABCDTransfer,
    ConstantAmplitudeShift,
    ConstantPhaseShift,
    Deflector,
    DeflectionBiprism,
    Detector,
    DistortedLens,
    ElectromagneticLens,
    HardEdgeSolenoid,
    KrivanekLens,
    LinearPhaseShift,
    Lens,
    PhaseBiprism,
    Plane,
    QuadraticAmplitudeShift,
    QuadraticPhaseShift,
    Rotator,
    SeidelLens,
    SigmoidAperture,
    Stigmator,
    ThickLens,
)
from temgym_core.constants import energy2wavelength
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.gaussian import make_gaussian
from temgym_core.run import run_to_end
from temgym_core.source import rectangular_input_wave
from temgym_core.wave import (
    Lattice,
    Wavefront,
    component_mask,
    gaussian_field,
    input_lattice,
    is_linear,
    propagate,
    split_component,
)
from temgym_core.wave.reference import chief_from_beam
from wave_validation import complex_q_chain, gaussian_Q, rel_err, rel_err_no_phase

VOLTAGE = 200e3
LAM = float(energy2wavelength(VOLTAGE))


# ------------------------------------------------------------------------------------------------
# the mask is exactly the action the Gaussian branch applies
# ------------------------------------------------------------------------------------------------
QUADRATIC_ACTIONS = [
    ConstantPhaseShift(z=0.0, constant_phase_shift=3e-12),
    LinearPhaseShift(z=0.0, linear_phase_shift=jnp.array([2e-4, -1e-4])),
    QuadraticPhaseShift(z=0.0, quadratic_phase_shift=jnp.array([[1.2e2, 3e1], [3e1, -7e1]])),
    Lens(z=0.0, focal_length=0.5, x0=3e-7, y0=-1e-7),
    Stigmator(z=0.0, focal_length_x=0.6, focal_length_y=0.9),
    QuadraticAmplitudeShift(z=0.0, quadratic_log_amplitude=jnp.array([[-1e8, 2e7], [2e7, -3e8]])),
    ConstantAmplitudeShift(z=0.0, amplitude=0.7),
]


@pytest.mark.parametrize("component", QUADRATIC_ACTIONS, ids=lambda c: type(c).__name__)
def test_mask_times_gaussian_equals_the_gaussian_branch(component):
    """For an action that is a polynomial of degree two the second-order Gaussian update is exact, so
    the mask times the input Gaussian is the output Gaussian of ``component(beam)``."""
    w = 2e-6
    beam = make_gaussian(x=0.0, y=0.0, dx=1e-5, dy=-2e-5, z=0.0, voltage=VOLTAGE, waist_x=w, waist_y=1.2 * w,
                         rcurv_x=0.4, rcurv_y=-0.7)
    lat = Lattice.regular((64, 64), 0.4 * w, centre=(1e-7, -2e-7))
    before = _beam_on(lat, beam)
    expected = _beam_on(lat, component(beam))
    got = before * component_mask(component, lat, LAM)
    assert rel_err(got, expected) < 1e-10


def _beam_on(lattice, beam):
    amp, centre, slopes, Q, path = chief_from_beam(beam)
    return gaussian_field(lattice.positions(), amp, centre, slopes, Q, LAM, path)


def test_phase_only_masks_have_unit_modulus_and_absorbing_masks_do_not():
    lat = Lattice.regular((32, 32), 1e-6)
    phase = component_mask(LinearPhaseShift(z=0.0, linear_phase_shift=jnp.array([2e-4, 0.0])), lat, LAM)
    assert float(jnp.max(jnp.abs(jnp.abs(phase) - 1.0))) < 1e-14
    absorbing = component_mask(ConstantAmplitudeShift(z=0.0, amplitude=0.5), lat, LAM)
    assert np.allclose(np.asarray(jnp.abs(absorbing)), 0.5)


def test_mask_follows_the_lattice_orientation():
    """The mask is evaluated at the physical positions of the samples, whatever the lattice."""
    ap = SigmoidAperture(z=0.0, radius=8e-6, edge_width=2e-6, x0=1e-6)
    rotated = Lattice.regular((40, 30), (0.7e-6, 0.5e-6), centre=(1e-6, 0.0), rotation=33.0)
    mask = component_mask(ap, rotated, LAM)
    x, y = rotated.coords()
    k = 2 * np.pi / LAM
    for iy, ix in ((0, 0), (7, 21), (29, 39), (15, 3)):
        direct = jnp.exp(1j * k * ap.complex_action(jnp.array([x[iy, ix], y[iy, ix]]), k))
        assert abs(complex(mask[iy, ix]) - complex(direct)) < 1e-14


# ------------------------------------------------------------------------------------------------
# roles: linear, mask, split, refused
# ------------------------------------------------------------------------------------------------
def test_linear_components_are_not_masks():
    linear = [Plane(z=0.1), Detector(z=0.1, pixel_size=(1e-6, 1e-6), shape=(4, 4)), Lens(z=0.1, focal_length=1.0),
              Stigmator(z=0.1, focal_length_x=1.0, focal_length_y=2.0), Deflector(z=0.1, def_x=1e-6, def_y=0.0),
              Rotator(z=0.1, angle=10.0), HardEdgeSolenoid(z=0.1, length=0.02, q=0.3),
              ElectromagneticLens(z=0.1, turns=10.0, current=0.1, Gc=1.0),
              QuadraticPhaseShift(z=0.1, quadratic_phase_shift=jnp.eye(2))]
    for component in linear:
        assert is_linear(component), type(component).__name__
        assert split_component(component) == (component, None) or split_component(component)[1] is None


def test_apertures_and_biprisms_are_masks():
    for component in (SigmoidAperture(z=0.1, radius=1e-3, edge_width=1e-4),
                      PhaseBiprism(z=0.1, strength=1e-5, width=1e-6),
                      ConstantAmplitudeShift(z=0.1, amplitude=0.5)):
        linear, mask = split_component(component)
        assert linear is None and mask is component and not is_linear(component)


def test_krivanek_lens_splits_into_the_ideal_lens_and_its_aberration():
    lens = KrivanekLens(z=0.2, focal_length=0.5, x0=1e-4, coeffs=KrivanekCoeffs(C10=1e-6, C12=2e-7, phi12=0.3))
    linear, mask = split_component(lens)
    assert type(linear) is Lens and linear.focal_length == 0.5 and linear.x0 == 1e-4 and float(mask.z) == 0.2
    xy = jnp.array([3e-4, -2e-4])
    ideal = linear.phase_shift(xy)
    assert abs(float(jnp.real(mask.complex_action(xy, 1.0) + ideal - lens.phase_shift(xy)))) < 1e-18


@pytest.mark.parametrize("component, message", [
    (SeidelLens(z=0.0, focal_length=1.0), "depends on the ray slope"),
    (DistortedLens(z=0.0, focal_length=1.0), "depends on the ray slope"),
    (ThickLens(z_po=0.0, z_pi=0.1, focal_length=1.0), "ray-only"),
    (DeflectionBiprism(z=0.0, def_x=1e-5), "PhaseBiprism"),
    (ABCDTransfer(A=jnp.eye(2), B=jnp.eye(2)), "phase reference"),
])
def test_non_local_and_ray_only_elements_are_refused_with_a_clear_error(component, message):
    with pytest.raises(NotImplementedError, match=message):
        split_component(component)
    field = Wavefront(jnp.ones((8, 8), dtype=complex), Lattice.regular((8, 8), 1e-6), 0.0, VOLTAGE)
    with pytest.raises(NotImplementedError, match=message):
        propagate(field, [component, Plane(z=0.1)])


# ------------------------------------------------------------------------------------------------
# thin lens plus aperture through wave.propagate against the Gaussian beamlet sum
# ------------------------------------------------------------------------------------------------
def _beamlet_field(beam, lattice):
    """Coherent field of a bundle at the samples of a lattice (``evaluate_gaussians`` needs coords and shape)."""
    positions = types.SimpleNamespace(coords=lattice.positions().reshape(-1, 2), shape=lattice.shape)
    return evaluate_gaussians(beam, positions, method="scan")


@functools.lru_cache(maxsize=None)
def _imaging_column(waist):
    """A square imaged 1:1 by a lens with a soft circular aperture at the lens plane, and both answers."""
    detector = Detector(z=0.8, pixel_size=(0.5e-6, 0.5e-6), shape=(256, 256))
    aperture = SigmoidAperture(z=0.4, radius=20e-6, edge_width=5e-6)
    column = (Lens(z=0.4, focal_length=0.2), aperture, detector)
    beam = rectangular_input_wave(aperture_width=40e-6, aperture_height=40e-6, waist=waist, voltage=VOLTAGE,
                                  overlap_factor=2.0, z0=0.0)
    beamlets = evaluate_gaussians(run_to_end(beam, column), detector, method="scan")
    lattice = Lattice.from_grid(detector)
    source = input_lattice(column, lattice, source_z=0.0, voltage=VOLTAGE)
    wave, reports = propagate(Wavefront(_beamlet_field(beam, source), source, 0.0, VOLTAGE), column, lattice,
                              tol=1e-10, return_reports=True)
    return wave.data, beamlets, reports


def test_lens_and_aperture_column_matches_the_beamlet_sum():
    """The wave is the reference (exact mask, exact Collins transform); the beamlet sum expands the
    aperture to second order about each beamlet. At 2 um waists against a 5 um edge they agree to a
    few parts in 1e4, and the difference shrinks as the beamlets shrink."""
    wave, beamlets, reports = _imaging_column(2e-6)
    assert [r.kind for r in reports] == ["near_image[I]", "near_image[I]"]     # drift, then lens and drift
    assert rel_err(jnp.abs(wave) ** 2, jnp.abs(beamlets) ** 2) < 2e-3
    assert rel_err_no_phase(wave, beamlets) < 2e-3
    # the aperture really acted: the corners of the 40 um square are outside the 20 um radius
    intensity = np.asarray(jnp.abs(wave) ** 2)
    assert intensity[128 + 60, 128 + 60] < 0.02 * intensity[128, 128]


@pytest.mark.slow
def test_beamlet_sum_converges_to_the_wave_as_the_beamlets_shrink():
    """Measured 8.5e-3, 6.9e-4 and 9.0e-5 for waists of 4, 2 and 1 um against a 5 um aperture edge."""
    errors = [rel_err(jnp.abs(w) ** 2, jnp.abs(b) ** 2) for w, b, _ in map(_imaging_column, (4e-6, 2e-6, 1e-6))]
    assert errors[0] > 5 * errors[1] and errors[1] > 4 * errors[2] and errors[2] < 3e-4, errors


# ------------------------------------------------------------------------------------------------
# an aberrated lens: ideal lens in the matrix, aberration as a mask
# ------------------------------------------------------------------------------------------------
def test_defocus_of_a_krivanek_lens_equals_a_lens_with_the_shifted_focal_length():
    """Only C10: the aberration is a quadratic phase, so the split (ideal lens in the span, aberration as a
    mask) must reproduce the single lens of focal length 1/f' = 1/f + C10/f^2, checked against the
    complex-q law for that lens (the two outputs live on different lattices: A P differs with f)."""
    f, c10 = 2e-3, 1.5e-4
    f_eff = 1.0 / (1.0 / f + c10 / f**2)
    N, p, u, v = 192, 10e-9, 3e-3, 4.5e-3
    lat = Lattice.regular((N, N), p)
    w0 = 1e-7
    Q0 = gaussian_Q(w0, w0, LAM)
    field = Wavefront(gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), Q0, LAM), lat, 0.0, VOLTAGE)
    Qo, amp = complex_q_chain(Q0, [("drift", u), ("lens", f_eff), ("drift", v)])
    aberrated = propagate(field, [KrivanekLens(z=u, focal_length=f, coeffs=KrivanekCoeffs(C10=c10)),
                                  Plane(z=u + v)], tol=1e-14)
    ideal = propagate(field, [Lens(z=u, focal_length=f_eff), Plane(z=u + v)], tol=1e-14)
    for result in (aberrated, ideal):
        reference = gaussian_field(result.lattice.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)
        assert rel_err_no_phase(result.data, reference) < 1e-9
    plain = propagate(field, [Lens(z=u, focal_length=f), Plane(z=u + v)], tol=1e-14)
    reference = gaussian_field(plain.lattice.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)
    assert rel_err_no_phase(plain.data, reference) > 1e-2            # without the aberration it is a different beam


def test_a_mask_at_a_lens_plane_sees_the_field_before_the_lens():
    """[Lens, aperture] and [aperture, Lens] are one physical element pair (both multiply by a function of
    position); the engine applies the mask on the smooth field before the lens chirp, so it needs no
    sampling of the chirp."""
    N, p = 192, 10e-9
    lat = Lattice.regular((N, N), p)
    source = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(1e-7, 1e-7, LAM), LAM)
    field = Wavefront(source, lat, 0.0, VOLTAGE)
    aperture = SigmoidAperture(z=3e-3, radius=1.5e-7, edge_width=1e-7)
    after = propagate(field, [Lens(z=3e-3, focal_length=2e-3), aperture, Plane(z=4.02e-3)], tol=1e-10)
    before = propagate(field, [aperture, Lens(z=3e-3, focal_length=2e-3), Plane(z=4.02e-3)], tol=1e-10)
    assert rel_err(after.data, before.data) < 1e-12
