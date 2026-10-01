"""Sampled-wave propagation against independent references (CPU).

References: ``propagator.FresnelPropagator`` and ``AngularSpectrumPropagator`` (free space), a
sequential complex-q chain (lens plus drift), and a chief ``GaussianBeam`` propagated through the
*same* components with the Gaussian engine (astigmatism, rotation, magnetic lens, offsets). The chief
supplies the physical Gouy/Maslov branch and constant path through ``reference.chief_phase0``; the
tests also assert that this ``phase0`` is only a sign (0 or pi) for drift, lens, stigmator and
rotation columns, which is the documented Gouy protocol.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import (
    Deflector,
    ElectromagneticLens,
    Lens,
    Plane,
    Rotator,
    Stigmator,
)
from temgym_core.constants import (
    compute_Rc_from_voltage,
    effective_accelerating_potential,
    energy2wavelength,
)
from temgym_core.gaussian import FreeSpacePropagator, make_gaussian
from temgym_core.propagator import AngularSpectrumPropagator, FresnelPropagator
from temgym_core.run import column_matrix, run_to_end
from temgym_core.wave import (
    Lattice,
    LatticeError,
    SamplingError,
    adapted_input_lattice,
    adapted_output_lattice,
    apply_lct,
    chief_phase0,
    collins_gaussian,
    gaussian_field,
    measure_support,
    plan_lct,
)
from temgym_core.wave.reference import chief_from_beam
from wave_validation import complex_q_chain, gaussian_Q, rel_err, rel_err_no_phase

VOLTAGE = 200e3
LAM = float(energy2wavelength(VOLTAGE))
FS = FreeSpacePropagator()


def chief(x=0.0, y=0.0, dx=0.0, dy=0.0, wx=1e-7, wy=1e-7, rx=np.inf, ry=np.inf):
    return make_gaussian(x=x, y=y, dx=dx, dy=dy, z=0.0, voltage=VOLTAGE, amp=1.0, phase=0.0,
                         waist_x=wx, waist_y=wy, rcurv_x=rx, rcurv_y=ry)


def run_case(comps, b0, N, p, *, method="auto", tol=1e-22):
    """Plan with the planner (field-measured support and band), apply, compare with the chief field."""
    M5 = np.asarray(column_matrix(comps, source_z=0.0, voltage=VOLTAGE))
    b1 = run_to_end(b0, comps, propagator=FS)
    cin, cout = chief_from_beam(b0), chief_from_beam(b1)
    lat = Lattice.regular((N, N), p)
    u = gaussian_field(lat.positions(), cin[0], cin[1], cin[2], cin[3], LAM, cin[4])
    sup, band = measure_support(u, tol=tol)
    # an MFT window follows the chief ray (the zero-slope image of the centre is the default otherwise)
    plan, rep = plan_lct(M5, LAM, lat, method=method, support=sup, band=band, check="raise",
                         out_centre=cout[1], return_report=True)
    ph0, mismatch = chief_phase0(M5, LAM, b0, b1)
    got = apply_lct(plan, u, phase0=ph0)
    ref = gaussian_field(plan.out_lattice.positions(), cout[0], cout[1], cout[2], cout[3], LAM, cout[4])
    info = dict(kind=plan.kind + plan.variant, e_phys=rel_err(got, ref),
                e_glob=rel_err_no_phase(apply_lct(plan, u), ref), phase0=float(ph0),
                chief_mismatch=float(mismatch), planner_ok=None if rep is None else bool(rep.ok))
    return plan, info


def _is_sign(phase0, tol=1e-9):
    """``phase0`` is 0 or pi modulo 2 pi."""
    return abs(np.sin(phase0)) < tol


N_, P_ = 192, 10e-9            # 1.92 um window, 10 nm pixels: F = 1 at 7.7 mm
U_, F_ = 3e-3, 2e-3            # object plane to lens distance, focal length
V_IMG = U_ * F_ / (U_ - F_)


# ------------------------------------------------------------------------------------------------
# free space
# ------------------------------------------------------------------------------------------------
def _drift(dz):
    M = np.eye(5)
    M[0, 2] = M[1, 3] = dz
    return M


@pytest.mark.parametrize("F", [0.1, 0.5, 0.9])
def test_free_space_vs_fresnel_propagator(F):
    N, p = 128, 25e-9
    ell = N * p * p / LAM
    lat = Lattice.regular((N, N), p)
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(8 * p, 7 * p, LAM), LAM)
    dz = F * ell
    ni = apply_lct(plan_lct(_drift(dz), LAM, lat, method="near_image", check="off"), u)
    fr = FresnelPropagator(u, N * p, LAM, dz)
    assert rel_err_no_phase(ni, fr) < 1e-10
    # even the carrier phase agrees: Fresnel includes exp(ikz), the engine does not
    assert rel_err(ni, fr * jnp.exp(-1j * 2 * jnp.pi * dz / LAM)) < 1e-10


def test_mft_at_unit_fresnel_number_vs_fresnel_propagator():
    """F = 1: the natural DFT pitch lambda*d/(n p) equals p, so MFT (FFT path) and TF live on one lattice."""
    N, p = 128, 25e-9
    ell = N * p * p / LAM
    lat = Lattice.regular((N, N), p)
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(8 * p, 7 * p, LAM), LAM)
    plan = plan_lct(_drift(ell), LAM, lat, method="mft", check="off")
    assert plan.mode_x == 1 and plan.mode_y == 1
    assert jnp.allclose(plan.out_lattice.pitch, lat.pitch, rtol=1e-12)
    assert rel_err_no_phase(apply_lct(plan, u), FresnelPropagator(u, N * p, LAM, ell)) < 1e-10


@pytest.mark.parametrize("F", [0.01, 0.03])
def test_free_space_vs_angular_spectrum(F):
    """Optical-scale parameters: the AS reference computes exp(i kz z) in float64, so it is only good to
    ~1e-16 * k z; with k z ~ 1e5 rad and lambda/p = 0.01 it reaches 1e-11 (at electron scale, k z ~ 1e13,
    its own floor is ~1e-3)."""
    N, p, lam = 256, 1.0, 0.01
    ell = N * p * p / lam
    lat = Lattice.regular((N, N), p)
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2),
                       gaussian_Q(24 * p, 21 * p, lam), lam)
    dz = F * ell
    ni = apply_lct(plan_lct(_drift(dz), lam, lat, method="near_image", check="off"), u)
    asp = AngularSpectrumPropagator(u, N * p, lam, dz, xp=jnp)
    assert rel_err_no_phase(ni, asp) < 1e-10


def test_angular_spectrum_floor_at_electron_scale_is_the_reference_not_the_engine():
    N, p = 256, 250e-9
    ell = N * p * p / LAM
    lat = Lattice.regular((N, N), p)
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(24 * p, 21 * p, LAM), LAM)
    dz = 0.5 * ell
    ni = apply_lct(plan_lct(_drift(dz), LAM, lat, method="near_image", check="off"), u)
    fr = FresnelPropagator(u, N * p, LAM, dz)
    asp = AngularSpectrumPropagator(u, N * p, LAM, dz, xp=jnp)
    assert rel_err_no_phase(ni, fr) < 1e-12          # engine == Fresnel to rounding
    e_as = rel_err_no_phase(ni, asp)
    kz_floor = 1e-16 * 2 * np.pi / LAM * dz
    assert e_as > 1e-6 and e_as < 100 * kz_floor      # bounded by float64 cancellation in the reference


# ------------------------------------------------------------------------------------------------
# thin lens plus drift versus the sequential complex-q result
# ------------------------------------------------------------------------------------------------
LENS_CASES = [
    ("BFP exactly", F_),
    ("near BFP +2%", 1.02 * F_),
    ("near BFP -2%", 0.98 * F_),
    ("near image +1%", 1.01 * V_IMG),
    ("near image -1e-3", (1 - 1e-3) * V_IMG),
    ("near image +1e-6", (1 + 1e-6) * V_IMG),
    ("exact image (B=0)", V_IMG),
    ("general 0.6 v_img", 0.6 * V_IMG),
    ("beyond image 1.5", 1.5 * V_IMG),
]


@pytest.mark.parametrize("label, v", LENS_CASES, ids=[c[0] for c in LENS_CASES])
def test_thin_lens_gaussian_vs_complex_q(label, v):
    w0 = 1.0e-7
    b0 = chief(wx=w0, wy=w0)
    comps = [Lens(z=U_, focal_length=F_), Plane(z=U_ + v)]
    plan, info = run_case(comps, b0, N_, P_)
    # independent reference: sequential complex-q law (drift u, lens f, drift v), no Collins algebra
    Q0 = gaussian_Q(w0, w0, LAM)
    Qo, amp = complex_q_chain(Q0, [("drift", U_), ("lens", F_), ("drift", v)])
    lat = Lattice.regular((N_, N_), P_)
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), Q0, LAM)
    got = apply_lct(plan, u)
    ref = gaussian_field(plan.out_lattice.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)
    assert rel_err_no_phase(got, ref) < 1e-9
    assert info["e_phys"] < 1e-9 and info["planner_ok"] in (True, None)
    assert _is_sign(info["phase0"])
    # the engine lift differs from the continuous Gouy branch only by the BFP crossing (v > f): exactly
    # one focal-plane transit flips the round-beam sign
    ph = rel_err(got, ref)                            # includes phase: 0 (same branch) or 2 (opposite)
    assert min(ph, abs(ph - 2.0)) < 1e-9


def test_branch_selection_regimes():
    """Auto planner: MFT near the back focal plane (A -> 0), near-image branch near and at the image plane."""
    b0 = chief(wx=1e-7, wy=1e-7)
    kinds = {}
    for label, v in LENS_CASES:
        _, info = run_case([Lens(z=U_, focal_length=F_), Plane(z=U_ + v)], b0, N_, P_)
        kinds[label] = info["kind"]
    assert kinds["BFP exactly"] == "mft" and kinds["near BFP +2%"] == "mft"
    assert kinds["exact image (B=0)"] == "exactI"
    assert kinds["near image +1%"] == "near_imageI" and kinds["near image +1e-6"] == "near_imageI"


# ------------------------------------------------------------------------------------------------
# astigmatism (aligned and rotated), asymmetric tilted and offset beam
# ------------------------------------------------------------------------------------------------
FX, FY = 2.0e-3, 2.6e-3
B_ASYM = dict(x=2e-8, y=-1e-8, dx=6e-6, dy=-9e-6, wx=1.0e-7, wy=0.85e-7)   # x/y-asymmetric, offset, tilted
STIG_PLANES = [("x-focal plane", FX), ("between foci", 0.5 * (FX + FY)), ("y-focal plane", FY),
               ("far", 3 * FY), ("near image x", U_ * FX / (U_ - FX))]
P3_ = 8e-9   # 8 nm pixels: at the y-focal plane the x axis sits at F = 1.2 (MFT-comfortable), not ~0.7


@pytest.mark.parametrize("label, v", STIG_PLANES, ids=[c[0] for c in STIG_PLANES])
def test_stigmator_aligned(label, v):
    comps = [Stigmator(z=U_, focal_length_x=FX, focal_length_y=FY), Plane(z=U_ + v)]
    _, info = run_case(comps, chief(**B_ASYM), N_, P3_)
    assert info["e_phys"] < 1e-9 and _is_sign(info["phase0"])


@pytest.mark.parametrize("angle", [30.0, 67.0])
@pytest.mark.parametrize("label, v", STIG_PLANES[:4], ids=[c[0] for c in STIG_PLANES[:4]])
def test_stigmator_rotated(label, v, angle):
    comps = [Rotator(z=U_, angle=angle), Stigmator(z=U_, focal_length_x=FX, focal_length_y=FY),
             Rotator(z=U_, angle=-angle), Plane(z=U_ + v)]
    _, info = run_case(comps, chief(**B_ASYM), N_, P3_)
    assert info["e_phys"] < 1e-9 and _is_sign(info["phase0"])


def test_x_and_y_are_not_swapped():
    """Swapping the two focal lengths must transpose the output (catches x/y vs y/x errors)."""
    v = 0.5 * (FX + FY)
    lat = Lattice.regular((N_, N_), P_)
    ua = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(1e-7, 0.7e-7, LAM), LAM)
    ub = ua.T                                            # swap the roles of x and y in the input
    Ma = np.asarray(column_matrix([Stigmator(z=U_, focal_length_x=FX, focal_length_y=FY), Plane(z=U_ + v)],
                                  source_z=0.0, voltage=VOLTAGE))
    Mb = np.asarray(column_matrix([Stigmator(z=U_, focal_length_x=FY, focal_length_y=FX), Plane(z=U_ + v)],
                                  source_z=0.0, voltage=VOLTAGE))
    pa = plan_lct(Ma, LAM, lat, method="mft", check="off")
    pb = plan_lct(Mb, LAM, lat, method="mft", check="off")
    oa, ob = apply_lct(pa, ua), apply_lct(pb, ub)
    assert rel_err(ob, oa.T) < 1e-12
    assert rel_err(oa, oa.T) > 0.1                       # and the output really is asymmetric


# ------------------------------------------------------------------------------------------------
# round magnetic lens with Larmor rotation
# ------------------------------------------------------------------------------------------------
def _em_lens(z, rotation_rad=0.7, focal=2.0e-3):
    Rc = float(compute_Rc_from_voltage(VOLTAGE))
    NI = rotation_rad / Rc
    Vstar = float(effective_accelerating_potential(VOLTAGE))
    return ElectromagneticLens(z=z, turns=100.0, current=NI / 100.0, Gc=Vstar / (focal * NI**2))


EM_PLANES = [("BFP", F_), ("near BFP +3%", 1.03 * F_), ("near image +0.2%", 1.002 * V_IMG),
             ("exact image", V_IMG), ("general 5 mm", 5e-3)]


@pytest.mark.parametrize("label, v", EM_PLANES, ids=[c[0] for c in EM_PLANES])
def test_electromagnetic_lens_with_larmor_rotation(label, v):
    lens = _em_lens(U_)
    assert abs(float(lens.rotation_angle(VOLTAGE)) - 0.7) < 1e-12
    comps = [lens, Plane(z=U_ + v)]
    plan, info = run_case(comps, chief(**B_ASYM), N_, P_)
    assert info["e_phys"] < 1e-9 and _is_sign(info["phase0"])
    # the output lattice is the input lattice rotated by the Larmor angle (times the magnification)
    A = np.asarray(column_matrix(comps, source_z=0.0, voltage=VOLTAGE))[:2, :2]
    if plan.kind != "mft":
        assert np.allclose(np.asarray(plan.out_lattice.pitch), A @ (P_ * np.eye(2)), rtol=1e-12)


def test_analytic_rotated_gaussian_at_the_image_plane():
    """Analytic rotated round Gaussian at the image plane: |g| rotates with the lens (chief-independent)."""
    lens = _em_lens(U_)
    v = V_IMG
    comps = [lens, Plane(z=U_ + v)]
    M5 = np.asarray(column_matrix(comps, source_z=0.0, voltage=VOLTAGE))
    w0, x0 = 1e-7, 4e-8
    lat = Lattice.regular((N_, N_), P_)
    u = gaussian_field(lat.positions(), 1.0, np.array([x0, 0.0]), np.zeros(2), gaussian_Q(w0, w0, LAM), LAM)
    plan = plan_lct(M5, LAM, lat)                       # exact image plane -> exact branch
    out = apply_lct(plan, u)
    # analytic: centre at A r_c, width magnified by |a|, image of a *round* beam is round
    A = M5[:2, :2]
    centre = A @ np.array([x0, 0.0])
    mag = np.sqrt(abs(np.linalg.det(A)))
    r2 = jnp.sum((plan.out_lattice.positions() - centre) ** 2, axis=-1)
    ref_abs = jnp.exp(-r2 / (mag * w0) ** 2) / mag
    assert float(jnp.max(jnp.abs(jnp.abs(out) - ref_abs))) < 1e-9
    # rotation angle of the image relative to the object: A = mag * R(phi)
    phi = np.arctan2(A[1, 0], A[0, 0])
    expected_rot = 0.7 + (np.pi if A[0, 0] / mag < 0 else 0.0)
    assert abs(((phi - expected_rot + np.pi) % (2 * np.pi)) - np.pi) < 1e-9


# ------------------------------------------------------------------------------------------------
# precision at k z ~ 1e12, mixed regimes, rotated lattices
# ------------------------------------------------------------------------------------------------
def test_metre_scale_electron_drift_has_no_phase_precision_floor():
    """k*d ~ 1e12 rad: only the excess phase is sampled, the carrier is not (contrast: the AS floor)."""
    N, p = 256, 2e-6
    d = 0.5
    ell = N * p * p / LAM
    lat = Lattice.regular((N, N), p)
    Q0 = gaussian_Q(10 * p, 9 * p, LAM)
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), Q0, LAM)
    plan = plan_lct(_drift(d), LAM, lat, field=u)                     # F = d / ell ~ 0.3 -> near-image branch
    Qo, amp = complex_q_chain(Q0, [("drift", d)])
    ref = gaussian_field(plan.out_lattice.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)
    assert d / ell < 1 and 2 * np.pi / LAM * d > 1e11
    assert rel_err(apply_lct(plan, u), ref) < 1e-12


def test_mixed_regime_astigmatic_case_is_flagged_and_cured_by_resampling_the_window():
    """y at its focal plane (A_y = 0, only the MFT works) while x sits at F ~ 0.7 (neither branch is comfortable
    with 10 nm pixels): the planner raises instead of returning aliased data; 8 nm pixels fix it."""
    comps = [Stigmator(z=U_, focal_length_x=FX, focal_length_y=FY), Plane(z=U_ + FY)]
    with pytest.raises(SamplingError, match="no branch"):
        run_case(comps, chief(**B_ASYM), N_, P_)
    _, info = run_case(comps, chief(**B_ASYM), N_, 8e-9)
    assert info["e_phys"] < 1e-9


def test_mixed_regime_case_is_rescued_by_composing_through_an_intermediate_plane():
    """A short drift first (near-image branch, lattice unchanged) makes B of the remaining map well
    conditioned, so the MFT applies: M = M2 D(d1), any d1 between 0.05 and 0.8 Fresnel lengths works."""
    comps = [Stigmator(z=U_, focal_length_x=FX, focal_length_y=FY), Plane(z=U_ + FY)]
    M5 = np.asarray(column_matrix(comps, source_z=0.0, voltage=VOLTAGE))
    b0 = chief(**B_ASYM)
    cin, cout = chief_from_beam(b0), chief_from_beam(run_to_end(b0, comps, propagator=FS))
    lat = Lattice.regular((N_, N_), P_)
    u = gaussian_field(lat.positions(), cin[0], cin[1], cin[2], cin[3], LAM, cin[4])
    ell = N_ * P_**2 / LAM
    for d1_over_ell in (0.05, 0.3, 0.8):
        D1 = np.eye(5)
        D1[0, 2] = D1[1, 3] = d1_over_ell * ell
        M2 = M5 @ np.linalg.inv(D1)
        p1 = plan_lct(D1, LAM, lat, field=u)
        u1 = apply_lct(p1, u)
        p2 = plan_lct(M2, LAM, p1.out_lattice, field=u1)
        assert p1.kind == "near_image" and p2.kind == "mft"
        ref = gaussian_field(p2.out_lattice.positions(), cout[0], cout[1], cout[2], cout[3], LAM, cout[4])
        assert rel_err_no_phase(apply_lct(p2, u1), ref) < 1e-12


def test_larmor_rotation_makes_the_adapted_input_lattice_rotated():
    """Cartesian in and Cartesian out is impossible without resampling one side: the engine raises
    LatticeError for unadapted lattices and the adapted input lattice is the Cartesian one rotated by
    the Larmor angle."""
    lens = _em_lens(U_, rotation_rad=0.7)
    v = 1.03 * F_                                                  # MFT regime
    M5 = np.asarray(column_matrix([lens, Plane(z=U_ + v)], source_z=0.0, voltage=VOLTAGE))
    out = Lattice.regular((64, 64), 5e-9)
    cart = Lattice.regular((64, 64), 5e-9)
    with pytest.raises(LatticeError):
        plan_lct(M5, LAM, cart, out, method="mft")
    inn = adapted_input_lattice(M5[:2, 2:4], LAM, out, (1 / 64, 1 / 64), (64, 64))
    P = np.asarray(inn.pitch)
    B = M5[:2, 2:4]
    Rot = P / np.sqrt(abs(np.linalg.det(P)))                       # pitch is scalar * rotation
    angle = np.arctan2(Rot[1, 0], Rot[0, 0])
    phi = np.arctan2(B[1, 0], B[0, 0])                              # B = b R(phi) up to sign: the Larmor rotation
    assert np.allclose(Rot @ Rot.T, np.eye(2), atol=1e-12)
    assert abs(np.sin(angle + phi)) < 1e-9                          # rotated by -phi (mod pi)


# ------------------------------------------------------------------------------------------------
# offsets (tilt, decentred lens, deflector): the offset column and the constant path
# ------------------------------------------------------------------------------------------------
def test_offset_column_decentred_lens_and_deflector():
    x0, y0 = 6e-8, -4e-8
    comps = [Deflector(z=0.5e-3, def_x=3e-6, def_y=-2e-6), Lens(z=U_, focal_length=F_, x0=x0, y0=y0),
             Plane(z=U_ + 1.02 * V_IMG)]
    M5 = np.asarray(column_matrix(comps, source_z=0.0, voltage=VOLTAGE))
    assert np.abs(M5[:4, 4]).max() > 0
    b0 = chief(x=1e-8, y=1e-8, dx=2e-6, dy=1e-6, wx=1.0e-7, wy=0.85e-7)
    _, info = run_case(comps, b0, N_, P_)
    assert info["e_phys"] < 1e-9
    # the matrix carries the slope kick x0/f of the decentred lens (plus the deflector), not the constant path
    Mc = np.asarray(column_matrix([Deflector(z=0.5e-3, def_x=3e-6, def_y=-2e-6), Lens(z=U_, focal_length=F_),
                                   Plane(z=U_ + 1.02 * V_IMG)], source_z=0.0, voltage=VOLTAGE))
    assert np.abs((M5 - Mc)[:4, :4]).max() == 0
    kick = np.array([x0 / F_, y0 / F_])
    d2 = 1.02 * V_IMG
    assert np.allclose((M5 - Mc)[:2, 4], d2 * kick, rtol=1e-12) and np.allclose((M5 - Mc)[2:4, 4], kick, rtol=1e-12)
    # the chief path is the Hamilton action along the chief ray: drift 1/2 d theta^2, lens -(r - r0)^2/2f, ...
    b1 = run_to_end(b0, comps, propagator=FS)
    th1 = np.array([2e-6, 1e-6]) + np.array([3e-6, -2e-6])            # slope after the deflector
    z2 = U_ - 0.5e-3
    r0 = np.array([1e-8, 1e-8])
    r_def = r0 + 0.5e-3 * np.array([2e-6, 1e-6])
    r_lens = r_def + z2 * th1
    th_out = th1 - (r_lens - np.array([x0, y0])) / F_
    kick_path = 3e-6 * r_def[0] - 2e-6 * r_def[1]                      # Deflector: linear action def . r
    hamilton = (kick_path + 0.5e-3 * 0.5 * np.sum(np.array([2e-6, 1e-6]) ** 2) + 0.5 * z2 * np.sum(th1**2)
                - 0.5 * np.sum((r_lens - np.array([x0, y0])) ** 2) / F_ + 0.5 * d2 * np.sum(th_out**2))
    assert abs(float(b1.pathlength) - hamilton) < 1e-12 * abs(hamilton) + 1e-25
    comps_c = [Deflector(z=0.5e-3, def_x=3e-6, def_y=-2e-6), Lens(z=U_, focal_length=F_), Plane(z=U_ + d2)]
    ph_off, _ = chief_phase0(M5, LAM, b0, b1)
    ph_cen, _ = chief_phase0(Mc, LAM, b0, run_to_end(b0, comps_c, propagator=FS))
    # kicks (deflector) and decentring both contribute constant path: phase0 is a sign only for offset-free columns
    assert abs(float(np.angle(np.exp(1j * (ph_off - ph_cen))))) > 1e-2


def test_phase0_depends_on_the_chief_ray_not_on_the_beam_shape():
    """The constant path and Gouy correction belong to the ray, so beams of different waists share it."""
    comps = [Lens(z=U_, focal_length=F_, x0=5e-8), Plane(z=U_ + 0.7 * V_IMG)]
    M5 = np.asarray(column_matrix(comps, source_z=0.0, voltage=VOLTAGE))
    phases = []
    for w in (0.8e-7, 1.0e-7, 1.3e-7):
        b0 = chief(x=1e-8, dx=3e-6, wx=w, wy=1.1 * w)
        b1 = run_to_end(b0, comps, propagator=FS)
        phases.append(float(chief_phase0(M5, LAM, b0, b1)[0]))
    d = np.angle(np.exp(1j * (np.array(phases) - phases[0])))
    assert np.abs(d).max() < 1e-8, phases


# ------------------------------------------------------------------------------------------------
# lift behaviour at image and Fourier planes
# ------------------------------------------------------------------------------------------------
def test_lift_is_continuous_through_the_image_plane_and_flips_across_the_bfp():
    b0 = chief(wx=1e-7, wy=1e-7)
    raw = []
    for eps in (-1e-2, -1e-4, -1e-6, 0.0, 1e-6, 1e-4, 1e-2):
        comps = [Lens(z=U_, focal_length=F_), Plane(z=U_ + V_IMG * (1 + eps))]
        _, info = run_case(comps, b0, N_, P_)
        raw.append(info["phase0"])
    # phase0 is the same branch bit on both sides of the image plane (no sign flip at B = 0)
    assert len({round(abs(np.sin(x)), 6) for x in raw}) == 1 and len({round(np.cos(x)) for x in raw}) == 1
    rawf = []
    for eps in (-1e-2, -1e-6, 1e-6, 1e-2):
        comps = [Lens(z=U_, focal_length=F_), Plane(z=U_ + F_ * (1 + eps))]
        _, info = run_case(comps, b0, N_, P_)
        rawf.append(round(np.cos(info["phase0"])))
    assert rawf[0] == rawf[1] and rawf[2] == rawf[3] and rawf[0] == -rawf[3]   # flips across the BFP


# ------------------------------------------------------------------------------------------------
# round trip and flux
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("label, v", [("MFT near BFP", 1.02 * F_), ("near image", 1.01 * V_IMG),
                                      ("exact image", V_IMG)])
def test_round_trip_with_inverse_matrix(label, v):
    comps = [_em_lens(U_), Plane(z=U_ + v)]
    M5 = np.asarray(column_matrix(comps, source_z=0.0, voltage=VOLTAGE))
    lat = Lattice.regular((N_, N_), P_)
    cin = chief_from_beam(chief(**B_ASYM))
    u = gaussian_field(lat.positions(), cin[0], cin[1], cin[2], cin[3], LAM, cin[4])
    sup, band = measure_support(u, tol=1e-22)
    fwd = plan_lct(M5, LAM, lat, support=sup, band=band, check="raise")
    out = apply_lct(fwd, u)
    # (a) inverse plan built from the forward plan: returns to exactly the original lattice
    inv = fwd.inverse()
    assert jnp.allclose(inv.out_lattice.pitch, lat.pitch, rtol=1e-12, atol=1e-14 * P_)
    assert rel_err(apply_lct(inv, out), u) < 1e-10
    # (b) independent plan for the inverse matrix, same target lattice. The planner (rightly, for an
    # arbitrary field) refuses the near-image inverse: the image-plane samples carry the lens chirp and
    # are Nyquist-saturated, yet multiplying by the exact conjugate chirp sample by sample restores a
    # smooth field. The planner cannot know that, so the check is switched off here.
    inv_method = {"mft": "mft", "exact": "exact"}.get(fwd.kind, "near_image_II")
    inv_b = plan_lct(np.linalg.inv(M5), LAM, fwd.out_lattice, lat, method=inv_method, check="off")
    assert rel_err_no_phase(apply_lct(inv_b, out), u) < 1e-10   # an offset-free column has no global phase


def test_round_trip_with_offsets_returns_the_weyl_constant_only():
    comps = [Deflector(z=0.5e-3, def_x=3e-6, def_y=-2e-6), Lens(z=U_, focal_length=F_, x0=6e-8, y0=-4e-8),
             Plane(z=U_ + 0.7 * V_IMG)]
    M5 = np.asarray(column_matrix(comps, source_z=0.0, voltage=VOLTAGE))
    lat = Lattice.regular((N_, N_), P_)
    cin = chief_from_beam(chief(x=1e-8, y=1e-8, dx=2e-6, dy=1e-6, wx=1.0e-7, wy=0.85e-7))
    u = gaussian_field(lat.positions(), cin[0], cin[1], cin[2], cin[3], LAM, cin[4])
    fwd = plan_lct(M5, LAM, lat, check="off")
    back = apply_lct(fwd.inverse(), apply_lct(fwd, u))
    ratio = back / u
    m = jnp.abs(u) > 1e-6 * jnp.abs(u).max()
    assert float(jnp.std(jnp.angle(ratio[m] / ratio[m][0]))) < 1e-9   # constant phase, nothing else
    assert float(jnp.max(jnp.abs(jnp.abs(ratio[m]) - 1))) < 1e-9


@pytest.mark.parametrize("label, v", [("mft", 1.02 * F_), ("near image", 1.01 * V_IMG), ("exact", V_IMG),
                                      ("general", 0.6 * V_IMG)])
def test_flux_conservation(label, v):
    comps = [_em_lens(U_), Plane(z=U_ + v)]
    M5 = np.asarray(column_matrix(comps, source_z=0.0, voltage=VOLTAGE))
    lat = Lattice.regular((N_, N_), P_)
    cin = chief_from_beam(chief(**B_ASYM))
    u = gaussian_field(lat.positions(), cin[0], cin[1], cin[2], cin[3], LAM, cin[4])
    sup, band = measure_support(u, tol=1e-22)
    plan = plan_lct(M5, LAM, lat, support=sup, band=band, check="raise")
    out = apply_lct(plan, u)
    flux_in = float(jnp.sum(jnp.abs(u) ** 2) * lat.cell_area)
    flux_out = float(jnp.sum(jnp.abs(out) ** 2) * plan.out_lattice.cell_area)
    assert abs(flux_out / flux_in - 1) < 1e-12


# ------------------------------------------------------------------------------------------------
# |B| sweep across the regime boundary
# ------------------------------------------------------------------------------------------------
def test_free_space_sweep_over_six_decades_of_B():
    N, p = 256, 10e-9
    ell = N * p * p / LAM
    lat = Lattice.regular((N, N), p)
    Q0 = gaussian_Q(10 * p, 9 * p, LAM)                      # sigma_x ~ 7 samples: narrow band, well inside the window
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), Q0, LAM)
    sup, band = measure_support(u, tol=1e-22)
    worst = 0.0
    kinds = set()
    for d_over_ell in np.logspace(-3, 3, 25):
        d = d_over_ell * ell
        Qo, amp = complex_q_chain(Q0, [("drift", d)])
        plan = plan_lct(_drift(d), LAM, lat, support=sup, band=band, check="raise")
        ref = gaussian_field(plan.out_lattice.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)
        worst = max(worst, rel_err(apply_lct(plan, u), ref))
        kinds.add(plan.kind)
    assert worst < 1e-9, worst
    assert kinds == {"near_image", "mft"}     # the sweep does cross the regime boundary


def test_branches_agree_where_both_are_valid_and_the_switch_is_continuous():
    N, p = 256, 10e-9
    ell = N * p * p / LAM
    lat = Lattice.regular((N, N), p)
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), gaussian_Q(10 * p, 9 * p, LAM), LAM)
    diffs = []
    for F in np.linspace(0.85, 1.6, 12):
        M5 = _drift(F * ell)
        ni = plan_lct(M5, LAM, lat, method="near_image", check="off")
        mft = plan_lct(M5, LAM, lat, ni.out_lattice, method="mft", check="off")
        diffs.append(rel_err(apply_lct(mft, u), apply_lct(ni, u)))
    assert max(diffs) < 1e-9, diffs


def test_lens_sweep_through_image_plane_including_B_equal_zero():
    """|B| from 1e-8 to 1e-1 of the Fresnel length on both sides of an image plane, and B = 0 itself."""
    Q0 = gaussian_Q(1e-7, 1e-7, LAM)
    lat = Lattice.regular((N_, N_), P_)
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), Q0, LAM)
    sup, band = measure_support(u, tol=1e-22)
    worst = 0.0
    for sgn in (-1, 1):
        for rel in [0.0] + list(np.logspace(-8, -1, 8)):
            v = V_IMG * (1 + sgn * rel)
            M5 = np.asarray(column_matrix([Lens(z=U_, focal_length=F_), Plane(z=U_ + v)], source_z=0.0,
                                          voltage=VOLTAGE))
            plan = plan_lct(M5, LAM, lat, support=sup, band=band, check="raise")
            Qo, amp = complex_q_chain(Q0, [("drift", U_), ("lens", F_), ("drift", v)])
            ref = gaussian_field(plan.out_lattice.positions(), amp, np.zeros(2), np.zeros(2), Qo, LAM)
            worst = max(worst, rel_err_no_phase(apply_lct(plan, u), ref))
    assert worst < 1e-9


def test_output_lattice_of_an_adapted_mft_can_be_requested_explicitly():
    """A requested output lattice is honoured by the MFT when the cross term is diagonal."""
    v = 1.02 * F_
    M5 = np.asarray(column_matrix([Lens(z=U_, focal_length=F_), Plane(z=U_ + v)], source_z=0.0, voltage=VOLTAGE))
    lat = Lattice.regular((N_, N_), P_)
    out = adapted_output_lattice(M5[:2, 2:4], LAM, lat, (0.8 / N_, 0.8 / N_), (N_, N_))
    plan = plan_lct(M5, LAM, lat, out, method="mft", check="off")
    assert jnp.allclose(plan.out_lattice.pitch, out.pitch)
    b0 = chief(wx=1e-7, wy=1e-7)
    cin = chief_from_beam(b0)
    u = gaussian_field(lat.positions(), cin[0], cin[1], cin[2], cin[3], LAM, cin[4])
    amp, rc, d, Qo, po = collins_gaussian(M5, LAM, cin[0], cin[1], cin[2], cin[3], cin[4])
    assert rel_err(apply_lct(plan, u), gaussian_field(out.positions(), amp, rc, d, Qo, LAM, po)) < 1e-9
