"""The sampling planner raises on aliasing geometry and its accepted cases are accurate."""

import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.wave import (
    Lattice,
    LatticeError,
    SamplingError,
    adapted_output_lattice,
    apply_lct,
    collins_gaussian,
    gaussian_field,
    measure_support,
    plan_lct,
)
from wave_validation import (
    brute_force_collins,
    drift4,
    gaussian_Q,
    lens4,
    lin4,
    matrix_from_factors,
    random_mft_case,
    random_symplectic,
    rel_err,
    rot2,
)

LAM = 1.0


def _drift_matrix(d):
    return matrix_from_factors(drift4(d * np.eye(2)))


def _gauss(lat, wx, wy, rc=(0.0, 0.0), d=(0.0, 0.0), Q=None):
    Q = gaussian_Q(wx, wy, LAM) if Q is None else Q
    return gaussian_field(lat.positions(), 1.0, np.asarray(rc), np.asarray(d), Q, LAM)


# ------------------------------------------------------------------------------------------------
# deliberately under-sampled geometry must raise (and really is inaccurate)
# ------------------------------------------------------------------------------------------------
N0, P0 = 128, 0.15
ELL = N0 * P0**2 / LAM          # Fresnel length of the window


def test_mft_in_the_near_image_regime_raises_and_is_really_aliased():
    lat = Lattice.regular((N0, N0), P0)
    u = _gauss(lat, 0.8, 0.9)
    M5 = _drift_matrix(0.05 * ELL)
    with pytest.raises(SamplingError, match="replica rule"):
        plan_lct(M5, LAM, lat, method="mft", field=u)
    plan, rep = plan_lct(M5, LAM, lat, method="mft", field=u, check="report", return_report=True)
    assert not rep.ok and rep.margins["x"] < 0
    amp, rc, d, Qo, po = collins_gaussian(M5, LAM, 1.0, np.zeros(2), np.zeros(2), gaussian_Q(0.8, 0.9, LAM))
    ref = gaussian_field(plan.out_lattice.positions(), amp, rc, d, Qo, LAM, po)
    assert rel_err(apply_lct(plan, u), ref) > 1e-2


def test_sampling_error_lists_the_failing_inequality_with_its_numbers():
    lat = Lattice.regular((N0, N0), P0)
    with pytest.raises(SamplingError) as info:
        plan_lct(_drift_matrix(0.05 * ELL), LAM, lat, method="mft", field=_gauss(lat, 0.8, 0.9))
    message = str(info.value)
    assert "c + |gamma| n'/2 <= 1" in message and "axis x" in message and "axis y" in message
    assert "intermediate plane" in message


def test_near_image_branch_wrap_raises_beyond_the_fresnel_length():
    lat = Lattice.regular((N0, N0), P0)
    u = _gauss(lat, 0.8, 0.9)
    with pytest.raises(SamplingError, match="wrap rule support \\+ kernel spread <= n/2"):
        plan_lct(_drift_matrix(3.0 * ELL), LAM, lat, method="near_image", field=u)


def test_undersampled_input_raises():
    """A field whose spectrum reaches Nyquist is aliased before any transform is applied."""
    lat = Lattice.regular((N0, N0), P0)
    u = _gauss(lat, 1.5 * P0, 1.5 * P0)
    with pytest.raises(SamplingError, match="Nyquist"):
        plan_lct(_drift_matrix(1.0 * ELL), LAM, lat, field=u)


def test_truncated_input_raises():
    lat = Lattice.regular((N0, N0), P0)
    u = _gauss(lat, 0.45 * N0 * P0, 0.45 * N0 * P0)          # field fills the window
    with pytest.raises(SamplingError, match="truncated"):
        plan_lct(_drift_matrix(1.0 * ELL), LAM, lat, field=u)


def test_auto_raises_when_no_branch_can_hold_the_field():
    """F ~ 0.7 with a field filling half the window: too far for the drift, too near for the MFT."""
    lat = Lattice.regular((N0, N0), P0)
    u = _gauss(lat, 0.25 * N0 * P0 / 3, 0.25 * N0 * P0 / 3)   # support ~ 0.5 window
    with pytest.raises(SamplingError, match="no branch"):
        plan_lct(_drift_matrix(0.62 * ELL), LAM, lat, field=u, support=(0.45 * N0, 0.45 * N0), band=(0.3, 0.3))


def test_check_off_skips_every_rule_for_an_explicit_method():
    lat = Lattice.regular((N0, N0), P0)
    u = _gauss(lat, 0.8, 0.9)
    plan, rep = plan_lct(_drift_matrix(0.05 * ELL), LAM, lat, method="mft", field=u, check="off",
                         return_report=True)
    assert rep is None and plan.kind == "mft"


def test_check_report_never_raises_and_check_off_auto_chooses_by_the_assumed_rules():
    lat = Lattice.regular((N0, N0), P0)
    u = _gauss(lat, 0.25 * N0 * P0 / 3, 0.25 * N0 * P0 / 3)
    kw = dict(support=(0.45 * N0, 0.45 * N0), band=(0.3, 0.3))
    plan, rep = plan_lct(_drift_matrix(0.62 * ELL), LAM, lat, field=u, check="report", return_report=True, **kw)
    assert not rep.ok and plan.kind == "mft"                 # nothing fits: falls back to the last candidate
    # off + auto: near-image is feasible for a small drift on the assumed content, so it is chosen
    assert plan_lct(_drift_matrix(0.05 * ELL), LAM, lat, check="off").kind == "near_image"
    # off + auto for a far span: the assumed rules choose the MFT rather than a hopeless near-image
    assert plan_lct(_drift_matrix(4.0 * ELL), LAM, lat, check="off").kind == "mft"


def test_unadapted_lattices_raise_lattice_error():
    lat = Lattice.regular((32, 32), 0.2)
    M = matrix_from_factors(random_symplectic(np.random.default_rng(3)))
    with pytest.raises(LatticeError):
        plan_lct(M, LAM, lat, lat, method="mft")


def test_invalid_options_are_rejected():
    lat = Lattice.regular((16, 16), 0.2)
    with pytest.raises(ValueError, match="check"):
        plan_lct(_drift_matrix(1.0), LAM, lat, check="maybe")
    with pytest.raises(ValueError, match="unknown method"):
        plan_lct(_drift_matrix(1.0), LAM, lat, method="fft")
    with pytest.raises(ValueError, match="exact needs B = 0|B = 0"):
        plan_lct(_drift_matrix(1.0), LAM, lat, method="exact")


# ------------------------------------------------------------------------------------------------
# the derived replica rule c + gamma n'/2 <= 1 is sharp
# ------------------------------------------------------------------------------------------------
def test_replica_rule_is_sharp_and_less_strict_than_half_the_period():
    """Sweep the output window at fixed input: the error crosses 1e-6 where the margin crosses zero."""
    M5 = matrix_from_factors(
        drift4(np.diag([2.0, 2.6])) @ lens4(np.array([[-0.25, 0.05], [0.05, -0.2]])) @ lin4(rot2(0.5))
        @ drift4(np.array([[3.0, 0.4], [0.4, 3.5]])))
    B = M5[:2, 2:4]
    N = 96
    in_lat = Lattice.regular((N, N), 0.11, rotation=np.degrees(0.3))
    u = _gauss(in_lat, 0.5, 0.42)
    amp, rc, d, Qo, po = collins_gaussian(M5, LAM, 1.0, np.zeros(2), np.zeros(2), gaussian_Q(0.5, 0.42, LAM))
    rows = []
    for g in np.linspace(0.3, 1.6, 27):
        out = adapted_output_lattice(B, LAM, in_lat, (g / N, g / N), (N, N), out_centre=tuple(np.asarray(rc)))
        plan, rep = plan_lct(M5, LAM, in_lat, out, method="mft", field=u, check="report", return_report=True)
        ref = gaussian_field(out.positions(), amp, rc, d, Qo, LAM, po)
        margin = min(rep.margins["x"], rep.margins["y"])
        rows.append((float(g), margin, rel_err(apply_lct(plan, u), ref), rep.ok))
    for g, m, e, ok in rows:
        if m >= 0.02:
            assert e < 1e-6, (g, m, e)
        if m <= -0.15:
            assert e > 1e-4, (g, m, e)
        assert ok == (m >= 0)
    # windows beyond 0.45 periods (the naive "window <= period / 2" rule would reject them) are exact
    # when the chirped field is compact: the exact rule c + gamma n'/2 <= 1 accepts them
    accurate_beyond_half = [g for g, m, e, ok in rows if ok and g / 2 > 0.45 and e < 1e-8]
    assert accurate_beyond_half, rows


# ------------------------------------------------------------------------------------------------
# random battery: accepted => matches 4x oversampled brute-force quadrature to 1e-6
# ------------------------------------------------------------------------------------------------
def test_brute_force_reference_matches_analytic_gaussian():
    rng = np.random.default_rng(5)
    M5, in_lat, out_lat, u, (rc, d, Q0) = random_mft_case(rng, 36)
    fine = Lattice(pitch=in_lat.pitch / 4, centre=in_lat.centre, shape=(4 * in_lat.shape[0], 4 * in_lat.shape[1]))
    bf = brute_force_collins(M5, LAM, lambda r: gaussian_field(r, 1.0, rc, d, Q0, LAM), fine.positions(),
                             float(in_lat.cell_area) / 16, out_lat.positions())
    amp, rco, do, Qo, po = collins_gaussian(M5, LAM, 1.0, rc, d, Q0)
    ref = gaussian_field(out_lat.positions(), amp, rco, do, Qo, LAM, po)
    assert rel_err(bf, ref) < 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_accepted_cases_meet_1e6_against_oversampled_brute_force(seed):
    rng = np.random.default_rng(seed)
    accepted, rejected = [], []
    for _ in range(40):
        M5, in_lat, out_lat, u, (rc, d, Q0) = random_mft_case(rng, 44)
        plan, rep = plan_lct(M5, LAM, in_lat, out_lat, method="mft", field=u, check="report", return_report=True)
        out = apply_lct(plan, u)
        fine = Lattice(pitch=in_lat.pitch / 4, centre=in_lat.centre,
                       shape=(4 * in_lat.shape[0], 4 * in_lat.shape[1]))
        bf = brute_force_collins(M5, LAM, lambda r: gaussian_field(r, 1.0, rc, d, Q0, LAM), fine.positions(),
                                 float(in_lat.cell_area) / 16, out_lat.positions())
        (accepted if rep.ok else rejected).append((rel_err(out, bf), rep.tail_energy))
    assert len(accepted) >= 8 and len(rejected) >= 8
    assert max(e for e, _ in accepted) < 1e-6, accepted
    # the diagnostic is informative: rejected cases carry (much) more spectral-tail energy on average
    assert np.mean([t for _, t in rejected]) > np.mean([t for _, t in accepted])


def _blobs_field(r, rc, width, phase_scale=1.0):
    """Three interfering, differently sized and tilted blobs with a cubic phase plate: smooth,
    band-limited, neither a Gaussian nor separable (a hard-edged aperture would need anti-aliasing
    to be sampled at all)."""
    d = jnp.asarray(r) - jnp.asarray(rc)
    x, y = d[..., 0] / width, d[..., 1] / width

    def blob(cx, cy, w, a, kx, ky):
        return a * jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / w**2) * jnp.exp(1j * (kx * x + ky * y))

    field = (blob(0.0, 0.0, 1.0, 1.0, 0.0, 0.0) + blob(0.9, -0.4, 0.75, 0.6, 1.5, -0.8)
             + blob(-0.7, 0.8, 0.6, 0.5j, -1.2, 1.6))
    return field * jnp.exp(1j * phase_scale * (0.35 * x**3 - 0.2 * x * y**2 + 0.15 * (x * x - y * y)))


@pytest.mark.slow
def test_non_gaussian_interfering_blobs_match_brute_force():
    """The planner and engine are not tuned to Gaussians: interfering blobs with a cubic phase plate."""
    rng = np.random.default_rng(21)
    errs = []
    for _ in range(18):
        M5, in_lat, out_lat, _, (rc, d, _Q) = random_mft_case(rng, 60, width_fraction=(1.0, 1.0),
                                                              pitch_range=(0.12, 0.12))
        px = np.sqrt(abs(np.linalg.det(np.asarray(in_lat.pitch))))
        width = 0.09 * min(in_lat.shape) * px                    # ~5 samples per unit width

        def fn(r, rc=rc, width=width):
            return _blobs_field(r, rc, width, phase_scale=0.25)

        u = fn(in_lat.positions())
        sup, band = measure_support(u, tol=1e-12)                # error ~ sqrt(tol) = 1e-6
        plan, rep = plan_lct(M5, LAM, in_lat, out_lat, method="mft", support=sup, band=band, field=u,
                             check="report", return_report=True)
        if not rep.ok:
            continue
        fine = Lattice(pitch=in_lat.pitch / 4, centre=in_lat.centre, shape=(4 * in_lat.shape[0], 4 * in_lat.shape[1]))
        bf = brute_force_collins(M5, LAM, fn, fine.positions(), float(in_lat.cell_area) / 16, out_lat.positions())
        errs.append(rel_err(apply_lct(plan, u), bf))
    assert len(errs) >= 3 and max(errs) < 1e-6, errs


def test_tail_energy_is_reported_and_grows_when_the_window_is_too_large():
    rng = np.random.default_rng(2)
    M5, in_lat, out_lat, u, _ = random_mft_case(rng, 40)
    B = M5[:2, 2:4]
    tails = []
    for g in (0.4, 0.8, 1.2, 1.6):
        out = adapted_output_lattice(B, LAM, in_lat, (g / 40, g / 40), (40, 40), out_centre=tuple(np.asarray(out_lat.centre)))
        _, rep = plan_lct(M5, LAM, in_lat, out, method="mft", field=u, check="report", return_report=True)
        assert rep.tail_energy is not None
        tails.append(rep.tail_energy)
    assert tails[0] <= tails[-1] and tails[-1] > 0
