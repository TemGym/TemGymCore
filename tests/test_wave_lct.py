"""Lattice-adapted Collins engine: algebra, branches, the Maslov lift and autodiff (CPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from temgym_core.wave import (
    Lattice,
    LatticeError,
    adapted_input_lattice,
    adapted_output_lattice,
    apply_lct,
    collins_gaussian,
    gaussian_field,
    plan_lct,
)
from wave_validation import (
    drift4,
    lens4,
    lin4,
    matrix_from_factors,
    random_symplectic,
    rel_err,
    rot2,
)

LAM = 1.0  # dimensionless units: pitch ~ 0.2, distances ~ 1


def _sym(rng, s):
    a = rng.normal(size=(2, 2)) * s
    return 0.5 * (a + a.T)


def _blocks(M):
    return M[:2, :2], M[:2, 2:], M[2:, :2], M[2:, 2:]


def _Q(im_x=0.02 + 1.2j, im_y=0.03 + 1.8j, cross=0.0005j):
    Q = np.array([[im_x, cross], [cross, im_y]])
    return 0.5 * (Q + Q.T)


# ------------------------------------------------------------------------------------------------
# algebra
# ------------------------------------------------------------------------------------------------
def test_factorisations_of_a_symplectic_matrix():
    rng = np.random.default_rng(0)
    M = random_symplectic(rng)
    A, B, C, D = _blocks(M)
    J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    assert np.abs(M.T @ J @ M - J).max() < 1e-13
    S, T = np.linalg.solve(A, B), C @ np.linalg.inv(A)
    assert np.abs(S - S.T).max() < 1e-13 and np.abs(T - T.T).max() < 1e-13
    assert np.abs(lens4(T) @ lin4(A) @ drift4(S) - M).max() < 1e-12            # drift first
    Dinv = np.linalg.inv(D)
    S2, T2 = B @ Dinv, Dinv @ C
    assert np.abs(drift4(S2) @ lin4(Dinv.T) @ lens4(T2) - M).max() < 1e-12     # drift last
    Bi = np.linalg.inv(B)
    FT = np.block([[np.zeros((2, 2)), B], [-Bi.T, np.zeros((2, 2))]])
    assert np.abs(lens4(D @ Bi) @ FT @ lens4(Bi @ A) - M).max() < 1e-12       # Collins kernel form


@pytest.mark.parametrize("seed", range(4))
def test_adapted_lattices_make_the_cross_term_diagonal(seed):
    rng = np.random.default_rng(seed)
    M = random_symplectic(rng)
    B = M[:2, 2:]
    P_out = rng.normal(size=(2, 2)) * 0.2 + np.diag([0.3, 0.25])     # sheared, rotated, anisotropic
    out = Lattice(pitch=jnp.asarray(P_out), centre=jnp.zeros(2), shape=(16, 24))
    gamma = (0.7 / 24, 1.3 / 16)
    inn = adapted_input_lattice(B, LAM, out, gamma, (16, 24))
    Gam = np.asarray(out.pitch).T @ np.linalg.inv(B).T @ np.asarray(inn.pitch) / LAM
    assert np.abs(Gam - np.diag(gamma)).max() < 1e-13
    out2 = adapted_output_lattice(B, LAM, inn, gamma, (16, 24))
    assert np.abs(np.asarray(out2.pitch) - P_out).max() < 1e-12


def test_unadapted_lattices_raise():
    rng = np.random.default_rng(1)
    M = matrix_from_factors(random_symplectic(rng))
    lat = Lattice.regular((32, 32), 0.2)
    with pytest.raises(LatticeError, match="adapted"):
        plan_lct(M, LAM, lat, lat, method="mft")


# ------------------------------------------------------------------------------------------------
# MFT on a general Sp(4) map (rotation + astigmatism + shear): lattice bookkeeping, no resampling
# ------------------------------------------------------------------------------------------------
def _general_case():
    rng = np.random.default_rng(7)
    M4 = (drift4(np.diag([2.0, 3.0])) @ lens4(_sym(rng, 0.2)) @ lin4(rot2(0.4))
          @ drift4(_sym(rng, 3.0)) @ lens4(_sym(rng, 0.2)) @ lin4(rot2(1.1) @ np.diag([1.2, 0.8]))
          @ drift4(_sym(rng, 2.0)))
    return matrix_from_factors(M4)


@pytest.mark.parametrize("gamma_scale, n_in, n_out", [
    ((1.0, 1.0), (192, 192), (192, 192)),       # exact DFT on both axes (FFT path)
    ((0.8, 0.8), (192, 192), (192, 192)),       # zoom, matrix path
    ((0.8, 0.8), (192, 200), (176, 208)),       # rectangular and different in/out shapes
])
def test_mft_general_sp4_matches_analytic_gaussian(gamma_scale, n_in, n_out):
    M5 = _general_case()
    B = M5[:2, 2:4]
    out = Lattice.regular(n_out, (0.25, 0.3))
    gamma = (gamma_scale[0] / n_out[1], gamma_scale[1] / n_out[0])
    inn = adapted_input_lattice(B, LAM, out, gamma, n_in)
    Q = _Q(0.02 + 2.4j, 0.03 + 3.4j)
    rc, d, a = np.array([0.3, -0.2]), np.array([0.01, -0.02]), 0.7 * np.exp(0.3j)
    u = gaussian_field(inn.positions(), a, rc, d, Q, LAM)
    plan, rep = plan_lct(M5, LAM, inn, out, method="mft", field=u, return_report=True, check="report")
    amp, rco, do, Qo, po = collins_gaussian(M5, LAM, a, rc, d, Q)
    ref = gaussian_field(out.positions(), amp, rco, do, Qo, LAM, po)
    err = rel_err(apply_lct(plan, u), ref)      # phase included: the engine lift is exact for Gaussians
    assert err < 1e-9, (err, rep.margins)
    assert plan.mode_x == (1 if gamma_scale == (1.0, 1.0) else 0)


def test_mft_offset_column_and_offset_modes():
    """Tilted/offset beam through the affine column: 'after' and 'before' differ by a constant phase."""
    rng = np.random.default_rng(21)
    M4 = drift4(np.diag([2.0, 3.0])) @ lens4(_sym(rng, 0.2)) @ lin4(rot2(0.4)) @ drift4(_sym(rng, 3.0))
    M5 = matrix_from_factors(M4, offset=[0.4, -0.3, 0.02, 0.05])
    N = 160
    Q = _Q(0.02 + 2.0j, 0.03 + 2.6j)
    rc, d, a = np.array([0.3, -0.2]), np.array([0.01, -0.02]), 0.7 * np.exp(0.3j)
    for mode in ("after", "before"):
        amp, rco, do, Qo, po = collins_gaussian(M5, LAM, a, rc, d, Q, offset_mode=mode)
        out = Lattice.regular((N, N), (0.25, 0.3), centre=tuple(np.asarray(rco)))
        B = M5[:2, 2:4]
        inn = adapted_input_lattice(B, LAM, out, (1 / N, 1 / N), (N, N), in_centre=tuple(rc))
        u = gaussian_field(inn.positions(), a, rc, d, Q, LAM)
        plan = plan_lct(M5, LAM, inn, out, method="mft", offset_mode=mode, check="off")
        got = apply_lct(plan, u)
        ref = gaussian_field(out.positions(), amp, rco, do, Qo, LAM, po)
        assert rel_err(got, ref) < 1e-8, mode


# ------------------------------------------------------------------------------------------------
# near-image branch: both factorisations, every signature of S, det A of both signs, offsets, B = 0
# ------------------------------------------------------------------------------------------------
def _ni_matrix(kind):
    A0 = rot2(0.5) @ np.diag([1.3, 0.9])
    S0 = np.array([[0.05, 0.02], [0.02, 0.08]])
    T0 = np.array([[-0.3, 0.1], [0.1, -0.2]])
    if kind == "S>0":
        return matrix_from_factors(lens4(T0), lin4(A0), drift4(S0))
    if kind == "S<0":
        return matrix_from_factors(lens4(T0), lin4(A0), drift4(-S0))
    if kind == "S indefinite":
        return matrix_from_factors(lens4(T0), lin4(A0), drift4(np.diag([0.05, -0.04])))
    if kind == "detA<0":
        return matrix_from_factors(lens4(T0), lin4(A0 @ np.diag([1, -1])), drift4(S0))
    if kind == "offset":
        return matrix_from_factors(lens4(T0), lin4(A0), drift4(S0), offset=[0.05, -0.03, 0.002, 0.001])
    if kind == "B=0":
        return matrix_from_factors(lens4(T0), lin4(A0))
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["S>0", "S<0", "S indefinite", "detA<0", "offset", "B=0"])
@pytest.mark.parametrize("method", ["near_image", "near_image_II", "exact"])
def test_near_image_matches_analytic_gaussian(kind, method):
    M5 = _ni_matrix(kind)
    if method == "exact" and kind != "B=0":
        pytest.skip("exact needs B = 0")
    N = 96
    lat = Lattice.regular((N, N), (0.15, 0.17), rotation=np.degrees(0.2))
    Q = _Q(0.02 + 0.3j, 0.03 + 0.4j)
    rc, d, a = np.array([0.3, -0.2]), np.array([0.01, -0.02]), 0.7 * np.exp(0.3j)
    u = gaussian_field(lat.positions(), a, rc, d, Q, LAM)
    plan = plan_lct(M5, LAM, lat, method=method, check="off")
    amp, rco, do, Qo, po = collins_gaussian(M5, LAM, a, rc, d, Q)
    ref = gaussian_field(plan.out_lattice.positions(), amp, rco, do, Qo, LAM, po)
    assert rel_err(apply_lct(plan, u), ref) < 1e-12


def test_exact_branch_is_a_pure_relabelling_of_the_lattice():
    """B = 0: the output samples ARE the input samples (scaled and rotated lattice, chirp only)."""
    M5 = _ni_matrix("B=0")
    lat = Lattice.regular((40, 48), (0.15, 0.17))
    rng = np.random.default_rng(3)
    u = jnp.asarray(rng.normal(size=(40, 48)) + 1j * rng.normal(size=(40, 48)))
    plan = plan_lct(M5, LAM, lat)          # auto -> exact
    assert plan.kind == "exact"
    out = apply_lct(plan, u)
    scale = jnp.sqrt(jnp.abs(jnp.linalg.det(jnp.asarray(M5[:2, :2]))))
    assert jnp.allclose(jnp.abs(out) * scale, jnp.abs(u), atol=1e-14)
    A = jnp.asarray(M5[:2, :2])
    assert jnp.allclose(plan.out_lattice.pitch, A @ lat.pitch)


def test_mft_and_near_image_are_the_same_operator_on_their_overlap():
    """F in [0.9, 1.2] is inside both validity regions (narrow-band field), forward and backward
    drifts included (Maslov signature of S = A^-1 B negative)."""
    N = 128
    p = 0.15
    for sgn, F in ((+1, 1.0), (-1, 1.0), (+1, 1.2), (-1, 0.9)):
        s = sgn * F * N * p * p / LAM
        A0 = rot2(0.6) * 1.25
        T0 = np.array([[-0.05, 0.02], [0.02, -0.03]])
        M5 = matrix_from_factors(lens4(T0), lin4(A0), drift4(s * np.eye(2)))
        lat = Lattice.regular((N, N), p)
        Q = _Q(0.02 + 0.3j, 0.02 + 0.4j)
        u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), Q, LAM)
        ni = plan_lct(M5, LAM, lat, method="near_image", check="off")
        mft = plan_lct(M5, LAM, lat, ni.out_lattice, method="mft", check="off")
        e = rel_err(apply_lct(mft, u), apply_lct(ni, u))     # no phase removal: identical lift
        assert e < 1e-9, (sgn, F, e)


# ------------------------------------------------------------------------------------------------
# the engine lift is a genuine (continuous) lift: c(Q2)/c(Q1) follows the continuous sqrt det
# ------------------------------------------------------------------------------------------------
def _continuous_sqrt_ratio(A, B, Q1, Q2, steps=400):
    t = np.linspace(0, 1, steps + 1)
    dets = np.array([np.linalg.det(A + B @ ((1 - s) * Q1 + s * Q2)) for s in t])
    phase = np.unwrap(np.angle(dets))
    return np.sqrt(abs(dets[-1] / dets[0])) * np.exp(0.5j * (phase[-1] - phase[0]))


@pytest.mark.parametrize("seed", range(6))
def test_engine_amplitude_is_a_continuous_lift(seed):
    rng = np.random.default_rng(100 + seed)
    M5 = matrix_from_factors(random_symplectic(rng, drift_scale=0.7, lens_scale=0.4))
    A, B = M5[:2, :2], M5[:2, 2:4]
    Q1, Q2 = _Q(0.3 + 0.9j, 0.2 + 0.4j), _Q(-0.4 + 0.2j, 0.5 + 1.3j, cross=0.05j)
    c1 = collins_gaussian(M5, LAM, 1.0, np.zeros(2), np.zeros(2), Q1)[0]
    c2 = collins_gaussian(M5, LAM, 1.0, np.zeros(2), np.zeros(2), Q2)[0]
    ratio_engine = c2 / c1
    ratio_cont = 1.0 / _continuous_sqrt_ratio(A, B, Q1, Q2)      # amplitude ~ det(A+BQ)^-1/2
    assert abs(ratio_engine - ratio_cont) < 1e-6 * abs(ratio_cont), (ratio_engine, ratio_cont)


# ------------------------------------------------------------------------------------------------
# autodiff, jit, batching
# ------------------------------------------------------------------------------------------------
def test_batched_jit_and_field_gradient():
    M5 = _ni_matrix("S>0")
    lat = Lattice.regular((48, 48), (0.15, 0.17))
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.normal(size=(3, 48, 48)) + 1j * rng.normal(size=(3, 48, 48)))
    plan = plan_lct(M5, LAM, lat, method="near_image", check="off")
    out = jax.jit(apply_lct)(plan, u)
    assert out.shape == u.shape
    for i in range(3):
        assert jnp.allclose(out[i], apply_lct(plan, u[i]), atol=1e-13)
    w = jnp.asarray(rng.normal(size=(48, 48)))

    def loss(uu):
        return jnp.sum(w * jnp.abs(apply_lct(plan, uu)) ** 2)

    check_grads(loss, (u[0],), order=1, modes=("rev",), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("method", ["near_image", "mft"])
def test_matrix_gradient_matches_finite_differences(method):
    lat = Lattice.regular((64, 64), (0.15, 0.15))
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64)))
    u = u * jnp.exp(-(lat.positions() ** 2).sum(-1) / (2 * 0.8**2))
    w = jnp.asarray(rng.normal(size=(64, 64)))
    d0 = 0.3 * 64 * 0.15**2 / LAM if method == "near_image" else 1.5 * 64 * 0.15**2 / LAM
    out_lat = None
    if method == "mft":
        # a fixed *output* lattice for the whole family (input lattice adapted at the base point)
        out_lat = Lattice.regular((64, 64), 0.15)
        B0 = jnp.eye(2) * d0
        lat = adapted_input_lattice(B0, LAM, out_lat, (1 / 64, 1 / 64), (64, 64))

    def loss(f):
        # thin lens of focal length f, then drift d0 (analytic matrix, differentiable in f)
        T = -1.0 / f
        M4 = jnp.array([[1 + d0 * T, 0, d0, 0], [0, 1 + d0 * T, 0, d0], [T, 0, 1, 0], [0, T, 0, 1.0]])
        M5 = jnp.eye(5).at[:4, :4].set(M4)
        plan = plan_lct(M5, LAM, lat, out_lat, method=method, check="off")
        return jnp.sum(w * jnp.abs(apply_lct(plan, u)) ** 2)

    f0 = 40.0
    g = jax.grad(loss)(f0)
    h = 1e-4 * f0
    fd = (loss(f0 + h) - loss(f0 - h)) / (2 * h)
    assert abs(g - fd) < 1e-6 * abs(fd), (float(g), float(fd))


# ------------------------------------------------------------------------------------------------
# Grid interoperability (axis-order trap) and the literal 1/(i sqrt(det B)) prefactor
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rotation, flip_y, pixel_size, shape, centre", [
    (0.0, False, (0.3, 0.5), (12, 20), (0.0, 0.0)),
    (23.0, False, (0.3, 0.5), (12, 20), (1.0, -2.0)),
    (23.0, True, (0.4, 0.4), (9, 14), (0.5, 0.25)),
    (-77.0, True, (0.2, 0.7), (16, 10), (-3.0, 2.0)),
])
def test_lattice_from_grid_reproduces_grid_coordinates(rotation, flip_y, pixel_size, shape, centre):
    from temgym_core.components import Detector

    det = Detector(z=0.0, pixel_size=pixel_size, shape=shape, rotation=rotation, centre=centre,
                   flip_y=flip_y)
    lat = Lattice.from_grid(det)
    x, y = lat.coords()
    coords = np.asarray(det.coords).reshape(shape[0], shape[1], 2)     # (x_m, y_m) per pixel, row-major
    assert np.allclose(np.asarray(x), coords[..., 0], atol=1e-12)
    assert np.allclose(np.asarray(y), coords[..., 1], atol=1e-12)


def test_regular_lattice_rotation_is_counter_clockwise_degrees():
    """``Lattice.regular`` is right handed (y up with iy); unlike ``Grid`` rows do not run towards -y."""
    lat = Lattice.regular((4, 6), (2.0, 3.0), rotation=90.0)
    assert np.allclose(np.asarray(lat.pitch), [[0.0, -3.0], [2.0, 0.0]], atol=1e-14)
    assert np.allclose(np.asarray(Lattice.regular((4, 6), 2.0).pitch), 2.0 * np.eye(2))


def test_literal_one_over_i_sqrt_det_b_prefactor_flips_sign_against_the_drift_branch(monkeypatch):
    """1/(i lambda sqrt(det B)) read literally (-i for det B > 0, -1 for det B < 0) is a lift only up
    to a sign that depends on sig(B^-1 A): with it the MFT branch disagrees with the drift branch by
    -1 for backward propagation, so continuity and the inverse fail; the Maslov rule in
    ``lift_phase`` does not."""
    import temgym_core.wave.lct as lct

    N, p = 128, 0.15
    lat = Lattice.regular((N, N), p)
    Q = _Q(0.02 + 0.3j, 0.02 + 0.4j)
    u = gaussian_field(lat.positions(), 1.0, np.zeros(2), np.zeros(2), Q, LAM)

    def literal(A, B):
        detB = jnp.linalg.det(B)
        return -1j * jnp.sqrt(jnp.abs(detB)) / jnp.sqrt(detB + 0j)   # phase of 1/(i sqrt(det B)), principal branch

    for sgn in (+1, -1):
        s = sgn * 1.0 * N * p * p / LAM
        M5 = matrix_from_factors(lens4(np.array([[-0.05, 0.02], [0.02, -0.03]])), lin4(rot2(0.6) * 1.25),
                                 drift4(s * np.eye(2)))
        ni = plan_lct(M5, LAM, lat, method="near_image", check="off")
        good = plan_lct(M5, LAM, lat, ni.out_lattice, method="mft", check="off")
        with monkeypatch.context() as mp:
            mp.setattr(lct, "lift_phase", literal)
            bad = plan_lct(M5, LAM, lat, ni.out_lattice, method="mft", check="off")
        e_good = rel_err(apply_lct(good, u), apply_lct(ni, u))
        e_literal = rel_err(apply_lct(bad, u), apply_lct(ni, u))
        assert e_good < 1e-9
        if sgn > 0:
            assert e_literal < 1e-9          # forward drift: the two prefactors coincide
        else:
            assert abs(e_literal - 2.0) < 1e-9   # backward: opposite sign, and the round trip would give -1


def test_gradient_with_respect_to_lattice_pitch_and_offset_column():
    """Everything upstream of the field is differentiable: pitch, wavelength, offset and the matrix."""
    rng = np.random.default_rng(4)
    M4 = np.asarray(random_symplectic(rng, drift_scale=0.8, lens_scale=0.1))
    N = 48
    Q = _Q(0.02 + 1.5j, 0.03 + 2.0j)
    w = jnp.asarray(rng.normal(size=(N, N)))

    def loss(theta):
        pitch_scale, lam, off = theta[0], theta[1], theta[2:6]
        M5 = jnp.eye(5).at[:4, :4].set(jnp.asarray(M4)).at[:4, 4].set(off)
        out = Lattice.regular((N, N), pitch_scale * 0.25)
        inn = adapted_input_lattice(M5[:2, 2:4], lam, out, (1 / N, 1 / N), (N, N))
        u = gaussian_field(inn.positions(), 1.0, jnp.zeros(2), jnp.zeros(2), Q, lam)
        plan = plan_lct(M5, lam, inn, out, method="mft", check="off")
        return jnp.sum(w * jnp.abs(apply_lct(plan, u)) ** 2)

    theta0 = jnp.array([1.0, 1.0, 0.05, -0.04, 0.002, 0.001])
    g = jax.grad(loss)(theta0)
    for i in range(6):
        h = 1e-6 * max(abs(float(theta0[i])), 1e-2)
        e = jnp.zeros(6).at[i].set(h)
        fd = (loss(theta0 + e) - loss(theta0 - e)) / (2 * h)
        fd_noise = 50 * 1.1e-16 * abs(float(loss(theta0))) / h          # the slope offset is a pure phase ramp: exact 0
        assert abs(float(g[i]) - float(fd)) < 1e-5 * abs(float(fd)) + fd_noise, (i, float(g[i]), float(fd))


@pytest.mark.parametrize("shape", [(65, 63), (64, 63), (63, 64), (129, 127)])
def test_odd_and_mixed_parity_shapes(shape):
    """The centred-index twist of the FFT path and the near-image lattice map for odd / mixed array sizes."""
    ny, nx = shape
    p = 0.15
    ell = max(shape) * p * p / LAM
    # mild map (F ~ 1.2-1.5, no rotation): a narrow-band Gaussian fits comfortably even at N ~ 64
    M5 = matrix_from_factors(lens4(np.array([[-0.03, 0.01], [0.01, -0.02]])), drift4(np.diag([1.3, 1.6]) * ell))
    lat = Lattice.regular(shape, p)
    Q = _Q(0.0 + 0.55j, 0.0 + 0.7j)
    rc, d, a = np.array([0.2, -0.1]), np.array([0.005, -0.01]), 0.7 * np.exp(0.3j)
    u = gaussian_field(lat.positions(), a, rc, d, Q, LAM)
    plan = plan_lct(M5, LAM, lat, method="mft", check="off")
    assert (plan.mode_x, plan.mode_y) == (1, 1)                      # DFT path with odd/even centred twists
    amp, rco, do, Qo, po = collins_gaussian(M5, LAM, a, rc, d, Q)
    ref = gaussian_field(plan.out_lattice.positions(), amp, rco, do, Qo, LAM, po)
    assert rel_err(apply_lct(plan, u), ref) < 1e-8
    # matrix path (zoom) on the same shape
    out = adapted_output_lattice(M5[:2, 2:4], LAM, lat, (0.9 / nx, 0.9 / ny), shape)
    pz = plan_lct(M5, LAM, lat, out, method="mft", check="off")
    assert (pz.mode_x, pz.mode_y) == (0, 0)
    assert rel_err(apply_lct(pz, u), gaussian_field(out.positions(), amp, rco, do, Qo, LAM, po)) < 1e-8
    # near-image branch on the same shape
    M5n = _ni_matrix("S>0")
    latn = Lattice.regular(shape, (0.15, 0.17), rotation=np.degrees(0.2))
    Qn = _Q(0.02 + 0.6j, 0.03 + 0.8j)
    un = gaussian_field(latn.positions(), a, rc, d, Qn, LAM)
    pn = plan_lct(M5n, LAM, latn, method="near_image", check="off")
    amp, rco, do, Qo, po = collins_gaussian(M5n, LAM, a, rc, d, Qn)
    assert rel_err(apply_lct(pn, un), gaussian_field(pn.out_lattice.positions(), amp, rco, do, Qo, LAM, po)) < 1e-9
