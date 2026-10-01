"""Independent references and helpers for the sampled-wavefront (``temgym_core.wave``) tests.

Nothing here uses the engine's algebra: a sequential complex-q chain (drift, lens and stigmator,
axis aligned, continuous Gouy branch axis by axis), a random symplectic generator, brute-force direct
quadrature of the Collins kernel, and small error metrics.
"""

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.wave import Lattice, adapted_output_lattice, collins_gaussian, gaussian_field, lift_phase
from temgym_core.wave.lct import split_matrix


def rel_err(x, ref):
    """Relative L2 difference ``|x - ref| / |ref|``."""
    x, ref = jnp.asarray(x), jnp.asarray(ref)
    return float(jnp.linalg.norm(x - ref) / jnp.linalg.norm(ref))


def rel_err_no_phase(x, ref):
    """Relative L2 difference after removing the best global phase (least squares)."""
    x, ref = jnp.asarray(x), jnp.asarray(ref)
    inner = jnp.vdot(ref, x)
    ph = inner / jnp.abs(inner)
    return float(jnp.linalg.norm(x - ref * ph) / jnp.linalg.norm(ref))


def gaussian_Q(waist_x, waist_y, wavelength, rcurv_x=np.inf, rcurv_y=np.inf):
    """``Q = 1/q`` matrix of an axis-aligned Gaussian, as ``make_gaussian`` builds it."""
    qx = 1.0 / rcurv_x + 1j * wavelength / (np.pi * waist_x**2)
    qy = 1.0 / rcurv_y + 1j * wavelength / (np.pi * waist_y**2)
    return np.array([[qx, 0.0], [0.0, qy]], dtype=complex)


def complex_q_chain(Q0, steps):
    """Sequential complex-q law per axis for an axis-aligned Gaussian.

    ``steps``: ``("drift", d)``, ``("lens", f)`` or ``("stig", fx, fy)``. Returns ``(Q, amp)`` where
    ``amp`` is the continuous (physical) amplitude factor; each axis keeps its own principal
    ``(1 + d Q_j)^-1/2`` (never crosses the cut for ``Im Q > 0``).
    """
    q = np.array([Q0[0, 0], Q0[1, 1]], dtype=complex)
    amp = 1.0 + 0j
    for step in steps:
        if step[0] == "drift":
            d = step[1]
            amp = amp / np.sqrt(1 + d * q[0]) / np.sqrt(1 + d * q[1])
            q = q / (1 + d * q)
        elif step[0] == "lens":
            q = q - 1.0 / step[1]
        elif step[0] == "stig":
            q = q - np.array([1.0 / step[1], 1.0 / step[2]])
        else:
            raise ValueError(step)
    return np.diag(q), amp


def rot2(a):
    """2x2 rotation matrix of angle ``a`` [rad]."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def drift4(S):
    """4x4 drift with symmetric 2x2 block ``S``."""
    M = np.eye(4)
    M[:2, 2:] = S
    return M


def lens4(T):
    """4x4 thin lens with symmetric 2x2 power block ``T``."""
    M = np.eye(4)
    M[2:, :2] = T
    return M


def lin4(A):
    """4x4 linear lattice map ``(A, A^-T)``."""
    M = np.zeros((4, 4))
    M[:2, :2] = A
    M[2:, 2:] = np.linalg.inv(A).T
    return M


def _sym(rng, scale):
    a = rng.normal(size=(2, 2)) * scale
    return 0.5 * (a + a.T)


def random_symplectic(rng, drift_scale=1.0, lens_scale=1.0, rot_deg=180.0):
    """``D L Lin D L Lin D`` with random symmetric drifts and lenses, rotations and anisotropic scalings."""
    def lin():
        return lin4(rot2(rng.uniform(0, np.radians(rot_deg))) @ np.diag(np.exp(rng.normal(size=2) * 0.25)))
    M = np.eye(4)
    for _ in range(3):
        M = drift4(_sym(rng, drift_scale) + 2.0 * drift_scale * np.eye(2)) @ M
        M = lens4(_sym(rng, lens_scale)) @ M
        M = lin() @ M
    return drift4(_sym(rng, drift_scale) + 2.0 * drift_scale * np.eye(2)) @ M


def matrix_from_factors(*mats, offset=None):
    """5x5 ray matrix from 4x4 factors applied right to left (``M = f1 @ f2 @ ...``)."""
    M4 = np.eye(4)
    for m in mats:
        M4 = M4 @ m
    M5 = np.eye(5)
    M5[:4, :4] = M4
    if offset is not None:
        M5[:4, 4] = offset
    return M5


def brute_force_collins(matrix, wavelength, u_fn, fine_positions, cell_area, out_positions, chunk=128):
    """Direct quadrature of the Collins kernel on a fine input lattice.

    ``u_fn(r[..., 2])`` evaluates the *continuous* input field at ``fine_positions`` (any lattice,
    for example ``Lattice(P/4, ...)`` for a 4x oversampled reference); ``cell_area`` is the fine cell
    area. The prefactor uses the engine's lift, so the result is directly comparable to
    ``apply_lct(..., phase0=0)``. ``matrix`` must have B invertible. O(n_out * n_fine): use for
    n <= 64 only.
    """
    A, B, C, D, a, b = [np.asarray(x) for x in split_matrix(matrix)]
    lam = wavelength
    k = 2 * np.pi / lam
    Bi = np.linalg.inv(B)
    G = 0.5 * (Bi @ A + (Bi @ A).T)
    H = 0.5 * (D @ Bi + (D @ Bi).T)
    r = jnp.asarray(np.asarray(fine_positions).reshape(-1, 2))
    u = jnp.asarray(u_fn(r))
    quad_in = jnp.einsum("ni,ij,nj->n", r, jnp.asarray(G), r)
    out_pos = np.asarray(out_positions).reshape(-1, 2)
    n_out = out_pos.shape[0]
    pad = (-n_out) % chunk
    out_pad = jnp.asarray(np.concatenate([out_pos, np.zeros((pad, 2))]).reshape(-1, chunk, 2))
    pref = complex(lift_phase(jnp.asarray(A), jnp.asarray(B))) / (lam * np.sqrt(abs(np.linalg.det(B)))) * cell_area
    a_j, b_j, Bi_j, H_j = map(jnp.asarray, (a, b, Bi, H))

    @jax.jit
    def block(op):
        rp = op - a_j
        cross = (rp @ Bi_j.T) @ r.T
        quad_out = jnp.einsum("ni,ij,nj->n", rp, H_j, rp)
        phase = 0.5 * quad_in[None, :] - cross + 0.5 * quad_out[:, None]
        return pref * jnp.exp(1j * k * (op @ b_j)) * jnp.sum(u[None, :] * jnp.exp(1j * k * phase), axis=1)

    res = jax.lax.map(block, out_pad).reshape(-1)[:n_out]
    return np.asarray(res).reshape(np.asarray(out_positions).shape[:-1])


def random_mft_case(rng, N, wavelength=1.0, width_fraction=(0.6, 1.0), window_over_waist=9.0,
                    pitch_range=(0.08, 0.2)):
    """Random general-Sp(4) MFT problem for planner validation.

    Returns ``(M5, in_lattice, out_lattice, u, (centre, slopes, Q))``: a random symplectic map
    (optionally with an offset column), an input lattice tailored to a random rotated astigmatic
    Gaussian (so the input is contained), and the *adapted* output lattice for a random cross-term
    ``gamma``. Whether the output window holds the output field is left to chance: that is what the
    planner has to decide.
    """
    M4 = random_symplectic(rng, drift_scale=rng.uniform(0.5, 2.5), lens_scale=rng.uniform(0.03, 0.3))
    M5 = matrix_from_factors(M4)
    if rng.random() < 0.5:
        M5[:4, 4] = rng.normal(size=4) * np.array([0.2, 0.2, 0.01, 0.01])
    B = M4[:2, 2:]
    Nin = (int(rng.integers(N - 8, N + 1)), int(rng.integers(N - 8, N + 1)))
    Nout = (int(rng.integers(N - 8, N + 1)), int(rng.integers(N - 8, N + 1)))
    pin = rng.uniform(*pitch_range, 2)
    ang = rng.uniform(0, np.pi)
    rc = rng.normal(size=2) * 0.1
    d = rng.normal(size=2) * 0.02
    in_lat = Lattice(pitch=jnp.asarray(rot2(ang) @ np.diag(pin)), centre=jnp.asarray(rc), shape=Nin)
    window = np.array([Nin[1] * pin[0], Nin[0] * pin[1]])
    fac = rng.uniform(*width_fraction, 2)
    w = window * fac / window_over_waist
    Q0 = rot2(ang) @ gaussian_Q(w[0], w[1], wavelength, rcurv_x=rng.choice([np.inf, 40.0, -25.0]),
                                rcurv_y=np.inf) @ rot2(ang).T
    Q0 = 0.5 * (Q0 + Q0.T)
    _, rc_o, _, _, _ = collins_gaussian(M5, wavelength, 1.0, rc, d, Q0)
    g = rng.uniform(0.5, 1.3, 2)
    gamma = (g[0] / Nout[1], g[1] / Nout[0])
    out_lat = adapted_output_lattice(B, wavelength, in_lat, gamma, Nout, out_centre=tuple(np.asarray(rc_o)))
    u = gaussian_field(in_lat.positions(), 1.0, rc, d, Q0, wavelength)
    return M5, in_lat, out_lat, u, (rc, d, Q0)
