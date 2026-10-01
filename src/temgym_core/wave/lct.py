"""Lattice-adapted Collins (linear canonical) transform of a sampled 2D field.

The operator implemented is the paraxial wave propagator of a symplectic ray map

.. code-block:: none

    [r; theta] -> M [r; theta] + [a; b],   M = [[A, B], [C, D]]   (2x2 blocks, (x, y) order)

that is ``U'(r') = (e^{i chi} / (lambda sqrt|det B|)) int U(r) exp{ik/2 [r^T B^-1 A r - 2 r'^T B^-T r
+ r'^T D B^-1 r']} d^2r`` with the Maslov phase ``e^{i chi}`` given by :func:`lift_phase` (not the
literal ``1/(i sqrt(det B))``, which has the wrong sign whenever ``B^-1 A`` is negative definite),
followed by the Heisenberg-Weyl offset ``(a, b)``: ``U_full u (r') = exp(i k b.r') (U_M u)(r' - a)``.
Three implementations of the same operator:

- ``mft``: for a well conditioned ``B``. With the input lattice adapted to the output lattice the
  cross term is ``exp(-2 pi i mu'^T diag(gamma) mu)``: two matrix products, or FFTs when
  ``gamma = 1/n``. Pitch and window are independent per axis; shear and rotation are lattice
  bookkeeping.
- ``near_image``: for a small ``B`` (Fresnel number ``F = lambda |B/A| / (n p^2) < 1``). The
  factorisation ``L(C A^-1) Lin(A) Fr(A^-1 B)``: an FFT transfer-function drift on the input lattice,
  the lattice map ``P -> A P`` (no interpolation), and an elementwise chirp. ``variant='II'`` is the
  mirror factorisation ``Fr(B D^-1) Lin(D^-T) L(D^-1 C)`` (drift last, lattice ``D^-T P``), which
  returns exactly to the original lattice for the inverse of a variant-I plan.
- ``exact``: ``B = 0`` exactly, a lattice map plus a chirp.

``plan_lct`` and ``apply_lct`` are pure JAX, jit-able and differentiable in the field and in the
matrix. The branch is a static choice made from concrete values: pass ``method=`` explicitly when
planning under a trace.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from ..config import require_x64
from ._terms import mft_terms, ni_terms, split_matrix
from .lattice import Lattice, adapted_output_lattice, is_concrete, near_image_lattice
from .planner import (LatticeError, SamplingError, _field_stats, _support_band,
                      check_mft, check_near_image)

__all__ = ["LCTPlan", "SepFactor", "plan_lct", "apply_lct", "lift_phase", "split_matrix"]

_TWO_PI = 2.0 * np.pi
_TIE = 1e-9   # dimensionless: |eigenvalue of A| below this counts as a Fourier plane, resolved as A -> 0+


def _reflect(A):
    """Extra Maslov factor ``-i`` when the lattice map reverses orientation (det A < 0)."""
    return jax.lax.stop_gradient(jnp.where(jnp.linalg.det(A) < 0, -1j, 1.0 + 0j))


def lift_phase(A, B):
    """Return the Maslov phase of the engine's lift of a ray matrix (``B`` must be invertible).

    The phase is ``exp(-i pi sig(B^-1 A) / 4) * (-i)^[det A < 0]`` with ``sig`` the signature of the
    symmetric matrix ``B^-1 A``. For ``A`` invertible this equals the kernel phase of the canonical
    composition ``L(T) Lin(A) Fr(S)`` (checked in the tests), so it is continuous through image
    planes and flips sign across round Fourier planes. Exactly at a Fourier plane (singular ``A``)
    the sign is fixed by the limit ``A -> A + 0+`` (implemented as ``A + 1e-9 I``), so rounding
    noise cannot change it. The lift of a symplectic matrix is only defined up to a sign
    (metaplectic double cover); the caller pins the physical branch with ``phase0`` (see
    :func:`temgym_core.wave.reference.chief_phase0`).

    Parameters
    ----------
    A, B : (2, 2) arrays
        Position-to-position [1] and slope-to-position [m/rad] blocks of the ray matrix.

    Returns
    -------
    complex
        A unit-modulus factor.
    """
    A_eff = A + _TIE * jnp.eye(2)
    G = jnp.linalg.solve(B, A_eff)
    G = 0.5 * (G + G.T)
    tr = G[0, 0] + G[1, 1]
    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    disc = jnp.sqrt(jnp.maximum(tr * tr / 4 - det, 0.0))
    n_pos = (tr / 2 + disc >= 0).astype(jnp.int32) + (tr / 2 - disc >= 0).astype(jnp.int32)
    sig = 2 * n_pos - 2
    ph = jnp.exp(-0.25j * jnp.pi * sig)
    ph = jnp.where(jnp.linalg.det(A_eff) < 0, ph * (-1j), ph)
    return jax.lax.stop_gradient(ph)


@jdc.pytree_dataclass
class SepFactor:
    """A separable elementwise factor ``exp(i phi)`` of a plan, with an optional bilinear term.

    The factor is ``fy[:, None] * fx`` times, when ``cross`` is set, ``exp(i cross * iy * ix)``.

    Parameters
    ----------
    fx, fy : (nx,), (ny,) arrays
        Complex per-axis factors.
    cross : float, optional
        Coefficient of the bilinear phase, rad per index squared.
    ix, iy : arrays, optional
        Index vectors the bilinear phase is evaluated on.
    """

    fx: jnp.ndarray
    fy: jnp.ndarray
    cross: Optional[jnp.ndarray] = None
    ix: Optional[jnp.ndarray] = None
    iy: Optional[jnp.ndarray] = None


def _mul(f, s: SepFactor, scale=None):
    fx = s.fx if scale is None else s.fx * scale
    f = f * s.fy[:, None] * fx
    if s.cross is not None:
        f = f * jnp.exp(1j * s.cross * s.iy[:, None] * s.ix)
    return f


def _quad_factor(ix, iy, Qk, lin, const, force_cross):
    """``exp(i [1/2 i^T Qk i + lin . i + const])`` split per axis (const folded into fx)."""
    px = 0.5 * Qk[0, 0] * ix**2 + lin[0] * ix + const
    py = 0.5 * Qk[1, 1] * iy**2 + lin[1] * iy
    cross = Qk[0, 1] if force_cross else None
    return SepFactor(fx=jnp.exp(1j * px), fy=jnp.exp(1j * py), cross=cross,
                     ix=ix if cross is not None else None, iy=iy if cross is not None else None)


def _wants_cross(coeff, ix, iy):
    """Keep a bilinear ``exp(i coeff ix iy)`` unless it is a provably negligible phase (concrete only)."""
    if not is_concrete(coeff, ix, iy):
        return True
    reach = float(np.max(np.abs(np.asarray(ix)))) * float(np.max(np.abs(np.asarray(iy))))
    return abs(float(np.asarray(coeff))) * reach > 1e-13


@jdc.pytree_dataclass
class LCTPlan:
    """Everything :func:`apply_lct` needs: static structure plus factor arrays (a pytree).

    Parameters
    ----------
    kind : {'mft', 'near_image', 'exact'}
        Branch of the transform (static).
    variant : {'', 'I', 'II'}
        Near-image factorisation; empty for ``mft`` (static).
    mode_x, mode_y : int
        MFT per-axis implementation: 0 matrix product, +1 FFT, -1 inverse FFT (static).
    yfirst : bool
        Order of the two MFT passes (static).
    offset_mode : {'after', 'before'}
        Operator ordering of the Heisenberg-Weyl offset (static).
    in_lattice, out_lattice : Lattice
        Lattice of the field before and after the transform.
    prefactor : complex
        Scalar factor: Maslov phase, amplitude and cell area.
    matrix : (5, 5) array
        The ray map, kept for :meth:`inverse` and diagnostics.
    wavelength : float
        Wavelength, metres.
    pre, post, tf : SepFactor, optional
        Elementwise factors before the transform, after it, and the Fresnel transfer function.
    Ex, Ey : (n', n) arrays, optional
        Matrix-product kernels of the MFT.
    """

    kind: jdc.Static[str]
    variant: jdc.Static[str]
    mode_x: jdc.Static[int]
    mode_y: jdc.Static[int]
    yfirst: jdc.Static[bool]
    offset_mode: jdc.Static[str]
    in_lattice: Lattice
    out_lattice: Lattice
    prefactor: jnp.ndarray
    matrix: jnp.ndarray
    wavelength: jnp.ndarray
    pre: Optional[SepFactor] = None
    post: Optional[SepFactor] = None
    tf: Optional[SepFactor] = None
    Ex: Optional[jnp.ndarray] = None
    Ey: Optional[jnp.ndarray] = None

    def inverse(self, **kw):
        """Return the plan of the inverse map, taking ``out_lattice`` back to exactly ``in_lattice``.

        A near-image plan is inverted with the mirror factorisation so that the lattice returns
        exactly. The samples of an imaging plane carry the Nyquist-saturated lens chirp that the
        exact conjugate chirp cancels sample by sample, which the sampling planner cannot know:
        the default is ``check='off'``.

        Parameters
        ----------
        **kw
            Passed to :func:`plan_lct` (for example ``check='raise'``).

        Returns
        -------
        LCTPlan
        """
        M = self.matrix
        Minv = jnp.linalg.inv(M)
        method = {"mft": "mft", "exact": "exact"}.get(self.kind)
        if method is None:
            method = "near_image_II" if self.variant == "I" else "near_image"
        mode = "before" if self.offset_mode == "after" else "after"
        return plan_lct(Minv, self.wavelength, self.out_lattice, self.in_lattice,
                        method=method, offset_mode=mode, check=kw.pop("check", "off"), **kw)


def _lattice_close(l1: Lattice, l2: Lattice, rtol=1e-9) -> bool:
    p1, p2 = np.asarray(l1.pitch), np.asarray(l2.pitch)
    c1, c2 = np.asarray(l1.centre), np.asarray(l2.centre)
    scale = max(np.abs(p2).max(), 1e-300)
    return (l1.shape == l2.shape and np.abs(p1 - p2).max() <= rtol * scale
            and np.abs(c1 - c2).max() <= rtol * scale * max(l2.shape))


def _boost_factor(ix, iy, P, b, centre, k):
    """``exp(i k b . r)`` on a lattice (separable linear phase; constant folded into x)."""
    lin = k * (P.T @ b)
    const = k * (b @ centre)
    return jnp.exp(1j * (lin[0] * ix + const)), jnp.exp(1j * lin[1] * iy)


def _build_mft(A, B, C, D, a, b, lam, in_lat: Lattice, out_lat: Lattice, M5, force_gemm=False,
               pre_boost=None, offset_mode="after", plan_in_lattice=None):
    Pin, rc, Pout, rcp = in_lat.pitch, in_lat.centre, out_lat.pitch, out_lat.centre
    t = mft_terms(jnp, A, B, D, lam, Pin, rc, Pout, rcp, a, b)
    nyi, nxi = in_lat.shape
    nyo, nxo = out_lat.shape
    mu_x, mu_y = in_lat.mu()
    mup_x, mup_y = out_lat.mu()
    gam = jnp.stack((t["Gam"][0, 0], t["Gam"][1, 1]))

    def axis_mode(g, n_in, n_out):
        if force_gemm or not is_concrete(g) or n_in != n_out:
            return 0
        gn = float(np.asarray(g)) * n_in
        if abs(gn - 1.0) < 1e-11:
            return 1
        if abs(gn + 1.0) < 1e-11:
            return -1
        return 0

    mode_x = axis_mode(gam[0], nxi, nxo)
    mode_y = axis_mode(gam[1], nyi, nyo)

    pre = _quad_factor(mu_x, mu_y, t["Qin_k"], t["lin_in"], 0.0, _wants_cross(t["Qin_k"][0, 1], mu_x, mu_y))
    post = _quad_factor(mup_x, mup_y, t["Qout_k"], t["lin_out"], t["const"],
                        _wants_cross(t["Qout_k"][0, 1], mup_x, mup_y))
    pref = (lift_phase(A, B) / (lam * jnp.sqrt(jnp.abs(jnp.linalg.det(B)))) * in_lat.cell_area)

    Ex = Ey = None
    fx_pre, fy_pre, fx_post, fy_post = pre.fx, pre.fy, post.fx, post.fy
    # FFT modes fold the centred-index twist into the elementwise factors
    for axis in ("x", "y"):
        mode = mode_x if axis == "x" else mode_y
        n = nxi if axis == "x" else nyi
        g = gam[0] if axis == "x" else gam[1]
        mu, mup = (mu_x, mup_x) if axis == "x" else (mu_y, mup_y)
        if mode == 0:
            E = jnp.exp(-1j * _TWO_PI * g * jnp.outer(mup, mu))
            if axis == "x":
                Ex = E
            else:
                Ey = E
        else:
            s = float(mode)
            m = jnp.arange(n, dtype=jnp.float64)
            c = (n - 1) / 2.0
            twist = jnp.exp(1j * _TWO_PI * s * c * m / n)        # exp(+2 pi i s c m / N), c = (N - 1) / 2
            pref = pref * jnp.exp(-1j * _TWO_PI * s * c * c / n)
            if axis == "x":
                fx_pre, fx_post = fx_pre * twist, fx_post * twist
            else:
                fy_pre, fy_post = fy_pre * twist, fy_post * twist
    if pre_boost is not None:
        fx_pre, fy_pre = fx_pre * pre_boost[0], fy_pre * pre_boost[1]
    pre = SepFactor(fx=fx_pre, fy=fy_pre, cross=pre.cross, ix=pre.ix, iy=pre.iy)
    post = SepFactor(fx=fx_post, fy=fy_post, cross=post.cross, ix=post.ix, iy=post.iy)

    # cheaper matrix-product order (static)
    cost_y_first = nyo * nyi * nxi + nyo * nxi * nxo
    cost_x_first = nyi * nxi * nxo + nyo * nyi * nxo
    return LCTPlan(kind="mft", variant="", mode_x=mode_x, mode_y=mode_y,
                   yfirst=bool(cost_y_first <= cost_x_first), offset_mode=offset_mode,
                   in_lattice=plan_in_lattice if plan_in_lattice is not None else in_lat, out_lattice=out_lat,
                   prefactor=pref, matrix=M5, wavelength=jnp.asarray(lam, dtype=jnp.float64),
                   pre=pre, post=post, Ex=Ex, Ey=Ey)


def _build_ni(variant, A, B, C, D, a, b, lam, in_lat: Lattice, M5, exact, pre_boost=None,
              offset_mode="after", plan_in_lattice=None):
    t = ni_terms(jnp, variant, A, B, C, D, lam, in_lat.pitch, in_lat.centre, a, b)
    ny, nx = in_lat.shape
    mu_x, mu_y = in_lat.mu()
    out_lat = Lattice(pitch=t["Pout"], centre=t["rc_out"], shape=in_lat.shape)
    Al = t["Al"]
    pref = _reflect(Al) / jnp.sqrt(jnp.abs(jnp.linalg.det(Al)))

    chirp = _quad_factor(mu_x, mu_y, t["Qc_k"], t["lin_c"], t["const_c"],
                         _wants_cross(t["Qc_k"][0, 1], mu_x, mu_y))
    # boost exp(i k b . r') on the output lattice is separable-linear
    boost = SepFactor(fx=jnp.exp(1j * (t["lin_b"][0] * mu_x + t["const_b"])),
                      fy=jnp.exp(1j * t["lin_b"][1] * mu_y))
    tf = None
    if not exact:
        nux = jnp.fft.fftfreq(nx)
        nuy = jnp.fft.fftfreq(ny)
        Lam = t["Lam"]
        cross = -2.0 * jnp.pi * Lam[0, 1]
        keep = _wants_cross(cross, nux, nuy)
        tf = SepFactor(fx=jnp.exp(-1j * jnp.pi * Lam[0, 0] * nux**2),
                       fy=jnp.exp(-1j * jnp.pi * Lam[1, 1] * nuy**2),
                       cross=cross if keep else None,
                       ix=nux if keep else None, iy=nuy if keep else None)
    if variant == "I":
        pre = None
        post = SepFactor(fx=chirp.fx * boost.fx, fy=chirp.fy * boost.fy, cross=chirp.cross,
                         ix=chirp.ix, iy=chirp.iy)
    else:
        pre = chirp
        post = boost
    if pre_boost is not None:
        if pre is None:
            pre = SepFactor(fx=pre_boost[0], fy=pre_boost[1])
        else:
            pre = SepFactor(fx=pre.fx * pre_boost[0], fy=pre.fy * pre_boost[1], cross=pre.cross,
                            ix=pre.ix, iy=pre.iy)
    kind = "exact" if exact else "near_image"
    return LCTPlan(kind=kind, variant=variant, mode_x=0, mode_y=0, yfirst=True, offset_mode=offset_mode,
                   in_lattice=plan_in_lattice if plan_in_lattice is not None else in_lat,
                   out_lattice=out_lat, prefactor=pref, matrix=M5,
                   wavelength=jnp.asarray(lam, dtype=jnp.float64), pre=pre, post=post, tf=tf)


def _matrix5(matrix):
    M = jnp.asarray(matrix, dtype=jnp.float64)
    if M.shape == (4, 4):
        M = jnp.eye(5).at[:4, :4].set(M)
    return M


def _f_length(in_lattice: Lattice, lam) -> float:
    """Distance at which the Fresnel number of the input window is one, ``n p^2 / lambda``."""
    p2 = abs(float(np.linalg.det(np.asarray(in_lattice.pitch))))
    return max(in_lattice.nx, in_lattice.ny) * p2 / float(lam)


_METHODS = ("mft", "near_image", "near_image_II", "exact")


def plan_lct(matrix, wavelength, in_lattice: Lattice, out_lattice: Lattice | None = None, *,
             method: str = "auto", gamma=None, out_shape=None, out_centre=None,
             support=None, band=None, field=None, tol: float = 1e-16, check: str = "raise",
             margin: float = 0.0, offset_mode: str = "after", return_report: bool = False):
    """Plan the propagation of a field on ``in_lattice`` through a ray matrix.

    Parameters
    ----------
    matrix : (5, 5) or (4, 4) array
        Ray map on ``[x, y, theta_x, theta_y, (1)]`` (``run.column_matrix``); must be symplectic.
        The last column is the affine offset ``t`` of ``z' = M z + t``.
    wavelength : float
        Wavelength, metres.
    in_lattice : Lattice
        Lattice of the input field.
    out_lattice : Lattice, optional
        Requested output lattice. ``near_image`` and ``exact`` reproduce only the mapped lattice
        (``A P`` or ``D^-T P``); ``mft`` accepts any lattice for which the cross term is diagonal
        (build the input lattice with :func:`~temgym_core.wave.lattice.adapted_input_lattice`),
        else :class:`LatticeError`. If omitted, the natural output lattice of the chosen branch
        is used.
    method : {'auto', 'mft', 'near_image', 'near_image_II', 'exact'}, default 'auto'
        ``auto`` needs a concrete ``matrix`` and picks the first feasible branch: ``exact`` for
        ``B = 0``, then ``near_image``, then ``mft``. Pass the branch explicitly under jit or grad.
    gamma : (2,) sequence, optional
        MFT cross-term parameters ``(gamma_x, gamma_y)`` when ``out_lattice`` is omitted (default
        ``1/n``: an exact DFT along each axis).
    out_shape : tuple[int, int], optional
        MFT output shape (default: the input shape).
    out_centre : (2,) sequence, optional
        MFT output centre, metres (default: the image of the input centre, ``A c + a``).
    support, band : ((x, y)), optional
        Assumed input half-width [samples] and bandwidth [cycles/sample]. ``field`` measures them.
    field : array, optional
        The field to be transformed. Its support and band are measured for the sampling checks and
        the report carries the spectral tail energy.
    tol : float, default 1e-16
        Fraction of the field's energy allowed outside the measured support and band. The relative
        L2 error of the transform is of order ``sqrt(tol)``: 1e-16 gives 1e-8, 1e-12 gives 1e-6.
    check : {'raise', 'report', 'off'}, default 'raise'
        ``raise``: choose the branch by the sampling rules and raise :class:`SamplingError` if the
        chosen branch would alias. ``report``: choose the same way but never raise. ``off``: no
        check at all for an explicit ``method`` (the expert override); with ``method='auto'`` the
        branch is still chosen by the rules on the assumed support and band, without measuring
        ``field``, and never raises.
    margin : float, default 0
        Extra slack required of every sampling rule.
    offset_mode : {'after', 'before'}, default 'after'
        Operator ordering of the Heisenberg-Weyl offset: ``T(t) U_M`` (the output samples are the
        image of the input samples under ``A r + a``) or ``U_M T(M^-1 t)`` (the input field is
        shifted and boosted first). Same operator up to a constant phase; the second is what
        :meth:`LCTPlan.inverse` needs to return exactly to the original lattice.
    return_report : bool, default False
        Also return the :class:`SamplingReport` (``None`` for ``exact``, ``check='off'`` or traced
        input).

    Returns
    -------
    plan : LCTPlan
    report : SamplingReport or None
        Only if ``return_report`` is true.

    Raises
    ------
    SamplingError
        If the checks fail (``check='raise'``).
    LatticeError
        If the requested lattices are not adapted to the matrix.
    """
    require_x64("plan_lct")
    if check not in ("raise", "report", "off"):
        raise ValueError(f"check must be 'raise', 'report' or 'off', got {check!r}")
    if offset_mode not in ("after", "before"):
        raise ValueError(f"offset_mode must be 'after' or 'before', got {offset_mode!r}")
    A, B, C, D, a, b = split_matrix(matrix)
    M5 = _matrix5(matrix)
    lam = wavelength
    concrete = is_concrete(M5, lam, in_lattice.pitch, in_lattice.centre)
    stats = None
    if field is not None and concrete and is_concrete(field) and check != "off":
        stats = _field_stats(field)
        if support is None and band is None:
            support, band = _support_band(stats, tol)
    lam_f = float(lam) if concrete else lam
    k = _TWO_PI / lam

    pre_boost = None
    in_eff = in_lattice
    if offset_mode == "before":
        t_pre = jnp.linalg.inv(M5[:4, :4]) @ M5[:4, 4]
        a_pre, b_pre = t_pre[:2], t_pre[2:]
        in_eff = in_lattice.with_centre(in_lattice.centre + a_pre)
        pre_boost = _boost_factor(*in_lattice.mu(), in_lattice.pitch, b_pre, in_eff.centre, k)
        if concrete and band is not None:  # the boost shifts the input band
            shift = np.abs(np.asarray(in_lattice.pitch).T @ np.asarray(b_pre)) / lam_f
            band = tuple(float(x) for x in (np.asarray(band) + shift))
        a = jnp.zeros(2)
        b = jnp.zeros(2)

    def report_for(meth, out_lat=None):
        if not concrete or (check == "off" and method != "auto"):
            return None
        if meth == "mft":
            return check_mft(A, B, D, lam_f, in_eff, out_lat, a, b, support=support, band=band,
                             margin=margin, field=stats)
        variant = "II" if meth == "near_image_II" else "I"
        return check_near_image(variant, A, B, C, D, lam_f, in_eff, a, b, support=support,
                                band=band, margin=margin, field=stats)

    def default_mft_out():
        shape = out_shape if out_shape is not None else in_lattice.shape
        g = gamma if gamma is not None else (1.0 / shape[1], 1.0 / shape[0])
        centre = out_centre if out_centre is not None else A @ in_eff.centre + a
        return adapted_output_lattice(B, lam, in_eff, g, shape, centre)

    b_is_zero = (concrete and
                 float(np.abs(np.asarray(B)).max()) <= 1e-13 * _f_length(in_lattice, lam_f))

    reports = []
    by_method = {}
    if method == "auto":
        if not concrete:
            raise ValueError("method='auto' needs a concrete matrix; pass method= explicitly under jit/grad")
        if b_is_zero:
            chosen = "exact"
        else:
            cands = []
            for meth, variant in (("near_image", "I"), ("near_image_II", "II")):
                if out_lattice is None:
                    if variant == "I":
                        cands.append(meth)
                elif _lattice_close(out_lattice, near_image_lattice(variant, A, D, a, in_eff)):
                    cands.append(meth)
            cands.append("mft")
            chosen = None
            for meth in cands:
                out_lat = (out_lattice if out_lattice is not None else default_mft_out()) if meth == "mft" else None
                try:
                    rep = report_for(meth, out_lat)
                except LatticeError:
                    if out_lattice is not None and meth == "mft":
                        raise
                    continue
                reports.append(rep)
                by_method[meth] = rep
                if rep.ok:
                    chosen = meth
                    break
            if chosen is None:
                if check != "raise":
                    chosen = cands[-1]
                else:
                    detail = " | ".join(f"{r.kind}: " + "; ".join(r.messages) for r in reports)
                    raise SamplingError(
                        "no branch can hold this field without aliasing: " + detail +
                        " -- split the span at an intermediate plane (propagate to it, then continue "
                        "from the returned field) or change the sampling")
    else:
        chosen = method
        if chosen not in _METHODS:
            raise ValueError(f"unknown method {method!r}; expected 'auto' or one of {_METHODS}")
        if chosen == "exact" and concrete and not b_is_zero:
            raise ValueError("method='exact' needs B = 0 (an image plane); use near_image")

    rep = by_method.get(chosen)
    if chosen == "mft":
        out_lat = out_lattice if out_lattice is not None else default_mft_out()
        rep = report_for("mft", out_lat) if rep is None else rep
        if rep is not None and check == "raise":
            rep.raise_if_bad()
        plan = _build_mft(A, B, C, D, a, b, lam, in_eff, out_lat, M5, pre_boost=pre_boost,
                          offset_mode=offset_mode, plan_in_lattice=in_lattice)
    else:
        variant = "II" if chosen == "near_image_II" else "I"
        exact = chosen == "exact"
        if not exact:
            rep = report_for(chosen) if rep is None else rep
            if rep is not None and check == "raise":
                rep.raise_if_bad()
        plan = _build_ni(variant, A, B, C, D, a, b, lam, in_eff, M5, exact, pre_boost=pre_boost,
                         offset_mode=offset_mode, plan_in_lattice=in_lattice)
        if out_lattice is not None and concrete and not _lattice_close(out_lattice, plan.out_lattice):
            raise LatticeError(
                f"branch {chosen!r} produces the lattice A P / D^-T P, which differs from the requested "
                "out_lattice; use method='mft' with an adapted lattice, or resample")
    return (plan, rep) if return_report else plan


def _axis_transform(f, axis, mode, E):
    n_axis = -2 if axis == "y" else -1
    if mode == 0:
        return jnp.matmul(E, f) if axis == "y" else jnp.matmul(f, E.T)
    if mode > 0:
        return jnp.fft.fft(f, axis=n_axis)
    return jnp.fft.ifft(f, axis=n_axis) * f.shape[n_axis]


def apply_lct(plan: LCTPlan, field, phase0=0.0):
    """Apply a planned transform to ``field[..., ny, nx]`` (leading axes are batch axes).

    Parameters
    ----------
    plan : LCTPlan
        From :func:`plan_lct`.
    field : array, shape (..., ny, nx)
        Complex field on ``plan.in_lattice``.
    phase0 : float, default 0
        Constant phase multiplied into the output, rad: the physical branch (Gouy/Maslov) and
        constant path of the span, taken from a chief Gaussian
        (:func:`temgym_core.wave.reference.chief_phase0` or ``column_transform(...).phase0``).

    Returns
    -------
    array, shape (..., ny', nx')
        The transformed field on ``plan.out_lattice``.
    """
    f = jnp.asarray(field)
    if f.dtype != jnp.complex128:
        f = f.astype(jnp.complex128)
    if f.shape[-2:] != tuple(plan.in_lattice.shape):
        raise ValueError(f"field shape {f.shape[-2:]} != plan input shape {tuple(plan.in_lattice.shape)}")
    pref = plan.prefactor * jnp.exp(1j * phase0)
    if plan.kind == "mft":
        f = _mul(f, plan.pre)
        order = ("y", "x") if plan.yfirst else ("x", "y")
        for axis in order:
            mode = plan.mode_y if axis == "y" else plan.mode_x
            f = _axis_transform(f, axis, mode, plan.Ey if axis == "y" else plan.Ex)
        return _mul(f, plan.post, scale=pref)
    if plan.pre is not None:
        f = _mul(f, plan.pre)
    if plan.tf is not None:
        f = jnp.fft.ifft2(_mul(jnp.fft.fft2(f), plan.tf))
    return _mul(f, plan.post, scale=pref)
