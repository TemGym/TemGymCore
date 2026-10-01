"""Sampling planner for the lattice-adapted Collins engine (host side).

Two levels:

- **geometry** (raise): the lattices are not adapted (cross term not diagonal), or the requested
  window and pitch cannot hold the field for the assumed support and bandwidth. That is a
  :class:`LatticeError` or :class:`SamplingError`; nothing is returned that would be silently aliased.
- **content** (report): :attr:`SamplingReport.tail_energy` is the fraction of the *actual* field's
  spectral energy outside the band the geometry can carry.

Derivation, in index space (``mu`` is the centred sample index, ``nu`` is in cycles per sample). The
discrete matrix Fourier transform of the samples equals the exact Collins integral periodised in the
output index with period ``1/|gamma_j|`` (Poisson summation). Write the true output extent as ``E'_j``
and the output window as ``n'_j``. Replicas do not enter the window iff
``n'_j/2 + E'_j/2 <= 1/|gamma_j|`` with::

    E'_j / 2 = (abs(K) delta)_j + nu_j / abs(gamma_j),   K = P'^-1 A P

and ``abs(K_jl) = abs(Qin_jl) / (lambda abs(gamma_j))`` on adapted lattices, which gives the per-axis
rule::

    c_j + abs(gamma_j) n'_j / 2 <= 1,
    c_j = nu_j + (1/lambda) sum_l abs(Qin_jl) delta_l + abs(lin_in_j) / (2 pi)

where ``Qin = P^T B^-1 A P``, ``lin_in`` is the input-chirp centre offset (``2 pi gamma_j`` times the
output-window offset), ``delta`` is the input support half-width and ``nu`` its band. The near-image
(drift-first) branch is a circular convolution and needs the Fresnel kernel spread to fit::

    delta_j + sum_l abs(Lambda_jl) nu_l <= n_j / 2,   Lambda = lambda P^-1 S P^-T
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ._terms import mft_terms, ni_terms

__all__ = [
    "SamplingError",
    "LatticeError",
    "SamplingReport",
    "measure_support",
    "check_mft",
    "check_near_image",
    "default_support_band",
]


class SamplingError(ValueError):
    """The requested sampling would alias the transformed field."""


class LatticeError(ValueError):
    """The input and output lattices are not adapted to the matrix (cross term not diagonal)."""


@dataclasses.dataclass(frozen=True)
class SamplingReport:
    """Outcome of a sampling check.

    Parameters
    ----------
    kind : str
        Branch that was checked: ``'mft'``, ``'near_image[I]'`` or ``'near_image[II]'``.
    ok : bool
        Whether every rule is satisfied.
    margins : dict
        Signed slack of each rule per axis; a value >= 0 means the rule holds. Cycles per sample
        for band rules, samples for window rules.
    details : dict
        The numbers behind the margins (support, band, chirp, window).
    tail_energy : float or None
        Fraction of the spectral energy of the checked field outside the admissible band, or
        ``None`` when no field was supplied.
    messages : tuple[str, ...]
        One line per failing rule, stating the inequality and the numbers.
    """

    kind: str
    ok: bool
    margins: dict
    details: dict
    tail_energy: float | None = None
    messages: tuple = ()

    def raise_if_bad(self):
        """Raise :class:`SamplingError` listing every failing rule; return ``self`` otherwise."""
        if not self.ok:
            raise SamplingError(f"{self.kind}: " + "; ".join(self.messages))
        return self


def default_support_band(shape, support=None, band=None):
    """Return the assumed field support and band as ``(x, y)`` pairs.

    Defaults: the field fills the central half of the window and uses half of Nyquist. Replace
    them with :func:`measure_support` values for a real field.

    Parameters
    ----------
    shape : tuple[int, int]
        Array shape ``(ny, nx)``.
    support : ((float, float)), optional
        Half-width of the field about the array centre, samples, ``(x, y)``.
    band : ((float, float)), optional
        Half-width of the field spectrum, cycles per sample, ``(x, y)``.

    Returns
    -------
    support, band : tuple[float, float]
    """
    ny, nx = shape
    if support is None:
        support = (0.25 * nx, 0.25 * ny)
    if band is None:
        band = (0.25, 0.25)
    return tuple(float(s) for s in support), tuple(float(b) for b in band)


class _FieldStats(NamedTuple):
    """Intensity and spectral-energy profiles of a field, summed over its batch axes."""

    intensity_x: np.ndarray
    intensity_y: np.ndarray
    spectrum: jnp.ndarray  # (ny, nx) energy per frequency bin, fft order


def _field_stats(field) -> _FieldStats:
    u = jnp.asarray(field)
    u = u.reshape((-1,) + u.shape[-2:])
    intensity = jnp.sum(jnp.abs(u) ** 2, axis=0)
    spectrum = jnp.sum(jnp.abs(jnp.fft.fft2(u)) ** 2, axis=0)
    return _FieldStats(np.asarray(intensity.sum(axis=0)), np.asarray(intensity.sum(axis=1)), spectrum)


def _half_width(profile, grid, tol):
    """Smallest ``|grid|`` cut outside which the energy of ``profile`` is at most ``tol`` of the total."""
    mu = np.abs(grid)
    total = profile.sum()
    for cut in np.unique(mu):
        if profile[mu > cut].sum() <= tol * total:
            return float(cut)
    return float(mu.max())


def measure_support(field, tol=1e-16):
    """Measure the support and band of a concrete sampled field.

    Parameters
    ----------
    field : array, shape (..., ny, nx)
        Complex field; leading axes are pooled.
    tol : float, default 1e-16
        Fraction of the total energy allowed outside the reported extents. The relative L2 error of
        a transform is of order ``sqrt(tol)``.

    Returns
    -------
    support : tuple[float, float]
        ``(delta_x, delta_y)``: smallest half-width, in samples about the array centre, outside
        which the energy is at most ``tol`` of the total.
    band : tuple[float, float]
        ``(nu_x, nu_y)``: the same for the spectrum, in cycles per sample (a carrier counts).
    """
    stats = _field_stats(field)
    return _support_band(stats, tol)


def _support_band(stats: _FieldStats, tol):
    ny, nx = stats.spectrum.shape
    spec = np.asarray(stats.spectrum)
    mux = np.arange(nx) - (nx - 1) / 2
    muy = np.arange(ny) - (ny - 1) / 2
    nux, nuy = np.fft.fftfreq(nx), np.fft.fftfreq(ny)
    support = (_half_width(stats.intensity_x, mux, tol), _half_width(stats.intensity_y, muy, tol))
    band = (_half_width(spec.sum(axis=0), nux, tol), _half_width(spec.sum(axis=1), nuy, tol))
    return support, band


def _spectral_tail(stats: _FieldStats, bad_mask_fn):
    ny, nx = stats.spectrum.shape
    nux, nuy = jnp.meshgrid(jnp.fft.fftfreq(nx), jnp.fft.fftfreq(ny))
    bad = bad_mask_fn(jnp.abs(nux), jnp.abs(nuy))
    total = jnp.sum(stats.spectrum)
    return 0.0 if float(total) == 0 else float(jnp.sum(jnp.where(bad, stats.spectrum, 0.0)) / total)


def _lat(lat):
    return (np.asarray(lat.pitch, dtype=float), np.asarray(lat.centre, dtype=float), lat.shape)


def check_mft(A, B, D, lam, in_lat, out_lat, a, b, *, support=None, band=None,
              margin=0.0, field=None, gamma_tol=1e-8):
    """Check the separable matrix Fourier transform between two lattices (concrete values).

    Parameters
    ----------
    A, B, D : (2, 2) arrays
        Blocks of the ray matrix.
    lam : float
        Wavelength, metres.
    in_lat, out_lat : Lattice
        Input and output lattices.
    a, b : (2,) arrays
        Position [m] and slope [rad] offsets of the ray matrix.
    support, band : ((x, y)), optional
        Assumed input half-width [samples] and band [cycles/sample]; default to
        :func:`default_support_band`, or to the measured values when ``field`` is given to the
        caller of :func:`~temgym_core.wave.lct.plan_lct`.
    margin : float, default 0
        Extra slack required of every rule.
    field : array, optional
        Input field; sets ``SamplingReport.tail_energy``.
    gamma_tol : float, default 1e-8
        Relative size of the off-diagonal cross term that still counts as adapted.

    Returns
    -------
    SamplingReport

    Raises
    ------
    LatticeError
        If the cross term ``Gamma = P'^T B^-T P / lambda`` is not diagonal.
    """
    stats = field if isinstance(field, _FieldStats) else (None if field is None else _field_stats(field))
    Pin, rc, (nyi, nxi) = _lat(in_lat)
    Pout, rcp, (nyo, nxo) = _lat(out_lat)
    t = mft_terms(np, *map(np.asarray, (A, B, D)), lam, Pin, rc, Pout, rcp, np.asarray(a), np.asarray(b))
    Gam = t["Gam"]
    diag = np.array([Gam[0, 0], Gam[1, 1]])
    off = max(abs(Gam[0, 1]), abs(Gam[1, 0]))
    if off > gamma_tol * max(np.abs(diag).max(), 1e-300):
        raise LatticeError(
            "input/output lattices are not adapted to this matrix: the cross-term matrix "
            f"Gamma = P'^T B^-T P / lambda has off-diagonal {off:.3g} vs diagonal {diag}. Build the input "
            "lattice with adapted_input_lattice(...) (or the output lattice with adapted_output_lattice).")
    support, band = default_support_band((nyi, nxi), support, band)
    n_out = np.array([nxo, nyo], dtype=float)
    Qin = t["Qin_k"] / (2 * np.pi)
    chirp = np.abs(Qin) @ np.array(support)  # cycles/sample at the support edge
    lin = np.abs(t["lin_in"]) / (2 * np.pi)
    c = np.array(band) + chirp + lin
    window = np.abs(diag) * n_out / 2
    margins = 1.0 - c - window - margin
    n_in = np.array([nxi, nyi], dtype=float)
    edge = n_in / 2 - np.array(support) - 1.0  # input support must stay inside the window
    nyq = 0.5 - 2.0 / n_in - np.array(band)    # input samples must resolve the field (2 bins of headroom)
    ok = bool(np.all(margins >= 0) and np.all(edge >= 0) and np.all(nyq >= 0))
    msgs = []
    for j, ax in enumerate("xy"):
        if nyq[j] < 0:
            msgs.append(f"axis {ax}: input band {band[j]:.4g} > {0.5 - 2.0 / n_in[j]:.4g} cycles/sample "
                        "(Nyquist less two bins): the samples themselves are aliased; refine the input sampling")
    for j, ax in enumerate("xy"):
        if edge[j] < 0:
            msgs.append(f"axis {ax}: input support {support[j]:.4g} > {n_in[j] / 2 - 1:.4g} samples (half "
                        "window less one): the field is truncated; enlarge the window or shrink gamma")
    for j, ax in enumerate("xy"):
        if margins[j] < 0:
            msgs.append(
                f"axis {ax}: replica rule c + |gamma| n'/2 <= 1 violated: {c[j]:.3g} (band + chirp + offset) "
                f"+ {window[j]:.3g} (half output window in replica periods) = {c[j] + window[j]:.3g}; reduce the "
                "output window, band or support, or split the span at an intermediate plane")
    tail = None
    if stats is not None:
        adm = 1.0 - window - (c - np.array(band))  # admissible |nu| per axis
        tail = _spectral_tail(stats, lambda ax_, ay_: (ax_ > adm[0]) | (ay_ > adm[1]))
    return SamplingReport(
        kind="mft", ok=ok,
        margins={"x": float(margins[0]), "y": float(margins[1]),
                 "edge_x": float(edge[0]), "edge_y": float(edge[1]),
                 "nyquist_x": float(nyq[0]), "nyquist_y": float(nyq[1])},
        details=dict(gamma=diag, c=c, window=window, chirp=chirp, lin=lin, support=support, band=band),
        tail_energy=tail, messages=tuple(msgs))


def check_near_image(variant, A, B, C, D, lam, in_lat, a, b, *, support=None, band=None,
                     margin=0.0, field=None):
    """Check the drift, lattice-map and chirp branch (circular-convolution wrap and band).

    Parameters
    ----------
    variant : {'I', 'II'}
        Drift-first or drift-last factorisation.
    A, B, C, D : (2, 2) arrays
        Blocks of the ray matrix.
    lam : float
        Wavelength, metres.
    in_lat : Lattice
        Input lattice.
    a, b : (2,) arrays
        Position [m] and slope [rad] offsets of the ray matrix.
    support, band : ((x, y)), optional
        Assumed input half-width [samples] and band [cycles/sample].
    margin : float, default 0
        Extra slack required of every rule.
    field : array, optional
        Input field; sets ``SamplingReport.tail_energy``.

    Returns
    -------
    SamplingReport
    """
    stats = field if isinstance(field, _FieldStats) else (None if field is None else _field_stats(field))
    Pin, rc, (ny, nx) = _lat(in_lat)
    lead = np.asarray(A if variant == "I" else D, dtype=float)
    if abs(np.linalg.det(lead)) < 1e-12 * max(1.0, np.abs(lead).max()) ** 2:
        who = "A" if variant == "I" else "D"
        return SamplingReport(kind=f"near_image[{variant}]", ok=False,
                              margins={"singular": -1.0}, details={},
                              messages=(f"{who} is singular (Fourier plane): the drift/lattice-map factorisation "
                                        "does not exist; use the MFT branch",))
    t = ni_terms(np, variant, *map(np.asarray, (A, B, C, D)), lam, Pin, rc, np.asarray(a), np.asarray(b))
    support, band = default_support_band((ny, nx), support, band)
    n = np.array([nx, ny], dtype=float)
    band = np.array(band)
    if variant == "II":  # the chirp is applied before the drift: it widens the band entering the FFT
        Qc = t["Qc_k"] / (2 * np.pi)
        band_eff = band + np.abs(Qc) @ np.array(support) + np.abs(t["lin_c"]) / (2 * np.pi)
    else:
        band_eff = band
    spread = np.abs(t["Lam"]) @ band_eff  # samples
    d = np.array(support) + spread
    wrap = n / 2 - d - margin
    nyq = 0.5 - band_eff - margin
    if variant == "I":  # the FFT sees the raw samples: require two bins of headroom below Nyquist
        nyq = nyq - 2.0 / n
    edge = n / 2 - np.array(support) - 1.0
    margins = {"wrap_x": float(wrap[0]), "wrap_y": float(wrap[1]),
               "band_x": float(nyq[0]), "band_y": float(nyq[1]),
               "edge_x": float(edge[0]), "edge_y": float(edge[1])}
    ok = all(v >= 0 for v in margins.values())
    msgs = []
    for j, ax in enumerate("xy"):
        if wrap[j] < 0:
            msgs.append(f"axis {ax}: wrap rule support + kernel spread <= n/2 violated: "
                        f"{support[j]:.3g} + {spread[j]:.3g} > {n[j] / 2:.3g} samples; use the MFT branch "
                        "or split the span at an intermediate plane")
        if nyq[j] < 0:
            msgs.append(f"axis {ax}: band entering the drift {band_eff[j]:.3g} cycles/sample exceeds Nyquist")
        if edge[j] < 0:
            msgs.append(f"axis {ax}: input support {support[j]:.4g} > {n[j] / 2 - 1:.4g} samples (half "
                        "window less one): the field is truncated")
    tail = None
    if stats is not None:
        Lam = np.abs(t["Lam"])
        sup = np.array(support)

        def bad(nx_, ny_):
            spread_x = Lam[0, 0] * nx_ + Lam[0, 1] * ny_
            spread_y = Lam[1, 0] * nx_ + Lam[1, 1] * ny_
            return (sup[0] + spread_x > n[0] / 2) | (sup[1] + spread_y > n[1] / 2)

        tail = _spectral_tail(stats, bad)
    return SamplingReport(
        kind=f"near_image[{variant}]", ok=ok, margins=margins,
        details=dict(Lam=t["Lam"], spread=spread, support=support, band=tuple(band_eff)),
        tail_energy=tail, messages=tuple(msgs))
