"""Band-limited (Fourier) resampling of a sampled field between two lattices.

The engine owns its lattices (rotated by the Larmor angle, scaled by the magnification, sheared by
astigmatism), so a field that comes from a Cartesian array, or must end on a Cartesian detector
grid, has to be interpolated. The array is treated as one period of a band-limited function and
evaluated exactly at the new sample positions: for a field that the planner accepts (compact,
resolved by its samples) the result is accurate to rounding, not to the order of an interpolation
kernel.

The map between the two lattices, ``mu = m0 + K mu'`` in centred sample indices, is applied in two
passes. The first pass shears and zooms each row: an FFT along x turns the row-dependent shift
``s * y`` into a phase ramp (an FFT shear) and a matrix Fourier evaluation lands the samples on the
output columns. The second pass does the same along y. The four entries of ``K`` are permuted so
that the largest one is the pivot, which keeps the shear ``s`` within one sample per row.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..config import require_x64
from .lattice import as_lattice, is_concrete
from .planner import SamplingError, _field_stats, _support_band

__all__ = ["resample_field"]

_MARGIN = 2.0  # samples of zero padding beyond the region the output reads


def _shift_phases(n, shifts):
    """``exp(2 pi i nu s / n)`` for every shift ``s`` (rows) and FFT frequency ``nu`` (columns).

    The Nyquist bin of an even ``n`` uses ``cos(pi s)``, the real interpolation of that bin.
    """
    nu = jnp.fft.fftfreq(n) * n
    phase = 2.0 * jnp.pi * jnp.asarray(shifts)[:, None] * nu[None, :] / n
    out = jnp.exp(1j * phase)
    return out if n % 2 else out.at[:, n // 2].set(jnp.cos(phase[:, n // 2]))


def _evaluation_matrix(n, positions):
    """Matrix ``E[j, nu] = exp(2 pi i nu (r_j + (n - 1) / 2) / n) / n`` evaluating an FFT-ordered spectrum.

    ``r_j`` are centred positions; multiplying the FFT of an ``n``-sample array by ``E.T`` gives the
    band-limited interpolant at those positions.
    """
    return _shift_phases(n, jnp.asarray(positions) + (n - 1) / 2.0) / n


def _is_smooth(n):
    for prime in (2, 3, 5, 7):
        while n % prime == 0:
            n //= prime
    return n == 1


def _padding(needed, n):
    """Symmetric zero padding (per side) so that an array of ``n`` samples, read up to ``needed``
    samples from its centre, sees no periodic image of its content, at an FFT-friendly length."""
    pad = max(0, int(np.ceil((needed + _MARGIN - n / 2.0) / 2.0)))
    while not _is_smooth(n + 2 * pad):
        pad += 1
    return pad


def _pads(K, m0, in_shape, out_shape):
    """Zero padding ``(pad_y, pad_x)`` per side that the two passes need for ``out(mu') = f(m0 + K mu')``."""
    (ny, nx), (ny_o, nx_o) = in_shape, out_shape
    kappa, shear = (K[0, 0] * K[1, 1] - K[0, 1] * K[1, 0]) / K[1, 1], K[0, 1] / K[1, 1]
    x0 = m0[0] - shear * m0[1]
    half_x_out, half_y_out = (nx_o - 1) / 2.0, (ny_o - 1) / 2.0
    rows_needed = abs(m0[1]) + abs(K[1, 0]) * half_x_out + abs(K[1, 1]) * half_y_out
    columns_needed = abs(x0) + abs(kappa) * half_x_out + abs(shear) * (ny - 1) / 2.0
    return _padding(rows_needed, ny), _padding(columns_needed, nx)


@partial(jax.jit, static_argnames=("out_shape", "pads"))
def _two_pass(f, K, m0, out_shape, pads):
    """Evaluate ``out(mu') = f(m0 + K mu')`` by the band-limited interpolant, ``K[1, 1]`` the pivot."""
    ny_o, nx_o = out_shape
    pad_y, pad_x = pads
    k12, k21, k22 = K[0, 1], K[1, 0], K[1, 1]
    kappa, shear = (K[0, 0] * k22 - k12 * k21) / k22, k12 / k22
    x0 = m0[0] - shear * m0[1]
    f = jnp.pad(f, [(0, 0)] * (f.ndim - 2) + [(pad_y, pad_y), (pad_x, pad_x)])
    ny_p, nx_p = f.shape[-2:]
    mu_x = jnp.arange(nx_o) - (nx_o - 1) / 2.0
    mu_y = jnp.arange(ny_o) - (ny_o - 1) / 2.0
    rows = jnp.arange(ny_p) - (ny_p - 1) / 2.0

    spectrum = jnp.fft.fft(f, axis=-1) * _shift_phases(nx_p, shear * rows)          # FFT shear of each row
    g = spectrum @ _evaluation_matrix(nx_p, x0 + kappa * mu_x).T                     # onto the output columns
    spectrum = jnp.fft.fft(g, axis=-2) * _shift_phases(ny_p, k21 * mu_x).T          # shear of each column
    return _evaluation_matrix(ny_p, m0[1] + k22 * mu_y) @ spectrum                   # onto the output rows


def _significant_frequencies(stats, tol):
    """Signed frequencies ``[nu_x; nu_y]`` (cycles per sample) of the spectral bins left after dropping
    the weakest ones that hold at most ``tol`` of the energy."""
    spectrum = np.asarray(stats.spectrum)
    flat = spectrum.ravel()
    order = np.argsort(flat)
    dropped = np.cumsum(flat[order]) <= tol * flat.sum()
    iy, ix = np.unravel_index(order[~dropped], spectrum.shape)
    ny, nx = spectrum.shape
    return np.stack((np.fft.fftfreq(nx)[ix], np.fft.fftfreq(ny)[iy]))


def _pivot(K):
    """Swap the input and/or output axes so that the largest entry of ``K`` is the pivot ``K[1, 1]``."""
    best = None
    for swap_in in (False, True):
        for swap_out in (False, True):
            Kp = K[::-1] if swap_in else K
            Kp = Kp[:, ::-1] if swap_out else Kp
            if best is None or abs(Kp[1, 1]) > abs(best[2][1, 1]):
                best = (swap_in, swap_out, Kp)
    return best


def _check(data, K, in_shape, out_shape, pivot, tol):
    """Raise ``SamplingError`` unless the samples resolve the field before, during and after the map."""
    stats = _field_stats(data)
    (support_x, support_y), _ = _support_band(stats, tol)
    nu = _significant_frequencies(stats, tol)
    (ny, nx), (ny_o, nx_o) = in_shape, out_shape
    swap_in, swap_out, Kp = pivot
    messages = []
    for axis, band, n in (("x", np.abs(nu[0]).max(), nx), ("y", np.abs(nu[1]).max(), ny)):
        if band > 0.5 - 2.0 / n:
            messages.append(f"axis {axis}: input band {band:.4g} > {0.5 - 2.0 / n:.4g} cycles/sample "
                            "(Nyquist less two bins): the samples are already aliased")
    for axis, support, n in (("x", support_x, nx), ("y", support_y, ny)):
        if support > n / 2.0 - 1.0:
            messages.append(f"axis {axis}: input support {support:.4g} > {n / 2.0 - 1.0:.4g} samples: the field "
                            "touches the array edge and would wrap around; pad it")
    output = K.T @ nu
    for axis, band, n in (("x", np.abs(output[0]).max(), nx_o), ("y", np.abs(output[1]).max(), ny_o)):
        if band > 0.5 - 2.0 / n:
            messages.append(f"axis {axis}: the output lattice does not resolve the field: band {band:.4g} > "
                            f"{0.5 - 2.0 / n:.4g} cycles/sample; use finer output samples")
    nu_p = nu[::-1] if swap_in else nu
    rows = nx if swap_in else ny
    columns = ny_o if swap_out else nx_o
    kappa, shear = (Kp[0, 0] * Kp[1, 1] - Kp[0, 1] * Kp[1, 0]) / Kp[1, 1], Kp[0, 1] / Kp[1, 1]
    if np.abs(kappa * nu_p[0]).max() > 0.5 - 2.0 / columns:
        messages.append("the column zoom of the first pass exceeds Nyquist")
    if np.abs(nu_p[1] + shear * nu_p[0]).max() > 0.5 - 2.0 / rows:
        messages.append("the row shear of the first pass raises the band beyond Nyquist")
    if messages:
        raise SamplingError("resample: " + "; ".join(messages))


def resample_field(data, in_lattice, out_lattice, *, check: str = "raise", tol: float = 1e-16):
    """Interpolate a field from one lattice to another with the exact band-limited interpolant.

    Parameters
    ----------
    data : array, shape (..., ny, nx)
        Field values at the samples of ``in_lattice``; leading axes are batch axes.
    in_lattice, out_lattice : Lattice or Grid
        Source and target lattices (concrete). A ``Detector`` or ``ScanGrid`` stands for its own
        lattice. They may differ in pitch, rotation, shear, centre and shape.
    check : {'raise', 'off'}, default 'raise'
        Raise :class:`~temgym_core.wave.planner.SamplingError` if the samples do not resolve the
        field before, during or after the map (spectrum at Nyquist, field touching the array edge,
        an output lattice too coarse). Not applied to traced ``data``.
    tol : float, default 1e-16
        Fraction of the field's energy allowed outside the band and support the checks use.

    Returns
    -------
    complex array, shape (..., ny', nx')
        The field values at the samples of ``out_lattice``. Positions outside the input window
        read as zero for a compact field.

    Raises
    ------
    SamplingError
        If ``check='raise'`` and the field is not resolved.
    ValueError
        If a lattice is traced or not invertible, or ``data`` does not match ``in_lattice``.

    Notes
    -----
    The array is treated as one period of a band-limited function, so the field must be compact
    (vanish towards the array edge) and smooth: a hard edge rings, and content that leaves the input
    window is not extrapolated. Values, not densities, are interpolated, so the pointwise field is
    preserved and the flux ``sum(|u|**2) * cell_area`` is conserved up to interpolation error.
    """
    require_x64("resample_field")
    if check not in ("raise", "off"):
        raise ValueError(f"check must be 'raise' or 'off', got {check!r}")
    src, dst = as_lattice(in_lattice), as_lattice(out_lattice)
    if not is_concrete(src.pitch, src.centre, dst.pitch, dst.centre):
        raise ValueError("resample needs concrete lattices")
    data = jnp.asarray(data)
    if data.dtype != jnp.complex128:
        data = data.astype(jnp.complex128)
    if tuple(data.shape[-2:]) != tuple(src.shape):
        raise ValueError(f"data shape {data.shape[-2:]} != in_lattice shape {tuple(src.shape)}")
    P_in, P_out = np.asarray(src.pitch, dtype=float), np.asarray(dst.pitch, dtype=float)
    if np.linalg.det(P_in) == 0 or np.linalg.det(P_out) == 0:
        raise ValueError("lattice pitch matrices must be invertible")
    K = np.linalg.solve(P_in, P_out)
    m0 = np.linalg.solve(P_in, np.asarray(dst.centre, dtype=float) - np.asarray(src.centre, dtype=float))
    if src.shape == dst.shape and np.allclose(K, np.eye(2), atol=1e-12) and np.allclose(m0, 0.0, atol=1e-9):
        return data
    pivot = _pivot(K)
    swap_in, swap_out, Kp = pivot
    if check == "raise" and is_concrete(data):
        _check(data, K, src.shape, dst.shape, pivot, tol)
    m0p = m0[::-1] if swap_in else m0
    in_shape = src.shape[::-1] if swap_in else src.shape
    out_shape = dst.shape[::-1] if swap_out else dst.shape
    out = _two_pass(jnp.swapaxes(data, -1, -2) if swap_in else data, jnp.asarray(Kp), jnp.asarray(m0p), out_shape,
                    _pads(Kp, m0p, in_shape, out_shape))
    return jnp.swapaxes(out, -1, -2) if swap_out else out
