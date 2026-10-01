"""The packed ``(N, 12)`` bundle, its pixel-index form and the policy for invalid beamlets.

Column layout of the packed float64 array (``k`` is folded into every phase coefficient so the
pair loops contain no multiplications by ``k``)::

    0 m_x   1 m_y   2 a_x   3 a_y   4 p_xx   5 q_xx   6 p_xy   7 q_xy   8 p_yy   9 q_yy
    10 Re(A)   11 Im(A)

with ``A = amplitude * exp(i k pathlength)``, ``phi = a . u + u^T P u`` and
``delta = u^T Q u`` (see :mod:`temgym_core.render.predicate`).

Invalid beamlets
    A beamlet with any non-finite field contributes zero and receives a zero gradient: it is
    replaced, before anything else touches it, by the *null beamlet* (zero amplitude, far
    outside every detector), so it never reaches a kernel and never poisons another beamlet.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

MX, MY, AX, AY, PXX, QXX, PXY, QXY, PYY, QYY, AR, AI = range(12)
N_COLUMNS = 12

# Null beamlet: zero amplitude, unit envelope, 1e30 away from any detector.
_NULL_FAR = 1e30
NULL_ROW = np.array([_NULL_FAR, _NULL_FAR, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def pack_gaussian_params(r_centre, dr, amplitude, pathlength, Q_inv, k):
    """Pack beamlet fields into the ``(N, 12)`` float64 array the kernels consume.

    The pathlength is folded into the amplitude (``amplitude * exp(i k pathlength)``), ``Q_inv``
    is symmetrised, and ``k`` multiplies every phase coefficient.  Ordinary JAX, so derivatives
    with respect to every beamlet field follow from the chain rule.

    Parameters
    ----------
    r_centre : array, shape (N, 2)
        Beamlet centres ``(x, y)``.
    dr : array, shape (N, 2)
        Ray slopes.
    amplitude : array, shape (N,)
        Complex amplitudes.
    pathlength : array, shape (N,)
        Optical path lengths (metres).
    Q_inv : array, shape (N, 2, 2)
        Complex inverse curvature matrices; ``Im Q_inv`` sets the envelope.
    k : array, shape (N,)
        Wavenumbers.

    Returns
    -------
    array, shape (N, 12)
        The packed bundle.
    """
    f = jnp.float64

    # Symmetrize Q_inv: after this Qxy == Qyx
    Q = jnp.asarray(Q_inv, jnp.complex128)
    Q = 0.5 * (Q + jnp.swapaxes(Q, -1, -2))

    kf = jnp.asarray(k, f)
    k_half = 0.5 * kf

    # Fold pathlength into amplitude: amp * exp(i*k*path)
    amp = jnp.asarray(amplitude, jnp.complex128) * jnp.exp(
        1j * kf * jnp.asarray(pathlength, f)
    )

    # Pack all per-beam scalars into (N, 12), pre-multiplying k into every
    # phase coefficient. After symmetrization the cross-term is 2*Qxy;
    # combined with the 0.5 prefactor: 0.5*k*2*Qxy = k*Qxy.
    rc = jnp.asarray(r_centre, f)
    drf = jnp.asarray(dr, f)
    return jnp.column_stack([
        rc[:, 0],                           # 0  mx
        rc[:, 1],                           # 1  my
        kf * drf[:, 0],                     # 2  k·drx
        kf * drf[:, 1],                     # 3  k·dry
        k_half * jnp.real(Q[:, 0, 0]),      # 4  ½k·Re(Qxx)
        k_half * jnp.imag(Q[:, 0, 0]),      # 5  ½k·Im(Qxx)
        kf * jnp.real(Q[:, 0, 1]),          # 6  k·Re(Qxy)  (= ½k·2Re(Qxy))
        kf * jnp.imag(Q[:, 0, 1]),          # 7  k·Im(Qxy)
        k_half * jnp.real(Q[:, 1, 1]),      # 8  ½k·Re(Qyy)
        k_half * jnp.imag(Q[:, 1, 1]),      # 9  ½k·Im(Qyy)
        jnp.real(amp),                      # 10 Re(amp)
        jnp.imag(amp),                      # 11 Im(amp)
    ])


def sanitise_packed(packed):
    """Replace every non-finite row of a packed bundle by the null beamlet.

    Parameters
    ----------
    packed : array, shape (N, 12)
        Packed bundle.

    Returns
    -------
    tuple of array
        The cleaned bundle and the boolean ``(N,)`` mask of rows that were valid.  ``jnp.where``
        gives the invalid rows a zero cotangent without multiplying by their non-finite values.
    """
    valid = jnp.all(jnp.isfinite(packed), axis=-1)
    return jnp.where(valid[:, None], packed, jnp.asarray(NULL_ROW)), valid


def sanitise_beam_fields(r_centre, dr, amplitude, pathlength, Q_inv, k):
    """Replace beamlets with any non-finite field by the null beamlet, field by field.

    Doing this before packing (and before the detector map) keeps the derivative of every
    other quantity finite: a non-finite beamlet gets an exactly zero gradient and does not
    turn the gradient with respect to shared detector parameters into NaN.

    Parameters
    ----------
    r_centre, dr, amplitude, pathlength, Q_inv, k : array
        Beamlet fields as returned by ``evaluate._prepare_gaussian_params``.

    Returns
    -------
    tuple of array
        The six cleaned fields, in the same order.
    """
    n = r_centre.shape[0]
    finite = jnp.stack([
        jnp.all(jnp.isfinite(r_centre), axis=-1),
        jnp.all(jnp.isfinite(dr), axis=-1),
        jnp.isfinite(amplitude.real) & jnp.isfinite(amplitude.imag),
        jnp.isfinite(pathlength),
        jnp.all(jnp.isfinite(Q_inv.real) & jnp.isfinite(Q_inv.imag), axis=(-2, -1)),
        jnp.isfinite(k),
    ]).all(axis=0)

    def pick(x, null):
        mask = finite.reshape((n,) + (1,) * (x.ndim - 1))
        return jnp.where(mask, x, null)

    return (
        pick(r_centre, jnp.full(2, _NULL_FAR)), pick(dr, 0.0), pick(amplitude, 0.0),
        pick(pathlength, 0.0), pick(Q_inv, 2j * jnp.eye(2)), pick(k, 1.0),
    )


def affine_from_grid(grid):
    """Pixel-to-metre map of a grid: ``(x, y) = J (ix, iy) + o``.

    Parameters
    ----------
    grid : Grid
        Detector or any object with ``pixels_to_metres_mat`` (a ``temgym_core`` ``Grid``).

    Returns
    -------
    tuple of array or None
        ``J`` with shape ``(2, 2)`` (pixel size, rotation and flip) and ``o`` with shape
        ``(2,)`` (the position of pixel ``(0, 0)``); ``None`` if ``grid`` has no affine map
        (only ``coords`` and ``shape``).
    """
    t = getattr(grid, "pixels_to_metres_mat", None)
    if t is None:
        return None
    t = jnp.asarray(t)  # [y_m, x_m, 1] = t [y_px, x_px, 1]
    j = jnp.array([[t[1, 1], t[1, 0]], [t[0, 1], t[0, 0]]])
    o = jnp.array([t[1, 2], t[0, 2]])
    return j, o


def index_coords(shape):
    """Row-major ``(H*W, 2)`` array of ``(ix, iy)`` pixel indices as float64.

    Parameters
    ----------
    shape : tuple of int
        ``(H, W)``.

    Returns
    -------
    array, shape (H*W, 2)
        Pixel-index coordinates.
    """
    h, w = shape
    iy, ix = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    return jnp.stack([ix.ravel(), iy.ravel()], axis=-1).astype(jnp.float64)


def _form(a, b, c):
    """Symmetric 2x2 forms of ``a u^2 + b u v + c v^2`` stacked over beams."""
    return jnp.stack([jnp.stack([a, 0.5 * b], -1), jnp.stack([0.5 * b, c], -1)], -2)


def to_index_space(packed, j, o):
    """Map a packed bundle into the pixel-index frame of the affine detector ``(J, o)``.

    Writing the offsets as index offsets ``s = J^-1 (r - m)`` maps every column exactly::

        m' = J^-1 (m - o)     a' = J^T a     P' = J^T P J     Q' = J^T Q J

    so kernels see integer pixel indices only, and derivatives with respect to the detector
    pose (rotation, flip, pixel size, centre) are ordinary autodiff of this map.

    Parameters
    ----------
    packed : array, shape (N, 12)
        Packed bundle in metre space.
    j : array, shape (2, 2)
        Pixel-to-metre matrix.
    o : array, shape (2,)
        Position of pixel ``(0, 0)`` in metres.

    Returns
    -------
    array, shape (N, 12)
        The bundle in pixel-index space.
    """
    packed = jnp.asarray(packed, jnp.float64)
    m = packed[:, [MX, MY]]
    a = packed[:, [AX, AY]]
    j_inv = jnp.linalg.inv(j)
    m_new = (m - o) @ j_inv.T
    a_new = a @ j
    phase = jnp.einsum("ji,njk,kl->nil", j, _form(packed[:, PXX], packed[:, PXY], packed[:, PYY]), j)
    decay = jnp.einsum("ji,njk,kl->nil", j, _form(packed[:, QXX], packed[:, QXY], packed[:, QYY]), j)
    out = packed
    out = out.at[:, MX].set(m_new[:, 0]).at[:, MY].set(m_new[:, 1])
    out = out.at[:, AX].set(a_new[:, 0]).at[:, AY].set(a_new[:, 1])
    out = out.at[:, PXX].set(phase[:, 0, 0]).at[:, PXY].set(2 * phase[:, 0, 1])
    out = out.at[:, PYY].set(phase[:, 1, 1])
    out = out.at[:, QXX].set(decay[:, 0, 0]).at[:, QXY].set(2 * decay[:, 0, 1])
    return out.at[:, QYY].set(decay[:, 1, 1])
