"""Sampling lattices of sampled wavefronts.

A field is an array ``u[..., iy, ix]``. Its samples sit at the physical positions

    r(ix, iy) = centre + pitch @ (mu_x, mu_y),   mu = index - (n - 1) / 2,

with ``r = (x, y)`` in the order of ``run.column_matrix`` (``[x, y, theta_x, theta_y, 1]``).
``pitch`` is a full 2x2 matrix whose columns are the lattice vectors of the x-index and the
y-index, so shear, rotation and anisotropy are all expressible (``temgym_core.grid.Grid`` cannot
express shear and acts on ``[y, x, 1]``). The centre convention ``(n - 1) / 2`` matches ``Grid``.

Nothing here interpolates: mapping a lattice through an affine ray map, or adapting one lattice
to another through a matrix, is bookkeeping on ``pitch`` and ``centre`` only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from ._terms import inv2, split_matrix

__all__ = [
    "Lattice",
    "adapted_input_lattice",
    "adapted_output_lattice",
    "near_image_lattice",
    "near_image_input_lattice",
    "lattice_chain",
    "as_lattice",
    "is_concrete",
]


def is_concrete(*values) -> bool:
    """Return whether none of ``values`` is an abstract tracer, so host-side checks are possible.

    Parameters
    ----------
    *values
        Arrays or scalars.
    """
    return not any(isinstance(v, jax.core.Tracer) for v in values)


def _rotation(angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]])


@jdc.pytree_dataclass
class Lattice:
    """Affine sampling lattice of a field of shape ``(ny, nx)``.

    Parameters
    ----------
    pitch : (2, 2) array
        Metres per index: ``r = centre + pitch @ [mu_x, mu_y]``. The columns are the x- and
        y-index vectors.
    centre : (2,) array
        Physical ``(x, y)`` of the geometric centre of the array, metres. This is the position
        of the index ``((nx - 1) / 2, (ny - 1) / 2)``.
    shape : tuple[int, int]
        Static array shape ``(ny, nx)``.
    """

    pitch: jnp.ndarray
    centre: jnp.ndarray
    shape: jdc.Static[tuple]

    @classmethod
    def regular(cls, shape, pitch, centre=(0.0, 0.0), rotation=0.0):
        """Build a Cartesian lattice, optionally rotated.

        Parameters
        ----------
        shape : tuple[int, int]
            Array shape ``(ny, nx)``.
        pitch : float or (2,) sequence
            Sample spacing ``(px, py)`` in metres, x first. A scalar sets both.
        centre : (2,) sequence, default (0, 0)
            Physical ``(x, y)`` of the array centre, metres.
        rotation : float, default 0
            Rotation of the lattice about its centre, degrees, counter-clockwise. The lattice is
            right handed: ``iy`` runs towards ``+y``. This is not the convention of
            :class:`temgym_core.grid.Grid`, whose rows run towards ``-y`` and whose rotation is
            clockwise; use :meth:`from_grid` for a ``Detector``.

        Returns
        -------
        Lattice
        """
        px, py = (pitch, pitch) if np.ndim(pitch) == 0 else pitch
        matrix = _rotation(jnp.deg2rad(rotation)) @ jnp.diag(jnp.asarray([px, py], dtype=jnp.float64))
        return cls(pitch=matrix, centre=jnp.asarray(centre, dtype=jnp.float64),
                   shape=(int(shape[0]), int(shape[1])))

    @classmethod
    def from_grid(cls, grid):
        """Return the lattice of a :class:`temgym_core.grid.Grid` (``Detector``, ``ScanGrid``).

        ``Grid`` maps ``[y_px, x_px, 1]`` to ``[y_m, x_m, 1]`` (rows first, degrees, optional
        flip); the pitch matrix is re-expressed here in ``(x, y)`` order acting on ``(ix, iy)``.
        This is axis-order bookkeeping only.

        Parameters
        ----------
        grid : Grid
            Any object with ``shape`` and ``pixels_to_metres_mat``.

        Returns
        -------
        Lattice
        """
        T = jnp.asarray(grid.pixels_to_metres_mat)
        ny, nx = grid.shape
        pitch = jnp.array([[T[1, 1], T[1, 0]], [T[0, 1], T[0, 0]]])
        cy, cx = (ny - 1) / 2, (nx - 1) / 2
        ym = T[0, 0] * cy + T[0, 1] * cx + T[0, 2]
        xm = T[1, 0] * cy + T[1, 1] * cx + T[1, 2]
        return cls(pitch=pitch, centre=jnp.stack((xm, ym)), shape=(int(ny), int(nx)))

    @property
    def ny(self) -> int:
        """Number of rows."""
        return self.shape[0]

    @property
    def nx(self) -> int:
        """Number of columns."""
        return self.shape[1]

    @property
    def cell_area(self):
        """Area of one sample cell, ``|det pitch|``, square metres."""
        return jnp.abs(jnp.linalg.det(self.pitch))

    def mu(self):
        """Return the centred index vectors ``(mu_x (nx,), mu_y (ny,))``, dimensionless."""
        mu_x = jnp.arange(self.nx, dtype=jnp.float64) - (self.nx - 1) / 2
        mu_y = jnp.arange(self.ny, dtype=jnp.float64) - (self.ny - 1) / 2
        return mu_x, mu_y

    def coords(self):
        """Return the physical ``(x, y)`` of every sample, two arrays of shape ``(ny, nx)``, metres."""
        mu_x, mu_y = self.mu()
        mx, my = jnp.meshgrid(mu_x, mu_y)  # (ny, nx), indexing='xy'
        x = self.centre[0] + self.pitch[0, 0] * mx + self.pitch[0, 1] * my
        y = self.centre[1] + self.pitch[1, 0] * mx + self.pitch[1, 1] * my
        return x, y

    def positions(self):
        """Return the physical positions as an ``(ny, nx, 2)`` array of ``(x, y)``, metres."""
        x, y = self.coords()
        return jnp.stack((x, y), axis=-1)

    def mapped(self, matrix, offset=None):
        """Return the lattice of the samples after the affine map ``r -> matrix @ r + offset``.

        Parameters
        ----------
        matrix : (2, 2) array
            Linear part of the position map.
        offset : (2,) array, optional
            Position offset, metres.

        Returns
        -------
        Lattice
            Same shape; ``pitch`` becomes ``matrix @ pitch``. No resampling takes place.
        """
        matrix = jnp.asarray(matrix, dtype=jnp.float64)
        centre = matrix @ self.centre
        if offset is not None:
            centre = centre + jnp.asarray(offset, dtype=jnp.float64)
        return Lattice(pitch=matrix @ self.pitch, centre=centre, shape=self.shape)

    def with_shape(self, shape):
        """Return the same pitch and centre with a new array ``shape`` ``(ny, nx)``."""
        return Lattice(pitch=self.pitch, centre=self.centre, shape=(int(shape[0]), int(shape[1])))

    def with_centre(self, centre):
        """Return the same pitch and shape with a new ``centre`` ``(x, y)``, metres."""
        return Lattice(pitch=self.pitch, centre=jnp.asarray(centre, dtype=jnp.float64),
                       shape=self.shape)


def as_lattice(obj) -> Lattice:
    """Return ``obj`` as a :class:`Lattice`, converting a ``Grid`` such as a ``Detector``.

    Parameters
    ----------
    obj : Lattice or Grid
        A lattice, or an object with ``shape`` and ``pixels_to_metres_mat``.

    Returns
    -------
    Lattice
    """
    if isinstance(obj, Lattice):
        return obj
    if hasattr(obj, "pixels_to_metres_mat") and hasattr(obj, "shape"):
        return Lattice.from_grid(obj)
    raise TypeError(f"expected a Lattice or a Grid such as Detector, got {type(obj).__name__}")


def adapted_input_lattice(B, wavelength, out_lattice: Lattice, gamma, in_shape, in_centre=(0.0, 0.0)):
    """Return the input lattice that makes the Collins cross term diagonal for an output lattice.

    ``Gamma = (1/lambda) P'^T B^{-T} P = diag(gamma)`` gives ``P = lambda B^T P'^{-T} diag(gamma)``.
    ``gamma`` is the free per-axis parameter: with ``gamma_j = 1 / n_j`` the transform is an exact
    DFT along axis ``j``.

    Parameters
    ----------
    B : (2, 2) array
        Slope-to-position block of the ray matrix, m/rad.
    wavelength : float
        Wavelength, metres.
    out_lattice : Lattice
        Lattice of the transformed field.
    gamma : (2,) sequence
        Cross-term parameters ``(gamma_x, gamma_y)``, cycles per sample squared.
    in_shape : tuple[int, int]
        Shape ``(ny, nx)`` of the input array.
    in_centre : (2,) sequence, default (0, 0)
        Centre ``(x, y)`` of the input lattice, metres.

    Returns
    -------
    Lattice
    """
    B = jnp.asarray(B, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    pitch = wavelength * B.T @ jnp.linalg.inv(out_lattice.pitch).T @ jnp.diag(gamma)
    return Lattice(pitch=pitch, centre=jnp.asarray(in_centre, dtype=jnp.float64),
                   shape=(int(in_shape[0]), int(in_shape[1])))


def adapted_output_lattice(B, wavelength, in_lattice: Lattice, gamma, out_shape, out_centre=(0.0, 0.0)):
    """Return the output lattice that makes the Collins cross term diagonal for an input lattice.

    ``P' = lambda B P^{-T} diag(gamma)``. For ``gamma_j = 1 / n_j`` this is the natural DFT output
    sampling ``lambda |B| / (n p)``, rotated and sheared with ``B``.

    Parameters
    ----------
    B : (2, 2) array
        Slope-to-position block of the ray matrix, m/rad.
    wavelength : float
        Wavelength, metres.
    in_lattice : Lattice
        Lattice of the input field.
    gamma : (2,) sequence
        Cross-term parameters ``(gamma_x, gamma_y)``, cycles per sample squared.
    out_shape : tuple[int, int]
        Shape ``(ny, nx)`` of the output array.
    out_centre : (2,) sequence, default (0, 0)
        Centre ``(x, y)`` of the output lattice, metres.

    Returns
    -------
    Lattice
    """
    B = jnp.asarray(B, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    pitch = wavelength * B @ jnp.linalg.inv(in_lattice.pitch).T @ jnp.diag(gamma)
    return Lattice(pitch=pitch, centre=jnp.asarray(out_centre, dtype=jnp.float64),
                   shape=(int(out_shape[0]), int(out_shape[1])))


def _near_image_map(variant, A, D):
    if variant == "I":
        return jnp.asarray(A, dtype=jnp.float64)
    if variant == "II":
        return inv2(jnp, jnp.asarray(D, dtype=jnp.float64)).T
    raise ValueError(f"unknown near-image variant {variant!r}")


def near_image_lattice(variant, A, D, a, in_lattice: Lattice) -> Lattice:
    """Return the output lattice of a near-image branch, ``A P`` or ``D^-T P``, by bookkeeping.

    Parameters
    ----------
    variant : {'I', 'II'}
        Drift-first (lattice ``A P``) or drift-last (lattice ``D^-T P``) factorisation.
    A, D : (2, 2) arrays
        Position-to-position and slope-to-slope blocks of the ray matrix.
    a : (2,) array
        Position offset of the ray matrix, metres.
    in_lattice : Lattice
        Lattice of the input field.

    Returns
    -------
    Lattice
    """
    return in_lattice.mapped(_near_image_map(variant, A, D), a)


def near_image_input_lattice(variant, A, D, a, out_lattice: Lattice) -> Lattice:
    """Return the input lattice that a near-image branch maps onto ``out_lattice``.

    This inverts :func:`near_image_lattice`: for the drift-first variant ``P = A^-1 P'`` and the
    centre is ``c = A^-1 (c' - a)``.

    Parameters
    ----------
    variant : {'I', 'II'}
        Drift-first or drift-last factorisation.
    A, D : (2, 2) arrays
        Position-to-position and slope-to-slope blocks of the ray matrix.
    a : (2,) array
        Position offset of the ray matrix, metres.
    out_lattice : Lattice
        Lattice of the transformed field.

    Returns
    -------
    Lattice
    """
    inverse = inv2(jnp, _near_image_map(variant, A, D))
    centre = inverse @ (out_lattice.centre - jnp.asarray(a, dtype=jnp.float64))
    return Lattice(pitch=inverse @ out_lattice.pitch, centre=centre, shape=out_lattice.shape)


def lattice_chain(matrices, wavelength, out_lattice: Lattice, methods, *, gammas=None, shapes=None,
                  centres=None):
    """Compute the lattices at every plane of a chain of spans, from the last plane backwards.

    A detector lattice fixes the lattice of the field before the last span (adapted to it), that
    one fixes the one before, and so on up to the source. A field generated on the first lattice
    then lands exactly on ``out_lattice``.

    Parameters
    ----------
    matrices : sequence of (5, 5) or (4, 4) arrays
        Ray matrices of the spans in propagation order (``run.column_matrix``).
    wavelength : float
        Wavelength, metres.
    out_lattice : Lattice
        Lattice after the last span.
    methods : sequence of {'exact', 'near_image', 'near_image_II', 'mft'}
        Branch of each span. The near-image branches map the lattice (``A P`` or ``D^-T P``);
        ``'mft'`` adapts it to the following lattice.
    gammas : sequence of (2,) or None, optional
        MFT cross-term parameters ``(gamma_x, gamma_y)`` per span; ``None`` entries mean
        ``(1/nx, 1/ny)`` of the following lattice, an exact DFT.
    shapes : sequence of (ny, nx) or None, optional
        Shape of the lattice before each MFT span; ``None`` means the shape after it.
    centres : sequence of (2,) or None, optional
        Centre, metres, of the lattice before each MFT span; ``None`` means the origin.

    Returns
    -------
    tuple[Lattice, ...]
        ``len(matrices) + 1`` lattices, the source lattice first and ``out_lattice`` last.
    """
    n = len(matrices)
    if len(methods) != n:
        raise ValueError(f"need one method per span: {len(methods)} methods for {n} spans")

    def entry(seq, i):
        return None if seq is None or seq[i] is None else seq[i]

    lattices = [None] * (n + 1)
    lattices[n] = out_lattice
    for i in reversed(range(n)):
        A, B, C, D, a, b = split_matrix(matrices[i])
        after = lattices[i + 1]
        method = methods[i]
        if method in ("exact", "near_image"):
            lattices[i] = near_image_input_lattice("I", A, D, a, after)
        elif method == "near_image_II":
            lattices[i] = near_image_input_lattice("II", A, D, a, after)
        elif method == "mft":
            shape = entry(shapes, i) or after.shape
            gamma = entry(gammas, i) or (1.0 / after.nx, 1.0 / after.ny)
            centre = entry(centres, i)
            lattices[i] = adapted_input_lattice(B, wavelength, after, gamma, shape,
                                                (0.0, 0.0) if centre is None else centre)
        else:
            raise ValueError(f"unknown method {method!r}")
    return tuple(lattices)
