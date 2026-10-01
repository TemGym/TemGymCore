"""Wave propagation through a column of components: masks and linear spans.

A column is a sequence of components. :func:`propagate` splits it at every thin
``GaussianActionComponent`` that is not linear (an aperture, a phase plate, a biprism, a specimen):
that element becomes a multiplicative mask on the field, and the linear elements between two masks
become one Collins transform each, planned from the 5x5 ray matrix of the span
(:func:`~temgym_core.run.column_matrix`). The physical phase of each span, which the matrix does not
fix, comes from a chief Gaussian propagated through the same components
(:func:`column_transform`), so the Gouy and Maslov conventions of the engine never reach the user.

The field is the paraxial wave amplitude with the axial carrier ``exp(i k (z - z0))`` removed, like
the field of a Gaussian beamlet: ``|data|**2 * lattice.cell_area`` summed over the array is the flux
and is conserved by propagation.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from ..components import HardEdgeSolenoid, Plane
from ..config import require_x64
from ..constants import energy2wavelength
from ..gaussian import FreeSpacePropagator, GaussianBeam, make_gaussian
from ..run import column_matrix, run_to_end
from ._terms import split_matrix
from .lattice import (
    Lattice,
    as_lattice,
    is_concrete,
    lattice_chain,
    near_image_input_lattice,
)
from .lct import _f_length, _lattice_close, apply_lct, plan_lct
from .masks import _commutes_with_masks, component_mask, split_component
from .planner import LatticeError, check_near_image
from .reference import chief_phase0

__all__ = ["Wavefront", "ColumnTransform", "column_transform", "propagate", "input_lattice"]

_CHIEF_WAIST = 1e-6           # metres; phase0 does not depend on it (a test asserts that)
_MISMATCH_TOLERANCE = 1e-6    # dimensionless, see reference.chief_phase0


@jdc.pytree_dataclass
class Wavefront:
    """A complex field sampled on a lattice at one plane of a column.

    Parameters
    ----------
    data : complex array, shape (..., ny, nx)
        The field, in (row, column) = (y, x) order; leading axes are batch axes. The axial carrier
        ``exp(i k (z - z0))`` is not included. ``sum(|data|**2) * lattice.cell_area`` is the flux.
    lattice : Lattice
        Where the samples sit; see :class:`~temgym_core.wave.lattice.Lattice`.
    z : float
        Axial position of the plane, metres.
    voltage : float
        Accelerating voltage, volts; it sets the wavelength.
    """

    data: jnp.ndarray
    lattice: Lattice
    z: float
    voltage: float

    @property
    def wavelength(self):
        """Electron wavelength, metres."""
        return energy2wavelength(self.voltage)


class ColumnTransform(NamedTuple):
    """The linear description of a span of components: ray matrix, chief Gaussian and phase.

    Parameters
    ----------
    matrix : (5, 5) array
        Ray matrix on ``[x, y, theta_x, theta_y, 1]`` (``run.column_matrix``).
    chief_in, chief_out : GaussianBeam
        The chief Gaussian before and after the span. ``chief_out.pathlength`` is the excess optical
        path (the Hamilton action of the chief ray) and its amplitude carries the continuous Gouy
        phase.
    phase0 : float
        Constant phase, rad: pass it to :func:`~temgym_core.wave.lct.apply_lct`. It is 0 or pi for an
        offset-free span and includes the Hamilton action of the chief ray otherwise.
    mismatch : float
        Dimensionless disagreement between the matrix and the chief; about 1e-12 for a consistent
        span (see :func:`~temgym_core.wave.reference.chief_phase0`).
    """

    matrix: jnp.ndarray
    chief_in: GaussianBeam
    chief_out: GaussianBeam
    phase0: jnp.ndarray
    mismatch: jnp.ndarray


def _require_first_half_period(component):
    """A hard-edge solenoid's coherent (Gouy) phase is only continued over |q| < pi."""
    if isinstance(component, HardEdgeSolenoid) and is_concrete(component.q):
        q = float(np.max(np.abs(np.asarray(component.q))))
        if q >= np.pi:
            raise ValueError(
                f"HardEdgeSolenoid with |q| = {q:.3g} >= pi: the coherent phase is only continued over the "
                "first radial half period; split the field into bodies with |q| < pi")


def column_transform(components, source_z=0.0, voltage=200e3, *, chief=None) -> ColumnTransform:
    """Return the ray matrix, chief Gaussian and constant phase of a linear span.

    The span must contain only linear elements (planes, lenses, stigmators, deflectors, rotators,
    solenoids, magnetic lenses); use :func:`propagate` for a column with masks. The function is
    pure JAX: it differentiates through the component parameters and the voltage, and ``vmap``
    over the voltage batches the energy spread (one matrix and one ``phase0`` per wavelength).

    Parameters
    ----------
    components : sequence of Component
        Elements ordered along ``z``.
    source_z : float, default 0
        Axial position where the span starts, metres.
    voltage : float, default 200e3
        Accelerating voltage, volts.
    chief : GaussianBeam, optional
        Chief Gaussian at ``source_z``. The default is on axis with zero slope and a 1 um waist;
        the result does not depend on it for a linear span.

    Returns
    -------
    ColumnTransform

    Raises
    ------
    ValueError
        If a component is not linear, or a ``HardEdgeSolenoid`` has ``|q| >= pi``.
    NotImplementedError
        If a component cannot be used in a wave column.
    """
    require_x64("column_transform")
    components = tuple(components)
    for component in components:
        _, mask = split_component(component)
        if mask is not None:
            raise ValueError(f"{type(component).__name__} acts as a mask, not as part of one Collins "
                             "transform; use propagate(), which splits the column at masks")
        _require_first_half_period(component)
    matrix = column_matrix(components, source_z=source_z, voltage=voltage)
    if chief is None:
        chief = make_gaussian(z=source_z, voltage=voltage, waist_x=_CHIEF_WAIST, waist_y=_CHIEF_WAIST)
    chief_out = run_to_end(chief, components, propagator=FreeSpacePropagator())
    phase0, mismatch = chief_phase0(matrix, energy2wavelength(voltage), chief, chief_out)
    return ColumnTransform(matrix, chief, chief_out, phase0, mismatch)


def _split_column(components):
    """Split a column into ``('span', [linear...])`` and ``('mask', mask)`` steps.

    Thin position-only elements (a lens, a deflector) at the plane of a mask commute with it, so they
    are applied after the mask: the mask then sees the field before, not after, the lens chirp, which
    a lattice that resolves the field can hold.
    """
    steps, current = [], []
    for component in components:
        linear, mask = split_component(component)
        if linear is not None:
            current.append(linear)
        if mask is not None:
            z, held = float(mask.z), []
            while current and _commutes_with_masks(current[-1], z):
                held.insert(0, current.pop())
            steps.append(("span", current + [Plane(z=z)]))
            steps.append(("mask", mask))
            current = held
    if current:
        steps.append(("span", current))
    return steps


def _is_zero_drift(span, z):
    return len(span) == 1 and isinstance(span[0], Plane) and float(span[0].z) == z


def _transforms(steps, source_z, voltage):
    """Attach the :class:`ColumnTransform` of every span; zero-length spans are dropped."""
    out, z = [], float(source_z)
    for kind, payload in steps:
        if kind == "mask":
            out.append((kind, payload))
        elif not _is_zero_drift(payload, z):
            transform = column_transform(payload, z, voltage)
            if is_concrete(transform.mismatch) and float(transform.mismatch) > _MISMATCH_TOLERANCE:
                raise ValueError(
                    f"the ray matrix and the Gaussian chief disagree by {float(transform.mismatch):.3g} over "
                    f"the span ending at z = {float(transform.chief_out.z):.6g} m: the ray and Gaussian branches "
                    "of one of its components are inconsistent (for example ElectromagneticLens with x0/y0)")
            z = float(transform.chief_out.z)
            out.append((kind, transform))
    return out, z


def _pick_method(matrix, wavelength, out_lattice, requested):
    """Choose the branch of one span from its output lattice, on the assumed field support and band."""
    if requested != "auto":
        return requested
    A, B, C, D, a, b = split_matrix(matrix)
    for variant, method in (("I", "near_image"), ("II", "near_image_II")):
        if abs(float(np.linalg.det(np.asarray(A if variant == "I" else D)))) < 1e-9:
            continue
        in_lattice = near_image_input_lattice(variant, A, D, a, out_lattice)
        if variant == "I" and float(np.abs(np.asarray(B)).max()) <= 1e-13 * _f_length(in_lattice, wavelength):
            return "exact"
        if check_near_image(variant, A, B, C, D, wavelength, in_lattice, a, b).ok:
            return method
    return "mft"


def _plan_chain(transforms, wavelength, out_lattice, method, in_shape, gamma=None):
    """Choose the branch of every span and return the lattices before each of them, backwards."""
    spans = [t for kind, t in transforms if kind == "span"]
    methods = _per_span(method, len(spans))
    lattice, chosen = out_lattice, [None] * len(spans)
    for i in reversed(range(len(spans))):
        requested = "mft" if i == 0 and gamma is not None and methods[i] == "auto" else methods[i]
        chosen[i] = _pick_method(spans[i].matrix, wavelength, lattice, requested)
        lattice = lattice_chain([spans[i].matrix], wavelength, lattice, [chosen[i]],
                                centres=[np.asarray(spans[i].chief_in.r_xy)])[0]
    shapes = [in_shape if i == 0 else None for i in range(len(spans))]
    gammas = [gamma if i == 0 else None for i in range(len(spans))]
    chain = lattice_chain([t.matrix for t in spans], wavelength, out_lattice, chosen, shapes=shapes,
                          gammas=gammas, centres=[np.asarray(t.chief_in.r_xy) for t in spans])
    return chain, chosen


def _per_span(method, count):
    if isinstance(method, str):
        return [method] * count
    if len(method) != count:
        raise ValueError(f"method has {len(method)} entries for {count} linear spans")
    return list(method)


def input_lattice(components, out_lattice, *, source_z=0.0, voltage=200e3, in_shape=None, gamma=None,
                  method="auto") -> Lattice:
    """Return the lattice a field must be sampled on so that it lands exactly on ``out_lattice``.

    The lattices of a column are adapted backwards from the last plane: the near-image branches map
    the lattice (``A P`` or ``D^-T P``, no interpolation) and the matrix Fourier transform adapts it
    to the following lattice. A source generated on the returned lattice, or an imported array
    resampled onto it (:func:`~temgym_core.wave.resample.resample_field`), then propagates to
    ``out_lattice`` without further interpolation.

    Parameters
    ----------
    components : sequence of Component
        The column, ordered along ``z``.
    out_lattice : Lattice or Grid
        Lattice of the field after the column, for example a ``Detector``.
    source_z : float, default 0
        Axial position of the source plane, metres.
    voltage : float, default 200e3
        Accelerating voltage, volts.
    in_shape : tuple[int, int], optional
        Array shape ``(ny, nx)`` of the source when the first span is a matrix Fourier transform
        (default: the shape of ``out_lattice``). Near-image spans keep the shape.
    gamma : (2,) sequence, optional
        Cross-term parameters ``(gamma_x, gamma_y)`` of the first span when it is a matrix Fourier
        transform (default ``(1/nx, 1/ny)`` of the following lattice, an exact DFT). A smaller
        ``gamma`` samples the source more finely and covers a smaller part of the output window;
        the source pitch is ``lambda |B| gamma / p_out``.
    method : str or sequence of str, default 'auto'
        Branch per linear span; ``'auto'`` picks it from the assumed field support and band.

    Returns
    -------
    Lattice
    """
    require_x64("input_lattice")
    out_lattice = as_lattice(out_lattice)
    transforms, _ = _transforms(_split_column(tuple(components)), source_z, voltage)
    chain, _ = _plan_chain(transforms, energy2wavelength(voltage), out_lattice, method, in_shape, gamma)
    return chain[0]


_apply_span = jax.jit(apply_lct)


@jax.jit
def _apply_mask_jit(mask, lattice, data, wavelength):
    return data * component_mask(mask, lattice, wavelength)


def _apply_mask(mask, lattice, data, wavelength):
    try:
        return _apply_mask_jit(mask, lattice, data, wavelength)
    except TypeError:  # a component holding a non-array leaf cannot cross the jit boundary
        return data * component_mask(mask, lattice, wavelength)


def _field_chief(data, lattice, wavelength, matrix):
    """Where the field's intensity centroid lands: ``A r + B theta + a`` for its centroid and mean slope."""
    u = np.asarray(data).reshape((-1,) + tuple(lattice.shape))
    intensity = np.sum(np.abs(u) ** 2, axis=0)
    total = intensity.sum()
    A, B, _, _, a, _ = map(np.asarray, split_matrix(matrix))
    if total == 0:
        return tuple(A @ np.asarray(lattice.centre) + a)
    x, y = map(np.asarray, lattice.coords())
    centroid = np.array([(x * intensity).sum(), (y * intensity).sum()]) / total
    frequency = np.array([np.angle(np.sum(np.conj(u[:, :, :-1]) * u[:, :, 1:])),
                          np.angle(np.sum(np.conj(u[:, :-1, :]) * u[:, 1:, :]))]) / (2 * np.pi)
    slopes = wavelength * np.linalg.inv(np.asarray(lattice.pitch)).T @ frequency
    return tuple(A @ centroid + B @ slopes + a)


def propagate(field: Wavefront, components: Sequence, out_lattice=None, *, check: str = "raise",
              method="auto", tol: float = 1e-16, return_reports: bool = False):
    """Propagate a wavefront through a column of components.

    The column is split at every thin ``GaussianActionComponent`` that is not linear: it multiplies
    the field on its lattice (the exact action its Gaussian branch approximates), and the linear
    elements between two masks form one Collins transform each, planned from their ray matrix with
    the physical phase from a chief Gaussian. ``KrivanekLens`` splits into its ideal lens (linear)
    and its aberration (a mask). Components that are neither raise ``NotImplementedError``.

    Parameters
    ----------
    field : Wavefront
        The field at ``field.z``.
    components : sequence of Component
        The column, ordered along ``z``. End it with a ``Plane`` or ``Detector`` to choose the
        output plane.
    out_lattice : Lattice or Grid, optional
        Lattice of the returned field. If omitted the engine chooses it (the natural lattice of
        each branch, centred on the beam). If given and the column has one linear span, the two
        lattices are used as they are (the transform must be adapted to them, else
        :class:`LatticeError`); with several spans ``field.lattice`` must be the lattice that
        lands exactly on ``out_lattice``, :func:`input_lattice`, and the lattices between the spans
        follow from it. To choose an intermediate lattice, call ``propagate`` once per span.
    check : {'raise', 'report', 'off'}, default 'raise'
        Sampling checks of every span, see :func:`~temgym_core.wave.lct.plan_lct`. ``'raise'`` raises
        :class:`SamplingError` listing the failing inequality; ``'off'`` is the expert override.
    method : str or sequence of str, default 'auto'
        Branch of every linear span (``'mft'``, ``'near_image'``, ``'near_image_II'``, ``'exact'``),
        or one per span.
    tol : float, default 1e-16
        Fraction of the field's energy allowed outside the support and band the planner measures at
        the start of each span; the transform error is of order ``sqrt(tol)``.
    return_reports : bool, default False
        Also return the :class:`~temgym_core.wave.planner.SamplingReport` of each span.

    Returns
    -------
    Wavefront
        The field at the end of the column, on the lattice the engine returns. With
        ``return_reports`` a tuple ``(wavefront, reports)``.

    Raises
    ------
    SamplingError
        If a span would alias (with ``check='raise'``). Split the column at an intermediate plane:
        propagate to it, then continue from the returned wavefront.
    LatticeError
        If ``field.lattice`` is not the lattice adapted to ``out_lattice``.
    ValueError
        If the ray matrix and the chief Gaussian of a span disagree.

    Notes
    -----
    Planning reads concrete values (the ray matrices, the support and band of the field), so
    ``propagate`` itself cannot be jitted or vmapped, but it is differentiable in the field when the
    branches are fixed (``check='off'``). For gradients with respect to component parameters or
    the voltage, and for ``vmap`` over an energy spread, use ``column_transform``, ``plan_lct(method=...)``
    and ``apply_lct``, which are pure JAX.
    """
    require_x64("propagate")
    if check not in ("raise", "report", "off"):
        raise ValueError(f"check must be 'raise', 'report' or 'off', got {check!r}")
    wavelength = float(field.wavelength)
    steps = _split_column(tuple(components))
    transforms, z_end = _transforms(steps, field.z, field.voltage)
    count = sum(kind == "span" for kind, _ in transforms)
    if out_lattice is not None:
        out_lattice = as_lattice(out_lattice)
        if count == 1:
            chain, methods = (field.lattice, out_lattice), _per_span(method, count)
        else:
            chain, methods = _plan_chain(transforms, wavelength, out_lattice, method, None)
        if not _lattice_close(field.lattice, chain[0]):
            raise LatticeError(
                "the field is not sampled on the lattice that lands exactly on out_lattice: expected "
                f"pitch {np.asarray(chain[0].pitch).tolist()}, centre {np.asarray(chain[0].centre).tolist()}, "
                f"shape {chain[0].shape}. Generate the source on temgym_core.wave.input_lattice(...) or "
                "resample it onto that lattice with temgym_core.wave.resample_field")
    else:
        chain, methods = None, _per_span(method, count)

    data, lattice, reports, index = field.data, field.lattice, [], 0
    for kind, payload in transforms:
        if kind == "mask":
            data = _apply_mask(payload, lattice, data, wavelength)
            continue
        target = None if chain is None else chain[index + 1]
        centre = (_field_chief(data, lattice, wavelength, payload.matrix)
                  if chain is None and is_concrete(data) else None)
        plan, report = plan_lct(payload.matrix, wavelength, lattice, target, method=methods[index],
                                field=data, tol=tol, check=check, out_centre=centre,
                                return_report=True)
        data = _apply_span(plan, data, payload.phase0)
        lattice = plan.out_lattice
        reports.append(report)
        index += 1
    result = Wavefront(data=data, lattice=lattice, z=z_end, voltage=field.voltage)
    return (result, reports) if return_reports else result
