"""Thin components as multiplicative masks on a lattice, and the split of a column into roles.

A thin :class:`~temgym_core.components.GaussianActionComponent` has a complex action
``complex_action(xy, k) = phase_shift(xy) - i log(T(xy)) / k`` and multiplies a field by
``exp(i k complex_action)``: this is exactly the action its Gaussian branch applies, which expands it
to second order around each beamlet's centre. On a lattice it is evaluated at every sample by
``vmap``. Linear elements (lenses, deflectors, rotations, solenoids) are not masks: they join the
Collins transform of the span between two masks through ``run.column_matrix``.

Elements that are neither are refused with a clear error: ``SeidelLens`` and ``DistortedLens`` have an
action that depends on the ray slope (a non-local kernel, not a function of position), ``ThickLens``
has no wave branch, and so on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from ..components import (
    ABCDTransfer,
    ConstantPhaseShift,
    Descanner,
    Detector,
    Deflector,
    DoubleDeflector,
    ElectromagneticLens,
    FourierTransform,
    GaussianActionComponent,
    HardEdgeSolenoid,
    LinearPhaseShift,
    Lens,
    Plane,
    QuadraticPhaseShift,
    RotatingLens,
    Rotator,
    ScanGrid,
    Scanner,
    SeidelLens,
    Stigmator,
    ThickLens,
)
from ..components._base import _uses_default_log_transmission
from .lattice import Lattice

__all__ = ["component_mask", "split_component", "is_linear"]

# Thin elements that only multiply the field by a function of position (or do nothing), so they commute
# with a mask at the same plane.
_MULTIPLICATIVE_TYPES = (
    Plane, ScanGrid, Detector, Lens, Stigmator, Deflector,
    ConstantPhaseShift, LinearPhaseShift, QuadraticPhaseShift,
)

_LINEAR_TYPES = (
    Plane, ScanGrid, Detector, Lens, Stigmator, Deflector, DoubleDeflector, Rotator, RotatingLens,
    ElectromagneticLens, HardEdgeSolenoid, Scanner, Descanner, FourierTransform,
    ConstantPhaseShift, LinearPhaseShift, QuadraticPhaseShift,
)


@jdc.pytree_dataclass
class _ResidualAction:
    """The action of a thin lens subclass beyond its ideal lens (for example the aberrations)."""

    component: GaussianActionComponent
    ideal: GaussianActionComponent
    z: float

    def complex_action(self, xy, k):
        """Complex action of the subclass minus the path of the ideal lens it contains."""
        return self.component.complex_action(xy, k) - self.ideal.phase_shift(xy)


def _commutes_with_masks(component, z) -> bool:
    """Whether ``component`` is a thin, position-only element at plane ``z``.

    Such an element acts on the field as a multiplication by a function of position, exactly like a
    mask, so at one plane the two may be applied in either order. Rotations, shifts and thick
    elements do not qualify.
    """
    return type(component) in _MULTIPLICATIVE_TYPES and float(getattr(component, "z", np.nan)) == float(z)


def _overrides_action(component, base):
    return (type(component).phase_shift is not base.phase_shift
            or not _uses_default_log_transmission(component))


def is_linear(component) -> bool:
    """Return whether a component is a linear element of a Collins span, not a mask.

    Parameters
    ----------
    component : Component
        Any column element.

    Returns
    -------
    bool
        ``True`` for planes, thin round or astigmatic lenses, deflectors, rotators, solenoids and
        magnetic lenses; ``False`` for masks such as apertures, phase plates and specimens.
    """
    return split_component(component)[1] is None


def split_component(component):
    """Split a column element into its linear part and its mask part.

    Parameters
    ----------
    component : Component
        Any element of a column.

    Returns
    -------
    linear : Component or None
        The part that joins the Collins transform of the surrounding span. For a ``Lens`` subclass
        that adds an action of its own (``KrivanekLens``) this is the ideal thin lens.
    mask : object or None
        The part that multiplies the field: the thin ``GaussianActionComponent`` itself or, for such
        a lens subclass, its action beyond the ideal lens (the aberrations). Has ``z`` and
        ``complex_action(xy, k)``.

    Raises
    ------
    NotImplementedError
        If the component is neither linear nor a thin ``GaussianActionComponent``, or its action is
        not a function of position alone.
    """
    if isinstance(component, SeidelLens):
        raise NotImplementedError(
            f"{type(component).__name__} has an action that depends on the ray slope (a non-local "
            "kernel), so it is not a mask on a plane; propagate Gaussian beamlets through it instead.")
    if isinstance(component, ThickLens):
        raise NotImplementedError("ThickLens is a ray-only principal-plane model with no wave branch; "
                                  "use Lens (thin) or ElectromagneticLens.")
    for base in (Lens, Stigmator):
        if isinstance(component, base):
            if not _overrides_action(component, base):
                return component, None
            ideal = (Lens(z=component.z, focal_length=component.focal_length, x0=component.x0,
                          y0=component.y0) if base is Lens else
                     Stigmator(z=component.z, focal_length_x=component.focal_length_x,
                               focal_length_y=component.focal_length_y, x0=component.x0, y0=component.y0))
            return ideal, _ResidualAction(component=component, ideal=ideal, z=component.z)
    if type(component) in _LINEAR_TYPES:
        return component, None
    if isinstance(component, GaussianActionComponent):
        return None, component
    if isinstance(component, ABCDTransfer):
        raise NotImplementedError(
            "ABCDTransfer adds neither optical path nor the Gouy amplitude to a Gaussian, so a span "
            "containing it has no physical phase reference; plan the transform with "
            "temgym_core.wave.plan_lct and its matrix instead.")
    raise NotImplementedError(
        f"{type(component).__name__} cannot be used in a wave column: only linear elements (planes, "
        "lenses, stigmators, deflectors, rotators, solenoids, magnetic lenses) and thin "
        "GaussianActionComponents (apertures, phase plates, biprisms, specimens) are supported. "
        "DeflectionBiprism is a ray-level element; use PhaseBiprism.")


def component_mask(component, lattice: Lattice, wavelength):
    """Evaluate a thin component's complex field multiplier on a lattice.

    Parameters
    ----------
    component : GaussianActionComponent
        A thin element with ``complex_action(xy, k)`` (or the residual returned by
        :func:`split_component`).
    lattice : Lattice
        Where to evaluate; the positions are the sample coordinates ``(x, y)`` in metres.
    wavelength : float
        Wavelength, metres.

    Returns
    -------
    complex array, shape ``lattice.shape``
        ``exp(i k complex_action(xy, k))`` at every sample: the same action the element's Gaussian
        branch applies, without the second-order expansion.
    """
    k = 2.0 * np.pi / wavelength
    positions = lattice.positions().reshape(-1, 2)
    action = jax.vmap(lambda xy: component.complex_action(xy, k))(positions)
    return jnp.exp(1j * k * action).reshape(lattice.shape)
