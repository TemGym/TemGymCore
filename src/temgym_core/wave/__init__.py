"""Sampled wavefronts: Collins transforms of complex fields between planes.

.. warning::

   **Experimental.** This package is new and its API may change until it has been validated
   against more columns. It lives at its final path, ``temgym_core.wave``.

A complex field sampled on a lattice is propagated through a column of the same components that
rays and Gaussian beamlets use. Every linear stretch of the column (lenses, deflectors, rotations,
solenoids, magnetic lenses, drifts) is one linear canonical (Collins) transform, so planes between
components are skipped, and every thin element that is not linear (an aperture, a biprism, a phase
plate, a specimen) multiplies the field. Two implementations of one operator cover the
regimes: a lattice-adapted matrix Fourier transform (far from an image plane) and a drift, lattice
map and chirp (near an image plane), and a sampling planner refuses any span that would alias.

- :func:`propagate` runs a column on a :class:`Wavefront`; :func:`input_lattice` gives the lattice a
  source must be sampled on to land on a detector grid.
- :func:`resample_field` interpolates a field between lattices, for arrays that arrive on or must leave
  on a Cartesian grid.
- :func:`column_transform`, :func:`plan_lct` and :func:`apply_lct` are the differentiable
  building blocks: the ray matrix and the physical phase of a span, the plan, and its application.

The field excludes the axial carrier ``exp(i k (z - z0))``, like the field of a Gaussian beamlet.
Images are in (row, column) = (y, x) order and lattices in ``(x, y)`` order. 64-bit JAX arithmetic
is required (``temgym_core.enable_x64()``).
"""

from .column import ColumnTransform, Wavefront, column_transform, input_lattice, propagate
from .lattice import (
    Lattice,
    adapted_input_lattice,
    adapted_output_lattice,
    as_lattice,
    lattice_chain,
    near_image_input_lattice,
    near_image_lattice,
)
from .lct import LCTPlan, apply_lct, lift_phase, plan_lct
from .masks import component_mask, is_linear, split_component
from .planner import (
    LatticeError,
    SamplingError,
    SamplingReport,
    check_mft,
    check_near_image,
    measure_support,
)
from .reference import chief_phase0, collins_gaussian, gaussian_field
from .resample import resample_field

__all__ = [
    "ColumnTransform",
    "LCTPlan",
    "Lattice",
    "LatticeError",
    "SamplingError",
    "SamplingReport",
    "Wavefront",
    "adapted_input_lattice",
    "adapted_output_lattice",
    "apply_lct",
    "as_lattice",
    "check_mft",
    "check_near_image",
    "chief_phase0",
    "collins_gaussian",
    "column_transform",
    "component_mask",
    "gaussian_field",
    "input_lattice",
    "is_linear",
    "lattice_chain",
    "lift_phase",
    "measure_support",
    "near_image_input_lattice",
    "near_image_lattice",
    "plan_lct",
    "propagate",
    "resample_field",
    "split_component",
]
