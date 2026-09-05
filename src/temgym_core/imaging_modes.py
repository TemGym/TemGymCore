"""Conjugate imaging settings for the thin, round microscope model.

This designs simulation currents, not a calibration of instrument DAC values.
An objective and final projector stay fixed while three intervening lenses
relay their conjugate planes with prescribed magnification and rotation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .constants import compute_Rc_from_voltage, voltage_scaling_ratio


@dataclass(frozen=True)
class ConjugateImagingConstraint:
    relay_lenses: tuple[str, str, str]
    current_bounds_a: tuple[float, float] = (0.01, 5.0)
    rotation_rad: float = 0.0

    def __post_init__(self):
        names = tuple(self.relay_lenses)
        bounds = tuple(float(v) for v in self.current_bounds_a)
        if len(names) != 3 or len(set(names)) != 3:
            raise ValueError("Conjugate imaging requires three distinct relay lenses.")
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError("Imaging current bounds must be finite and ordered.")
        if not np.isfinite(self.rotation_rad):
            raise ValueError("Imaging rotation must be finite.")
        object.__setattr__(self, "relay_lenses", names)
        object.__setattr__(self, "current_bounds_a", bounds)


def drift(distance):
    return np.array(((1.0, distance), (0.0, 1.0)))


def radial_chain(positions, powers, start_z, end_z):
    """Compose a thin-lens radial map in physical propagation order."""
    matrix = np.eye(2)
    previous_z = start_z
    for z, power in zip(positions, powers, strict=True):
        matrix = np.array(((1.0, 0.0), (-power, 1.0))) @ drift(z - previous_z) @ matrix
        previous_z = z
    return drift(end_z - previous_z) @ matrix


def imaging_geometry(model, currents, gc_scales=None):
    """Return the five downstream lenses and their fixed conjugate planes.

    Reject unsupported optics rather than optimizing a surrogate inconsistent
    with ``MicroscopeModel.build_components``.
    """
    start = float(model.auxiliary["sample"]["z"])
    end = float(model.auxiliary["detector"]["z"])
    indices = np.array(sorted(
        (i for i, lens in enumerate(model.lenses) if start < lens.z_position < end),
        key=lambda i: model.lenses[i].z_position,
    ))
    if len(indices) != 5:
        raise ValueError("Conjugate imaging requires an objective, three relay lenses, and a projector.")
    lenses = tuple(model.lenses[i] for i in indices)
    if any(l.Tc != 0 or l.stigmator_strength_x != 0 or l.stigmator_strength_y != 0 for l in lenses):
        raise ValueError("Conjugate imaging currently supports only thin, round lenses.")
    if any(start < d.z_position < end for d in model.deflectors):
        raise ValueError("Conjugate imaging does not support downstream deflectors.")
    z = np.array([l.z_position for l in lenses])
    if np.any(np.diff(z) <= 0):
        raise ValueError("Downstream lens positions must be strictly increasing.")
    turns = np.array([l.turns for l in lenses])
    scales = np.ones(len(model.lenses)) if gc_scales is None else np.asarray(gc_scales)
    gains = (np.array([l.Gc for l in lenses]) * turns**2 * scales[indices]
             * float(voltage_scaling_ratio(model.voltage, model.reference_voltage)))
    if np.any(gains <= 0) or not np.all(np.isfinite(gains)):
        raise ValueError("Imaging focal-power gains must be finite and positive.")
    downstream_currents = np.asarray(currents, dtype=float)[indices]
    powers = gains * downstream_currents**2
    u = z[0] - start
    v = end - z[-1]
    objective_denominator = powers[0] - 1 / u
    projector_denominator = powers[-1] - 1 / v
    if objective_denominator <= 0 or projector_denominator <= 0:
        raise ValueError("The fixed objective and projector must have real conjugate planes.")
    objective_v = 1 / objective_denominator
    projector_u = 1 / projector_denominator
    object_plane = z[0] + objective_v
    image_plane = z[-1] - projector_u
    if not z[0] < object_plane < z[1] or not z[-2] < image_plane < z[-1]:
        raise ValueError("The fixed conjugate planes must lie outside the three-lens relay.")
    return {
        "indices": indices, "lenses": lenses, "z": z, "turns": turns,
        "gains": gains, "currents": downstream_currents,
        "sample_z": start, "detector_z": end,
        "objective_image_z": object_plane, "projector_object_z": image_plane,
        "objective_magnification": -objective_v / u,
        "projector_magnification": -v / projector_u,
    }


def refine_conjugate_imaging_currents(model, currents, target, constraint, gc_scales=None):
    """Refine a table/continuation seed to A = target*R and B = 0.

    The solve uses the intermediate relay's conjugate planes to avoid the
    severe cancellation in a high-magnification sample-to-detector residual.
    Focusing and zero net rotation give three equations for three currents.
    Geometry, both end lenses, and every upstream current are held fixed.
    """
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Target magnification must be finite and positive.")
    geometry = imaging_geometry(model, currents, gc_scales)
    if tuple(l.name for l in geometry["lenses"][1:-1]) != constraint.relay_lenses:
        raise ValueError("The declared relay lenses must be the three interior downstream lenses in order.")
    relay_target = target / (geometry["objective_magnification"] * geometry["projector_magnification"])
    rotation_per_at = float(compute_Rc_from_voltage(model.voltage))
    target_ampturns = constraint.rotation_rad / rotation_per_at
    fixed_ampturns = geometry["turns"][[0, -1]] @ geometry["currents"][[0, -1]]
    ampturn_scale = max(abs(fixed_ampturns), 1.0)
    length_scale = geometry["objective_image_z"] - geometry["z"][0]

    def residual(relay_currents):
        matrix = radial_chain(
            geometry["z"][1:-1], geometry["gains"][1:-1] * relay_currents**2,
            geometry["objective_image_z"], geometry["projector_object_z"],
        )
        return np.array((
            matrix[0, 0] / relay_target - 1,
            matrix[0, 1] / length_scale,
            (geometry["turns"][1:-1] @ relay_currents + fixed_ampturns - target_ampturns) / ampturn_scale,
        ))

    lower, upper = constraint.current_bounds_a
    seed = np.clip(geometry["currents"][1:-1], lower, upper)
    result = least_squares(
        residual, seed, bounds=(lower, upper),
        xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=300,
    )
    if not result.success or np.max(np.abs(residual(result.x))) > 1e-10:
        raise ValueError(f"No converged conjugate imaging solution at {target:g}x within the current bounds.")
    output = np.array(currents, dtype=float, copy=True)
    output[geometry["indices"][1:-1]] = result.x
    return output
