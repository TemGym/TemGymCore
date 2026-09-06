"""Screen simpler acquisitions before re-running the complete pixel experiment.

These are local information forecasts and exact-symmetry checks, not completed
reduced-data fits. Only already measured data and an earlier fitted model enter.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_aperture_experiment import (
    aperture_specification,
    equivalent_aperture_model,
)
from condenser_fitting import beam_to_moments, model_moments
from condenser_objective_experiment import predict_pattern
from condenser_rotation_measurements import make_focus_transmission

QUANTITIES = (
    "source_z_mm",
    "CL1_z_mm",
    "CL3_z_mm",
    "mini_z_mm",
    "prefield_z_mm",
    "CL1_length_mm",
    "CL3_length_mm",
    "mini_length_mm",
    "prefield_length_mm",
    "aperture_z_mm",
    "aperture_x_um",
    "aperture_y_um",
)


def physical_geometry(parameters):
    """The twelve physical quantities that the user asked us to profile."""
    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    source = -jnp.sum(gaps) - jnp.sum(lengths)
    centres = (
        source
        + jnp.cumsum(gaps[:-1])
        + jnp.r_[0.0, jnp.cumsum(lengths[:-1])]
        + lengths / 2
    )
    aperture = centres[1] + lengths[1] / 2 + jax.nn.sigmoid(p[19]) * gaps[2]
    return jnp.r_[
        source * 1000, centres * 1000, lengths * 1000, aperture * 1000, p[20:22]
    ]


def physical_tolerances(parameters):
    """User-accepted absolute tolerances, using fitted lengths for 5% widths."""
    geometry = np.asarray(physical_geometry(parameters))
    return np.r_[1.0, np.full(4, 0.1), 0.05 * geometry[5:9], 0.1, 0.2, 0.2]


def measurement_blocks(*, generated_directory):
    """Return independently whitened image, focus-current and objective blocks."""
    root = Path(generated_directory)
    beam = dict(
        np.load(root / "condenser_calibration_run/calibration/measurements.npz")
    )
    aperture = dict(np.load(root / "condenser_rotation/calibration_measurements.npz"))
    counts = dict(np.load(root / "condenser_rotation/calibration_counts.npz"))
    objective = dict(np.load(root / "condenser_objective/calibration_measurements.npz"))
    views = json.loads(
        (root / "condenser_objective/reference_projector.json").read_text()
    )
    detector = json.loads((root / "condenser_objective/detector.json").read_text())
    convert = np.asarray(jax.vmap(jax.jacfwd(beam_to_moments))(beam["values"]))
    beam_cov = convert @ beam["covariance"] @ convert.transpose(0, 2, 1)
    bw = jnp.asarray(np.linalg.inv(np.linalg.cholesky(beam_cov)))
    bv = jax.vmap(beam_to_moments)(beam["values"])
    bc = jnp.asarray(beam["controls"])
    transmit = make_focus_transmission(specification=aperture_specification())
    ac = jnp.asarray(aperture["controls"])
    oc, ou = (
        jnp.asarray(objective["controls"]),
        jnp.asarray(objective["objective_commands"]),
    )
    ow = jnp.asarray(np.linalg.inv(np.linalg.cholesky(objective["covariance"])))
    ov = jnp.asarray(objective["values"])
    projector = jnp.asarray(views["calibrated_projector_parameters"])

    def images(p):
        predicted = jax.vmap(model_moments, in_axes=(None, 0))(p[:19], bc)
        return jnp.einsum("nij,nj->ni", bw, predicted - bv)

    def currents(p):
        return jax.vmap(transmit, in_axes=(None, 0))(p[:22], ac)

    def changed_objective(p):
        predicted = jax.vmap(
            lambda c, u: predict_pattern(
                p[:22], c, u, projector, detector["sample_kernel_sigma_m"], p[22:25]
            )
        )(oc, ou)
        return jnp.einsum("nij,nj->ni", ow, predicted - ov)

    return {
        "images": jax.jit(images),
        "currents": jax.jit(currents),
        "objective": jax.jit(changed_objective),
        "image_data": beam,
        "current_data": aperture,
        "current_groups": counts["groups"],
        "objective_data": objective,
    }


def make_selected_residual(
    *, blocks, image_indices, current_indices, aperture_indices, objective_indices=()
):
    """Use only specified measured blocks; subset covariance before whitening.

    Image blocks still contain all five views until independently re-analysed.
    No rotation measurement or old monotonic aperture scan is used here.
    """
    image_indices = jnp.asarray(image_indices, dtype=int)
    current_indices = np.asarray(current_indices, dtype=int)
    radii = np.asarray(aperture_indices, dtype=int)
    objective_indices = jnp.asarray(objective_indices, dtype=int)
    data = blocks["current_data"]
    values = jnp.asarray(data["values"][np.ix_(current_indices, radii)])
    cov = data["covariance"][current_indices][:, radii][:, :, radii]
    weight = jnp.asarray(np.linalg.inv(np.linalg.cholesky(cov)))

    def residual(p):
        transmitted = blocks["currents"](p)[current_indices][:, radii]
        parts = [
            blocks["images"](p)[image_indices].ravel(),
            jnp.einsum("nij,nj->ni", weight, transmitted - values).ravel(),
        ]
        if len(objective_indices):
            parts.append(blocks["objective"](p)[objective_indices].ravel())
        return jnp.concatenate(parts)

    return jax.jit(residual)


def design_information(*, derivative, parameters, fixed_objective):
    """Report full rank; conditional errors do not erase the exact ambiguity."""
    p = np.asarray(parameters)
    jac = np.asarray(derivative(p))
    scaled = np.linalg.svd(
        jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-30), compute_uv=False
    )
    rank = int(np.sum(scaled > scaled[0] * 1e-9))
    # At fixed objective choose a numerical representative solely for local
    # precision. An explicit exact-family range is reported separately.
    free = np.delete(np.arange(len(p)), 12) if fixed_objective else np.arange(len(p))
    _, s, vt = np.linalg.svd(jac[:, free], full_matrices=False)
    local_rank = np.linalg.matrix_rank(
        jac[:, free] / np.maximum(np.linalg.norm(jac[:, free], axis=0), 1e-30),
        tol=scaled[0] * 1e-9,
    )
    if local_rank == len(free):
        covariance = (vt.T / s) @ (vt / s[:, None])
        g = np.asarray(jax.jacfwd(physical_geometry)(p))[:, free]
        errors = np.sqrt(np.maximum(np.diag(g @ covariance @ g.T), 0)).tolist()
    else:
        errors = [None] * len(QUANTITIES)
    return {
        "rank": rank,
        "parameter_count": len(p),
        "standard_errors": dict(zip(QUANTITIES, errors)),
        "errors_conditioned_on_prefield_strength": fixed_objective,
        "scope": "Local forecast at the earlier fitted geometry, not a reduced-data fit.",
    }


def audit_exact_family(*, residual, parameters, bounds_lower, bounds_upper):
    """Find physically allowed, exactly equivalent geometries; retain each one."""
    p = np.asarray(parameters)[:22]
    reference = np.asarray(residual(p))
    lower, upper = np.asarray(bounds_lower)[:22], np.asarray(bounds_upper)[:22]
    family = []
    for strength in np.linspace(np.exp(lower[12]), np.exp(upper[12]), 201):
        try:
            alternative = equivalent_aperture_model(p, objective_strength=-strength)
        except ValueError:
            continue
        if np.any(alternative < lower) or np.any(alternative > upper):
            continue
        family.append(
            {
                "parameters": alternative.tolist(),
                "geometry": np.asarray(physical_geometry(alternative)).tolist(),
                "maximum_residual_change": float(
                    np.max(np.abs(np.asarray(residual(alternative)) - reference))
                ),
            }
        )
    if not family:
        raise RuntimeError("No valid family members; check declared bounds.")
    values = np.asarray([r["geometry"] for r in family])
    tolerances = physical_tolerances(p)
    return {
        "members": family,
        "quantities": list(QUANTITIES),
        "ranges": {
            name: {
                "minimum": float(values[:, i].min()),
                "maximum": float(values[:, i].max()),
                "span": float(np.ptp(values[:, i])),
                "accepted_tolerance": float(tolerances[i]),
                "span_exceeds_tolerance": bool(np.ptp(values[:, i]) > tolerances[i]),
            }
            for i, name in enumerate(QUANTITIES)
        },
        "maximum_residual_change": max(r["maximum_residual_change"] for r in family),
        "scope": "Exact family within engineering bounds; this is a lower bound on the full uncertainty, not a full profile over all other parameters.",
    }
