"""Joint condenser fitting with explicit rotation and aperture-focus measurements."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_aperture_experiment import (
    aperture_specification,
    equivalent_aperture_model,
    make_joint_residual,
    named_aperture_parameters,
)
from condenser_objective_fitting import make_objective_residual
from condenser_rotation_measurements import make_focus_transmission
from scipy.optimize import least_squares


def fit_rotation_slopes(*, measurements_path):
    """Measured ellipse angles -> three gains and one unknown angle intercept."""
    data = np.load(measurements_path)
    angles = np.unwrap(2 * data["angles"]) / 2
    design = np.c_[np.ones(len(angles)), -data["controls"]]
    weighted = design / np.sqrt(data["variance"])[:, None]
    coefficients = np.linalg.lstsq(
        weighted, angles / np.sqrt(data["variance"]), rcond=None
    )[0]
    covariance = np.linalg.inv(weighted.T @ weighted)
    return {
        "intercept": float(coefficients[0]),
        "gains": coefficients[1:].tolist(),
        "covariance": covariance.tolist(),
        "rms": float(
            np.sqrt(
                np.mean(
                    ((design @ coefficients - angles) / np.sqrt(data["variance"])) ** 2
                )
            )
        ),
    }


def make_rotation_joint_residual(
    *,
    rotation_directory,
    baseline_directory,
    baseline_aperture_directory,
    objective_directory,
    include_objective,
):
    """Return residual and derivative for 23 (fixed) or 26 (varied) unknowns."""
    folder = Path(rotation_directory)
    baseline = Path(baseline_directory)
    aperture = Path(baseline_aperture_directory)
    objective = Path(objective_directory)
    common = {
        "baseline_beam_path": baseline / "calibration/measurements.npz",
        "aperture_measurements_path": aperture / "calibration_measurements.npz",
        "aperture_specification": aperture_specification(),
    }
    if include_objective:
        original, _ = make_objective_residual(
            **common,
            objective_measurements_path=objective / "calibration_measurements.npz",
            projector_views_path=objective / "reference_projector.json",
            detector_path=objective / "detector.json",
            fit_post_response=True,
        )
    else:
        original = make_joint_residual(
            beam_measurements_path=common["baseline_beam_path"],
            aperture_measurements_path=common["aperture_measurements_path"],
            specification=aperture_specification(),
            side="after",
        )
    measured = np.load(folder / "calibration_measurements.npz")
    rotation = np.load(folder / "rotation_measurements.npz")
    transmit = make_focus_transmission(specification=aperture_specification())
    whitening = jnp.asarray(np.linalg.inv(np.linalg.cholesky(measured["covariance"])))
    controls = jnp.asarray(measured["controls"])
    values = jnp.asarray(measured["values"])
    rotation_controls = jnp.asarray(rotation["controls"])
    angles = jnp.asarray(rotation["angles"])
    sigma = jnp.sqrt(rotation["variance"])

    def residual(p):
        currents = jax.vmap(transmit, in_axes=(None, 0))(p[:22], controls)
        predicted_angle = p[-1] - rotation_controls @ jnp.exp(p[9:12])
        delta = predicted_angle - angles
        angular_error = (
            0.5 * jnp.arctan2(jnp.sin(2 * delta), jnp.cos(2 * delta)) / sigma
        )
        return jnp.r_[
            original(p[:-1]),
            jnp.einsum("nij,nj->ni", whitening, currents - values).ravel(),
            angular_error,
        ]

    return jax.jit(residual), jax.jit(jax.jacfwd(residual))


def fit_with_rotation(
    *,
    rotation_directory,
    baseline_directory,
    baseline_aperture_directory,
    objective_directory,
    output_directory,
    include_objective=True,
    number_of_starts=6,
    max_evaluations=5000,
    starting_fit_path=None,
):
    """Fit measurements from data-derived guesses, retaining every start.

    Rotation supplies an unknown angle intercept. With objective images, the
    previous objective fit supplies the starting microscope; the joint fit
    releases all 26 parameters. No generating parameters enter initialization.
    """
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    rotation = Path(rotation_directory)
    baseline = Path(baseline_aperture_directory)
    previous = json.loads((baseline / "fit/joint_fit_summary.json").read_text())
    slopes = fit_rotation_slopes(
        measurements_path=rotation / "rotation_measurements.npz"
    )
    p0 = np.asarray(previous["best_fit"]["parameters"])
    lo = np.asarray(previous["bounds_lower"])
    hi = np.asarray(previous["bounds_upper"])
    if include_objective:
        lo = np.r_[lo, 0.2, -3.0, -1.0]
        hi = np.r_[hi, 2.0, 3.0, 1.0]
    lo = np.r_[lo, -np.pi]
    hi = np.r_[hi, np.pi]
    residual, derivative = make_rotation_joint_residual(
        rotation_directory=rotation,
        baseline_directory=baseline_directory,
        baseline_aperture_directory=baseline,
        objective_directory=objective_directory,
        include_objective=include_objective,
    )
    # The fixed-objective fit has an exact one-dimensional family. Keeping the
    # pre-field strength at each starting value only chooses a representative;
    # the full unrestrained data Jacobian is used for the identifiability audit.
    free = np.array([i for i in range(len(lo)) if include_objective or i != 12])
    seed_geometry = p0.copy()
    seed_geometry[9:12] = np.log(slopes["gains"])
    if include_objective:
        starting_fit_path = Path(
            starting_fit_path
            or Path(objective_directory) / "fit_free_post/fit_summary.json"
        )
        prior = json.loads(starting_fit_path.read_text())
        best_seed = np.r_[prior["best_fit"]["parameters"], slopes["intercept"]]
    rng = np.random.default_rng(481)
    records = []
    for i in range(number_of_starts):
        if include_objective:
            start = best_seed.copy()
            if i:
                start[:12] += rng.normal(0, 0.005, 12)
        else:
            q = equivalent_aperture_model(
                seed_geometry,
                objective_strength=-np.linspace(0.35, 0.65, number_of_starts)[i],
            )
            start = np.r_[q, slopes["intercept"]]
        start = np.clip(start, lo + 1e-8, hi - 1e-8)

        def unpack(v, start=start):
            return jnp.asarray(start).at[free].set(v)

        fn = lambda v: residual(unpack(v))
        jac = lambda v: derivative(unpack(v))[:, free]
        fit = least_squares(
            lambda v, fn=fn: np.asarray(fn(v)),
            start[free],
            jac=lambda v, jac=jac: np.asarray(jac(v)),
            bounds=(lo[free], hi[free]),
            max_nfev=max_evaluations,
            x_scale="jac",
            ftol=1e-9,
            xtol=1e-11,
            gtol=1e-5,
        )
        p = np.asarray(unpack(fit.x))
        record = {
            "index": i,
            "parameters": p.tolist(),
            "starting_parameters": start.tolist(),
            "success": bool(fit.success),
            "evaluations": int(fit.nfev),
            "message": str(fit.message),
            "chi_squared": float(2 * fit.cost),
            "residual_rms": float(np.sqrt(np.mean(fit.fun**2))),
            "active_bounds": free[np.flatnonzero(fit.active_mask)].tolist(),
            "geometry": named_aperture_parameters(p[:22], "after"),
        }
        records.append(record)
        (folder / f"start_{i:02d}.json").write_text(json.dumps(record, indent=2))
        print(
            f"Objective={include_objective} start {i + 1}: RMS {record['residual_rms']:.4f}, success {fit.success}, nfev {fit.nfev}",
            flush=True,
        )
    best = min(records, key=lambda x: x["chi_squared"])
    p = np.asarray(best["parameters"])
    jac = np.asarray(derivative(p))
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    scaled = np.linalg.svd(
        jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-30), compute_uv=False
    )
    rank = int(np.sum(scaled > scaled[0] * 1e-9))
    result = {
        "best_fit": best,
        "starts": records,
        "parameter_count": len(p),
        "rank": rank,
        "soft_priors": False,
        "include_objective": include_objective,
        "fit_post_response": include_objective,
        "covariance": ((vt.T / s) @ (vt / s[:, None])).tolist()
        if rank == len(p)
        else None,
        "singular_values": s.tolist(),
        "column_normalized_singular_values": scaled.tolist(),
        "bounds_lower": lo.tolist(),
        "bounds_upper": hi.tolist(),
        "rotation_regression": slopes,
        "fixed_for_numerical_representative": []
        if include_objective
        else ["pre-field strength, varied between starts"],
        "initialization": (
            "Previous objective fit supplies a data-derived starting microscope; all parameters released, with 0.5% log-geometry perturbations across starts. Not a blind global search."
            if include_objective
            else "Rotation regression and previous aperture fit supply starting values. Pre-field strength fixed at a different representative per start; full Jacobian used for rank."
        ),
    }
    (folder / "fit_summary.json").write_text(json.dumps(result, indent=2))
    return result
