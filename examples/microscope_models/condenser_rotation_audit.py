"""Check physical ambiguity and predictions after fitting measured rotations."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_aperture_experiment import equivalent_aperture_model
from condenser_rotation_fitting import make_rotation_joint_residual
from scipy.optimize import brentq, least_squares


def profile_prefield_length(
    *,
    fitted,
    rotation_directory,
    baseline_directory,
    baseline_aperture_directory,
    objective_directory,
    output_directory,
    max_evaluations=2000,
    reuse_completed=True,
):
    """Fix each pre-field length and refit ALL other parameters, without priors.

    A finite grid and one continuation start per point cannot prove a global
    minimum. Every convergence flag and bound contact is retained.
    """
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    from condenser_rotation_workflow import fit_signature, input_signature

    measurement_signature = fit_signature(
        rotation_directory=rotation_directory,
        baseline_directory=baseline_directory,
        baseline_aperture_directory=baseline_aperture_directory,
        objective_directory=objective_directory,
        include_objective=True,
        number_of_starts=len(fitted["starts"]),
        max_evaluations=5000,
    )
    signature = input_signature(
        paths=[__file__],
        settings={
            "measurements": measurement_signature,
            "parameters": fitted["best_fit"]["parameters"],
            "max_evaluations": max_evaluations,
        },
    )
    manifest = folder / "profile_manifest.json"
    if (
        reuse_completed
        and manifest.exists()
        and (folder / "profile.json").exists()
        and json.loads(manifest.read_text())["signature"] == signature
    ):
        return json.loads((folder / "profile.json").read_text())
    residual, derivative = make_rotation_joint_residual(
        rotation_directory=rotation_directory,
        baseline_directory=baseline_directory,
        baseline_aperture_directory=baseline_aperture_directory,
        objective_directory=objective_directory,
        include_objective=True,
    )
    best = np.asarray(fitted["best_fit"]["parameters"])
    covariance = np.asarray(fitted["covariance"])
    sigma = np.sqrt(covariance[8, 8])
    offsets = sorted(
        {-0.4, -0.2, -3 * sigma, -sigma, 0.0, sigma, 3 * sigma, 0.2, 0.4}, key=abs
    )
    lower = np.asarray(fitted["bounds_lower"])
    upper = np.asarray(fitted["bounds_upper"])
    free = np.delete(np.arange(len(best)), 8)
    records = []
    for offset in offsets:
        fixed = best[8] + offset
        if not lower[8] < fixed < upper[8]:
            continue

        # Start on the old ambiguity family, as close to this length as possible.
        def difference(strength, fixed=fixed):
            return float(
                equivalent_aperture_model(best[:22], objective_strength=-strength)[8]
                - fixed
            )

        strength = brentq(difference, 0.15, 0.9)
        start = best.copy()
        start[:22] = equivalent_aperture_model(best[:22], objective_strength=-strength)
        start[8] = fixed
        start = np.clip(start, lower + 1e-9, upper - 1e-9)

        def unpack(values, start=start):
            return jnp.asarray(start).at[free].set(values)

        fit = least_squares(
            lambda v, unpack=unpack: np.asarray(residual(unpack(v))),
            start[free],
            jac=lambda v, unpack=unpack: np.asarray(derivative(unpack(v)))[:, free],
            bounds=(lower[free], upper[free]),
            x_scale="jac",
            max_nfev=max_evaluations,
            ftol=1e-8,
            xtol=1e-10,
            gtol=1e-5,
        )
        record = {
            "length_mm": float(np.exp(fixed) * 1000),
            "chi_squared": float(2 * fit.cost),
            "delta_chi_squared": float(
                2 * fit.cost - fitted["best_fit"]["chi_squared"]
            ),
            "success": bool(fit.success),
            "evaluations": int(fit.nfev),
            "active_bounds": free[np.flatnonzero(fit.active_mask)].tolist(),
            "parameters": np.asarray(unpack(fit.x)).tolist(),
        }
        records.append(record)
        (folder / "profile.json").write_text(json.dumps(records, indent=2))
        print(
            f"Profile length {record['length_mm']:.6f} mm: delta chi2 {record['delta_chi_squared']:.3f}, success {fit.success}, nfev {fit.nfev}",
            flush=True,
        )
    records = sorted(records, key=lambda r: r["length_mm"])
    (folder / "profile.json").write_text(json.dumps(records, indent=2))
    manifest.write_text(json.dumps({"signature": signature}, indent=2))
    return records


def check_exact_rotation_family(
    *,
    fitted,
    rotation_directory,
    baseline_directory,
    baseline_aperture_directory,
    objective_directory,
):
    """Audit the full fixed-objective measurement vector along its exact family."""
    residual, _ = make_rotation_joint_residual(
        rotation_directory=rotation_directory,
        baseline_directory=baseline_directory,
        baseline_aperture_directory=baseline_aperture_directory,
        objective_directory=objective_directory,
        include_objective=False,
    )
    p = np.asarray(fitted["best_fit"]["parameters"])
    reference = np.asarray(residual(p))
    records = []
    for strength in [0.35, 0.45, 0.55, 0.65]:
        alternative = np.r_[
            equivalent_aperture_model(p[:22], objective_strength=-strength), p[-1]
        ]
        error = np.asarray(residual(alternative)) - reference
        records.append(
            {
                "prefield_strength": -strength,
                "prefield_length_mm": float(np.exp(alternative[8]) * 1000),
                "max_change_in_noise_units": float(np.max(np.abs(error))),
            }
        )
    return records


def compare_measurement_information(
    *,
    fitted,
    rotation_directory,
    baseline_directory,
    baseline_aperture_directory,
    objective_directory,
):
    """Compare local precision at ONE fitted geometry; these are not separate fits."""
    _, derivative = make_rotation_joint_residual(
        rotation_directory=rotation_directory,
        baseline_directory=baseline_directory,
        baseline_aperture_directory=baseline_aperture_directory,
        objective_directory=objective_directory,
        include_objective=True,
    )
    p = np.asarray(fitted["best_fit"]["parameters"])
    jacobian = np.asarray(derivative(p))
    rotation = np.load(Path(rotation_directory) / "rotation_measurements.npz")
    currents = np.load(Path(rotation_directory) / "calibration_measurements.npz")
    nr, nc = len(rotation["angles"]), currents["values"].size
    nb = len(jacobian) - nr - nc
    geometry = np.asarray(
        jax.jacfwd(lambda p: jnp.r_[-jnp.sum(jnp.exp(p[:9])), jnp.exp(p[8])])(
            jnp.asarray(p)
        )
    )
    records = []
    for label, rows, columns in [
        ("Previous data", np.arange(nb), np.arange(25)),
        (
            "Previous + rotation",
            np.r_[np.arange(nb), np.arange(nb + nc, nb + nc + nr)],
            np.arange(26),
        ),
        ("Previous + focus sweeps", np.arange(nb + nc), np.arange(25)),
        ("All measurements", np.arange(len(jacobian)), np.arange(26)),
    ]:
        j = jacobian[np.ix_(rows, columns)]
        _, s, vt = np.linalg.svd(j, full_matrices=False)
        covariance = (vt.T / s) @ (vt / s[:, None])
        g = geometry[:, columns]
        errors = np.sqrt(np.diag(g @ covariance @ g.T)) * 1000
        records.append(
            {
                "measurements": label,
                "source_z_sigma_mm": float(errors[0]),
                "prefield_length_sigma_mm": float(errors[1]),
            }
        )
    return records


def profile_aperture_direction(
    *,
    fitted,
    rotation_directory,
    baseline_directory,
    baseline_aperture_directory,
    objective_directory,
    output_directory,
    max_evaluations=600,
    reuse_completed=True,
):
    """Fix aperture-centre direction; refit its radius and all other parameters."""
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    from condenser_rotation_workflow import fit_signature, input_signature

    measurement_signature = fit_signature(
        rotation_directory=rotation_directory,
        baseline_directory=baseline_directory,
        baseline_aperture_directory=baseline_aperture_directory,
        objective_directory=objective_directory,
        include_objective=True,
        number_of_starts=len(fitted["starts"]),
        max_evaluations=5000,
    )
    signature = input_signature(
        paths=[__file__],
        settings={
            "measurements": measurement_signature,
            "parameters": fitted["best_fit"]["parameters"],
            "max_evaluations": max_evaluations,
        },
    )
    manifest = folder / "aperture_profile_manifest.json"
    if (
        reuse_completed
        and manifest.exists()
        and (folder / "aperture_direction_profile.json").exists()
        and json.loads(manifest.read_text())["signature"] == signature
    ):
        return json.loads((folder / "aperture_direction_profile.json").read_text())
    residual, derivative = make_rotation_joint_residual(
        rotation_directory=rotation_directory,
        baseline_directory=baseline_directory,
        baseline_aperture_directory=baseline_aperture_directory,
        objective_directory=objective_directory,
        include_objective=True,
    )
    p = np.asarray(fitted["best_fit"]["parameters"])
    lower = np.asarray(fitted["bounds_lower"])
    upper = np.asarray(fitted["bounds_upper"])
    free = np.delete(np.arange(len(p)), 21)
    angles = np.r_[
        np.linspace(-np.pi, np.pi, 16, endpoint=False), np.arctan2(p[21], p[20])
    ]
    records = []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        maximum = min(
            (upper[20 + i] if d > 0 else -lower[20 + i]) / abs(d)
            for i, d in enumerate(direction)
            if abs(d) > 1e-12
        )
        start = p[free].copy()
        start[20] = np.log(np.hypot(p[20], p[21]))
        lo = lower[free].copy()
        hi = upper[free].copy()
        lo[20] = np.log(1e-6)
        hi[20] = np.log(maximum)

        def unpack(v, direction=direction):
            full = jnp.asarray(p).at[free].set(v)
            return full.at[20:22].set(jnp.exp(v[20]) * direction)

        def jacobian(v, unpack=unpack, direction=direction):
            full = np.asarray(derivative(unpack(v)))
            reduced = full[:, free].copy()
            reduced[:, 20] = full[:, 20:22] @ (np.exp(v[20]) * direction)
            return reduced

        fit = least_squares(
            lambda v, unpack=unpack: np.asarray(residual(unpack(v))),
            start,
            jac=jacobian,
            bounds=(lo, hi),
            x_scale="jac",
            max_nfev=max_evaluations,
            ftol=1e-8,
            xtol=1e-10,
            gtol=1e-5,
        )
        record = {
            "angle_degrees": float(np.rad2deg(angle)),
            "radius_um": float(np.exp(fit.x[20])),
            "delta_chi_squared": float(
                2 * fit.cost - fitted["best_fit"]["chi_squared"]
            ),
            "success": bool(fit.success),
            "evaluations": int(fit.nfev),
            "parameters": np.asarray(unpack(fit.x)).tolist(),
        }
        records.append(record)
        (folder / "aperture_direction_profile.json").write_text(
            json.dumps(records, indent=2)
        )
        print(
            f"Aperture direction {record['angle_degrees']:.1f}: delta chi2 {record['delta_chi_squared']:.3f}, success {fit.success}",
            flush=True,
        )
    manifest.write_text(json.dumps({"signature": signature}, indent=2))
    return records
