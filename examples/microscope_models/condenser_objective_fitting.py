"""Fit objective sweeps to independently analysed images and aperture currents."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_aperture_experiment import (
    equivalent_aperture_model,
    make_joint_residual,
    named_aperture_parameters,
)
from condenser_objective_experiment import predict_pattern
from scipy.optimize import least_squares


def make_objective_residual(
    *,
    baseline_beam_path,
    aperture_measurements_path,
    objective_measurements_path,
    projector_views_path,
    detector_path,
    aperture_specification,
    fit_post_response=True,
):
    """Measured baseline beams/currents and new pattern quantities -> residual.

    All 22 condenser/aperture parameters remain free. Optionally fit three
    post-field response slopes as parameters 22:25, without soft priors.
    The aperture side was established by the baseline CL3 current sweep.
    """
    baseline = make_joint_residual(
        beam_measurements_path=baseline_beam_path,
        aperture_measurements_path=aperture_measurements_path,
        specification=aperture_specification,
        side="after",
    )
    data = np.load(objective_measurements_path)
    views = json.loads(Path(projector_views_path).read_text())
    detector = json.loads(Path(detector_path).read_text())
    projector = jnp.asarray(views["calibrated_projector_parameters"])
    values = jnp.asarray(data["values"])
    whitening = jnp.asarray(np.linalg.inv(np.linalg.cholesky(data["covariance"])))
    controls, currents = (
        jnp.asarray(data["controls"]),
        jnp.asarray(data["objective_commands"]),
    )

    def pattern_residual(p):
        response = p[22:25] if fit_post_response else jnp.array([1.0, 0.0, 0.0])
        predicted = jax.vmap(
            lambda c, u: predict_pattern(
                p[:22], c, u, projector, detector["sample_kernel_sigma_m"], response
            )
        )(controls, currents)
        return jnp.einsum("nij,nj->ni", whitening, predicted - values).ravel()

    return jax.jit(lambda p: jnp.r_[baseline(p[:22]), pattern_residual(p)]), jax.jit(
        pattern_residual
    )


def fit_objective_sweep(
    *,
    baseline_fit_path,
    baseline_beam_path,
    aperture_measurements_path,
    objective_measurements_path,
    projector_views_path,
    detector_path,
    aperture_specification,
    output_directory,
    number_of_starts=6,
    max_evaluations=4000,
    fit_post_response=True,
    reuse_completed=True,
):
    """Supply measured data and the previous fit; return independent starts and errors."""
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [
        baseline_fit_path,
        baseline_beam_path,
        aperture_measurements_path,
        objective_measurements_path,
        projector_views_path,
        detector_path,
        Path(__file__).with_name("condenser_objective_experiment.py"),
        Path(__file__).with_name("condenser_calibration_model.py"),
        Path(__file__).with_name("condenser_aperture_experiment.py"),
    ]
    settings = {
        "starts": number_of_starts,
        "max_evaluations": max_evaluations,
        "fit_post_response": fit_post_response,
        "aperture_specification": aperture_specification,
    }
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in paths:
        digest.update(Path(path).read_bytes())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    signature = digest.hexdigest()
    output = folder / "fit_summary.json"
    if (
        reuse_completed
        and output.exists()
        and json.loads(output.read_text()).get("signature") == signature
    ):
        return json.loads(output.read_text())
    baseline = json.loads(Path(baseline_fit_path).read_text())
    p0 = np.asarray(baseline["best_fit"]["parameters"])
    lower, upper = (
        np.asarray(baseline["bounds_lower"]),
        np.asarray(baseline["bounds_upper"]),
    )
    if fit_post_response:
        lower = np.r_[lower, 0.2, -3.0, -1.0]
        upper = np.r_[upper, 2.0, 3.0, 1.0]
    residual, _ = make_objective_residual(
        baseline_beam_path=baseline_beam_path,
        aperture_measurements_path=aperture_measurements_path,
        objective_measurements_path=objective_measurements_path,
        projector_views_path=projector_views_path,
        detector_path=detector_path,
        aperture_specification=aperture_specification,
        fit_post_response=fit_post_response,
    )
    derivative = jax.jit(jax.jacfwd(residual))
    rng = np.random.default_rng(317)
    records = []
    for i in range(number_of_starts):
        if i == 0:
            guess = p0.copy()
        else:
            # These equally good baseline models deliberately span its exact
            # ambiguity. They are generated from the previous fit, not truth.
            guess = equivalent_aperture_model(
                p0,
                objective_strength=-np.linspace(
                    0.30, 0.70, max(number_of_starts - 1, 1)
                )[i - 1],
            )
            guess[:12] += rng.normal(0, 0.015, 12)
        if fit_post_response:
            guess = np.r_[guess, 1.0, 0.0, 0.0]
        guess = np.clip(guess, lower + 1e-8, upper - 1e-8)
        fit = least_squares(
            lambda p: np.asarray(residual(p)),
            guess,
            jac=lambda p: np.asarray(derivative(p)),
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=max_evaluations,
            ftol=1e-9,
            xtol=1e-11,
            gtol=1e-5,
        )
        record = {
            "index": i,
            "starting_parameters": guess.tolist(),
            "parameters": fit.x.tolist(),
            "success": bool(fit.success),
            "message": str(fit.message),
            "evaluations": int(fit.nfev),
            "chi_squared": float(2 * fit.cost),
            "residual_rms": float(np.sqrt(np.mean(fit.fun**2))),
            "active_bounds": np.flatnonzero(fit.active_mask).tolist(),
            "geometry": named_aperture_parameters(fit.x[:22], "after"),
            "post_response": (
                fit.x[22:25] if fit_post_response else np.array([1.0, 0.0, 0.0])
            ).tolist(),
        }
        records.append(record)
        (folder / f"start_{i:02d}.json").write_text(json.dumps(record, indent=2))
        print(
            f"Free post={fit_post_response}, start {i + 1}/{number_of_starts}: RMS {record['residual_rms']:.4f}, success {fit.success}, nfev {fit.nfev}",
            flush=True,
        )
    best = min(records, key=lambda x: x["chi_squared"])
    p = np.asarray(best["parameters"])
    jac = np.asarray(derivative(p))
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    # Report both raw and column-normalized ranks; units alone must not decide it.
    normalized = np.linalg.svd(jac / np.linalg.norm(jac, axis=0), compute_uv=False)
    covariance = (vt.T / s) @ (vt / s[:, None])
    result = {
        "signature": signature,
        "best_fit": best,
        "starts": records,
        "soft_priors": False,
        "free_parameter_count": len(p),
        "fit_post_response": fit_post_response,
        "singular_values": s.tolist(),
        "column_normalized_singular_values": normalized.tolist(),
        "rank": int(np.sum(normalized > normalized[0] * 1e-9)),
        "covariance": covariance.tolist(),
        "bounds_lower": lower.tolist(),
        "bounds_upper": upper.tolist(),
        "assumptions": [
            "Reference post-field geometry and downstream maps are fixed to their calibration at u=1.",
            "Source and condenser geometry are shared across currents; pre-field q is proportional to recorded relative objective current.",
            "Post response may change strength, effective length about a fixed centre, and rotation; it has three shared slopes.",
            "The aperture lies after CL3, as established by the baseline current scans.",
        ],
    }
    output.write_text(json.dumps(result, indent=2))
    return result


def evaluate_objective_sweep(
    *,
    fitted,
    objective_image_directory,
    baseline_beam_path,
    aperture_measurements_path,
    aperture_specification,
    baseline_fit_path,
    aperture_truth_path,
):
    """Check new objective settings without refitting, then compare with hidden geometry."""
    folder = Path(objective_image_directory)
    _, prediction = make_objective_residual(
        baseline_beam_path=baseline_beam_path,
        aperture_measurements_path=aperture_measurements_path,
        objective_measurements_path=folder / "prediction_measurements.npz",
        projector_views_path=folder / "reference_projector.json",
        detector_path=folder / "detector.json",
        aperture_specification=aperture_specification,
        fit_post_response=fitted["fit_post_response"],
    )
    p = np.asarray(fitted["best_fit"]["parameters"])
    errors = np.asarray(prediction(p))
    truth = json.loads(Path(aperture_truth_path).read_text())
    objective_truth = json.loads((folder / "scoring_truth.json").read_text())
    actual = named_aperture_parameters(np.asarray(truth["parameters"]), "after")
    estimated = named_aperture_parameters(p[:22], "after")
    baseline = json.loads(Path(baseline_fit_path).read_text())["best_fit"]["model"]
    comparisons = [
        {
            "parameter": name,
            "truth": value,
            "baseline": baseline[name],
            "fit": estimated[name],
            "relative_error_percent": 100 * (estimated[name] - value) / abs(value)
            if value
            else None,
        }
        for name, value in actual.items()
    ]
    # The covariance is local, conditional on the selected solution and model.
    jac_geometry = jax.jacfwd(
        lambda p: jnp.r_[-jnp.sum(jnp.exp(p[:9])), jnp.exp(p[5:9]), jnp.exp(p[:5])]
    )(jnp.asarray(p))
    uncertainty = np.sqrt(
        np.maximum(
            np.diag(
                np.asarray(jac_geometry)
                @ np.asarray(fitted["covariance"])
                @ np.asarray(jac_geometry).T
            ),
            0,
        )
    )
    return {
        "new_objective_measurement_rms": float(np.sqrt(np.mean(errors**2))),
        "new_images": int(
            len(np.load(folder / "prediction_measurements.npz")["values"]) * 5
        ),
        "post_response_truth": objective_truth["post_response"],
        "parameter_comparison": comparisons,
        "local_geometry_standard_errors_m": dict(
            zip(
                [
                    "source_z",
                    "CL1_length",
                    "CL3_length",
                    "mini_length",
                    "prefield_length",
                    "source_CL1_gap",
                    "CL1_CL3_gap",
                    "CL3_mini_gap",
                    "mini_prefield_gap",
                    "prefield_specimen_gap",
                ],
                uncertainty.tolist(),
            )
        ),
        "scope": "Prediction uses measured covariance, without fitted-parameter uncertainty. Truth is used only for this final parameter comparison.",
    }


def check_old_family(
    *,
    fitted,
    baseline_beam_path,
    aperture_measurements_path,
    objective_measurements_path,
    projector_views_path,
    detector_path,
    aperture_specification,
):
    """Test three old-family alternatives, allowing their post response to refit.

    This is a targeted ambiguity check, not a profile over all other geometry
    parameters and not a count of every minimum of the nonlinear problem.
    """
    residual, _ = make_objective_residual(
        baseline_beam_path=baseline_beam_path,
        aperture_measurements_path=aperture_measurements_path,
        objective_measurements_path=objective_measurements_path,
        projector_views_path=projector_views_path,
        detector_path=detector_path,
        aperture_specification=aperture_specification,
        fit_post_response=True,
    )
    p = np.asarray(fitted["best_fit"]["parameters"])
    records = []
    for strength in [-0.35, -np.exp(p[12]), -0.60]:
        geometry = equivalent_aperture_model(p[:22], objective_strength=strength)
        fn = jax.jit(
            lambda response, geometry=geometry: residual(jnp.r_[geometry, response])
        )
        jac = jax.jit(jax.jacfwd(fn))
        fit = least_squares(
            lambda r, fn=fn: np.asarray(fn(r)),
            p[22:],
            jac=lambda r, jac=jac: np.asarray(jac(r)),
            bounds=(
                np.asarray(fitted["bounds_lower"])[22:],
                np.asarray(fitted["bounds_upper"])[22:],
            ),
            max_nfev=500,
            ftol=1e-10,
            xtol=1e-11,
            gtol=1e-5,
            x_scale="jac",
        )
        records.append(
            {
                "prefield_strength": strength,
                "prefield_length_mm": float(np.exp(geometry[8]) * 1000),
                "chi_squared": float(2 * fit.cost),
                "post_response": fit.x.tolist(),
                "success": bool(fit.success),
            }
        )
    minimum = min(r["chi_squared"] for r in records)
    for record in records:
        record["delta_chi_squared"] = record["chi_squared"] - minimum
    return records


def show_objective_results(
    *, fitted, checks, fixed_post_fit, fixed_post_checks, family_check, image_directory
):
    """One compact comparison, image panel and physical geometry table."""
    import matplotlib.pyplot as plt
    from condenser_objective_experiment import make_pattern_image_operator
    from IPython.display import Markdown, display

    data = np.load(Path(image_directory) / "calibration_images.npz")
    measured = np.load(Path(image_directory) / "calibration_measurements.npz")
    good = sum(r["success"] for r in fitted["starts"])
    near = sum(
        r["chi_squared"] - fitted["best_fit"]["chi_squared"] < 1
        for r in fitted["starts"]
    )
    display(
        Markdown(
            f"**Objective sweep: {fitted['rank']}/{fitted['free_parameter_count']} locally constrained directions.** "
            f"{good}/{len(fitted['starts'])} starts converged; {near}/{len(fitted['starts'])} reached within one chi-squared unit of the best fit.\n\n"
            "| Post-field response | Calibration RMS | New-image measurement RMS |\n|---|---:|---:|\n"
            f"| Nominal response held fixed | {fixed_post_fit['best_fit']['residual_rms']:.3f} | {fixed_post_checks['new_objective_measurement_rms']:.3f} |\n"
            f"| Three response terms fitted | {fitted['best_fit']['residual_rms']:.3f} | {checks['new_objective_measurement_rms']:.3f} |\n\n"
            "RMS values use the measurement uncertainties. Full local rank does not prove global uniqueness."
        )
    )
    folder = Path(image_directory)
    detector = json.loads((folder / "detector.json").read_text())
    operator = make_pattern_image_operator(
        projector_matrices=data["projector_matrices"][0],
        template=dict(np.load(folder / "template.npz")),
        detector=detector,
    )
    figure, axes = plt.subplots(2, 3, figsize=(10, 5.5), constrained_layout=True)
    for row, index in enumerate([0, 1]):
        frames = data["images"][index]
        predicted = np.asarray(
            operator(
                jnp.asarray(measured["image_parameters"][index]),
                jnp.asarray(data["projector_matrices"][index]),
            )
        )
        image = frames[0]
        vmax = np.percentile(image, 99.5)
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=vmax)
        axes[row, 1].imshow(predicted[0], cmap="gray", vmin=0, vmax=vmax)
        residual = (image - predicted[0]) / np.sqrt(
            np.maximum(image, 1) + detector["read_noise_e"] ** 2
        )
        axes[row, 2].imshow(residual, cmap="coolwarm", vmin=-3, vmax=3)
        for ax, title in zip(
            axes[row], ["Measured", "Independent image fit", "Residual / noise"]
        ):
            ax.set_title(
                f"{100 * data['objective_commands'][index]:.0f}% objective: {title}"
            )
            ax.set_axis_off()
    display(figure)
    figure.savefig(folder / "objective_image_check.png", dpi=160)
    plt.close(figure)
    lines = [
        "| Physical quantity | Truth | Fixed-objective fit | Objective-sweep fit |",
        "|---|---:|---:|---:|",
    ]
    for row in checks["parameter_comparison"]:
        if row["parameter"].endswith(("z_mm", "length_mm")):
            lines.append(
                f"| {row['parameter']} | {row['truth']:.5g} | {row['baseline']:.5g} | {row['fit']:.5g} |"
            )
    display(Markdown("All distances below are in **mm**.\n\n" + "\n".join(lines)))
    display(
        Markdown(
            "Local standard errors, conditional on this model: "
            f"pre-field length **{checks['local_geometry_standard_errors_m']['prefield_length'] * 1000:.4g} mm**; "
            f"source position **{checks['local_geometry_standard_errors_m']['source_z'] * 1000:.4g} mm**."
        )
    )
    lines = [
        "| Alternative pre-field length (mm) | Extra chi-squared after refitting post response |",
        "|---|---:|",
    ]
    for row in family_check:
        lines.append(
            f"| {row['prefield_length_mm']:.4f} | {row['delta_chi_squared']:.2f} |"
        )
    display(
        Markdown(
            "These alternatives were indistinguishable at fixed objective current.\n\n"
            + "\n".join(lines)
        )
    )
