"""Four readable notebook stages for the fixed-objective voltage experiment."""

import json
from contextlib import redirect_stdout
from pathlib import Path

import jax
import numpy as np
from condenser_rotation_workflow import input_signature
from condenser_simplification_design import (
    QUANTITIES,
    physical_geometry,
    physical_tolerances,
)
from condenser_voltage_experiment import (
    analyse_voltage_experiment,
    simulate_voltage_experiment,
    simulate_voltage_prediction_images,
)


def _record_call(log_path, function, **inputs):
    """Keep detailed solver output on disk so the notebook shows measurements/results."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as log, redirect_stdout(log):
        return function(**inputs)


def acquisition_signature(
    *,
    protocol_path,
    projector_views_path,
    voltages_v,
    view_indices,
    random_seed,
    voltage_offset_v,
    centre_aperture=False,
):
    here = Path(__file__).parent
    return input_signature(
        paths=[
            protocol_path,
            projector_views_path,
            here / "condenser_voltage_experiment.py",
            here / "condenser_aperture_alignment.py",
            here / "condenser_calibration_model.py",
            here / "condenser_image_pipeline.py",
            here / "condenser_rotation_measurements.py",
            here / "condenser_aperture_experiment.py",
            here / "finite_projector_pipeline.py",
        ],
        settings={
            "voltages_v": [float(v) for v in voltages_v],
            "view_indices": [int(v) for v in view_indices],
            "random_seed": random_seed,
            "voltage_offset_v": float(voltage_offset_v),
            "centre_aperture": centre_aperture,
        },
    )


def prepare_measurements(
    *,
    protocol_path,
    projector_views_path,
    output_directory,
    voltages_v=(200000.0, 198000.0),
    view_indices=(0, 1, 2),
    random_seed=13001,
    voltage_offset_v=0.0,
    centre_aperture=False,
    reuse_completed=True,
):
    """Known specimen/calibration + recorded protocol -> new pixels/counts and measurements."""
    if not jax.config.x64_enabled:
        raise RuntimeError(
            'Enable JAX 64-bit arithmetic before acquisition: jax.config.update("jax_enable_x64", True).'
        )
    voltages_v = tuple(float(v) for v in voltages_v)
    voltage_offset_v = float(voltage_offset_v)
    folder = Path(output_directory)
    signature = acquisition_signature(
        protocol_path=protocol_path,
        projector_views_path=projector_views_path,
        voltages_v=voltages_v,
        view_indices=view_indices,
        random_seed=random_seed,
        voltage_offset_v=voltage_offset_v,
        centre_aperture=centre_aperture,
    )
    manifest = folder / "acquisition_manifest.json"
    required = [
        folder / f"voltage_{i}" / name
        for i in range(len(voltages_v))
        for name in [
            "images.npz",
            "template.npz",
            "detector.json",
            "counts.npz",
            "aperture.json",
        ]
    ]
    required += [
        folder / name
        for name in [
            "experiment.json",
            "reference_projector.json",
            "scoring_truth.json",
        ]
    ]
    if centre_aperture:
        required.append(folder / "alignment/alignment_constraint.json")
    if not (
        reuse_completed
        and manifest.exists()
        and all(p.exists() for p in required)
        and json.loads(manifest.read_text())["signature"] == signature
    ):
        simulate_voltage_experiment(
            protocol_path=protocol_path,
            projector_views_path=projector_views_path,
            output_directory=folder,
            voltages_v=voltages_v,
            view_indices=view_indices,
            random_seed=random_seed,
            voltage_offset_v=voltage_offset_v,
            centre_aperture=centre_aperture,
        )
        manifest.write_text(json.dumps({"signature": signature}))
    analyse_voltage_experiment(experiment_directory=folder, calibrate_voltage=True)
    from condenser_voltage_calibration import measure_voltage_correction

    target = folder / "voltage_1"
    calibration_signature = input_signature(
        paths=[
            projector_views_path,
            target / "images.npz",
            target / "measurements.npz",
            target / "template.npz",
            target / "detector.json",
            Path(__file__).with_name("condenser_voltage_calibration.py"),
            Path(__file__).with_name("condenser_image_pipeline.py"),
        ],
        settings={},
    )
    calibration_manifest = target / "voltage_calibration_manifest.json"
    if not (
        reuse_completed
        and calibration_manifest.exists()
        and (target / "voltage_calibration.json").exists()
        and (target / "voltage_calibrated_measurements.npz").exists()
        and json.loads(calibration_manifest.read_text())["signature"]
        == calibration_signature
    ):
        calibrated = measure_voltage_correction(
            image_directory=target,
            projector_views_path=projector_views_path,
            reference_voltage_v=voltages_v[0],
        )
        if not calibrated["converged"] or calibrated["pixel_rms"] > 1.5:
            raise RuntimeError(
                "Corrected pixel fit failed the convergence/noise check."
            )
        calibration_manifest.write_text(
            json.dumps({"signature": calibration_signature})
        )
    return folder


def fit_geometry(
    *,
    measurements_directory,
    starting_guess_path,
    number_of_broad_starts=4,
    max_evaluations=2000,
    reuse_completed=True,
):
    """Independent measurements + bounded guesses -> all fitted microscope geometries."""
    folder = Path(measurements_directory)
    files = [
        folder / "experiment.json",
        starting_guess_path,
        Path(__file__).with_name("condenser_voltage_experiment.py"),
        Path(__file__).with_name("condenser_geometry_audit.py"),
        Path(__file__).with_name("condenser_voltage_calibration.py"),
        folder / "voltage_1/voltage_calibrated_measurements.npz",
    ]
    count = len(json.loads((folder / "experiment.json").read_text())["voltages_v"])
    files += [
        folder / f"voltage_{i}" / name
        for i in range(count)
        for name in ["measurements.npz", "current_measurements.npz", "aperture.json"]
    ]
    if json.loads((folder / "experiment.json").read_text()).get(
        "aperture_centred_by_current_scan"
    ):
        files.append(folder / "alignment/alignment_constraint.json")
    signature = input_signature(
        paths=files,
        settings={
            "number_of_broad_starts": number_of_broad_starts,
            "max_evaluations": max_evaluations,
        },
    )
    manifest, result = (
        folder / "geometry_fit/manifest.json",
        folder / "geometry_fit/search.json",
    )
    if (
        reuse_completed
        and manifest.exists()
        and result.exists()
        and json.loads(manifest.read_text())["signature"] == signature
    ):
        return json.loads(result.read_text())
    from condenser_voltage_calibration import fit_calibrated_experiment

    fitted = _record_call(
        folder / "geometry_fit/fitting.log",
        fit_calibrated_experiment,
        experiment_directory=folder,
        starting_guess_path=starting_guess_path,
        number_of_broad_starts=number_of_broad_starts,
        max_evaluations=max_evaluations,
    )
    manifest.write_text(json.dumps({"signature": signature}))
    return fitted


def geometry_report(*, fitted, scoring_truth_path=None):
    """Compare acceptable alternatives. Hidden truth, if supplied, is used only here."""
    accepted = [r for r in fitted["starts"] if r["acceptable"]]
    p = np.asarray(fitted["best_fit"]["parameters"])
    values = np.asarray(physical_geometry(p))
    derivative = np.asarray(jax.jacfwd(physical_geometry)(p))
    sigma = (
        np.sqrt(
            np.maximum(
                np.diag(derivative @ np.asarray(fitted["covariance"]) @ derivative.T), 0
            )
        )
        if fitted["covariance"] is not None
        else np.full(len(values), np.nan)
    )
    tolerances = physical_tolerances(p)
    geometries = np.asarray([r["geometry"] for r in accepted])
    spans = np.ptp(geometries, axis=0) if accepted else np.full(len(values), np.nan)
    lateral_distance = (
        float(
            np.max(
                np.linalg.norm(
                    geometries[:, None, -2:] - geometries[None, :, -2:], axis=-1
                )
            )
        )
        if accepted
        else None
    )
    report = {
        "acceptable_starts": len(accepted),
        "attempted_starts": len(fitted["starts"]),
        "fit_rms": fitted["best_fit"]["residual_rms"],
        "rank": fitted["rank"],
        "parameter_count": fitted["parameter_count"],
        "aperture_centre_maximum_distance_um": lateral_distance,
        "point_solutions_agree_within_tolerances": bool(
            accepted
            and np.all(spans[:10] <= tolerances[:10])
            and lateral_distance <= 0.2
        ),
        "quantities": {
            name: {
                "fitted": float(values[i]),
                "local_sigma": float(sigma[i]),
                "acceptable_fit_span": float(spans[i]),
                "tolerance": float(tolerances[i]),
            }
            for i, name in enumerate(QUANTITIES)
        },
        "uncertainty_scope": "Local errors are not profile intervals or proof of uniqueness. Repeated-noise coverage is preliminary.",
    }
    if len(p) >= 29:
        from condenser_geometry_audit import physical_quantity

        name = "source_z_second_voltage_mm"
        value = float(physical_quantity(p, name))
        row_derivative = np.asarray(jax.grad(lambda v: physical_quantity(v, name))(p))
        uncertainty = (
            float(
                np.sqrt(
                    row_derivative @ np.asarray(fitted["covariance"]) @ row_derivative
                )
            )
            if fitted["covariance"] is not None
            else float("nan")
        )
        alternatives = [
            float(physical_quantity(np.asarray(r["parameters"]), name))
            for r in accepted
        ]
        span = float(np.ptp(alternatives)) if alternatives else float("nan")
        report["quantities"][name] = {
            "fitted": value,
            "local_sigma": uncertainty,
            "acceptable_fit_span": span,
            "tolerance": 1.0,
        }
        report["point_solutions_agree_within_tolerances"] &= bool(span <= 1.0)
    if scoring_truth_path is not None:
        truth = np.asarray(
            json.loads(Path(scoring_truth_path).read_text())["parameters"]
        )
        geometry = np.asarray(physical_geometry(truth))
        for i, name in enumerate(QUANTITIES):
            report["quantities"][name].update(
                truth=float(geometry[i]),
                error=float(values[i] - geometry[i]),
                truth_inside_local_95_percent=bool(
                    abs(values[i] - geometry[i]) <= 1.96 * sigma[i]
                ),
            )
        if len(p) >= 29:
            name = "source_z_second_voltage_mm"
            row = report["quantities"][name]
            actual = float(physical_quantity(truth, name))
            row.update(
                truth=actual,
                error=row["fitted"] - actual,
                truth_inside_local_95_percent=bool(
                    abs(row["fitted"] - actual) <= 1.96 * row["local_sigma"]
                ),
            )
    return report


def profile_geometry(
    *, measurements_directory, fitted, max_iterations=1500, reuse_completed=True
):
    """Fix each requested physical quantity, refit all others from two starts."""
    from condenser_geometry_audit import profile_native_quantity
    from condenser_voltage_calibration import make_calibrated_voltage_residual

    folder = Path(measurements_directory)
    output = folder / "geometry_profiles"
    report = geometry_report(fitted=fitted)
    p = np.asarray(fitted["best_fit"]["parameters"])
    accepted = [r for r in fitted["starts"] if r["acceptable"]]
    # Include the accepted solution farthest in aperture centre, if there is one.
    other = (
        max(
            accepted,
            key=lambda r: np.linalg.norm(
                np.asarray(r["geometry"])[-2:] - np.asarray(physical_geometry(p))[-2:]
            ),
        )
        if accepted
        else fitted["best_fit"]
    )
    third = p.copy()
    third[:13] += 0.0005 * np.random.default_rng(731).normal(size=13)
    starts = [p, np.asarray(other["parameters"])]
    if np.allclose(starts[0], starts[1], atol=1e-8, rtol=0):
        starts[1] = np.clip(
            third,
            np.asarray(fitted["bounds_lower"]) + 1e-9,
            np.asarray(fitted["bounds_upper"]) - 1e-9,
        )
    residual = make_calibrated_voltage_residual(experiment_directory=folder)
    np.asarray([r["geometry"] for r in accepted] or [physical_geometry(p)])
    result = {}
    for name in report["quantities"]:
        row = report["quantities"][name]
        sigma = row["local_sigma"]
        from condenser_geometry_audit import physical_quantity

        other_value = float(physical_quantity(np.asarray(other["parameters"]), name))
        alternative_values = [
            float(physical_quantity(np.asarray(r["parameters"]), name))
            for r in accepted
        ] or [row["fitted"]]
        targets = np.unique(
            np.r_[
                row["fitted"],
                row["fitted"] - 2 * sigma,
                row["fitted"] + 2 * sigma,
                min(alternative_values) - 4 * sigma,
                max(alternative_values) + 4 * sigma,
                other_value,
            ]
        )
        destination = output / f"{name}.json"
        saved = json.loads(destination.read_text()) if destination.exists() else None
        # Reuse only profiles with matching targets and the same reference fit.
        reference_path = output / "reference_fit.json"
        reference_matches = reference_path.exists() and np.array_equal(
            json.loads(reference_path.read_text()), p
        )
        if not (
            reuse_completed
            and reference_matches
            and saved
            and np.array_equal([r["target"] for r in saved["points"]], targets)
        ):
            _record_call(
                output / f"{name}.log",
                profile_native_quantity,
                residual=residual,
                fitted_parameters=p,
                bounds_lower=fitted["bounds_lower"],
                bounds_upper=fitted["bounds_upper"],
                quantity=name,
                targets=targets,
                starting_parameters=starts,
                output_directory=output,
                max_evaluations=max_iterations,
            )
            saved = json.loads(destination.read_text())
        result[name] = saved
    (output / "reference_fit.json").write_text(json.dumps(p.tolist()))
    return result


def show_experiment(*, measurements_directory):
    """Show real generated MAGICAL frames and the one-aperture current scan."""
    import matplotlib.pyplot as plt

    folder = Path(measurements_directory)
    experiment = json.loads((folder / "experiment.json").read_text())
    fig, axes = plt.subplots(
        2, len(experiment["voltages_v"]), figsize=(9, 6), squeeze=False
    )
    for i, voltage in enumerate(experiment["voltages_v"]):
        target = folder / f"voltage_{i}"
        beams = np.load(target / "measurements.npz")["values"]
        setting = int(
            np.argmax(beams[:, 4])
        )  # Show a broad illuminated region of MAGICAL.
        image = np.load(target / "images.npz")["images"][setting, 0]
        axes[0, i].imshow(image, cmap="gray")
        axes[0, i].set_title(f"{voltage / 1000:g} kV, setting {setting + 1}")
        axes[0, i].set_axis_off()
        data = np.load(target / "current_measurements.npz")
        for cl3 in np.unique(data["controls"][:, 1]):
            mask = data["controls"][:, 1] == cl3
            axes[1, i].plot(data["controls"][mask, 0], data["values"][mask, 0], ".-")
        axes[1, i].set(xlabel="CL1 control", ylabel="Transmitted fraction")
    fig.tight_layout()
    return fig


def finish_profile_intervals(
    *,
    measurements_directory,
    fitted,
    profiles,
    confidence=0.95,
    reuse_completed=True,
    max_evaluations=2000,
):
    """Locate the confidence limits by refitting at each bisection point.

    Limits are bracketed by a resolved inside and outside point. The reported
    outer bracket is conservative at the chosen numerical resolution. Failed
    fits never supply bounds. These are finite-start profile intervals, not a
    certificate that every global minimum has been found.
    """
    from condenser_geometry_audit import profile_native_quantity
    from condenser_voltage_calibration import make_calibrated_voltage_residual
    from scipy.stats import chi2

    folder = Path(measurements_directory)
    output = folder / "geometry_intervals"
    output.mkdir(exist_ok=True)
    threshold = float(chi2.ppf(confidence, 1))
    best = np.asarray(fitted["best_fit"]["parameters"])
    residual = make_calibrated_voltage_residual(experiment_directory=folder)
    report = geometry_report(fitted=fitted)
    results = {}
    for name, profile in profiles.items():
        signature = input_signature(
            paths=[
                Path(__file__).with_name("condenser_geometry_audit.py"),
                Path(__file__).with_name("condenser_voltage_calibration.py"),
            ],
            settings={
                "profile": profile,
                "fit_parameters": best.tolist(),
                "confidence": confidence,
            },
        )
        destination = output / f"{name}.json"
        if reuse_completed and destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("signature") == signature:
                results[name] = saved
                continue
        points = sorted(
            [point for point in profile["points"] if point["resolved"]],
            key=lambda point: point["target"],
        )
        if any(point["best"]["delta_chi_squared"] < -0.001 for point in points):
            raise RuntimeError(
                f"{name} profile found a lower fit. Add it to the alternative search before reporting intervals."
            )
        inside = [
            point for point in points if point["best"]["delta_chi_squared"] <= threshold
        ]
        brackets = []
        extra = []
        for direction in [-1, 1]:
            if not inside:
                brackets.append(None)
                continue
            inner = min(inside, key=lambda point: direction * -point["target"])
            outside = [
                point
                for point in points
                if direction * (point["target"] - inner["target"]) > 0
                and point["best"]["delta_chi_squared"] > threshold
            ]
            if not outside:
                brackets.append(None)
                continue
            outer = min(
                outside, key=lambda point: abs(point["target"] - inner["target"])
            )
            precision = max(report["quantities"][name]["local_sigma"] * 0.01, 1e-8)
            failed = False
            for iteration in range(12):
                if abs(outer["target"] - inner["target"]) <= precision:
                    break
                target = 0.5 * (outer["target"] + inner["target"])
                # Both sides provide physically valid, different starting guesses.
                starts = [
                    np.asarray(point["best"]["parameters"]) for point in [inner, outer]
                ]
                trial = _record_call(
                    output / "trials" / name / "progress.log",
                    profile_native_quantity,
                    residual=residual,
                    fitted_parameters=best,
                    bounds_lower=fitted["bounds_lower"],
                    bounds_upper=fitted["bounds_upper"],
                    quantity=name,
                    targets=[target],
                    starting_parameters=starts,
                    output_directory=output
                    / "trials"
                    / name
                    / f"{direction}_{iteration}",
                    max_evaluations=max_evaluations,
                )[0]
                extra.append(trial)
                if not trial["resolved"]:
                    failed = True
                    break
                if trial["best"]["delta_chi_squared"] < -0.001:
                    raise RuntimeError(
                        f"{name} refinement found a lower fit; repeat alternative search."
                    )
                if trial["best"]["delta_chi_squared"] <= threshold:
                    inner = trial
                else:
                    outer = trial
            brackets.append(
                None
                if failed
                else {"inside": inner["target"], "outside": outer["target"]}
            )
        complete = all(bracket is not None for bracket in brackets)
        interval = (
            [brackets[0]["outside"], brackets[1]["outside"]] if complete else None
        )
        result = {
            "quantity": name,
            "resolved": complete,
            "confidence": confidence,
            "delta_chi_squared": threshold,
            "interval": interval,
            "boundary_brackets": brackets,
            "additional_points": extra,
            "tolerance": report["quantities"][name]["tolerance"],
            "width_within_tolerance": bool(
                complete
                and interval[1] - interval[0] <= report["quantities"][name]["tolerance"]
            ),
            "scope": "Outer numerical bracket of the sampled profile confidence set; all other unknowns refitted from multiple starts.",
            "signature": signature,
        }
        destination.write_text(json.dumps(result, indent=2))
        results[name] = result
    return results


def show_geometry(*, fitted, scoring_truth_path=None):
    """One compact table, retaining the practical agreement result."""
    from IPython.display import Markdown, display

    report = geometry_report(fitted=fitted, scoring_truth_path=scoring_truth_path)
    lines = [
        "| Quantity | Fitted | Local ±1σ | Span of acceptable fits |",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["quantities"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {row['fitted']:.5g} | {row['local_sigma']:.3g} | {row['acceptable_fit_span']:.3g} |"
        )
    display(Markdown("\n".join(lines)))
    print(
        f"{report['acceptable_starts']}/{report['attempted_starts']} fitting attempts were acceptable; {fitted.get('initial_guess_count', report['attempted_starts'])} initial guesses."
    )
    print(
        "Point solutions agree within tolerances:",
        report["point_solutions_agree_within_tolerances"],
    )
    return report


def _run_robustness_checks(
    *,
    protocol_path,
    projector_views_path,
    starting_guess_path,
    output_directory,
    seeds=(18101, 18121, 18141),
    voltages_v=(200000.0, 195000.0),
    centre_aperture=True,
    number_of_broad_starts=8,
    voltage_offset_v=20.0,
    reuse_completed=True,
):
    """Repeat the complete pixel/count/analysis/fit pipeline; then challenge calibration.

    Three repeats are a preliminary coverage check, not a calibrated 95% coverage
    estimate. The 20 V readback error applies only at the second voltage; it is
    hidden from inference. A failure is reported rather than removed from the study.
    """
    from condenser_voltage_calibration import check_calibrated_predictions

    root = Path(output_directory)
    rows = []
    cases = [(f"voltage_{i + 1:02d}", seed, 0.0) for i, seed in enumerate(seeds)]
    cases.append(("voltage_readback_error", 13061, voltage_offset_v))
    for name, seed, offset in cases:
        folder = root / name
        try:
            prepare_measurements(
                protocol_path=protocol_path,
                projector_views_path=projector_views_path,
                output_directory=folder,
                random_seed=seed,
                voltages_v=voltages_v,
                centre_aperture=centre_aperture,
                voltage_offset_v=offset,
                reuse_completed=reuse_completed,
            )
            fitted = fit_geometry(
                measurements_directory=folder,
                starting_guess_path=starting_guess_path,
                number_of_broad_starts=number_of_broad_starts,
                reuse_completed=reuse_completed,
            )
            report = geometry_report(
                fitted=fitted, scoring_truth_path=folder / "scoring_truth.json"
            )
            profiles = profile_geometry(
                measurements_directory=folder,
                fitted=fitted,
                reuse_completed=reuse_completed,
            )
            intervals = finish_profile_intervals(
                measurements_directory=folder,
                fitted=fitted,
                profiles=profiles,
                reuse_completed=reuse_completed,
            )
            for quantity, interval in intervals.items():
                row = report["quantities"][quantity]
                row["profile_interval"] = interval["interval"]
                row["profile_width_within_tolerance"] = interval[
                    "width_within_tolerance"
                ]
                row["truth_inside_profile_95_percent"] = (
                    interval["interval"][0] <= row["truth"] <= interval["interval"][1]
                    if interval["resolved"]
                    else None
                )
            report["profile_widths_within_tolerances"] = all(
                v["width_within_tolerance"] for v in intervals.values()
            )
            (folder / "geometry_report.json").write_text(json.dumps(report, indent=2))
            row = {
                "case": name,
                "completed": True,
                "voltage_offset_v": offset,
                "geometry": report,
            }
            if fitted["covariance"] is not None:
                prediction_signature = input_signature(
                    paths=[
                        protocol_path,
                        folder / "scoring_truth.json",
                        folder / "geometry_fit/search.json",
                        Path(__file__).with_name("condenser_voltage_experiment.py"),
                        Path(__file__).with_name("condenser_voltage_calibration.py"),
                    ],
                    settings={"prediction_seed": seed + 1000},
                )
                manifest = folder / "prediction_manifest.json"
                output = folder / "prediction_check.json"
                if not (
                    reuse_completed
                    and manifest.exists()
                    and output.exists()
                    and json.loads(manifest.read_text())["signature"]
                    == prediction_signature
                ):
                    simulate_voltage_prediction_images(
                        experiment_directory=folder,
                        protocol_path=protocol_path,
                        random_seed=seed + 1000,
                    )
                    check_calibrated_predictions(
                        experiment_directory=folder, fitted=fitted
                    )
                    manifest.write_text(json.dumps({"signature": prediction_signature}))
                row["prediction"] = json.loads(output.read_text())
        except (RuntimeError, ValueError) as error:
            row = {
                "case": name,
                "completed": False,
                "voltage_offset_v": offset,
                "failure": str(error),
            }
        rows.append(row)
        (root / "voltage_robustness.json").write_text(
            json.dumps(
                {
                    "cases": rows,
                    "scope": "Complete-noise repeats with individual 95% profile intervals and a voltage-calibration challenge; finite starts do not prove global uniqueness.",
                },
                indent=2,
            )
        )
    return rows


def run_robustness_checks(
    *,
    protocol_path,
    projector_views_path,
    starting_guess_path,
    output_directory,
    seeds=(18101, 18121, 18141),
    voltages_v=(200000.0, 195000.0),
    centre_aperture=True,
    number_of_broad_starts=8,
    voltage_offset_v=20.0,
    reuse_completed=True,
):
    """Complete fresh-noise experiments and calibration challenge; retain a detailed log."""
    return _record_call(
        Path(output_directory) / "robustness.log",
        _run_robustness_checks,
        protocol_path=protocol_path,
        projector_views_path=projector_views_path,
        starting_guess_path=starting_guess_path,
        output_directory=output_directory,
        seeds=seeds,
        voltages_v=voltages_v,
        centre_aperture=centre_aperture,
        number_of_broad_starts=number_of_broad_starts,
        voltage_offset_v=voltage_offset_v,
        reuse_completed=reuse_completed,
    )


def show_checks(*, profiles, repetitions):
    """Short notebook assessment that distinguishes failed checks from uncertainty bounds."""
    from IPython.display import Markdown, display

    points = [point for profile in profiles.values() for point in profile["points"]]
    resolved = sum(point["resolved"] for point in points)
    print(
        f"Profile points resolved: {resolved}/{len(points)}. Unfinished points do not bound uncertainty."
    )
    lines = [
        "| Experiment | Fit RMS | Geometry meets tolerances? | Truth in profile 95% intervals | New-image RMS | New-current RMS |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in repetitions:
        if not row["completed"]:
            lines.append(f"| {row['case']} | failed | — | — | — | — |")
            continue
        report = row["geometry"]
        coverage = sum(
            v.get("truth_inside_profile_95_percent") is True
            for v in report["quantities"].values()
        )
        prediction = row.get("prediction", {}).get("pixel_rms")
        pixel = f"{prediction:.3f}" if prediction is not None else "—"
        current = row.get("prediction", {}).get("current_prediction_rms")
        current_text = f"{current:.3f}" if current is not None else "—"
        geometry_pass = (
            report["point_solutions_agree_within_tolerances"]
            and report["profile_widths_within_tolerances"]
        )
        lines.append(
            f"| {row['case']} | {report['fit_rms']:.3f} | {geometry_pass} | {coverage}/{len(report['quantities'])} | {pixel} | {current_text} |"
        )
    display(Markdown("\n".join(lines)))
    print(
        "These are individual profile intervals from a finite search. A few noise repeats cannot establish 95% coverage."
    )


def show_profile_geometry(*, fitted, intervals, scoring_truth_path=None):
    """Display measured physical intervals and the simulation's separate truth check."""
    from IPython.display import Markdown, display

    report = geometry_report(fitted=fitted, scoring_truth_path=scoring_truth_path)
    lines = [
        "| Quantity | Fitted | 95% profile interval | Error from truth | Width meets tolerance? |",
        "|---|---:|---|---:|---|",
    ]
    for name, row in report["quantities"].items():
        interval = intervals[name]
        limits = interval["interval"]
        text = f"[{limits[0]:.6g}, {limits[1]:.6g}]" if limits else "unresolved"
        error = f"{row['error']:.3g}" if "error" in row else "—"
        lines.append(
            f"| {name.replace('_', ' ')} | {row['fitted']:.6g} | {text} | {error} | {interval['width_within_tolerance']} |"
        )
    display(Markdown("\n".join(lines)))
    print(
        f"Initial guesses: {fitted.get('initial_guess_count', report['attempted_starts'])}. Acceptable fitted geometries agree within tolerances: {report['point_solutions_agree_within_tolerances']}."
    )
    return report
