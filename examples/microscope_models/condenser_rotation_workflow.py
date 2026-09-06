"""Readable notebook steps for rotation measurement, fitting and checks."""

import hashlib
import json
from pathlib import Path

import numpy as np


def input_signature(*, paths, settings):
    """Fingerprint array contents, avoiding irrelevant NPZ zip timestamps."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for filename in paths:
        path = Path(filename)
        if path.suffix == ".npz":
            data = np.load(path)
            for key in sorted(data.files):
                value = np.asarray(data[key])
                digest.update(key.encode())
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(value.tobytes())
        else:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_rotation_measurements(
    *,
    projector_views_path,
    protocol_path,
    output_directory,
    random_seed=9101,
    reuse_completed=True,
):
    """Known projector and recorded protocol -> beam images, current readings and measurements."""
    from condenser_rotation_measurements import (
        generate_rotation_and_focus,
        measure_rotation_and_focus,
    )

    folder = Path(output_directory)
    signature = input_signature(
        paths=[
            projector_views_path,
            protocol_path,
            Path(__file__).with_name("condenser_rotation_measurements.py"),
        ],
        settings={"random_seed": random_seed},
    )
    manifest = folder / "measurement_manifest.json"
    if (
        reuse_completed
        and manifest.exists()
        and json.loads(manifest.read_text())["signature"] == signature
        and all(
            (folder / name).exists()
            for name in [
                "rotation_images.npz",
                "rotation_measurements.npz",
                "calibration_counts.npz",
                "calibration_measurements.npz",
            ]
        )
    ):
        return folder
    generate_rotation_and_focus(
        projector_views_path=projector_views_path,
        protocol_path=protocol_path,
        output_directory=folder,
        random_seed=random_seed,
    )
    measure_rotation_and_focus(input_directory=folder)
    manifest.write_text(
        json.dumps({"signature": signature, "random_seed": random_seed}, indent=2)
    )
    return folder


def cached_rotation_fit(
    *,
    rotation_directory,
    baseline_directory,
    baseline_aperture_directory,
    objective_directory,
    output_directory,
    include_objective,
    number_of_starts=4,
    max_evaluations=5000,
    reuse_completed=True,
):
    """Measured quantities and earlier fitted guesses -> every fitted start and local uncertainty."""
    from condenser_rotation_fitting import fit_with_rotation

    arguments = {
        "rotation_directory": rotation_directory,
        "baseline_directory": baseline_directory,
        "baseline_aperture_directory": baseline_aperture_directory,
        "objective_directory": objective_directory,
        "output_directory": output_directory,
        "include_objective": include_objective,
        "number_of_starts": number_of_starts,
        "max_evaluations": max_evaluations,
    }
    signature = fit_signature(**arguments)
    folder = Path(output_directory)
    manifest = folder / "fit_manifest.json"
    output = folder / "fit_summary.json"
    if (
        reuse_completed
        and manifest.exists()
        and output.exists()
        and json.loads(manifest.read_text())["signature"] == signature
    ):
        return json.loads(output.read_text())
    result = fit_with_rotation(**arguments)
    manifest.write_text(json.dumps({"signature": signature}, indent=2))
    return result


def fit_signature(
    *,
    rotation_directory,
    baseline_directory,
    baseline_aperture_directory,
    objective_directory,
    include_objective,
    number_of_starts,
    max_evaluations,
    output_directory=None,
):
    rotation, baseline, aperture, objective = map(
        Path,
        [
            rotation_directory,
            baseline_directory,
            baseline_aperture_directory,
            objective_directory,
        ],
    )
    paths = [
        rotation / "rotation_measurements.npz",
        rotation / "calibration_measurements.npz",
        baseline / "calibration/measurements.npz",
        aperture / "calibration_measurements.npz",
        aperture / "fit/joint_fit_summary.json",
    ]
    if include_objective:
        paths += [
            objective / "calibration_measurements.npz",
            objective / "reference_projector.json",
            objective / "detector.json",
            objective / "fit_free_post/fit_summary.json",
        ]
    for name in [
        "condenser_rotation_fitting.py",
        "condenser_rotation_measurements.py",
        "condenser_objective_fitting.py",
        "condenser_objective_experiment.py",
        "condenser_aperture_experiment.py",
        "condenser_calibration_model.py",
    ]:
        paths.append(Path(__file__).with_name(name))
    return input_signature(
        paths=paths,
        settings={
            "include_objective": include_objective,
            "number_of_starts": number_of_starts,
            "max_evaluations": max_evaluations,
        },
    )


def show_rotation_measurements(*, measurement_directory):
    """Show one actual ellipse image, focus sweeps and measured angle errors."""
    import matplotlib.pyplot as plt
    from condenser_rotation_fitting import fit_rotation_slopes
    from IPython.display import Markdown, display

    folder = Path(measurement_directory)
    images = np.load(folder / "rotation_images.npz")
    angles = np.load(folder / "rotation_measurements.npz")
    currents = np.load(folder / "calibration_measurements.npz")
    counts = np.load(folder / "calibration_counts.npz")
    slopes = fit_rotation_slopes(measurements_path=folder / "rotation_measurements.npz")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    axes[0].imshow(images["images"][0], cmap="gray")
    axes[0].set_title("Elliptical illumination", fontsize=11)
    axes[0].set_axis_off()
    for group in np.unique(counts["groups"]):
        use = counts["groups"] == group
        axes[1].plot(
            currents["controls"][use, 0],
            currents["values"][use, -1],
            ".-",
            label=f"CL3 = {currents['controls'][use, 1][0]:.2g}",
        )
    axes[1].set(
        xlabel="CL1 command",
        ylabel="Transmitted / open current",
        title="20 µm aperture: both sides of focus",
    )
    axes[1].legend(fontsize=8)
    predicted = slopes["intercept"] - angles["controls"] @ slopes["gains"]
    error = 0.5 * np.arctan2(
        np.sin(2 * (angles["angles"] - predicted)),
        np.cos(2 * (angles["angles"] - predicted)),
    )
    axes[2].errorbar(
        np.arange(len(error)),
        error * 1000,
        yerr=np.sqrt(angles["variance"]) * 1000,
        fmt=".",
    )
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set(
        xlabel="Recorded lens setting",
        ylabel="Angle residual (mrad)",
        title="Rotation measured from pixels",
    )
    figure.savefig(folder / "rotation_measurement_check.png", dpi=160)
    display(figure)
    plt.close(figure)
    display(
        Markdown(
            "**24 ellipse images and 492 new current readings.** Each angle comes from a separate pixel fit. The microscope fit receives those measured angles and their errors, not the generating ellipse angle."
        )
    )


def show_rotation_fit(
    *,
    fixed_fit,
    objective_fit,
    checks,
    family_check,
    profile,
    aperture_profile,
    information,
    measurement_directory,
):
    """One comparison table, a thickness profile and the physical geometry errors."""
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    lines = [
        "| Measurements | Local rank | Fit RMS | Converged starts |",
        "|---|---:|---:|---:|",
    ]
    for label, result in [
        ("Fixed objective + rotation + focus sweeps", fixed_fit),
        ("Also vary objective", objective_fit),
    ]:
        good = sum(s["success"] for s in result["starts"])
        lines.append(
            f"| {label} | {result['rank']}/{result['parameter_count']} | {result['best_fit']['residual_rms']:.3f} | {good}/{len(result['starts'])} |"
        )
    display(Markdown("\n".join(lines)))
    lengths = [x["prefield_length_mm"] for x in family_check]
    change = max(x["max_change_in_noise_units"] for x in family_check)
    display(
        Markdown(
            f"At fixed objective current, pre-field lengths from **{min(lengths):.3f} to {max(lengths):.3f} mm** give the same observations to within **{change:.1g} noise units**. Rotation does not remove this exact ambiguity."
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    profile = sorted(profile, key=lambda x: x["length_mm"])
    x = np.array([r["length_mm"] for r in profile])
    y = np.array([r["delta_chi_squared"] for r in profile])
    for ax in axes[:2]:
        ax.plot(x, y, ".-")
        ax.axhline(1, color="gray", ls="--", lw=0.8)
        ax.set(xlabel="Fixed pre-field thickness (mm)", ylabel="Extra chi-squared")
        for row in profile:
            if not row["success"]:
                ax.plot(row["length_mm"], row["delta_chi_squared"], "rx")
    axes[0].set_title("All other parameters refitted")
    axes[0].set_yscale("symlog", linthresh=2)
    axes[0].set_ylim(-0.05, max(y) * 1.2)
    centre = np.exp(objective_fit["best_fit"]["parameters"][8]) * 1000
    sigma = centre * np.sqrt(objective_fit["covariance"][8][8])
    axes[1].set(
        xlim=(centre - 4 * sigma, centre + 4 * sigma),
        ylim=(-0.1, 12),
        title="Near the fitted thickness",
    )
    aperture_profile = sorted(aperture_profile, key=lambda r: r["angle_degrees"])
    axes[2].plot(
        [r["angle_degrees"] for r in aperture_profile],
        [r["delta_chi_squared"] for r in aperture_profile],
        ".-",
    )
    axes[2].axhline(3.84, color="gray", ls="--", lw=0.8)
    axes[2].set(
        xlabel="Aperture-centre direction (degrees)",
        ylabel="Extra chi-squared",
        title="Centre direction remains uncertain",
        yscale="symlog",
    )
    axes[2].set_yscale("symlog", linthresh=2)
    axes[2].set_ylim(-0.05, max(r["delta_chi_squared"] for r in aperture_profile) * 1.2)
    figure.savefig(Path(measurement_directory) / "prefield_profile.png", dpi=160)
    display(figure)
    plt.close(figure)
    lines = [
        "| Data at the same fitted geometry | Source-position error (mm) | Pre-field thickness error (mm) |",
        "|---|---:|---:|",
    ]
    for r in information:
        lines.append(
            f"| {r['measurements']} | {r['source_z_sigma_mm']:.4g} | {r['prefield_length_sigma_mm']:.4g} |"
        )
    display(
        Markdown(
            "Local **one-standard-deviation** errors, conditional on the model:\n\n"
            + "\n".join(lines)
        )
    )
    lines = ["| Geometry (mm) | Hidden truth | Fitted |", "|---|---:|---:|"]
    for r in checks["parameter_comparison"]:
        if r["parameter"].endswith(("z_mm", "length_mm")):
            lines.append(f"| {r['parameter']} | {r['truth']:.5g} | {r['fit']:.5g} |")
    display(Markdown("\n".join(lines)))
