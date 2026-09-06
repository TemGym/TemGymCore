"""Readable notebook steps for the condenser image experiment."""

import hashlib
import json
from contextlib import redirect_stdout
from pathlib import Path

import jax
import numpy as np
from condenser_calibration_model import (
    equivalent_parameters,
    matrix5,
    moment_observables,
    named_parameters,
)
from condenser_fitting import beam_to_moments, fit_condenser, model_moments
from condenser_image_pipeline import measure_condenser_images, simulate_condenser_images


def _fingerprint(paths, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _cached(folder, stage, signature, outputs):
    record = Path(folder) / f"{stage}_cache.json"
    return (
        record.exists()
        and json.loads(record.read_text())["signature"] == signature
        and all(Path(path).exists() for path in outputs)
    )


def _record(folder, stage, signature):
    (Path(folder) / f"{stage}_cache.json").write_text(
        json.dumps({"signature": signature})
    )


def make_images(
    *,
    protocol_path,
    projector_views_path,
    split,
    output_directory,
    reuse_completed=True,
):
    """Supply recorded controls/maps; return a folder of noisy MAGICAL images."""
    if split not in ("calibration", "prediction"):
        raise ValueError("Choose calibration or prediction.")
    folder = Path(output_directory)
    protocol = json.loads(Path(protocol_path).read_text())
    views = json.loads(Path(projector_views_path).read_text())
    controls = np.asarray(protocol[f"{split}_controls"])
    matrices = np.asarray([v["matrix"] for v in views["views"]])
    seed = protocol[f"{split}_noise_seed"]
    here = Path(__file__).parent
    signature = _fingerprint(
        [
            protocol_path,
            projector_views_path,
            here / "condenser_calibration_model.py",
            here / "condenser_image_pipeline.py",
            here / "magical_projector_pipeline.py",
        ],
        {"split": split, "seed": seed},
    )
    outputs = [
        folder / name
        for name in (
            "images.npz",
            "template.npz",
            "detector.json",
            "scoring_truth.json",
        )
    ]
    if not reuse_completed or not _cached(folder, "generation", signature, outputs):
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "generation.log").open("w") as log, redirect_stdout(log):
            simulate_condenser_images(
                projector_matrices=matrices,
                controls=controls,
                random_seed=seed,
                output_directory=folder,
            )
        _record(folder, "generation", signature)
    print(f"{len(controls) * len(matrices)} {split} images ready.")
    return folder


def measure_images(*, image_directory, reuse_completed=True, initialization_only=False):
    """Supply images and known imaging model; return measured beams with covariance."""
    folder = Path(image_directory)
    signature = _fingerprint(
        [
            folder / "images.npz",
            folder / "template.npz",
            folder / "detector.json",
            Path(__file__).with_name("condenser_image_pipeline.py"),
        ],
        {"initialization_only": initialization_only},
    )
    output = folder / "measurements.npz"
    if not reuse_completed or not _cached(folder, "measurement", signature, [output]):
        with (folder / "measurement.log").open("w") as log, redirect_stdout(log):
            measure_condenser_images(
                input_directory=folder, initialization_only=initialization_only
            )
        _record(folder, "measurement", signature)
    print(
        f"Measured beams ready for {len(np.load(output)['controls'])} condenser settings."
    )
    return output


def fit_microscope(
    *,
    measurements_path,
    starting_parameters,
    output_directory,
    number_of_starts=12,
    reuse_completed=True,
):
    """Supply measured beams and guesses; return a representative plus equivalent models."""
    folder = Path(output_directory)
    signature = _fingerprint(
        [
            measurements_path,
            Path(__file__).with_name("condenser_fitting.py"),
            Path(__file__).with_name("condenser_calibration_model.py"),
        ],
        {
            "starting_parameters": np.asarray(starting_parameters).tolist(),
            "starts": number_of_starts,
        },
    )
    output = folder / "fit_summary.json"
    if not reuse_completed or not _cached(folder, "fit", signature, [output]):
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "fitting.log").open("w") as log, redirect_stdout(log):
            fit_condenser(
                measurements_path=measurements_path,
                starting_parameters=starting_parameters,
                number_of_starts=number_of_starts,
                output_directory=folder,
            )
        _record(folder, "fit", signature)
    result = json.loads(output.read_text())
    print(
        f"Fit ready: {result['acceptable_starts']}/{number_of_starts} starts converged; "
        f"data rank {result['data_rank']}/19."
    )
    return result


def check_predictions(*, fitted, measurements_path):
    """Compare a frozen fit with separately measured beams; no refitting or truth access."""
    data = np.load(measurements_path)
    transform = np.asarray(jax.vmap(jax.jacfwd(beam_to_moments))(data["values"]))
    measured = np.asarray(jax.vmap(beam_to_moments)(data["values"]))
    cov = transform @ data["covariance"] @ transform.transpose(0, 2, 1)
    predicted = np.asarray(
        jax.vmap(model_moments, in_axes=(None, 0))(
            np.asarray(fitted["best_fit"]["parameters"]), data["controls"]
        )
    )
    normalized = np.linalg.solve(
        np.linalg.cholesky(cov), (predicted - measured)[..., None]
    )[..., 0]
    return {
        "measurement_rms": float(np.sqrt(np.mean(normalized**2))),
        "setting_count": len(measured),
        "image_count": len(measured) * 5,
        "includes_fit_uncertainty": False,
    }


def score_against_truth(*, fitted, scoring_truth_path):
    """Only this final scoring step opens the generating parameter file."""
    truth = np.asarray(json.loads(Path(scoring_truth_path).read_text())["parameters"])
    best = np.asarray(fitted["best_fit"]["parameters"])
    # Align the two arbitrary coordinate choices before measuring finite-noise
    # parameter error. This alignment uses truth only for scoring.
    aligned = equivalent_parameters(
        truth,
        upstream_scale=np.exp(best[0] - truth[0]),
        objective_strength=-np.exp(best[12]),
    )
    truth_named, aligned_named = named_parameters(truth), named_parameters(aligned)
    best_named = named_parameters(best)
    rows = [
        {
            "parameter": name,
            "truth": value,
            "fit": best_named[name],
            "truth_in_same_representative": aligned_named[name],
        }
        for name, value in truth_named.items()
    ]
    return {
        "parameters": rows,
        "maximum_controlled_strength_error_percent": float(
            100 * np.max(abs(np.exp(best[9:12] - truth[9:12]) - 1))
        ),
        "source_emittance_error_percent": float(
            100 * np.expm1(best[17:].sum() - truth[17:].sum())
        ),
    }


def check_identifiability(*, parameters, controls):
    """Ideal-information check, not an additional simulated measurement."""
    p = np.asarray(parameters)

    def beams(p):
        return jax.vmap(moment_observables, in_axes=(None, 0))(p, controls).ravel()

    def beams_and_known_source_angle_change(p):
        import jax.numpy as jnp

        response = (
            matrix5(p, controls[0])[:4, 2] * 1e-5 / jnp.array([1e-7, 1e-7, 1e-3, 1e-3])
        )
        return jnp.r_[beams(p), response]

    result = {}
    for name, fun in [
        ("current_measurements", beams),
        ("hypothetical_source_angle_change", beams_and_known_source_angle_change),
    ]:
        singular_values = np.linalg.svd(jax.jacfwd(fun)(p), compute_uv=False)
        result[name] = {
            "rank": int(np.sum(singular_values > singular_values[0] * 1e-12)),
            "singular_values": singular_values.tolist(),
        }
    return result


def show_images(*, image_directory, setting_index=0):
    import matplotlib.pyplot as plt

    data = np.load(Path(image_directory) / "images.npz")
    _fig, axes = plt.subplots(1, 6, figsize=(15, 3), constrained_layout=True)
    template = np.load(Path(image_directory) / "template.npz")
    axes[0].imshow(template["intensity"], origin="lower", cmap="gray")
    axes[0].set_title("Known specimen")
    axes[0].set_axis_off()
    labels = ["Specimen focus", "Defocus +", "Defocus −", "Lower mag +", "Lower mag −"]
    for ax, frame, label in zip(axes[1:], data["images"][setting_index], labels):
        ax.imshow(
            frame, origin="lower", cmap="gray", vmin=0, vmax=np.percentile(frame, 99.8)
        )
        ax.set_title(label)
        ax.set_axis_off()
    plt.show()


def show_results(*, fitted, checks, scores):
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    best = fitted["best_fit"]
    display(
        Markdown(
            f"**Calibration:** {fitted['image_count']} images; measurement residual RMS {best['data_rms']:.3f}. "
            f"**Predictions:** {checks['image_count']} new images; RMS {checks['measurement_rms']:.3f}.\n\n"
            f"{fitted['acceptable_starts']}/{fitted['number_of_starts']} starts met the optimizer stopping criteria. "
            "The other starts reached the evaluation limit; this is not a completed search for all minima.\n\n"
            f"The three controlled strength gains have a maximum error of "
            f"{scores['maximum_controlled_strength_error_percent']:.3f}%. "
            "The physical layout is still ambiguous. Prediction RMS uses measurement uncertainty only."
        )
    )
    rows = scores["parameters"]
    table = "| Parameter | Generating model | Fit (one representative) | Truth with same coordinate choices |\n|---|---:|---:|---:|\n"
    table += "\n".join(
        f"| {r['parameter']} | {r['truth']:.5g} | {r['fit']:.5g} | {r['truth_in_same_representative']:.5g} |"
        for r in rows
    )
    display(Markdown(table))
    _fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    singular = np.asarray(fitted["singular_values"])
    axes[0].semilogy(np.arange(1, 20), singular / singular[0], "o-")
    axes[0].set(
        xlabel="Sensitivity direction",
        ylabel="Relative sensitivity",
        title="Two unconstrained directions",
    )
    colours = ["C0", "C1", "C2", "C3"]
    for member, colour in zip(fitted["equivalent_models"], colours):
        model = member["model"]
        z = [
            model["source_z_mm"],
            *[
                model[f"{n}.z_mm"]
                for n in ("CL1", "CL3", "Mini", "Objective pre-field")
            ],
            0.0,
        ]
        row = colours.index(colour)
        axes[1].plot(z, np.repeat(row, len(z)), "o-", color=colour)
    axes[1].set(
        xlabel="Position relative to specimen (mm)",
        yticks=range(4),
        yticklabels=["Fit", "Equivalent 1", "Equivalent 2", "Equivalent 3"],
        title="Identical predicted specimen beams",
    )
    plt.show()


def save_summary(*, fitted, checks, scores, identifiability, output_path):
    """Save the reviewable numbers without bundling the raw image arrays."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fit": fitted,
                "predictions": checks,
                "truth_comparison": scores,
                "ideal_information_check": identifiability,
            },
            indent=2,
        )
    )
    return path
