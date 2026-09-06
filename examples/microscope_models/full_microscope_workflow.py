"""One notebook entry point for the complete two-voltage microscope experiment."""

import inspect
import json
from contextlib import redirect_stdout
from pathlib import Path

import jax
import numpy as np
from condenser_objective_experiment import choose_objective_projector_views
from condenser_rotation_workflow import input_signature
from condenser_voltage_experiment import (
    simulate_voltage_experiment,
    simulate_voltage_prediction_images,
)
from finite_projector_pipeline import _generator_parameters
from full_microscope_apertures import simulate_aligned_apertures
from full_microscope_checks import check_microscope as _check_microscope
from full_microscope_checks import show_result
from full_microscope_inference import (
    assemble_full_fit,
    fit_condenser_block,
    measure_specimen_beams,
    prepare_condenser_analysis,
)
from full_microscope_projector import (
    fit_projector_block,
    load_projector_images,
    measure_projector_block,
    simulate_projector_block,
)

__all__ = [
    "check_full_microscope",
    "fit_microscope",
    "fit_projector",
    "measure_condenser",
    "measure_projector",
    "show_images",
    "show_result",
    "simulate_experiment",
]

HERE = Path(__file__).parent


def _record_step(
    *, directory, name, signature, required, function, reuse_completed, **inputs
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{name}_manifest.json"
    if (
        reuse_completed
        and marker.exists()
        and json.loads(marker.read_text())["signature"] == signature
        and all(Path(p).exists() for p in required)
    ):
        return
    marker.unlink(missing_ok=True)
    with (directory / f"{name}.log").open("a") as log, redirect_stdout(log):
        function(**inputs)
    if not all(Path(p).exists() for p in required):
        raise RuntimeError(f"{name} did not produce all required outputs; see its log.")
    marker.write_text(json.dumps({"signature": signature}, indent=2))


def _generate_full(
    *, root, voltages_v, random_seed, aperture_diameters_um, centering_sigma_um
):
    simulate_projector_block(
        protocol_path=HERE / "projector_measurement_protocol.json",
        output_directory=root / "projector",
        voltages_v=voltages_v,
        random_seed=random_seed,
    )
    # Two distinct downstream settings at each voltage provide separate tests.
    prediction_protocol = json.loads(
        (HERE / "projector_prediction_protocol.json").read_text()
    )
    for key in ["lens_controls", "nominal_magnifications", "labels"]:
        prediction_protocol[key] = prediction_protocol[key][:2]
    prediction_protocol_path = root / "projector_prediction_protocol.json"
    prediction_protocol_path.write_text(json.dumps(prediction_protocol, indent=2))
    simulate_projector_block(
        protocol_path=prediction_protocol_path,
        output_directory=root / "projector_predictions",
        voltages_v=voltages_v,
        random_seed=random_seed + 2000,
    )

    # Both blocks use the SAME generating downstream microscope. Its parameters
    # are only an instrument-generation input, never an inference calibration.
    p, _ = _generator_parameters()
    views = json.loads((HERE / "condenser_projector_views.json").read_text())
    views["calibrated_projector_parameters"] = p.tolist()
    views["source"] = (
        "Simulation instrument parameters; not supplied to analysis or fitting."
    )
    maps, commands = choose_objective_projector_views(views, 1.0)
    for view, matrix, command in zip(views["views"], maps, commands):
        view.update(matrix=matrix.tolist(), controls=command.tolist())
    generating_path = root / "generating_projector.json"
    generating_path.write_text(json.dumps(views, indent=2))
    simulate_voltage_experiment(
        protocol_path=HERE / "condenser_voltage_protocol.json",
        projector_views_path=generating_path,
        output_directory=root / "condenser",
        voltages_v=voltages_v,
        random_seed=random_seed + 100,
        centre_aperture=False,
    )
    simulate_voltage_prediction_images(
        experiment_directory=root / "condenser",
        protocol_path=HERE / "condenser_voltage_protocol.json",
        random_seed=random_seed + 1100,
    )
    simulate_aligned_apertures(
        experiment_directory=root / "condenser",
        diameters_um=aperture_diameters_um,
        centering_sigma_um=centering_sigma_um,
        random_seed=random_seed + 8000,
    )
    (root / "experiment.json").write_text(
        json.dumps(
            {
                "voltages_v": voltages_v,
                "random_seed": random_seed,
                "objective_current": 1.0,
                "projector_calibration_images": 198,
                "condenser_calibration_images": 144,
                "calibration_current_readings": 78 * (len(aperture_diameters_um) + 1),
                "aperture_diameters_um": aperture_diameters_um,
                "centering_sigma_um": centering_sigma_um,
                "movement_scan": False,
                "prediction_images": 72,
                "prediction_current_readings": 16 * (len(aperture_diameters_um) + 1),
                "scope": "Known voltages and independently calibrated specimen-plane shifts/tilts at each voltage. One shared geometry; objective pre/post halves have separate effective field lengths.",
            },
            indent=2,
        )
    )


def simulate_experiment(
    *,
    output_directory,
    voltages_v=(200000.0, 195000.0),
    random_seed=24001,
    aperture_diameters_um=(4.0, 10.0),
    centering_sigma_um=0.2,
    reuse_completed=True,
):
    """Supply voltages and a noise seed; get all pixels/counts for calibration and checks."""
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "Enable JAX 64-bit arithmetic before running this experiment."
        )
    voltages = tuple(float(v) for v in voltages_v)
    if len(voltages) != 2 or min(voltages) <= 0 or voltages[0] == voltages[1]:
        raise ValueError("Supply two distinct positive voltages.")
    diameters = tuple(float(d) for d in aperture_diameters_um)
    if (
        len(diameters) not in (2, 3)
        or min(diameters) <= 0
        or len(set(diameters)) != len(diameters)
    ):
        raise ValueError("Supply two or three distinct positive aperture diameters.")
    if centering_sigma_um <= 0:
        raise ValueError("Supply a positive centering uncertainty in micrometres.")
    root = Path(output_directory)
    paths = [
        HERE / name
        for name in [
            "full_microscope_projector.py",
            "full_microscope_apertures.py",
            "finite_projector_pipeline.py",
            "condenser_voltage_experiment.py",
            "condenser_calibration_model.py",
            "condenser_aperture_alignment.py",
            "condenser_image_pipeline.py",
            "magical_projector_pipeline.py",
            "projector_tilt_pipeline.py",
            "projector_beam_shift_pipeline.py",
            "condenser_objective_experiment.py",
            "projector_measurement_protocol.json",
            "projector_prediction_protocol.json",
            "condenser_voltage_protocol.json",
            "condenser_projector_views.json",
        ]
    ]
    signature = input_signature(
        paths=paths,
        settings={
            "voltages_v": voltages,
            "random_seed": random_seed,
            "generator": inspect.getsource(_generate_full),
            "aperture_diameters_um": diameters,
            "centering_sigma_um": centering_sigma_um,
        },
    )
    required = [
        root / "experiment.json",
        root / "projector/scoring_truth.json",
        root / "condenser/scoring_truth.json",
    ]
    required += [
        root / block / f"voltage_{index}" / name
        for block in ["projector", "projector_predictions"]
        for index in [0, 1]
        for name in [
            "images.npz",
            "tilt_images_3mrad.npz",
            "beam_shift_images_10nm.npz",
            "specimen_template.npz",
            "known_settings.json",
        ]
    ]
    required += [
        root / "condenser" / f"voltage_{index}" / name
        for index in [0, 1]
        for name in [
            "images.npz",
            "counts.npz",
            "template.npz",
            "detector.json",
            "aperture.json",
            "prediction/images.npz",
            "prediction/current_counts.npz",
        ]
    ]
    required.append(root / "condenser/alignment/alignment_constraint.json")
    _record_step(
        directory=root,
        name="generation",
        signature=signature,
        required=required,
        function=_generate_full,
        reuse_completed=reuse_completed,
        root=root,
        voltages_v=voltages,
        random_seed=random_seed,
        aperture_diameters_um=diameters,
        centering_sigma_um=centering_sigma_um,
    )
    return root


def measure_projector(*, experiment_directory, reuse_completed=True):
    """Supply images; get scale, rotation, blur and signed displacements with covariance."""
    root = Path(experiment_directory) / "projector"
    paths = [
        root / f"voltage_{i}" / name
        for i in [0, 1]
        for name in [
            "images.npz",
            "tilt_images_3mrad.npz",
            "beam_shift_images_10nm.npz",
            "specimen_template.npz",
            "known_settings.json",
        ]
    ]
    signature = input_signature(
        paths=paths
        + [
            HERE / "full_microscope_projector.py",
            HERE / "projector_calibration_workflow.py",
            HERE / "projector_beam_shift_pipeline.py",
            HERE / "projector_tilt_pipeline.py",
            HERE / "magical_projector_pipeline.py",
        ],
        settings={},
    )
    _record_step(
        directory=root,
        name="analysis",
        signature=signature,
        required=[root / f"voltage_{i}/measurements.npz" for i in [0, 1]],
        function=measure_projector_block,
        reuse_completed=reuse_completed,
        images_directory=root,
    )
    return root


def fit_projector(
    *,
    measurements_directory,
    starting_guess_path,
    bounds_path,
    number_of_starts=8,
    reuse_completed=True,
):
    """Fit one shared projector geometry from two-voltage measured responses."""
    root = Path(measurements_directory)
    signature = input_signature(
        paths=[
            root / "experiment.json",
            root / "voltage_0/measurements.npz",
            root / "voltage_1/measurements.npz",
            starting_guess_path,
            bounds_path,
            HERE / "full_microscope_projector.py",
        ],
        settings={"starts": number_of_starts},
    )
    path = root / "fit/result.json"
    _record_step(
        directory=root,
        name="fitting",
        signature=signature,
        required=[path],
        function=fit_projector_block,
        reuse_completed=reuse_completed,
        measurements_directory=root,
        starting_guess_path=starting_guess_path,
        bounds_path=bounds_path,
        number_of_starts=number_of_starts,
    )
    return json.loads(path.read_text())


def _measure_condenser(*, root, projector_fit, bounds_path):
    experiment = json.loads((root / "experiment.json").read_text())
    analysis = prepare_condenser_analysis(
        images_directory=root / "condenser",
        projector_fit=projector_fit,
        reference_voltage_v=experiment["voltages_v"][0],
    )
    measure_specimen_beams(
        analysis_directory=analysis,
        projector_measurements_directory=root / "projector",
        projector_fit=projector_fit,
        bounds_path=bounds_path,
    )


def measure_condenser(
    *, experiment_directory, projector_fit, projector_bounds_path, reuse_completed=True
):
    """Supply condenser images/counts and a fitted projector; get correlated beam measurements."""
    root = Path(experiment_directory)
    paths = [
        root / "condenser" / f"voltage_{i}" / name
        for i in [0, 1]
        for name in [
            "images.npz",
            "template.npz",
            "detector.json",
            "counts.npz",
            "aperture.json",
        ]
    ]
    paths += [
        root / "condenser/experiment.json",
        root / "condenser/alignment/alignment_constraint.json",
        root / "projector/fit/result.json",
        root / "projector/voltage_0/measurements.npz",
        root / "projector/voltage_1/measurements.npz",
        projector_bounds_path,
        HERE / "full_microscope_inference.py",
        HERE / "full_microscope_apertures.py",
        HERE / "full_microscope_projector.py",
        HERE / "condenser_image_pipeline.py",
    ]
    signature = input_signature(paths=paths, settings={"projector_fit": projector_fit})
    analysis = root / "condenser/analysis"
    _record_step(
        directory=root,
        name="beam_analysis",
        signature=signature,
        required=[
            analysis / "joint_beam_measurements.npz",
            analysis / "beam_analysis.json",
        ],
        function=_measure_condenser,
        reuse_completed=reuse_completed,
        root=root,
        projector_fit=projector_fit,
        bounds_path=projector_bounds_path,
    )
    return analysis


def fit_microscope(
    *,
    beam_measurements_directory,
    condenser_starting_guess_path,
    projector_bounds_path,
    number_of_broad_starts=4,
    reuse_completed=True,
):
    """Fit the condenser, update the shared projector, and return the full microscope/covariance."""
    root = Path(beam_measurements_directory)

    def run():
        condenser = fit_condenser_block(
            analysis_directory=root,
            starting_guess_path=condenser_starting_guess_path,
            number_of_broad_starts=number_of_broad_starts,
        )
        assemble_full_fit(
            analysis_directory=root,
            condenser_fit=condenser,
            bounds_path=projector_bounds_path,
        )

    paths = [
        root / "joint_beam_measurements.npz",
        root / "experiment.json",
        root / "alignment/alignment_constraint.json",
        condenser_starting_guess_path,
        projector_bounds_path,
        HERE / "full_microscope_inference.py",
        HERE / "full_microscope_apertures.py",
        HERE / "condenser_geometry_audit.py",
    ]
    paths += [
        root / f"voltage_{i}" / name
        for i in [0, 1]
        for name in ["current_measurements.npz", "aperture.json"]
    ]
    signature = input_signature(
        paths=paths, settings={"broad_starts": number_of_broad_starts}
    )
    _record_step(
        directory=root,
        name="full_fitting",
        signature=signature,
        required=[root / "full_fit.json"],
        function=run,
        reuse_completed=reuse_completed,
    )
    return json.loads((root / "full_fit.json").read_text())


def check_full_microscope(*, experiment_directory, microscope, projector_bounds_path):
    """Measure new projector images, test frozen predictions, then compare geometry with truth."""
    root = Path(experiment_directory)
    with (root / "checks.log").open("a") as log, redirect_stdout(log):
        measure_projector_block(images_directory=root / "projector_predictions")
        result = _check_microscope(
            experiment_directory=root,
            fitted=microscope,
            bounds_path=projector_bounds_path,
        )
    return result


def show_images(*, experiment_directory, fitted_predictions=False):
    import matplotlib.pyplot as plt

    root = Path(experiment_directory)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for i in [0, 1]:
        if fitted_predictions:
            folder = root / "condenser/voltage_1/prediction"
            filename = "images.npz" if i == 0 else "full_microscope_predictions.npz"
            image = np.load(folder / filename)["images"][2, 0]
            title = "New recorded image" if i == 0 else "Full microscope prediction"
        else:
            images = load_projector_images(root / "projector" / f"voltage_{i}")
            image = images.ordinary.images[0]
            title = f"{images.ordinary.specimen_and_detector['voltage_v'] / 1000:g} kV, projector calibration"
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")
    fig.tight_layout()
    return fig
