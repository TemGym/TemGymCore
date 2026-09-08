"""Public acquisition, measurement, fitting and prediction steps."""

import json
from contextlib import redirect_stdout
from pathlib import Path

import jax
import numpy as np

from .acquisition import simulate_voltage_experiment, simulate_voltage_prediction_images
from .apertures import simulate_aligned_apertures
from .checks import check_microscope as _check_microscope
from .inference import (
    assemble_full_fit,
    fit_condenser_block,
    measure_specimen_beams,
    prepare_condenser_analysis,
)
from .projector import (
    fit_projector_block,
    load_projector_images,
    measure_projector_block,
    simulate_projector_block,
)
from .projector_model import _generator_parameters, choose_objective_projector_views
from .storage import input_signature as _input_signature

HERE = Path(__file__).parent
PRESETS = HERE / "presets"


def input_signature(*, paths, settings):
    """Invalidate saved stages when the shared TemGym physics changes too."""
    source = [*sorted(HERE.parent.glob("*.py")), *sorted(HERE.glob("*.py"))]
    return _input_signature(paths=[*paths, *source], settings=settings)


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
    *,
    root,
    voltages_v,
    random_seed,
    aperture_diameters_um,
    centering_sigma_um,
    image_model="coherent_gaussian",
):
    simulate_projector_block(
        protocol_path=PRESETS / "projector_measurement_protocol.json",
        output_directory=root / "projector",
        voltages_v=voltages_v,
        random_seed=random_seed,
    )
    # Two distinct downstream settings at each voltage provide separate tests.
    prediction_protocol = json.loads(
        (PRESETS / "projector_prediction_protocol.json").read_text()
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
    views = json.loads((PRESETS / "condenser_projector_views.json").read_text())
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
        protocol_path=PRESETS / "condenser_voltage_protocol.json",
        projector_views_path=generating_path,
        output_directory=root / "condenser",
        voltages_v=voltages_v,
        random_seed=random_seed + 100,
    )
    simulate_voltage_prediction_images(
        experiment_directory=root / "condenser",
        protocol_path=PRESETS / "condenser_voltage_protocol.json",
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

    if image_model == "coherent_gaussian":
        from .gaussian import (
            ensure_gaussian_moment_numerics,
            prepare_condenser_moment_specimens,
            refine_condenser_pixel_integration,
            replace_aperture_with_gaussian_counts,
            replace_condenser_with_gaussian_images,
            replace_projector_with_gaussian_images,
        )

        replace_projector_with_gaussian_images(
            directory=root / "projector", random_seed=random_seed + 17000
        )
        replace_projector_with_gaussian_images(
            directory=root / "projector_predictions", random_seed=random_seed + 19000
        )
        replace_condenser_with_gaussian_images(
            directory=root / "condenser", random_seed=random_seed + 18000
        )
        prepare_condenser_moment_specimens(directory=root / "condenser")
        refine_condenser_pixel_integration(
            directory=root / "condenser", random_seed=random_seed + 18000
        )
        ensure_gaussian_moment_numerics(
            directory=root / "condenser", random_seed=random_seed + 18000
        )
        replace_aperture_with_gaussian_counts(
            directory=root / "condenser", random_seed=random_seed + 20000
        )
    metadata = json.loads((root / "experiment.json").read_text())
    metadata["image_model"] = image_model
    (root / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")


def simulate_experiment(
    *,
    output_directory,
    voltages_v=(200000.0, 195000.0),
    random_seed=24001,
    aperture_diameters_um=(4.0, 10.0),
    centering_sigma_um=0.2,
    image_model="coherent_gaussian",
    reuse_completed=True,
):
    """Supply voltages and a noise seed; get all pixels/counts for calibration and checks."""
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "Enable JAX 64-bit arithmetic before running this experiment."
        )
    if image_model not in ("coherent_gaussian", "intensity"):
        raise ValueError("Use coherent_gaussian or intensity for image_model.")
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
    paths = sorted(PRESETS.glob("*.json"))
    signature = input_signature(
        paths=paths,
        settings={
            "voltages_v": voltages,
            "random_seed": random_seed,
            "aperture_diameters_um": diameters,
            "centering_sigma_um": centering_sigma_um,
            "image_model": image_model,
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
    if image_model == "coherent_gaussian":
        required += [
            root / block / "gaussian_generation.json"
            for block in ("projector", "projector_predictions", "condenser")
        ]
    if image_model == "coherent_gaussian":
        required += [
            root / "condenser" / name
            for name in (
                "gaussian_pixel_refinement.json",
                "gaussian_current_generation.json",
                "moment_quadrature.json",
                "gaussian_moment_validation.json",
            )
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
        image_model=image_model,
    )
    return root


def measure_projector(*, experiment_directory, reuse_completed=True):
    """Supply images; get scale, rotation, blur and signed displacements with covariance."""
    root = Path(experiment_directory) / "projector"
    gaussian = (
        json.loads((root / "experiment.json").read_text()).get("image_model")
        == "coherent_gaussian"
    )
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
        paths=paths,
        settings={},
    )
    _record_step(
        directory=root,
        name="analysis",
        signature=signature,
        required=(
            [
                root / f"voltage_{i}" / ("moments_" + name)
                for i in (0, 1)
                for name in (
                    "images.npz",
                    "tilt_images_3mrad.npz",
                    "beam_shift_images_10nm.npz",
                )
            ]
            if gaussian
            else [root / f"voltage_{i}/measurements.npz" for i in (0, 1)]
        ),
        function=measure_projector_block,
        reuse_completed=reuse_completed,
        images_directory=root,
    )
    return root


def fit_projector(
    *,
    measurements_directory,
    starting_guess_path=PRESETS / "projector_starting_guess.json",
    bounds_path=PRESETS / "projector_position_priors.json",
    number_of_starts=4,
    reuse_completed=True,
):
    """Fit one shared projector geometry from two-voltage measured responses."""
    root = Path(measurements_directory)
    gaussian = (
        json.loads((root / "experiment.json").read_text()).get("image_model")
        == "coherent_gaussian"
    )
    measured_files = [
        root / f"voltage_{i}" / name
        for i in (0, 1)
        for name in (
            (
                "moments_images.npz",
                "moments_tilt_images_3mrad.npz",
                "moments_beam_shift_images_10nm.npz",
                "moment_specimen.npz",
                "known_settings.json",
            )
            if gaussian
            else ("measurements.npz",)
        )
    ]
    signature = input_signature(
        paths=[
            root / "experiment.json",
            *measured_files,
            starting_guess_path,
            bounds_path,
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
    *,
    experiment_directory,
    projector_fit,
    projector_bounds_path=PRESETS / "projector_position_priors.json",
    reuse_completed=True,
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
        *(
            [
                root / "projector" / f"voltage_{i}" / name
                for i in (0, 1)
                for name in (
                    "moments_images.npz",
                    "moments_tilt_images_3mrad.npz",
                    "moments_beam_shift_images_10nm.npz",
                    "moment_specimen.npz",
                )
            ]
            if json.loads((root / "projector/experiment.json").read_text()).get(
                "image_model"
            )
            == "coherent_gaussian"
            else [root / "projector" / f"voltage_{i}/measurements.npz" for i in (0, 1)]
        ),
        projector_bounds_path,
    ]
    if (
        json.loads((root / "condenser/experiment.json").read_text()).get("image_model")
        == "coherent_gaussian"
    ):
        paths += [
            root / "condenser" / f"voltage_{i}/moment_specimen.npz" for i in (0, 1)
        ]
        paths += list((root / "condenser").glob("voltage_*/moment_specimen_local.npz"))
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
    condenser_starting_guess_path=PRESETS / "microscope_condenser_starting_guess.json",
    projector_bounds_path=PRESETS / "projector_position_priors.json",
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
    ]
    paths += [
        root / f"voltage_{i}" / name
        for i in [0, 1]
        for name in ["current_measurements.npz", "aperture.json"]
    ]
    if (
        json.loads((root / "experiment.json").read_text()).get("image_model")
        == "coherent_gaussian"
    ):
        paths += [
            root / f"voltage_{i}" / name
            for i in (0, 1)
            for name in (
                "images.npz",
                "measurements.npz",
                "detector.json",
                "moment_specimen.npz",
            )
        ]
        paths += list(root.glob("voltage_*/moment_specimen_local.npz"))
        paths += [
            root.parent.parent / "projector" / f"voltage_{i}" / name
            for i in (0, 1)
            for name in (
                "moments_images.npz",
                "moments_tilt_images_3mrad.npz",
                "moments_beam_shift_images_10nm.npz",
                "moment_specimen.npz",
            )
        ]
        paths += [root.parent.parent / "projector/experiment.json"]
        paths += list(
            (root.parent.parent / "projector").glob("voltage_*/known_settings.json")
        )
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


def check_full_microscope(
    *,
    experiment_directory,
    microscope,
    projector_bounds_path=PRESETS / "projector_position_priors.json",
    reuse_completed=True,
):
    """Test frozen predictions, then score geometry; cache only matching inputs."""
    root = Path(experiment_directory)
    paths = []
    paths += [Path(projector_bounds_path), root / "experiment.json"]
    paths += list((root / "projector_predictions").glob("voltage_*/*.npz"))
    paths += [
        root / block / "scoring_truth.json" for block in ("projector", "condenser")
    ]
    paths += [
        root / "condenser" / f"voltage_{i}" / name
        for i in (0, 1)
        for name in (
            "images.npz",
            "detector.json",
            "aperture.json",
            "prediction/images.npz",
            "prediction/current_counts.npz",
        )
    ]
    if (
        json.loads((root / "condenser/experiment.json").read_text()).get("image_model")
        == "coherent_gaussian"
    ):
        paths += [
            root / "condenser" / f"voltage_{i}" / name
            for i in (0, 1)
            for name in ("moment_specimen.npz", "gaussian_specimen.npz")
        ]
        paths += list((root / "condenser").glob("voltage_*/moment_specimen_local.npz"))
        paths += [
            root / "condenser/analysis" / f"voltage_{i}/measurements.npz"
            for i in (0, 1)
        ]
    # Generated predictions and image-analysis outputs are not acquisition inputs.
    paths = [
        p
        for p in paths
        if not p.name.startswith(
            ("moments_", "scoring_gaussian_", "full_microscope_predictions")
        )
    ]
    signature = input_signature(paths=paths, settings={"microscope": microscope})

    def run():
        measure_projector_block(images_directory=root / "projector_predictions")
        return _check_microscope(
            experiment_directory=root,
            fitted=microscope,
            bounds_path=projector_bounds_path,
        )

    _record_step(
        directory=root,
        name="checks",
        signature=signature,
        required=[root / "checks.json"],
        function=run,
        reuse_completed=reuse_completed,
    )
    return json.loads((root / "checks.json").read_text())


def show_images(*, experiment_directory, fitted_predictions=False):
    """Display the recorded beam region; saved acquisitions retain the full camera."""
    import matplotlib.pyplot as plt

    from temgym_core.coherent_moments import moment_regions

    root = Path(experiment_directory)
    frames, titles = [], []
    for index in (0, 1):
        if fitted_predictions:
            folder = root / "condenser/voltage_1/prediction"
            filename = "images.npz" if index == 0 else "full_microscope_predictions.npz"
            frames.append(np.load(folder / filename)["images"][2, 0])
            titles.append(
                "New recorded image" if index == 0 else "Full microscope prediction"
            )
        else:
            images = load_projector_images(root / "projector" / f"voltage_{index}")
            frames.append(images.ordinary.images[0])
            titles.append(
                f"{images.ordinary.specimen_and_detector['voltage_v'] / 1000:g} kV: Gaussian image"
            )
    masks = moment_regions(
        images=np.asarray(frames[:1] if fitted_predictions else frames),
        background_counts=3.0,
        read_noise_counts=2.0,
    )
    yy, xx = np.nonzero(np.any(masks, axis=0))
    crop = np.s_[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for axis, frame, title in zip(axes, frames, titles):
        axis.imshow(frame[crop], cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    return fig


def copy_recorded_experiment(*, source_directory, output_directory):
    """Copy raw acquisition, known settings and scoring model, without old fits.

    Returns the new experiment directory. Repeated calls preserve new analysis
    files; stage signatures invalidate them when their inputs change.
    """
    import hashlib
    import shutil

    source, output = Path(source_directory).resolve(), Path(output_directory).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            "Use non-overlapping directories to preserve the reference acquisition."
        )
    paths = [
        "experiment.json",
        "projector/experiment.json",
        "projector/scoring_truth.json",
        "projector_predictions/experiment.json",
        "projector_predictions/scoring_truth.json",
        "condenser/experiment.json",
        "condenser/scoring_truth.json",
        "condenser/alignment/alignment_constraint.json",
    ]
    for block in ("projector", "projector_predictions"):
        for index in (0, 1):
            paths.extend(
                f"{block}/voltage_{index}/{name}"
                for name in (
                    "images.npz",
                    "tilt_images_3mrad.npz",
                    "beam_shift_images_10nm.npz",
                    "specimen_template.npz",
                    "known_settings.json",
                    "scoring_truth.json",
                )
            )
    for index in (0, 1):
        paths.extend(
            f"condenser/voltage_{index}/{name}"
            for name in (
                "images.npz",
                "counts.npz",
                "template.npz",
                "detector.json",
                "aperture.json",
                "prediction/images.npz",
                "prediction/current_counts.npz",
                "prediction/template.npz",
                "prediction/detector.json",
            )
        )
    # Coherent fitting needs the independently known specimen quadrature too.
    coherent = (
        json.loads((source / "experiment.json").read_text()).get("image_model")
        == "coherent_gaussian"
    )
    if coherent:
        for block in ("projector", "projector_predictions", "condenser"):
            for index in (0, 1):
                folder = source / block / f"voltage_{index}"
                paths.append(f"{block}/voltage_{index}/moment_specimen.npz")
                for name in ("gaussian_specimen.npz", "moment_specimen_local.npz"):
                    if (folder / name).exists():
                        paths.append(f"{block}/voltage_{index}/{name}")
    if not all((source / name).is_file() for name in paths):
        missing = [name for name in paths if not (source / name).is_file()]
        raise FileNotFoundError(f"Incomplete recorded experiment: {missing}")
    hashes = {}
    for name in paths:
        original, target = source / name, output / name
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if (
            not target.exists()
            or hashlib.sha256(target.read_bytes()).hexdigest() != digest
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        hashes[name] = digest
    (output / "raw_acquisition.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "sha256": hashes,
                "previous_measurements_or_fits_copied": False,
            },
            indent=2,
        )
    )
    return output
