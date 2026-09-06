"""Repeat the complete original pixel/count experiment with fresh noise.

This audits the conditional objective-sweep result; it does not recommend
cycling an objective whose response is not known experimentally. Hidden lens
response changes are used only in generation, never supplied to image analysis.
"""

import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_aperture_experiment import (
    aperture_specification,
    generate_aperture_currents,
    measure_aperture_currents,
)
from condenser_calibration_workflow import make_images, measure_images
from condenser_objective_experiment import (
    make_pattern_image_operator,
    measure_objective_images,
    predict_pattern,
)
from condenser_rotation_measurements import (
    generate_rotation_and_focus,
    measure_rotation_and_focus,
)
from condenser_rotation_workflow import input_signature


def make_noise_experiment(
    *,
    original_directory,
    example_directory,
    output_directory,
    random_seed,
    prefield_response_error=0.0,
    postfield_curvature=0.0,
    extra_blur_pixels=0.0,
    reuse_completed=True,
):
    """Draw all new pixels and counts from the forward model at recorded settings.

    prefield_response_error=0.05 makes the field change 5% more per command
    excursion than assumed by inference; it does not change reference strength.
    postfield_curvature is the coefficient of (command-1)^2 in post-field q.
    Extra Gaussian blur is a limited mismatch test, not a full aberration model.
    """
    root, example, folder = map(
        Path, (original_directory, example_directory, output_directory)
    )
    folder.mkdir(parents=True, exist_ok=True)
    settings = {
        "random_seed": random_seed,
        "prefield_response_error": prefield_response_error,
        "postfield_curvature": postfield_curvature,
        "extra_blur_pixels": extra_blur_pixels,
    }
    signature = input_signature(
        paths=[
            __file__,
            example / "condenser_measurement_protocol.json",
            example / "condenser_projector_views.json",
            root / "condenser_objective/calibration_images.npz",
            root / "condenser_objective/prediction_images.npz",
        ],
        settings=settings,
    )
    manifest = folder / "generation.json"
    if (
        reuse_completed
        and manifest.exists()
        and json.loads(manifest.read_text())["signature"] == signature
    ):
        return folder
    protocol = json.loads((example / "condenser_measurement_protocol.json").read_text())
    protocol["calibration_noise_seed"] = int(random_seed)
    protocol["prediction_noise_seed"] = int(random_seed + 1)
    protocol_path = folder / "recorded_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2))
    reference = make_images(
        protocol_path=protocol_path,
        projector_views_path=example / "condenser_projector_views.json",
        split="calibration",
        output_directory=folder / "condenser_calibration_run/calibration",
        reuse_completed=reuse_completed,
    )
    generate_aperture_currents(
        output_directory=folder / "condenser_aperture",
        specification=aperture_specification(),
        random_seed=random_seed + 2,
    )
    generate_rotation_and_focus(
        projector_views_path=example / "condenser_projector_views.json",
        protocol_path=protocol_path,
        output_directory=folder / "condenser_rotation",
        random_seed=random_seed + 3,
    )
    objective = folder / "condenser_objective"
    objective.mkdir(exist_ok=True)
    for name in ("template.npz", "detector.json"):
        shutil.copyfile(reference / name, objective / name)
    shutil.copyfile(
        example / "condenser_projector_views.json",
        objective / "reference_projector.json",
    )
    views = json.loads((objective / "reference_projector.json").read_text())
    projector = np.asarray(views["calibrated_projector_parameters"])
    # This is generator truth, not a fitted image descriptor or fitted model.
    truth = json.loads((reference / "scoring_truth.json").read_text())
    p = np.asarray(truth["parameters"])
    detector = json.loads((objective / "detector.json").read_text())
    actual_detector = dict(detector)
    actual_detector["psf_sigma_pixels"] = np.hypot(
        detector["psf_sigma_pixels"], extra_blur_pixels
    )
    operator = make_pattern_image_operator(
        projector_matrices=np.asarray([v["matrix"] for v in views["views"]]),
        template=dict(np.load(objective / "template.npz")),
        detector=actual_detector,
    )
    base_response = np.array([1.08, 0.2, 0.03])

    def actual_pattern(controls, command):
        delta = command - 1.0
        effective = 1.0 + (1.0 + prefield_response_error) * delta
        changed = jnp.asarray(p).at[12].add(jnp.log(effective / command))
        response = jnp.asarray(base_response).at[0].add(postfield_curvature * delta)
        return predict_pattern(
            changed,
            controls,
            command,
            projector,
            detector["sample_kernel_sigma_m"],
            response,
        )

    actual_pattern = jax.jit(actual_pattern)
    rng = np.random.default_rng(random_seed + 4)
    for split in ("calibration", "prediction"):
        recorded = dict(np.load(root / f"condenser_objective/{split}_images.npz"))
        images = []
        for controls, command, maps in zip(
            recorded["controls"],
            recorded["objective_commands"],
            recorded["projector_matrices"],
        ):
            pattern = actual_pattern(controls, command)
            exposures = np.tile(np.log([4e7, 3.0]), (len(maps), 1)).ravel()
            mean = np.asarray(operator(jnp.r_[pattern, exposures], jnp.asarray(maps)))
            images.append(
                rng.poisson(mean) + rng.normal(0, detector["read_noise_e"], mean.shape)
            )
        recorded["images"] = np.asarray(images)
        np.savez_compressed(objective / f"{split}_images.npz", **recorded)
    (objective / "scoring_truth.json").write_text(
        json.dumps(
            {
                "condenser_parameters": p.tolist(),
                "post_response": base_response.tolist(),
                "generation_errors": settings,
            },
            indent=2,
        )
    )
    manifest.write_text(
        json.dumps(
            {
                "signature": signature,
                "settings": settings,
                "scope": "All calibration and objective prediction images and fitted current readings have independently drawn noise.",
            },
            indent=2,
        )
    )
    return folder


def analyse_noise_experiment(*, experiment_directory, reuse_completed=True):
    """Analyse every new image/count dataset; no generating parameters are read."""
    folder = Path(experiment_directory)
    beams = measure_images(
        image_directory=folder / "condenser_calibration_run/calibration",
        reuse_completed=reuse_completed,
    )
    measure_aperture_currents(input_directory=folder / "condenser_aperture")
    measure_rotation_and_focus(input_directory=folder / "condenser_rotation")
    for split in ("calibration", "prediction"):
        measure_objective_images(
            image_directory=folder / "condenser_objective",
            reference_measurements_path=beams,
            split=split,
            reuse_completed=reuse_completed,
        )
    return folder


def fit_noise_experiment(
    *,
    experiment_directory,
    starting_fit_path,
    number_of_broad_starts=4,
    max_evaluations=1200,
):
    """Fit a new measured experiment from a prior data fit and varied guesses.

    No truth file is read. The prior fit supplies one starting guess, not an
    additional residual or prior. Other starts include an engineering guess,
    two members of the earlier ambiguous family and a Latin-hypercube sample.
    """
    from condenser_aperture_experiment import equivalent_aperture_model
    from condenser_calibration_model import initial_parameters
    from condenser_geometry_audit import search_alternatives
    from condenser_rotation_fitting import (
        fit_rotation_slopes,
        make_rotation_joint_residual,
    )
    from scipy.stats import qmc

    folder = Path(experiment_directory)
    earlier = json.loads(Path(starting_fit_path).read_text())
    p = np.asarray(earlier["best_fit"]["parameters"])
    lower, upper = (
        np.asarray(earlier["bounds_lower"]),
        np.asarray(earlier["bounds_upper"]),
    )
    beta = fit_rotation_slopes(
        measurements_path=folder / "condenser_rotation/rotation_measurements.npz"
    )["intercept"]
    starts = [p, np.r_[initial_parameters(), -3.0, 0.1, -0.1, 1.0, 0.0, 0.0, beta]]
    starts += [
        np.r_[equivalent_aperture_model(p[:22], objective_strength=-q), p[22:]]
        for q in [0.35, 0.65]
    ]
    if number_of_broad_starts:
        starts += list(
            qmc.scale(
                qmc.LatinHypercube(d=len(p), seed=511).random(number_of_broad_starts),
                lower + 1e-7,
                upper - 1e-7,
            )
        )
    residual, _ = make_rotation_joint_residual(
        rotation_directory=folder / "condenser_rotation",
        baseline_directory=folder / "condenser_calibration_run",
        baseline_aperture_directory=folder / "condenser_aperture",
        objective_directory=folder / "condenser_objective",
        include_objective=True,
    )
    return search_alternatives(
        residual=residual,
        starting_parameters=starts,
        bounds_lower=lower,
        bounds_upper=upper,
        output_directory=folder / "fit",
        max_evaluations=max_evaluations,
    )
