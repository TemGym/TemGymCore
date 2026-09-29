"""Generate measurements, fit native components and check predictions."""

import hashlib
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from ..config import require_x64

PRESETS = Path(__file__).parent / "presets"


def input_signature(*, paths, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    here = Path(__file__).parent
    # Presentation and orchestration edits do not change numerical results.
    sources = (sorted(here.glob("*.py")) + sorted(here.parent.glob("*.py"))
               + sorted((here.parent / "components").glob("*.py")))
    numerical = [
        p
        for p in sources
        if p.name not in {"viewer.py", "plotting.py", "workflow.py", "__init__.py"}
    ]
    for path in [*paths, *numerical]:
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path) as data:
                for name in sorted(data.files):
                    a = np.asarray(data[name])
                    digest.update(str((name, a.shape, a.dtype)).encode())
                    digest.update(a.tobytes())
        else:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _record_step(
    *, directory, name, signature, required, function, reuse_completed, **inputs
):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / f"{name}_manifest.json"
    if (
        reuse_completed
        and marker.exists()
        and json.loads(marker.read_text())["signature"] == signature
        and all(Path(p).exists() for p in required)
    ):
        print(f"Using saved {name}: {root}", flush=True)
        return
    marker.unlink(missing_ok=True)
    print(f"Running {name}; progress: {root / (name + '.log')}", flush=True)
    with (root / f"{name}.log").open("w") as log, redirect_stdout(log):
        function(**inputs)
    if not all(Path(p).exists() for p in required):
        raise RuntimeError(f"{name} did not produce all required outputs; see its log.")
    marker.write_text(json.dumps({"signature": signature}, indent=2))
    print(f"Completed {name}", flush=True)


def simulate_experiment(
    *,
    output_directory,
    voltages_v=(200000.0, 195000.0),
    aperture_diameters_um=(4.0, 10.0),
    random_seed=24001,
    gaussians_across=512,
    samples_per_pixel=4,
    reuse_completed=True,
):
    """Supply voltages/apertures; receive recorded beam images with the aperture in."""

    from .acquisition import prepare_experiment
    from .gaussian import generate_images

    require_x64("this calibration")
    if len(voltages_v) != 2 or min(voltages_v) <= 0 or voltages_v[0] == voltages_v[1]:
        raise ValueError("Supply two distinct positive voltages.")
    if (
        len(aperture_diameters_um) != 2
        or min(aperture_diameters_um) <= 0
        or aperture_diameters_um[0] == aperture_diameters_um[1]
    ):
        raise ValueError("Supply two distinct positive aperture diameters.")
    if gaussians_across < 8 or samples_per_pixel < 1:
        raise ValueError(
            "Supply at least eight beamlets across and positive pixel sampling."
        )
    root = Path(output_directory)
    settings = {
        "voltages_v": list(voltages_v),
        "aperture_diameters_um": list(aperture_diameters_um),
        "random_seed": random_seed,
        "gaussians_across": gaussians_across,
        "samples_per_pixel": samples_per_pixel,
    }
    signature = input_signature(paths=sorted(PRESETS.glob("*.json")), settings=settings)

    def run():
        prepare_experiment(
            directory=root,
            voltages_v=voltages_v,
            aperture_diameters_um=aperture_diameters_um,
            random_seed=random_seed,
        )
        generate_images(
            directory=root,
            gaussians_across=gaussians_across,
            samples_per_pixel=samples_per_pixel,
        )

    _record_step(
        directory=root,
        name="generation",
        signature=signature,
        required=[
            root / "images.npz",
            root / "settings.npz",
            root / "experiment.json",
            root / "scoring_truth.json",
            root / "numerical_checks.json",
        ],
        function=run,
        reuse_completed=reuse_completed,
    )
    return root


def simulate_ray_experiment(
    *,
    reference_directory,
    output_directory,
    number_of_rays=262144,
    detector_samples_per_pixel=8,
    add_noise=False,
    random_seed=24004,
    reuse_completed=True,
):
    """Save native ray images for a recorded acquisition; return their directory.

    The source is sampled deterministically. Noiseless images still have finite
    ray-sampling and pixel errors. Optional noise uses the recorded exposure,
    Poisson counts and detector read noise. The matching fit omits aperture
    diffraction, which geometric rays cannot generate.
    """

    from .acquisition import render_ray_image

    reference, root = Path(reference_directory), Path(output_directory)
    if reference.resolve() == root.resolve():
        raise ValueError("Keep ray images separate from the reference experiment.")
    require_x64("this calibration")
    power = int(np.log2(number_of_rays)) if number_of_rays > 0 else -1
    if power < 3 or 2**power != number_of_rays:
        raise ValueError("number_of_rays must be a power of two, at least eight.")
    if (
        detector_samples_per_pixel < 1
        or int(detector_samples_per_pixel) != detector_samples_per_pixel
    ):
        raise ValueError("detector_samples_per_pixel must be a positive integer.")
    paths = [
        reference / name
        for name in ("experiment.json", "settings.npz", "scoring_truth.json")
    ]
    signature = input_signature(
        paths=paths,
        settings={
            "number_of_rays": number_of_rays,
            "detector_samples_per_pixel": detector_samples_per_pixel,
            "detector_psf_truncate_sigmas": 8.0,
            "add_noise": add_noise,
            "random_seed": random_seed,
        },
    )

    def generate():
        spec = json.loads(paths[0].read_text())
        settings = dict(np.load(paths[1]))
        truth = json.loads(paths[2].read_text())
        spec.update(
            data_source="native weighted ray images",
            aperture_diffraction=False,
            ray_sample_power=power,
            detector_samples_per_pixel=int(detector_samples_per_pixel),
            detector_psf_truncate_sigmas=8.0,
            moment_margin_pixels=spec.get("moment_margin_pixels", 8),
            measurement_noise_added=bool(add_noise),
            measurement_noise_seed=random_seed,
        )
        spec.pop("uncertainty_reference", None)
        spec.pop("uncertainty_assumption", None)
        rng = np.random.default_rng(random_seed)
        frames = []
        for row in range(len(settings["voltage_index"])):
            pair = []
            for aperture in range(len(spec["aperture_diameters_um"])):
                frame = (
                    render_ray_image(spec, settings, truth, row, aperture)
                    + spec["background_counts"]
                )
                if add_noise:
                    frame = rng.poisson(frame) + rng.normal(
                        0, spec["read_noise_counts"], frame.shape
                    )
                pair.append(frame)
            frames.append(pair)
            if row % 20 == 0:
                print(
                    f"Rendered {row + 1}/{len(settings['voltage_index'])} aperture pairs",
                    flush=True,
                )
        (root / "experiment.json").write_text(json.dumps(spec, indent=2) + "\n")
        (root / "scoring_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
        np.savez_compressed(root / "settings.npz", **settings)
        np.savez_compressed(root / "images.npz", images=np.asarray(frames))

    _record_step(
        directory=root,
        name="generation",
        signature=signature,
        required=[
            root / name
            for name in (
                "images.npz",
                "settings.npz",
                "experiment.json",
                "scoring_truth.json",
            )
        ],
        function=generate,
        reuse_completed=reuse_completed,
    )
    return root


def measure_images(*, experiment_directory, reuse_completed=True):
    """Supply recorded images; receive centres, covariance and paired intensities."""
    from .inference import measure_beam_images

    root = Path(experiment_directory)
    signature = input_signature(
        paths=[root / "images.npz", root / "settings.npz", root / "experiment.json"],
        settings={},
    )
    _record_step(
        directory=root,
        name="measurement",
        signature=signature,
        required=[root / "measurements.npz", root / "projector_responses.npz"],
        function=measure_beam_images,
        reuse_completed=reuse_completed,
        directory_path=root,
    )
    return root


def simulate_measurements(
    *, reference_directory, output_directory, add_noise=False, random_seed=24002
):
    """Generate moments directly for a saved acquisition, without rendering images.

    Native matrices and aperture moments supply the signal. Saved image-derived
    covariances supply weights (and optional Gaussian measurement noise). Reuse
    these uncertainties only for this same geometry, exposure and acquisition.
    This tests inverse recovery within the model, not image analysis or realism.
    """
    import jax.numpy as jnp

    from .inference import make_predictions, save_projector_responses
    from .projector_model import projector_coordinates

    reference, root = Path(reference_directory), Path(output_directory)
    require_x64("this calibration")
    if reference.resolve() == root.resolve():
        raise ValueError(
            "Keep the synthetic measurements separate from their reference."
        )
    if (root / "images.npz").exists():
        raise ValueError("Choose a separate output directory without rendered images.")
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads((reference / "experiment.json").read_text())
    truth = json.loads((reference / "scoring_truth.json").read_text())
    with np.load(reference / "settings.npz") as data:
        settings = dict(data)
    with np.load(reference / "measurements.npz") as data:
        covariance = data["covariance"]
    encode, _, _, _ = projector_coordinates(PRESETS / "projector_position_priors.json")
    parameters = jnp.r_[
        encode(jnp.asarray(truth["projector_parameters"]))[:16],
        jnp.asarray(truth["condenser_parameters"]),
    ]
    values = np.array(make_predictions(spec, settings)(parameters))
    values[..., 5] *= spec["nominal_incident_electrons_per_image"]
    if add_noise:
        noise = np.random.default_rng(random_seed).normal(size=values.shape)
        values += np.einsum("...ij,...j->...i", np.linalg.cholesky(covariance), noise)
    spec["data_source"] = "native matrix and aperture moments; no rendered images"
    spec["measurement_noise_added"] = bool(add_noise)
    spec["measurement_noise_seed"] = random_seed
    spec["uncertainty_reference"] = str(reference.resolve())
    spec["uncertainty_assumption"] = (
        "Fixed image-derived covariance for the identical acquisition; optional multivariate Gaussian measurement noise."
    )
    (root / "experiment.json").write_text(json.dumps(spec, indent=2) + "\n")
    (root / "scoring_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    np.savez_compressed(root / "settings.npz", **settings)
    np.savez_compressed(root / "measurements.npz", values=values, covariance=covariance)
    save_projector_responses(root, settings, values, covariance)
    print(f"Generated {values.size} moment/intensity values; noise added: {add_noise}.")
    return root


def fit_microscope(
    *,
    measurements_directory,
    number_of_starts=4,
    projector_aperture_um=None,
    reuse_completed=True,
):
    """Fit measured quantities from independent guesses; return the fitted column.

    ``projector_aperture_um`` selects one recorded aperture for projector images.
    None uses both. Condenser images always retain both apertures.
    """
    from .inference import fit_beam_microscope

    root = Path(measurements_directory)
    paths = [
        root / name
        for name in [
            "measurements.npz",
            "projector_responses.npz",
            "settings.npz",
            "experiment.json",
        ]
    ]
    paths += [
        PRESETS / name
        for name in [
            "projector_position_priors.json",
            "projector_starting_guess.json",
            "microscope_condenser_starting_guess.json",
        ]
    ]
    signature = input_signature(
        paths=paths,
        settings={
            "number_of_starts": number_of_starts,
            "projector_aperture_um": projector_aperture_um,
        }
    )
    _record_step(
        directory=root,
        name="fitting",
        signature=signature,
        required=[root / "fit.json"],
        function=fit_beam_microscope,
        reuse_completed=reuse_completed,
        directory_path=root,
        number_of_starts=number_of_starts,
        projector_aperture_um=projector_aperture_um,
    )
    return json.loads((root / "fit.json").read_text())


def check_microscope(
    *, experiment_directory, microscope, render_images=True, reuse_completed=True
):
    """Check unused measurements and geometry; optionally render prediction images."""
    from .checks import check_beam_microscope

    root = Path(experiment_directory)
    signature = input_signature(
        paths=[
            root / name
            for name in (["images.npz"] if render_images else [])
            + [
                "measurements.npz",
                "settings.npz",
                "experiment.json",
            ]
            + (["scoring_truth.json"] if (root / "scoring_truth.json").exists() else [])
        ],
        settings={"microscope": microscope, "render_images": render_images},
    )
    _record_step(
        directory=root,
        name="checks",
        signature=signature,
        required=[root / "checks.json"]
        + ([root / "predicted_images.npz"] if render_images else []),
        function=check_beam_microscope,
        reuse_completed=reuse_completed,
        directory_path=root,
        microscope=microscope,
        render_images=render_images,
    )
    return json.loads((root / "checks.json").read_text())
