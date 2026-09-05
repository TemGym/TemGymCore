"""MAGICAL texture in the finite-solenoid image/measurement/fit tutorial.

This uses the intensity of the existing synthetic atlas as a positive Gaussian
mixture. It is a prescribed partially coherent intensity target, not coherent
propagation of the complex Si/SiGe exit wave or a measured specimen certificate.
"""
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from temgym_core.multiscale_sample import MAGICALSampleAtlas, SampleView
from temgym_core.gaussian_image_likelihood import normal_interval_probability
from finite_projector_pipeline import predict_observables


def make_magical_template(voltage=200000., pitch_m=2.5e-9, *, illumination_radius_m=160e-9):
    """Same clean nominal atlas as magical_gaussian_imaging.ipynb.

    The ROI is fixed at a multilayer edge so the 600k view is not the central
    perforation or an effectively one-dimensional periodic stripe pattern.
    A soft circular illumination envelope is part of the public template.
    """
    if not np.isfinite(illumination_radius_m) or illumination_radius_m <= 0:
        raise ValueError("Illumination radius must be finite and positive.")
    atlas = MAGICALSampleAtlas.nominal(
        shape_yx=(384, 384), region_shape_yx=(128, 128),
        authoritative_multilayer_shape_yx=(256, 256), max_levels=3, voltage=voltage,
    )
    certificate = replace(atlas.certificate, features=tuple(
        replace(f, phase_amplitude_rad=0., log_amplitude_depth=0.)
        for f in atlas.certificate.features))
    atlas = MAGICALSampleAtlas.from_certificate(
        certificate, shape_yx=(384, 384), region_shape_yx=(128, 128),
        authoritative_multilayer_shape_yx=(256, 256), max_levels=3,
        global_support_margin_m=atlas.global_support_margin_m,
    )
    sample = SampleView.from_level(atlas.pyramid.levels[0], z=0.)
    region = next(r for r in certificate.regions if r.name == "multilayer_region_3")
    roi = np.array([4e-9, region.centre_xy_m[1] + region.size_xy_m[1] / 2 - 2e-9])
    n = int(np.ceil(2 * (illumination_radius_m + 30e-9) / pitch_m))
    axis = (np.arange(n) - (n - 1) / 2) * pitch_m
    xx, yy = np.meshgrid(axis, axis)
    coordinates = np.stack((xx, yy), axis=-1)
    actions = np.asarray(sample.evaluate_actions(jnp.asarray(coordinates + roi)))
    intensity = np.exp(2 * actions[..., 1])
    radius = np.hypot(xx, yy)
    illumination = 1 / (1 + np.exp((radius - illumination_radius_m) / 2.5e-9))
    included = radius <= illumination_radius_m + 25e-9
    weights = (intensity * illumination)[included] * pitch_m**2
    weights /= weights.sum()
    metadata = {
        "source": "MAGICALSampleAtlas.nominal: nominal synthetic certificate",
        "phase_contrast": False, "roi_centre_atlas_m": roi.tolist(),
        "silicon_width_m": certificate.silicon_layer_width_m,
        "sige_width_m": certificate.silicon_germanium_layer_width_m,
        "repeat_m": certificate.silicon_layer_width_m + certificate.silicon_germanium_layer_width_m,
        "template_pitch_m": pitch_m, "gaussian_component_sigma_m": 2e-9,
        "illumination_radius_m": float(illumination_radius_m), "illumination_edge_m": 2.5e-9,
        "description": "Known clean atlas intensity and illumination represented by a positive 2 nm-sigma Gaussian mixture; no background texture or fiducial spots",
    }
    return dict(centres_m=coordinates[included], weights=weights,
                axis_m=axis, intensity=intensity, illumination=illumination), metadata


def make_magical_image_operator(spec, template):
    """Seven fitted quantities: log scale, angle, log width_px, dx, dy, log dose, log background."""
    centres = jnp.asarray(template["centres_m"]) / spec["pixel_pitch_m"]
    weights = jnp.asarray(template["weights"])
    ny, nx = spec["shape_yx"]
    xe = jnp.arange(nx + 1) - nx / 2
    ye = jnp.arange(ny + 1) - ny / 2

    def image(p):
        c, s = jnp.cos(p[1]), jnp.sin(p[1])
        rotation = jnp.array([[c, -s], [s, c]])
        xy = jnp.exp(p[0]) * (centres @ rotation.T) + p[3:5]
        width = jnp.exp(p[2])
        px = normal_interval_probability((xe[:-1, None] - xy[:, 0]) / width,
                                         (xe[1:, None] - xy[:, 0]) / width)
        py = normal_interval_probability((ye[:-1, None] - xy[:, 1]) / width,
                                         (ye[1:, None] - xy[:, 1]) / width)
        return jnp.exp(p[5]) * ((py * weights) @ px.T) + jnp.exp(p[6])

    return jax.jit(image)


def generate_magical_dataset(base_directory, directory, seed=7391, *, illumination_radius_m=160e-9):
    """Reuse the synthetic controls and generating column, replace only the target.

    This is a generation function, so it may read the generating truth. Neither
    image analysis nor optical fitting calls this function or reads that file.
    """
    base_directory, directory = Path(base_directory), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = json.loads((base_directory / "acquisition.json").read_text())
    truth_record = json.loads((base_directory / "scoring_truth.json").read_text())
    truth = np.asarray(truth_record["parameters"])
    template, metadata = make_magical_template(spec["voltage_v"], illumination_radius_m=illumination_radius_m)
    spec.pop("feature_centres_m", None)
    spec.update(scope=metadata["description"], specimen=metadata,
                feature_sigma_m=metadata["gaussian_component_sigma_m"])
    np.savez_compressed(directory / "specimen_template.npz", **template)
    operator = make_magical_image_operator(spec, template)
    rng = np.random.default_rng(seed)
    images, expected, image_parameters = [], [], []
    for d, nominal in zip(spec["controls"], spec["nominal_magnifications"]):
        y = np.asarray(predict_observables(truth, np.array(d), spec))
        # Exposure increases with magnification so the cropped high-mag region
        # still has useful counts. Each frame's dose is estimated from pixels.
        dose = 4e7 * (nominal / 40000.)**2
        p = np.r_[y, rng.normal(0, .3, 2), np.log(dose), np.log(3.)]
        mean = np.asarray(operator(p))
        expected.append(mean)
        image_parameters.append(p)
        images.append(rng.poisson(mean).astype(float) + rng.normal(0, 2., mean.shape))
    (directory / "acquisition.json").write_text(json.dumps(spec, indent=2))
    np.savez_compressed(directory / "images.npz", images=np.asarray(images))
    (directory / "initial_guess.json").write_text((base_directory / "initial_guess.json").read_text())
    (directory / "scoring_truth.json").write_text(json.dumps(truth_record, indent=2))
    np.savez_compressed(directory / "scoring_expected_images.npz", expected=np.asarray(expected),
                        image_parameters=np.asarray(image_parameters))
    return spec


def analyse_magical_images(directory, verbose=False):
    """Register the public texture against noisy images, without lens parameters."""
    directory = Path(directory)
    spec = json.loads((directory / "acquisition.json").read_text())
    template = dict(np.load(directory / "specimen_template.npz"))
    images = np.load(directory / "images.npz")["images"]
    result = analyse_magical_arrays(images=images, nominal_magnifications=spec["nominal_magnifications"],
                                   specimen_template=template, specimen_and_detector=spec, verbose=verbose)
    np.savez_compressed(directory / "measurements.npz", **result)
    return result["values"], result["covariance"], result["image_residual_rms"]


def analyse_magical_arrays(*, images, nominal_magnifications, specimen_template,
                          specimen_and_detector, verbose=False, initial_image_parameters=None):
    """Measure scale, rotation, and blur from the supplied arrays; read no files."""
    spec, template = specimen_and_detector, specimen_template
    if len(images) != len(nominal_magnifications):
        raise ValueError("Supply one nominal magnification for each image.")
    if initial_image_parameters is not None and np.shape(initial_image_parameters) != (len(images), 7):
        raise ValueError("Supply seven image-registration starting values per image.")
    operator = make_magical_image_operator(spec, template)
    jacobian = jax.jit(jax.jacfwd(operator))
    measured, covariance, costs, image_parameters, parameter_covariance = [], [], [], [], []
    references = {}
    for index, (image, nominal) in enumerate(zip(images, nominal_magnifications)):
        sigma = np.sqrt(np.maximum(image, 1.) + spec["read_noise_std_e"]**2)
        lower = np.r_[np.log(nominal / 2), -4*np.pi, np.log(.15), -20., -20., np.log(1e4), np.log(.01)]
        upper = np.r_[np.log(nominal * 2), 4*np.pi, np.log(60.), 20., 20., np.log(1e13), np.log(1e5)]

        def residual(p):
            return ((np.asarray(operator(p)) - image) / sigma).ravel()

        def jac(p):
            return np.asarray(jacobian(p)).reshape(-1, 7) / sigma.ravel()[:, None]

        if initial_image_parameters is not None:
            seeds = [np.clip(initial_image_parameters[index], lower+1e-8, upper-1e-8)]
        elif nominal in references:
            seeds = [references[nominal].copy()]
        else:
            seeds = []
            for angle in np.linspace(-np.pi, np.pi, 72, endpoint=False):
                p = np.r_[np.log(nominal), angle,
                          np.log(nominal * spec["feature_sigma_m"] / spec["pixel_pitch_m"]),
                          0., 0., 0., np.log(.01)]
                template_image = np.maximum(np.asarray(operator(p)) - .01, 0.)
                design = np.column_stack((template_image.ravel(), np.ones(image.size))) / sigma.ravel()[:, None]
                coefficients = np.linalg.lstsq(design, image.ravel() / sigma.ravel(), rcond=None)[0]
                p[5:] = np.log(np.maximum(coefficients, [1e4, .011]))
                p = np.clip(p, lower+1e-8, upper-1e-8)
                seeds.append((float(np.mean(residual(p)**2)), p))
            seeds = [item[1] for item in sorted(seeds, key=lambda item: item[0])[:2]]
        results = [least_squares(residual, seed, jac=jac, bounds=(lower, upper),
                                  x_scale="jac", max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-5)
                   for seed in seeds]
        result = min(results, key=lambda r: np.sum(r.fun**2))
        rms = np.sqrt(np.mean(result.fun**2))
        if not result.success or rms > 1.5:
            raise RuntimeError(f"MAGICAL registration failed at frame {index}: {result.message}; RMS {rms:.3f}")
        if nominal not in references:
            references[nominal] = result.x.copy()
        measured.append(result.x[:3])
        full_covariance = np.linalg.inv(result.jac.T @ result.jac)
        covariance.append(full_covariance[:3, :3])
        parameter_covariance.append(full_covariance)
        costs.append(rms)
        image_parameters.append(result.x)
        if verbose:
            print(f"MAGICAL frame {index+1}/{len(images)}: weighted pixel RMS {rms:.3f}", flush=True)
    return dict(values=np.asarray(measured), covariance=np.asarray(covariance),
                image_residual_rms=np.asarray(costs), image_parameters=np.asarray(image_parameters),
                parameter_covariance=np.asarray(parameter_covariance))
