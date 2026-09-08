"""Coherent Gaussian acquisition, known specimen sampling and convergence checks."""

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from temgym_core.multiscale_sample import MAGICALSampleAtlas, SampleView
from temgym_core.run import column_matrix


def make_magical_sample(voltage=200000.0):
    """Prescribed complex MAGICAL transmission, including optical path in metres."""
    atlas = MAGICALSampleAtlas.nominal(
        shape_yx=(384, 384),
        region_shape_yx=(128, 128),
        authoritative_multilayer_shape_yx=(256, 256),
        max_levels=3,
        voltage=voltage,
    )
    certificate = replace(
        atlas.certificate,
        features=tuple(
            replace(f, phase_amplitude_rad=0.0, log_amplitude_depth=0.0)
            for f in atlas.certificate.features
        ),
    )
    atlas = MAGICALSampleAtlas.from_certificate(
        certificate,
        shape_yx=(384, 384),
        region_shape_yx=(128, 128),
        authoritative_multilayer_shape_yx=(256, 256),
        max_levels=3,
        global_support_margin_m=atlas.global_support_margin_m,
    )
    sample = SampleView.from_level(atlas.pyramid.levels[0], z=0.0)
    region = next(r for r in certificate.regions if r.name == "multilayer_region_3")
    roi = np.array([4e-9, region.centre_xy_m[1] + region.size_xy_m[1] / 2 - 2e-9])
    return sample, certificate, roi


def prepare_known_transmission(
    *, template, voltage_v, kernel_sigma_m, step_nm=0.5, half_width_nm=192.0
):
    """Sample the same prescribed MAGICAL amplitude and phase for both paths.

    The amplitude target is the earlier known smoothed intensity template;
    the optical path is the native MAGICAL specimen. No unknown microscope
    quantity enters this preparation. Derivatives are taken from the known
    specimen, not estimated from the generating beam.
    """
    import jax

    from temgym_core.image import SampledTransmission

    half = int(np.ceil(half_width_nm / step_nm))
    axis = np.arange(-half, half + 1) * step_nm * 1e-9
    old = np.asarray(template["axis_m"])
    delta = axis[:, None] - old[None, :]
    variance = kernel_sigma_m**2
    g = np.exp(-0.5 * delta**2 / variance) / np.sqrt(2 * np.pi * variance)
    gradient = -delta * g / variance
    intensity = g @ template["intensity"] @ g.T
    dx = g @ template["intensity"] @ gradient.T
    dy = gradient @ template["intensity"] @ g.T
    xy = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    sample, _, roi = make_magical_sample(voltage_v)

    def actions(point):
        return sample.evaluate_actions(point + jnp.asarray(roi))[0]

    path = jax.jit(jax.vmap(actions))(jnp.asarray(xy))
    path_gradient = jax.jit(jax.vmap(jax.jacfwd(actions)))(jnp.asarray(xy))
    amplitude_gradient = 0.5 * np.stack((dx, dy), axis=-1).reshape(-1, 2)
    amplitude_gradient /= np.maximum(intensity.ravel(), 1e-300)[:, None]
    maximum = intensity.max()
    sampled = SampledTransmission(
        jnp.asarray(xy),
        jnp.asarray(np.maximum(intensity.ravel() / maximum, 1e-300)),
        jnp.asarray(amplitude_gradient),
        path_gradient,
    )
    return {
        "sampled": sampled,
        "path_m": np.asarray(path),
        "axis_m": axis,
        "voltage_v": voltage_v,
        "step_nm": step_nm,
    }


def gaussian_exit_wave(
    *,
    transmission,
    profile,
    position_m=(0.0, 0.0),
    angle_rad=(0.0, 0.0),
    curvature_per_m=0.0,
    rms_size_m=None,
    radius_m=None,
    edge_m=None,
    specimen_rotation_rad=0.0,
    overlap=1.1,
):
    """Represent a prescribed coherent exit wave with actual TemGym Gaussians.

    Beamlets share a common quadratic phase; cardinal coefficients represent
    the complex specimen and illumination amplitude. Padding to size buckets
    reuses compiled GPU kernels while zero-amplitude entries carry no field.
    """
    from temgym_core.constants import energy2wavelength
    from temgym_core.gaussian import make_gaussian
    from temgym_core.source import _gaussian_cardinal_coefficients

    sampled = transmission["sampled"]
    c, s = np.cos(specimen_rotation_rad), np.sin(specimen_rotation_rad)
    rotation = np.array([[c, -s], [s, c]])
    xy = np.asarray(sampled.coordinates_m) @ rotation.T
    delta = xy - np.asarray(position_m)
    if profile == "gaussian":
        illumination = np.exp(-0.5 * np.sum(delta**2, axis=1) / rms_size_m**2)
    elif profile == "circle":
        from scipy.special import expit

        illumination = expit((radius_m - np.linalg.norm(delta, axis=1)) / edge_m)
    else:
        raise ValueError("Choose gaussian or circle illumination.")
    intensity = np.asarray(sampled.intensity) * illumination
    step = transmission["step_nm"] * 1e-9
    intensity /= intensity.sum() * step**2
    k = 2 * np.pi / float(energy2wavelength(transmission["voltage_v"]))
    shape = (len(transmission["axis_m"]),) * 2
    target = (np.sqrt(intensity) * np.exp(1j * k * transmission["path_m"])).reshape(
        shape
    )
    coefficients = _gaussian_cardinal_coefficients(
        target, width_over_step=overlap
    ).ravel()
    keep = np.flatnonzero(np.abs(coefficients) > np.max(np.abs(coefficients)) * 1e-12)
    count = len(keep)
    size = (
        4096 * 2 ** int(np.ceil(np.log2(max(1.0, count / 4096))))
        if count < 65536
        else 65536 * int(np.ceil(count / 65536))
    )
    selected = keep[np.arange(size) % count]
    amplitude = coefficients[selected].copy()
    amplitude[count:] = 0
    positions = xy[selected]
    theta = np.asarray(angle_rad) - curvature_per_m * np.asarray(position_m)
    slopes = curvature_per_m * positions + theta
    phase = k * (
        0.5 * curvature_per_m * np.sum(positions**2, axis=1) + positions @ theta
    )
    return make_gaussian(
        x=positions[:, 0],
        y=positions[:, 1],
        dx=slopes[:, 0],
        dy=slopes[:, 1],
        amp=amplitude,
        phase=phase,
        waist_x=overlap * step,
        waist_y=overlap * step,
        rcurv_x=np.inf if curvature_per_m == 0 else 1 / curvature_per_m,
        rcurv_y=np.inf if curvature_per_m == 0 else 1 / curvature_per_m,
        voltage=transmission["voltage_v"],
    )


def condenser_gaussian_state(
    *, parameters, controls, voltage_v, aperture_fraction=None
):
    """Generate the specimen state by native Gaussian source/lens propagation."""
    from temgym_core.source import gaussian_waist
    from temgym_core.image import propagate_gaussian_mixture
    from temgym_core.constants import energy2wavelength
    from temgym_core.components import solenoid_column

    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    strengths = jnp.exp(p[9:13]) * jnp.r_[jnp.asarray(controls), -1.0]
    z = -(jnp.sum(gaps) + jnp.sum(lengths))
    source = gaussian_waist(
        z=z, voltage=voltage_v,
        x=p[13] * 1e-8,
        y=p[14] * 1e-8,
        theta_x=p[15] * 1e-4,
        theta_y=p[16] * 1e-4,
        rms_size=jnp.exp(p[17]),
    )
    floor = (energy2wavelength(voltage_v) / (4 * jnp.pi * jnp.exp(p[17]))) ** 2
    extra = jnp.eye(4).at[:2, :2].set(0) * (jnp.exp(2 * p[18]) - floor)
    column = tuple(
        replace(lens, include_axial_path=False) if hasattr(lens, "q") else lens
        for lens in solenoid_column(gaps, lengths, strengths, source_z=z)
    )
    if aperture_fraction is not None:
        from temgym_core.components import Plane

        aperture_z = column[1].z + column[1].length + aperture_fraction * gaps[2]
        column = (*column[:2], Plane(z=aperture_z))
    return propagate_gaussian_mixture(source, column, extra)


def save_known_transmission(path, transmission):
    """Save only known specimen/grid quantities, never lens or source parameters."""
    t = transmission["sampled"]
    np.savez_compressed(
        path,
        coordinates_m=t.coordinates_m,
        intensity=t.intensity,
        log_amplitude_gradient=t.log_amplitude_gradient,
        path_gradient=t.path_gradient,
        path_m=transmission["path_m"],
        axis_m=transmission["axis_m"],
        voltage_v=transmission["voltage_v"],
        step_nm=transmission["step_nm"],
    )


def load_known_transmission(path):
    from temgym_core.image import SampledTransmission

    data = dict(np.load(path))
    sampled = SampledTransmission(
        *(
            jnp.asarray(data[name])
            for name in (
                "coordinates_m",
                "intensity",
                "log_amplitude_gradient",
                "path_gradient",
            )
        )
    )
    return {
        "sampled": sampled,
        **{
            name: data[name]
            for name in (
                "path_m",
                "axis_m",
                "voltage_v",
                "step_nm",
            )
        },
    }


def make_gaussian_renderer(
    *,
    shape_yx,
    pixel_pitch_m,
    samples_per_pixel,
    psf_sigma_pixels=0.0,
    padding_pixels=8,
):
    """Native field renderer with optional padded regions inside the full camera.

    Region choice only saves work in negligible signal tails. Pixel pitch and
    coherent quadrature stay fixed; verify the omitted signal independently.
    """
    import jax

    from temgym_core.components import Detector
    from temgym_core.evaluate import gaussian_image, gpu_kernel_available
    from temgym_core.gaussian import FreeSpacePropagator
    from temgym_core.run import run_to_end

    method = "spatial" if gpu_kernel_available() else "scan"
    compiled = {}

    def render(
        source, components, angular_variance, offset_pixels, *, region_shape_yx=None
    ):
        shape = tuple(shape_yx if region_shape_yx is None else region_shape_yx)
        if any(n > full or n < 2 or (full - n) % 2 for n, full in zip(shape, shape_yx)):
            raise ValueError(
                "Use a centred region with even padding inside the camera."
            )
        if shape not in compiled:
            sensor = Detector(z=0.0, shape=shape, pixel_size=(pixel_pitch_m, pixel_pitch_m))

            def evaluate(source, components, angular_variance, offset_pixels):
                output = run_to_end(
                    source,
                    components,
                    propagator=FreeSpacePropagator(include_axial_path=False),
                )
                output = output.derive(
                    x=output.x + offset_pixels[0] * pixel_pitch_m,
                    y=output.y + offset_pixels[1] * pixel_pitch_m,
                )
                matrix = column_matrix(components)
                blur = jnp.sqrt(
                    angular_variance * jnp.sum(matrix[:2, 2] ** 2)
                    + (psf_sigma_pixels * pixel_pitch_m) ** 2
                )
                return gaussian_image(
                    output,
                    sensor,
                    samples_per_pixel=samples_per_pixel,
                    blur_sigma_m=blur,
                    padding_pixels=padding_pixels,
                    method=method,
                )

            compiled[shape] = jax.jit(evaluate)
        frame = compiled[shape](source, components, angular_variance, offset_pixels)
        return jnp.pad(
            frame,
            tuple(
                ((full - n) // 2, (full - n) // 2) for n, full in zip(shape, shape_yx)
            ),
        )

    return render


def gaussian_rendering_region(*, detector_moments, shape_yx, standard_deviations=8.0):
    """Propose a centred integration region; convergence checks remain mandatory."""
    value = np.asarray(detector_moments)
    extent = np.abs(value[:2]) + standard_deviations * np.sqrt(
        np.maximum(value[[2, 4]], 0)
    )
    width = max(32, 16 * int(np.ceil(2 * extent.max() / 16)))
    return tuple(min(width, n) for n in shape_yx)


def replace_projector_with_gaussian_images(
    *,
    directory,
    random_seed=41001,
    step_nm=0.5,
    shape_yx=(256, 256),
    samples_per_pixel=4,
):
    """Generate all original projector controls, tilts and shifts with Gaussians."""
    from time import perf_counter

    import jax

    from .projector_model import projector_components

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    report = {
        "image_model": "coherent_gaussian",
        "frames": [],
        "source_construction_seconds": [],
    }
    for voltage_index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{voltage_index}"
        spec = json.loads((folder / "known_settings.json").read_text())
        truth = np.asarray(
            json.loads((folder / "scoring_truth.json").read_text())["parameters"]
        )
        spec.update(
            shape_yx=list(shape_yx),
            image_model="coherent_gaussian",
            psf_sigma_pixels=0.0,
            background_counts=3.0,
            moment_regions=True,
        )
        spec["specimen"].update(
            phase_contrast=True,
            description="Known smoothed MAGICAL intensity target and native optical-path phase, represented by coherent Gaussian beamlets.",
        )
        (folder / "known_settings.json").write_text(json.dumps(spec, indent=2) + "\n")
        material = prepare_known_transmission(
            template=dict(np.load(folder / "specimen_template.npz")),
            voltage_v=voltage,
            kernel_sigma_m=spec["feature_sigma_m"],
            step_nm=step_nm,
        )
        save_known_transmission(folder / "moment_specimen.npz", material)
        renderer = make_gaussian_renderer(
            shape_yx=shape_yx,
            pixel_pitch_m=spec["pixel_pitch_m"],
            samples_per_pixel=samples_per_pixel,
        )
        source_cache = {}
        rng = np.random.default_rng(random_seed + voltage_index)
        for filename, kind in (
            ("images.npz", "ordinary"),
            ("tilt_images_3mrad.npz", "tilts"),
            ("beam_shift_images_10nm.npz", "shifts"),
        ):
            data = dict(np.load(folder / filename))
            positions = data.get(
                "input_positions_m", np.zeros((len(data["images"]), 2))
            )
            angles = data.get(
                "input_angles_rad",
                data.get("input_tilts_rad", np.zeros_like(positions)),
            )
            frames, means = [], []
            common = np.zeros(2)
            for index, (controls, position, angle, mag) in enumerate(
                zip(
                    data["lens_controls"],
                    positions,
                    angles,
                    data["nominal_magnifications"],
                )
            ):
                key = tuple(position)
                if key not in source_cache:
                    start = perf_counter()
                    source_cache[key] = gaussian_exit_wave(
                        transmission=material,
                        profile="circle",
                        position_m=position,
                        curvature_per_m=truth[16] * 1e4,
                        radius_m=spec["specimen"]["illumination_radius_m"],
                        edge_m=spec["specimen"]["illumination_edge_m"],
                        specimen_rotation_rad=truth[18],
                    )
                    report["source_construction_seconds"].append(perf_counter() - start)
                source = source_cache[key]
                source = source.derive(
                    dx=source.dx + angle[0],
                    dy=source.dy + angle[1],
                    amplitude=source.amplitude
                    * jnp.exp(
                        1j * source.k * (source.x * angle[0] + source.y * angle[1])
                    ),
                )
                column = tuple(
                    replace(c, include_axial_path=False) if hasattr(c, "q") else c
                    for c in projector_components(
                        jnp.asarray(truth), jnp.asarray(controls)
                    )
                )
                if kind == "ordinary" or index % 2 == 0:
                    common = rng.normal(0, 0.3, 2)
                offset = common + (
                    rng.normal(0, float(data["drift_std_pixels"]), 2)
                    if kind != "ordinary"
                    else 0.0
                )
                start = perf_counter()
                mean = np.asarray(
                    renderer(
                        source, column, jnp.exp(2 * truth[17]), jnp.asarray(offset)
                    )
                )
                mean = mean * (4e7 * (mag / 40000.0) ** 2) + 3.0
                elapsed = perf_counter() - start
                frames.append(
                    rng.poisson(mean).astype(float) + rng.normal(0, 2.0, mean.shape)
                )
                means.append(mean)
                report["frames"].append(
                    {
                        "voltage_index": voltage_index,
                        "kind": kind,
                        "index": index,
                        "seconds": elapsed,
                        "gaussians": int(source.x.size),
                    }
                )
                if index % 5 == 0:
                    print(
                        f"Gaussian projector {voltage / 1000:g} kV, {kind}: {index + 1}/{len(positions)} ({elapsed:.2f}s)",
                        flush=True,
                    )
            data["images"] = np.asarray(frames)
            np.savez_compressed(folder / filename, **data)
            np.savez_compressed(folder / ("scoring_gaussian_" + filename), images=means)
        del source_cache
        jax.clear_caches()
    experiment.update(image_model="coherent_gaussian", detector_shape_yx=list(shape_yx))
    (root / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    (root / "gaussian_generation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def replace_condenser_with_gaussian_images(
    *,
    directory,
    random_seed=42001,
    step_nm=0.5,
    shape_yx=(256, 256),
    samples_per_pixel=8,
):
    """Generate all condenser settings from native Gaussian source/lens transport."""
    import hashlib
    import inspect
    from time import perf_counter

    import jax

    from temgym_core.image import detector_moments
    from temgym_core.constants import energy2wavelength

    from .condenser_images import beam_coordinates
    from .condenser_model import condenser_at_voltage
    from .projector import at_voltage
    from .projector_model import projector_components

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    truth = np.asarray(
        json.loads((root / "scoring_truth.json").read_text())["parameters"]
    )
    projector = np.asarray(
        json.loads((root.parent / "projector/scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    native_state = jax.jit(
        lambda p, c, v: condenser_gaussian_state(parameters=p, controls=c, voltage_v=v)
    )
    report = {"image_model": "coherent_gaussian", "frames": [], "beam_states": []}
    for voltage_index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{voltage_index}"
        detector = json.loads((folder / "detector.json").read_text())
        detector.update(
            shape_yx=list(shape_yx),
            image_model="coherent_gaussian",
            background_counts=3.0,
            moment_regions=True,
        )
        material = prepare_known_transmission(
            template=dict(np.load(folder / "template.npz")),
            voltage_v=voltage,
            kernel_sigma_m=detector["sample_kernel_sigma_m"],
            step_nm=step_nm,
        )
        save_known_transmission(folder / "moment_specimen.npz", material)
        effective = condenser_at_voltage(
            truth, voltage, source_parameters=truth[22:29] if voltage_index else None
        )
        pp = at_voltage(projector, voltage, experiment["reference_voltage_v"])
        renderer = make_gaussian_renderer(
            shape_yx=shape_yx,
            pixel_pitch_m=detector["pixel_pitch_m"],
            samples_per_pixel=samples_per_pixel,
            psf_sigma_pixels=detector["psf_sigma_pixels"],
        )
        for prediction in (False, True):
            target = folder / "prediction" if prediction else folder
            data = dict(np.load(target / "images.npz"))
            (target / "detector.json").write_text(json.dumps(detector, indent=2) + "\n")
            rng = np.random.default_rng(
                random_seed + 100 * voltage_index + 1000 * prediction
            )
            images, means = [], []
            checkpoint = target / "gaussian_image_progress.npz"
            digest = hashlib.sha256()
            for array in (
                truth,
                projector,
                data["controls"],
                data["projector_controls"],
            ):
                digest.update(np.asarray(array).tobytes())
            digest.update(
                json.dumps(
                    {
                        "voltage": voltage,
                        "step_nm": step_nm,
                        "shape": shape_yx,
                        "samples": samples_per_pixel,
                        "detector": detector,
                    },
                    sort_keys=True,
                ).encode()
            )
            digest.update(
                (
                    inspect.getsource(gaussian_exit_wave)
                    + inspect.getsource(make_gaussian_renderer)
                ).encode()
            )
            signature = digest.hexdigest()
            if checkpoint.exists():
                saved = np.load(checkpoint)
                if str(saved["signature"]) == signature:
                    means = list(saved["images"])
            for index, controls in enumerate(data["controls"]):
                if index < len(means):
                    continue
                mean, covariance = native_state(
                    effective, jnp.asarray(controls), jnp.asarray(voltage)
                )
                beam = np.asarray(beam_coordinates(mean, covariance))
                floor = float(energy2wavelength(voltage)) / (
                    4 * np.pi * np.exp(beam[4])
                )
                extra = np.exp(2 * beam[6]) - floor**2
                if extra < 0:
                    raise ValueError(
                        "Generating source violates the Gaussian uncertainty bound."
                    )
                source = gaussian_exit_wave(
                    transmission=material,
                    profile="gaussian",
                    position_m=beam[:2] * 1e-7,
                    angle_rad=beam[2:4] * 1e-3,
                    curvature_per_m=beam[5] * 1e4,
                    rms_size_m=np.exp(beam[4]),
                )
                exit_mean, exit_covariance = material["sampled"].gaussian_moments(
                    mean, covariance, energy2wavelength(voltage)
                )
                row = []
                for view, command in enumerate(data["projector_controls"]):
                    column = tuple(
                        replace(c, include_axial_path=False) if hasattr(c, "q") else c
                        for c in projector_components(pp, jnp.asarray(command))
                    )
                    moments = detector_moments(
                        column_matrix(column),
                        exit_mean,
                        exit_covariance,
                        pixel_pitch_m=detector["pixel_pitch_m"],
                        psf_sigma_pixels=detector["psf_sigma_pixels"],
                    )
                    region = gaussian_rendering_region(
                        detector_moments=moments, shape_yx=shape_yx
                    )
                    start = perf_counter()
                    frame = (
                        4e7
                        * np.asarray(
                            renderer(
                                source,
                                column,
                                extra,
                                jnp.zeros(2),
                                region_shape_yx=region,
                            )
                        )
                        + 3.0
                    )
                    elapsed = perf_counter() - start
                    row.append(frame)
                    report["frames"].append(
                        {
                            "voltage_index": voltage_index,
                            "prediction": prediction,
                            "setting": index,
                            "view": view,
                            "seconds": elapsed,
                            "gaussians": int(source.x.size),
                            "rendered_region_yx": list(region),
                        }
                    )
                report["beam_states"].append(
                    {
                        "voltage_index": voltage_index,
                        "prediction": prediction,
                        "setting": index,
                        "coordinates": beam.tolist(),
                    }
                )
                means.append(row)
                np.savez_compressed(checkpoint, signature=signature, images=means)
                print(
                    f"Gaussian condenser {voltage / 1000:g} kV, prediction={prediction}: {index + 1}/{len(data['controls'])}",
                    flush=True,
                )
            images = [
                rng.poisson(row).astype(float)
                + rng.normal(0, detector["read_noise_e"], np.shape(row))
                for row in means
            ]
            data["images"] = np.asarray(images)
            np.savez_compressed(target / "images.npz", **data)
            np.savez_compressed(target / "scoring_gaussian_images.npz", images=means)
        jax.clear_caches()
    experiment.update(image_model="coherent_gaussian", detector_shape_yx=list(shape_yx))
    (root / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    (root / "gaussian_generation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def replace_aperture_with_gaussian_counts(
    *, directory, random_seed=44001, radial_order=256, azimuthal_order=128
):
    """Propagate a native Gaussian source to each aperture and integrate |t psi|²."""
    from types import SimpleNamespace

    import jax
    from numpy.polynomial.legendre import leggauss

    from temgym_core.source import gaussian_waist
    from temgym_core.components import SigmoidAperture
    from temgym_core.evaluate import evaluate_gaussians

    from .apertures import aperture_centres, effective_parameters

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    truth = np.asarray(
        json.loads((root / "scoring_truth.json").read_text())["parameters"]
    )
    nodes, weights = leggauss(radial_order)
    nodes, weights = jnp.asarray((nodes + 1) / 2), jnp.asarray(weights / 2)
    theta = jnp.arange(azimuthal_order) * 2 * jnp.pi / azimuthal_order
    report = {
        "forward_model": "native Gaussian transport and squared SigmoidAperture transmission",
        "radial_order": radial_order,
        "azimuthal_order": azimuthal_order,
        "blocks": [],
    }
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        spec = json.loads((folder / "aperture.json").read_text())
        p = effective_parameters(
            truth, voltage, experiment["reference_voltage_v"], index
        )
        centres = aperture_centres(p, len(spec["radii_m"])) * 1e-6

        def transmit(command, p=p, voltage=voltage, spec=spec, centres=centres):
            mean, covariance = condenser_gaussian_state(
                parameters=p,
                controls=command,
                voltage_v=voltage,
                aperture_fraction=jax.nn.sigmoid(p[19]),
            )
            # The transported mixture has an exact Gaussian intensity at this
            # plane. Its phase does not affect an integrated aperture current.
            incident = gaussian_waist(
                z=0.0, voltage=voltage, x=mean[0], y=mean[1],
                rms_size=jnp.sqrt(covariance[0, 0])
            )
            fractions = []
            for radius, centre in zip(spec["radii_m"], centres):
                aperture = SigmoidAperture(
                    z=0.0,
                    radius=radius,
                    edge_width=spec["edge_width_m"],
                    x0=centre[0],
                    y0=centre[1],
                )
                narrow = jnp.sqrt(covariance[0, 0]) < radius / 3
                extent = jnp.where(
                    narrow,
                    8 * jnp.sqrt(covariance[0, 0]),
                    radius + 16 * spec["edge_width_m"],
                )
                origin = jnp.where(narrow, mean[:2], centre)
                r = nodes * extent
                xx, yy = (
                    origin[0] + r[:, None] * jnp.cos(theta),
                    origin[1] + r[:, None] * jnp.sin(theta),
                )
                coordinates = jnp.stack((xx, yy), axis=-1).reshape(-1, 2)
                grid = SimpleNamespace(coords=coordinates, shape=xx.shape)
                intensity = (
                    jnp.abs(evaluate_gaussians(incident, grid, method="scan")) ** 2
                )
                transmission = jnp.exp(
                    2 * aperture.log_transmission(coordinates)
                ).reshape(xx.shape)
                fractions.append(
                    jnp.sum(
                        intensity
                        * transmission
                        * (weights * nodes * extent**2)[:, None]
                    )
                    * 2
                    * jnp.pi
                    / azimuthal_order
                )
            return jnp.stack(fractions)

        calculate = jax.jit(jax.vmap(transmit))
        for prediction in (False, True):
            path = folder / (
                "prediction/current_counts.npz" if prediction else "counts.npz"
            )
            data = dict(np.load(path))
            fractions = np.asarray(calculate(jnp.asarray(data["controls"])))
            if (
                not np.isfinite(fractions).all()
                or fractions.min() < 0
                or fractions.max() > 1.000001
            ):
                raise ValueError("Invalid native Gaussian aperture transmission.")
            rng = np.random.default_rng(random_seed + 100 * index + 1000 * prediction)
            dose = 2e7 * np.exp(rng.normal(0, 0.1, len(fractions)))
            data["reference_electrons"] = rng.poisson(dose)
            data["transmitted_electrons"] = rng.poisson(dose[:, None] * fractions)
            np.savez_compressed(path, **data)
            report["blocks"].append(
                {
                    "voltage_index": index,
                    "prediction": prediction,
                    "fractions": fractions.tolist(),
                }
            )
    (root / "gaussian_current_generation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def refine_condenser_pixel_integration(
    *,
    directory,
    random_seed=42001,
    variance_threshold_pixels=4.0,
    samples_per_pixel=16,
    selected_frames=None,
    gaussian_overlap=1.1,
):
    """Refine narrow coherent spots before drawing the final detector noise.

    The physical experiment is unchanged. Smaller coherent structures require
    finer integration inside each recorded pixel, including before detector blur.
    """
    import jax

    from temgym_core.image import detector_moments
    from temgym_core.constants import energy2wavelength

    from .condenser_images import beam_coordinates
    from .condenser_model import condenser_at_voltage
    from .projector import at_voltage
    from .projector_model import projector_components

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    truth = jnp.asarray(
        json.loads((root / "scoring_truth.json").read_text())["parameters"]
    )
    projector = jnp.asarray(
        json.loads((root.parent / "projector/scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    native = jax.jit(
        lambda p, c, v: condenser_gaussian_state(parameters=p, controls=c, voltage_v=v)
    )
    report = {
        "samples_per_pixel": samples_per_pixel,
        "variance_threshold_pixels": variance_threshold_pixels,
        "frames": [],
    }
    previous_path = root / "gaussian_pixel_refinement.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for row in previous["frames"]:
            row.setdefault("samples_per_pixel", previous["samples_per_pixel"])
        report["frames"] = previous["frames"]
    selected = (
        None if selected_frames is None else {tuple(row) for row in selected_frames}
    )
    for index, voltage in enumerate(experiment["voltages_v"]):
        parent = root / f"voltage_{index}"
        detector = json.loads((parent / "detector.json").read_text())
        material = load_known_transmission(
            parent
            / (
                "gaussian_specimen.npz"
                if (parent / "gaussian_specimen.npz").exists()
                else "moment_specimen.npz"
            )
        )
        wave = energy2wavelength(voltage)
        sample_moments = jax.jit(
            lambda mean, cov, sample=material["sampled"], wave=wave: (
                sample.gaussian_moments(mean, cov, wave)
            )
        )
        effective = condenser_at_voltage(
            truth, voltage, source_parameters=truth[22:29] if index else None
        )
        pp = at_voltage(projector, voltage, experiment["reference_voltage_v"])
        render = make_gaussian_renderer(
            shape_yx=detector["shape_yx"],
            pixel_pitch_m=detector["pixel_pitch_m"],
            samples_per_pixel=samples_per_pixel,
            psf_sigma_pixels=detector["psf_sigma_pixels"],
        )
        for prediction in (False, True):
            folder = parent / "prediction" if prediction else parent
            data = dict(np.load(folder / "images.npz"))
            images = np.load(folder / "scoring_gaussian_images.npz")["images"]
            for setting, controls in enumerate(data["controls"]):
                mean, cov = native(
                    effective, jnp.asarray(controls), jnp.asarray(voltage)
                )
                exit_mean, exit_cov = sample_moments(mean, cov)
                source = None
                for view, command in enumerate(data["projector_controls"]):
                    if (
                        selected is not None
                        and (index, prediction, setting, view) not in selected
                    ):
                        continue
                    column = tuple(
                        replace(c, include_axial_path=False) if hasattr(c, "q") else c
                        for c in projector_components(pp, jnp.asarray(command))
                    )
                    moments = detector_moments(
                        column_matrix(column),
                        exit_mean,
                        exit_cov,
                        pixel_pitch_m=detector["pixel_pitch_m"],
                        psf_sigma_pixels=detector["psf_sigma_pixels"],
                    )
                    if (
                        selected is None
                        and min(np.asarray(moments)[[2, 4]])
                        >= variance_threshold_pixels
                    ):
                        continue
                    if source is None:
                        beam = np.asarray(beam_coordinates(mean, cov))
                        extra = (
                            np.exp(2 * beam[6])
                            - (float(wave) / (4 * np.pi * np.exp(beam[4]))) ** 2
                        )
                        source = gaussian_exit_wave(
                            transmission=material,
                            profile="gaussian",
                            position_m=beam[:2] * 1e-7,
                            angle_rad=beam[2:4] * 1e-3,
                            curvature_per_m=beam[5] * 1e4,
                            rms_size_m=np.exp(beam[4]),
                            overlap=gaussian_overlap,
                        )
                    region = gaussian_rendering_region(
                        detector_moments=moments, shape_yx=detector["shape_yx"]
                    )
                    updated = (
                        4e7
                        * np.asarray(
                            render(
                                source,
                                column,
                                extra,
                                jnp.zeros(2),
                                region_shape_yx=region,
                            )
                        )
                        + 3
                    )
                    difference = (updated - images[setting, view]) / np.sqrt(
                        updated + detector["read_noise_e"] ** 2
                    )
                    report["frames"].append(
                        {
                            "samples_per_pixel": samples_per_pixel,
                            "gaussian_overlap": gaussian_overlap,
                            "voltage_index": index,
                            "prediction": prediction,
                            "setting": setting,
                            "view": view,
                            "pixel_change_rms": float(np.sqrt(np.mean(difference**2))),
                            "region_yx": list(region),
                        }
                    )
                    images[setting, view] = updated
                    print(
                        f"Refined Gaussian pixels {voltage / 1000:g} kV, new={prediction}, setting {setting + 1}, view {view + 1}",
                        flush=True,
                    )
            rng = np.random.default_rng(random_seed + 100 * index + 1000 * prediction)
            data["images"] = np.asarray(
                [
                    rng.poisson(row).astype(float)
                    + rng.normal(0, detector["read_noise_e"], np.shape(row))
                    for row in images
                ]
            )
            np.savez_compressed(folder / "images.npz", **data)
            np.savez_compressed(folder / "scoring_gaussian_images.npz", images=images)
            checkpoint = dict(np.load(folder / "gaussian_image_progress.npz"))
            np.savez_compressed(
                folder / "gaussian_image_progress.npz",
                signature=str(checkpoint["signature"]) + "; refined pixels",
                images=images,
            )
        jax.clear_caches()
    (root / "gaussian_pixel_refinement.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def prepare_condenser_moment_specimens(*, directory, step_nm=0.25):
    """Refine known-specimen quadrature independently of the Gaussian image basis."""
    import shutil

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        coarse = folder / "gaussian_specimen.npz"
        if not coarse.exists():
            shutil.copyfile(folder / "moment_specimen.npz", coarse)
        detector = json.loads((folder / "detector.json").read_text())
        fine = prepare_known_transmission(
            template=dict(np.load(folder / "template.npz")),
            voltage_v=voltage,
            kernel_sigma_m=detector["sample_kernel_sigma_m"],
            step_nm=step_nm,
        )
        save_known_transmission(folder / "moment_specimen.npz", fine)
        local = prepare_known_transmission(
            template=dict(np.load(folder / "template.npz")),
            voltage_v=voltage,
            kernel_sigma_m=detector["sample_kernel_sigma_m"],
            step_nm=0.0625,
            half_width_nm=32.0,
        )
        save_known_transmission(folder / "moment_specimen_local.npz", local)
    (root / "moment_quadrature.json").write_text(
        json.dumps(
            {
                "step_nm": step_nm,
                "local_step_nm": 0.0625,
                "local_half_width_nm": 32.0,
                "scope": "Known specimen integrals only. Gaussian image basis is retained independently.",
            },
            indent=2,
        )
        + "\n"
    )


def check_gaussian_moment_numerics(
    *, directory, rms_tolerance=0.1, coordinate_tolerance=0.3
):
    """Generator audit: compare two forward calculations before any inverse fit.

    Generating parameters are used only to assess integration accuracy. The
    fitting measurements are always extracted from the independent noisy pixels.
    """
    import jax

    from temgym_core.image import measure_detector_moments, moment_regions

    from .condenser_images import beam_coordinates
    from .condenser_model import condenser_at_voltage, projector_at_voltage
    from .inference import make_condenser_moment_predictor

    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text())
    truth = jnp.asarray(
        json.loads((root / "scoring_truth.json").read_text())["parameters"]
    )
    pp = jnp.asarray(
        json.loads((root.parent / "projector/scoring_truth.json").read_text())[
            "parameters"
        ]
    )
    native = jax.jit(
        lambda p, c, v: condenser_gaussian_state(parameters=p, controls=c, voltage_v=v)
    )
    report = {
        "frames": [],
        "passed": True,
        "rms_tolerance": rms_tolerance,
        "coordinate_tolerance": coordinate_tolerance,
    }
    for index, voltage in enumerate(experiment["voltages_v"]):
        parent = root / f"voltage_{index}"
        detector = json.loads((parent / "detector.json").read_text())
        predict = jax.jit(make_condenser_moment_predictor(folder=parent))
        effective = condenser_at_voltage(
            truth, voltage, source_parameters=truth[22:29] if index else None
        )
        for prediction in (False, True):
            folder = parent / "prediction" if prediction else parent
            data = np.load(folder / "images.npz")
            means = np.load(folder / "scoring_gaussian_images.npz")["images"]
            shape = (-1, *means.shape[-2:])
            regions = moment_regions(
                images=data["images"].reshape(shape),
                background_counts=detector["background_counts"],
                read_noise_counts=detector["read_noise_e"],
            )
            observed = measure_detector_moments(
                images=data["images"].reshape(shape),
                background_counts=detector["background_counts"],
                read_noise_counts=detector["read_noise_e"],
                regions=regions,
            )
            noiseless = measure_detector_moments(
                images=means.reshape(shape),
                background_counts=detector["background_counts"],
                read_noise_counts=detector["read_noise_e"],
                regions=regions,
            )
            white = np.linalg.inv(np.linalg.cholesky(observed["covariance"]))
            maps = jnp.asarray(
                [
                    projector_at_voltage(
                        pp,
                        c,
                        voltage,
                        reference_voltage_v=experiment["reference_voltage_v"],
                    )
                    for c in data["projector_controls"]
                ]
            )
            expected = []
            for command in data["controls"]:
                mean, covariance = native(
                    effective, jnp.asarray(command), jnp.asarray(voltage)
                )
                expected.append(
                    np.asarray(predict(beam_coordinates(mean, covariance), maps))
                )
            error = np.einsum(
                "nij,nj->ni",
                white,
                np.asarray(expected).reshape(-1, 5) - noiseless["values"],
            )
            for setting, row in enumerate(error.reshape(len(means), -1, 5)):
                for view, values in enumerate(row):
                    rms, maximum = (
                        float(np.sqrt(np.mean(values**2))),
                        float(abs(values).max()),
                    )
                    passed = rms < rms_tolerance and maximum < coordinate_tolerance
                    report["frames"].append(
                        {
                            "voltage_index": index,
                            "prediction": prediction,
                            "setting": setting,
                            "view": view,
                            "rms": rms,
                            "maximum": maximum,
                            "passed": passed,
                        }
                    )
                    report["passed"] = report["passed"] and passed
        jax.clear_caches()
    report["rms"] = float(
        np.sqrt(np.mean([row["rms"] ** 2 for row in report["frames"]]))
    )
    (root / "gaussian_moment_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def ensure_gaussian_moment_numerics(*, directory, random_seed=42001):
    """Refine detector quadrature and Gaussian interpolation before fitting."""
    root = Path(directory)
    report = check_gaussian_moment_numerics(directory=root)

    def key(row):
        return (row["voltage_index"], row["prediction"], row["setting"], row["view"])

    for samples in (16, 32, 64):
        if report["passed"]:
            return report
        history = json.loads((root / "gaussian_pixel_refinement.json").read_text())
        latest = {
            key(row): row.get("samples_per_pixel", history["samples_per_pixel"])
            for row in history["frames"]
        }
        selected = [
            key(row)
            for row in report["frames"]
            if not row["passed"] and latest.get(key(row), 8) < samples
        ]
        if not selected:
            continue
        print(
            f"Refining {len(selected)} flagged Gaussian images at {samples} samples per pixel.",
            flush=True,
        )
        refine_condenser_pixel_integration(
            directory=root,
            random_seed=random_seed,
            samples_per_pixel=samples,
            selected_frames=selected,
        )
        report = check_gaussian_moment_numerics(directory=root)
    if not report["passed"]:
        selected = [key(row) for row in report["frames"] if not row["passed"]]
        print(f"Refining Gaussian interpolation in {len(selected)} images.", flush=True)
        refine_condenser_pixel_integration(
            directory=root,
            random_seed=random_seed,
            samples_per_pixel=32,
            selected_frames=selected,
            gaussian_overlap=1.5,
        )
        report = check_gaussian_moment_numerics(directory=root)
    if not report["passed"]:
        raise RuntimeError(
            "Gaussian image/moment numerical checks failed; see gaussian_moment_validation.json before fitting."
        )
    return report
