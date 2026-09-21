"""Native coherent Gaussian images for the aperture-in calibration."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from temgym_core.run import column_matrix


def make_magical_sample(voltage=200000.0, *, z=0.0, shape_yx=(384, 384)):
    """Return (sample component, known dimensions, calibration centre in metres).

    This is the same nominal synthetic MAGICAL specimen used by the calibration:
    four Si/SiGe strips, a central hole and a small silicon region. The amplitude
    and phase are prescribed, not a measured commercial-specimen certificate.
    One fixed map covers the field; no resolution pyramid or region selection
    is built. ``shape_yx`` sets the sampling density over the 352 nm field.
    """
    from temgym_core.components import InterpolatedSample2D
    from temgym_core.constants import energy2wavelength

    if not np.isfinite(voltage) or voltage <= 0:
        raise ValueError("voltage must be finite and positive.")
    shape = np.asarray(shape_yx)
    if (
        shape.shape != (2,)
        or not np.all(np.isfinite(shape))
        or np.any(shape < 4)
        or np.any(shape != shape.astype(int))
    ):
        raise ValueError("shape_yx must contain two integer sizes of at least four.")
    field_width = 352e-9
    silicon_width, sige_width = 13e-9, 10e-9
    interface_width = 1e-9
    repeats = 8
    stack_width = repeats * (silicon_width + sige_width)
    strip_height = 0.11 * field_width
    lattice_spacing = 0.31356e-9
    wavelength = float(np.asarray(energy2wavelength(voltage)))
    path_step = 0.06 * wavelength / (2 * np.pi)
    log_amplitude_step = 0.035

    # Cover a rotated field plus 8 nm on each side. Keep the original knot
    # positions (including four-knot size rounding) to preserve interpolation.
    map_width = np.sqrt(2.0) * field_width + 2 * 8e-9
    ny, nx = (int(np.ceil(np.ceil(n * map_width / field_width) / 4) * 4) for n in shape)
    dx, dy = map_width / nx, map_width / ny
    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    def smoothstep(value):
        t = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
        return t**3 * (10.0 + t * (-15.0 + 6.0 * t))

    def interval(coordinate, lower, upper):
        return smoothstep((coordinate - lower) / interface_width) * smoothstep(
            (upper - coordinate) / interface_width
        )

    # Accumulate the physical boundary coordinates in the original order.
    boundaries, coordinate = [], 0.0
    for _ in range(repeats):
        coordinate += sige_width
        boundaries.append(coordinate)
        coordinate += silicon_width
        boundaries.append(coordinate)
    sige_fraction = np.zeros_like(xx)
    origin = -0.5 * stack_width
    for repeat in range(repeats):
        start = origin + (0.0 if repeat == 0 else boundaries[2 * repeat - 1])
        stop = origin + boundaries[2 * repeat]
        sige_fraction += 0.5 * (
            np.tanh((xx - start) / interface_width)
            - np.tanh((xx - stop) / interface_width)
        )
    sige_fraction = np.clip(sige_fraction, 0.0, 1.0)
    path, log_amplitude = np.zeros_like(xx), np.zeros_like(xx)
    for layers, fraction_y in enumerate((-0.36, -0.12, 0.12, 0.36), start=1):
        cy = fraction_y * field_width
        mask = interval(xx, -0.5 * stack_width, 0.5 * stack_width) * interval(
            yy, cy - 0.5 * strip_height, cy + 0.5 * strip_height
        )
        material = mask * (0.55 + 0.45 * sige_fraction)
        path += layers * path_step * material
        log_amplitude -= layers * log_amplitude_step * material

    cx, size = 0.36 * field_width, 80 * lattice_spacing
    mask = interval(xx, cx - 0.5 * size, cx + 0.5 * size) * interval(
        yy, -0.5 * size, 0.5 * size
    )
    path += 0.5 * path_step * mask
    log_amplitude -= 0.5 * log_amplitude_step * mask
    if lattice_spacing >= 4.0 * max(dx, dy):
        lattice = np.cos(2 * np.pi * (xx - cx) / lattice_spacing) + np.cos(
            2 * np.pi * yy / lattice_spacing
        )
        path += 0.15 * path_step * mask * lattice
    outside_hole = smoothstep((np.hypot(xx, yy) - 20e-9) / interface_width)
    path *= outside_hole
    log_amplitude *= outside_hole

    sample = InterpolatedSample2D.from_action_maps(
        pathlength_yx_m=path,
        log_amplitude_yx=log_amplitude,
        x_coords_m=x,
        y_coords_m=y,
        z=z,
    )
    strip_top = 0.12 * field_width + strip_height / 2
    centre = np.array([4e-9, strip_top - 2e-9])
    dimensions = {
        "silicon_width_m": silicon_width,
        "sige_width_m": sige_width,
        "repeat_m": silicon_width + sige_width,
        "interface_width_m": interface_width,
        "perforation_diameter_m": 40e-9,
        "pixel_size_yx_m": (dy, dx),
        "third_strip_top_m": strip_top,
        "third_strip_centre_m": (0.0, 0.12 * field_width),
        "strip_size_xy_m": (stack_width, strip_height),
        "source": "nominal synthetic MAGICAL specimen",
    }
    return sample, dimensions, centre


def condenser_gaussian_state(
    *, parameters, controls, voltage_v, aperture_fraction=None
):
    """Generate the specimen state by native Gaussian source/lens propagation."""
    from temgym_core.run import propagate_gaussian_mixture
    from temgym_core.source import gaussian_waist

    from .condenser_model import condenser_components

    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    z = -(jnp.sum(gaps) + jnp.sum(lengths))
    source = gaussian_waist(
        z=z,
        voltage=voltage_v,
        x=p[13] * 1e-8,
        y=p[14] * 1e-8,
        theta_x=p[15] * 1e-4,
        theta_y=p[16] * 1e-4,
        rms_size=jnp.exp(p[17]),
    )
    extra = jnp.eye(4).at[:2, :2].set(0) * jnp.exp(2 * p[18])
    column = tuple(
        replace(lens, include_axial_path=False) if hasattr(lens, "q") else lens
        for lens in condenser_components(p, controls)
    )
    if aperture_fraction is not None:
        from temgym_core.components import Plane

        aperture_z = column[1].z + column[1].length + aperture_fraction * gaps[2]
        column = (*column[:2], Plane(z=aperture_z))
    return propagate_gaussian_mixture(source, column, extra)


def make_gaussian_renderer(
    *,
    shape_yx,
    pixel_pitch_m,
    samples_per_pixel,
    psf_sigma_pixels=0.0,
    padding_pixels=8,
):
    """Sum the coherent field over the full camera, including faint diffraction tails."""
    import jax

    from temgym_core.components import Detector
    from temgym_core.evaluate import gaussian_image, gpu_kernel_available
    from temgym_core.gaussian import FreeSpacePropagator
    from temgym_core.run import run_to_end

    method = "spatial" if gpu_kernel_available() else "scan"
    sensor = Detector(z=0.0, shape=shape_yx, pixel_size=(pixel_pitch_m, pixel_pitch_m))

    @jax.jit
    def render(source, components, angular_variance, offset_pixels):
        output = run_to_end(
            source, components, propagator=FreeSpacePropagator(include_axial_path=False)
        )
        output = output.derive(
            x=output.x + offset_pixels[0] * pixel_pitch_m,
            y=output.y + offset_pixels[1] * pixel_pitch_m,
        )
        matrix = column_matrix(
            components,
            source_z=jnp.ravel(jnp.asarray(source.z))[0],
            voltage=jnp.ravel(jnp.asarray(source.voltage))[0],
        )
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

    return render


def aperture_wave(
    parameters,
    controls,
    *,
    voltage_v,
    radius_m,
    edge_width_m,
    centre_m,
    gaussians_across,
):
    """Equivalent coherent aperture-plane mode plus its independent tilt variance.

    A Gaussian source ensemble remains Gaussian before the aperture. At this
    plane it is exactly a coherent Gaussian with an incoherent tilt distribution.
    The aperture multiplies every tilt mode identically; downstream vacuum optics
    turn that distribution into a detector blur. All beamlets interfere.
    """
    import jax

    from temgym_core.constants import energy2wavelength
    from temgym_core.source import decompose_gaussian, gaussian_waist

    from .condenser_model import condenser_with_aperture

    column = condenser_with_aperture(
        parameters,
        controls,
        radius_m=radius_m,
        edge_width_m=edge_width_m,
        centre_m=centre_m,
    )
    aperture = column[2]
    mean, covariance = condenser_gaussian_state(
        parameters=parameters,
        controls=controls,
        voltage_v=voltage_v,
        aperture_fraction=jax.nn.sigmoid(parameters[19]),
    )
    size = float(jnp.sqrt(covariance[0, 0]))
    curvature = covariance[0, 2] / covariance[0, 0]
    floor = (energy2wavelength(voltage_v) / (4 * jnp.pi * size)) ** 2
    extra = covariance[2, 2] - covariance[0, 2] ** 2 / covariance[0, 0] - floor
    if float(extra) < -1e-20:
        raise ValueError("Nonphysical source coherence at the aperture.")
    source = gaussian_waist(
        z=aperture.z,
        voltage=voltage_v,
        rms_size=size,
        x=mean[0],
        y=mean[1],
        theta_x=mean[2],
        theta_y=mean[3],
    )
    source = source.derive(Q_inv=source.Q_inv + curvature * jnp.eye(2))
    extent = min(
        7.0,
        float(
            (radius_m + 16 * edge_width_m + jnp.max(jnp.abs(mean[:2] - centre_m)))
            / size
        ),
    )
    bundle = decompose_gaussian(source, num_across=gaussians_across, extent_rms=extent)
    return bundle, column[2:], jnp.maximum(extra, 0.0)


def render_beam_images(
    *,
    specification,
    settings,
    projector_parameters,
    condenser_parameters,
    gaussians_across=512,
    samples_per_pixel=4,
    selected=None,
):
    """Render both inserted apertures for each recorded setting using native components."""
    from .apertures import aperture_centres, effective_parameters
    from .condenser_model import condenser_with_steering
    from .projector_model import at_voltage, projector_components

    render = make_gaussian_renderer(
        shape_yx=tuple(specification["shape_yx"]),
        pixel_pitch_m=specification["pixel_pitch_m"],
        samples_per_pixel=samples_per_pixel,
        psf_sigma_pixels=specification["psf_sigma_pixels"],
    )
    indices = (
        np.arange(len(settings["voltage_index"]))
        if selected is None
        else np.asarray(selected)
    )
    previous, cache = None, {}
    for index in indices:
        vi = int(settings["voltage_index"][index])
        voltage = specification["voltages_v"][vi]
        control = settings["condenser_controls"][index]
        key = (vi, *control)
        if key != previous:
            cache.clear()
            previous = key
        cp = effective_parameters(
            jnp.asarray(condenser_parameters),
            voltage,
            specification["voltages_v"][0],
            vi,
        )
        pp = at_voltage(
            jnp.asarray(projector_parameters), voltage, specification["voltages_v"][0]
        )
        centres = aperture_centres(cp, 2) * 1e-6
        kick = settings["specimen_kick"][index]
        projector = projector_components(
            pp, jnp.asarray(settings["projector_controls"][index])
        )
        frames = []
        for ai, diameter in enumerate(specification["aperture_diameters_um"]):
            if ai not in cache:
                wave = aperture_wave(
                    cp,
                    jnp.asarray(control),
                    voltage_v=voltage,
                    radius_m=diameter * 0.5e-6,
                    edge_width_m=specification["aperture_edge_width_m"],
                    centre_m=centres[ai],
                    gaussians_across=gaussians_across,
                )
                cache[ai] = wave
            beam, condenser, angular_variance = cache[ai]
            steered = condenser_with_steering(
                condenser,
                kick,
                voltage_v=voltage,
                gap_fractions=specification.get("deflector_gap_fractions", (0.3, 0.7)),
            )
            column = tuple(
                replace(c, include_axial_path=False) if hasattr(c, "q") else c
                for c in (*steered, *projector)
            )
            frames.append(
                np.asarray(render(beam, column, angular_variance, jnp.zeros(2)))
            )
        print(
            f"Gaussian beam pair {int(index) + 1}/{len(settings['voltage_index'])}",
            flush=True,
        )
        yield int(index), np.asarray(frames)


def generate_images(*, directory, gaussians_across=512, samples_per_pixel=4):
    """Save noisy Gaussian images; refine pixel integration where the audit requires it."""
    import json
    from pathlib import Path

    from .inference import numerical_image_check

    root = Path(directory)
    spec = json.loads((root / "experiment.json").read_text())
    settings = dict(np.load(root / "settings.npz"))
    truth = json.loads((root / "scoring_truth.json").read_text())
    count = len(settings["voltage_index"])
    means = np.zeros((count, 2, *spec["shape_yx"]))
    used_sampling = np.zeros(count, dtype=int)
    selected = np.arange(count)
    for refinement in range(3):
        sampling = samples_per_pixel * 2**refinement
        for index, pair in render_beam_images(
            specification=spec,
            settings=settings,
            projector_parameters=truth["projector_parameters"],
            condenser_parameters=truth["condenser_parameters"],
            gaussians_across=gaussians_across,
            samples_per_pixel=sampling,
            selected=selected,
        ):
            means[index] = pair
            used_sampling[index] = sampling
        # Incident dose is unknown to the fitter and shared within each aperture pair.
        rng = np.random.default_rng(spec["random_seed"])
        dose = spec.get("nominal_incident_electrons_per_image", 1e8) * np.exp(
            rng.normal(0.0, 0.05, count)
        )
        expected = means * dose[:, None, None, None]
        images = rng.poisson(expected + spec["background_counts"]).astype(float)
        images += rng.normal(0.0, spec["read_noise_counts"], images.shape)
        np.savez_compressed(root / "images.npz", images=images)
        np.savez_compressed(root / "scoring_images.npz", images=expected)
        report = numerical_image_check(directory_path=root, means=expected, truth=truth)
        report.update(
            gaussians_across=gaussians_across,
            samples_per_pixel=int(used_sampling.max()),
            samples_per_pixel_per_pair=used_sampling.tolist(),
        )
        (root / "numerical_checks.json").write_text(json.dumps(report, indent=2) + "\n")
        if report["passed"]:
            return
        selected = np.asarray(report["flagged_pairs"])
        print(
            f"Refining pixel integration for {len(selected)} aperture pairs", flush=True
        )
    raise RuntimeError(
        "Aperture image/moment numerical checks failed; refine Gaussian sampling "
        "or inspect detector clipping before fitting."
    )
