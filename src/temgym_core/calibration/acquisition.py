"""Record the beam-only experiment: two apertures stay inserted throughout."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .condenser_model import design_condenser_curve, magnetic_strength_ratio
from .condenser_model import generating_parameters as generating_condenser
from .projector_model import choose_objective_projector_views
from .projector_model import generating_parameters as generating_projector

PRESETS = Path(__file__).parent / "presets"


def prepare_experiment(*, directory, voltages_v, aperture_diameters_um, random_seed):
    """Save recorded controls/calibrations separately from simulation parameters."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    pp = generating_projector()
    base = np.r_[generating_condenser(), np.log(0.0025 / (0.056 - 0.0025)), 0.2, -0.15]
    # Synthetic pre-field: f ≈ 2.2 mm keeps the aperture halo on the camera.
    # This is a generating parameter, never supplied as a fitting constraint.
    base[12] = np.log(0.70)
    second = np.r_[base[0], base[13:19]]
    second[0] = np.log(np.exp(base[0]) + 0.05e-3)
    second[1:5] += [0.02, -0.01, 0.001, -0.001]
    second[5:] += np.log([1.03, 0.98])
    cp = np.r_[base, second, 0.12, -0.08]

    protocol = json.loads((PRESETS / "beam_measurement_protocol.json").read_text())
    views = [
        {"controls": list(protocol["projector_reference_controls"]), "target": [m, 0.0]}
        for m in protocol["nominal_magnifications"]
    ]
    illumination = design_condenser_curve(
        base,
        image_magnifications=[protocol["condenser_image_magnification"]],
        mini_control=protocol["mini_control"],
    )["controls"][0]
    curve = design_condenser_curve(
        base,
        image_magnifications=protocol["condenser_curve_magnifications"],
        mini_control=protocol["mini_control"],
    )
    rows = []
    group = 0

    def record(vi, controls, command, kick, purpose, group, mag, change):
        rows.append(
            (
                vi,
                controls.copy(),
                command.copy(),
                np.asarray(kick),
                purpose,
                group,
                mag,
                change,
            )
        )

    for vi, voltage in enumerate(voltages_v):
        ratio = float(magnetic_strength_ratio(voltage, voltages_v[0]))
        # Follow the focused branch in voltage while holding objective current fixed.
        for intermediate in np.linspace(
            voltages_v[0], voltage, max(2, int(abs(voltage - voltages_v[0]) / 1000) + 1)
        ):
            r = float(magnetic_strength_ratio(intermediate, voltages_v[0]))
            adjusted = pp.copy()
            adjusted[11] *= r
            adjusted[12:16] += np.log(r)
            _, commands = choose_objective_projector_views(
                {"calibrated_projector_parameters": adjusted.tolist(), "views": views},
                1.0,
            )
            for view, command in zip(views, commands):
                view["controls"] = command.tolist()
        baseline = illumination / ratio
        for mag, command in zip(protocol["nominal_magnifications"], commands):
            projector_states = [("Baseline", command)]
            for lens, fraction in protocol["projector_wobble_fractions"].items():
                index = ["Objective", "IL1", "IL2", "IL3", "PL1"].index(lens)
                for sign in [-1, 1]:
                    control = command.copy()
                    control[index] *= 1 + sign * fraction
                    projector_states.append((f"{lens} {sign * fraction:+.2%}", control))
            for label, control in projector_states:
                # Round lenses: X shifts/tilts determine the two isotropic 2x2 blocks.
                # Independent Y inputs are reserved for prediction checks below.
                for kick in (
                    [protocol["shift_m"], 0.0, 0.0, 0.0],
                    [0.0, 0.0, protocol["tilt_rad"], 0.0],
                ):
                    for sign in [1, -1]:
                        record(
                            vi,
                            baseline,
                            control,
                            sign * np.asarray(kick),
                            "projector",
                            group,
                            mag,
                            label,
                        )
                    group += 1
            record(vi, baseline, command, np.zeros(4), "condenser", -1, mag, "Baseline")
            for index, lens in enumerate(["CL1", "CL3", "Mini"]):
                for fraction in protocol["condenser_wobble_fractions"][lens]:
                    for sign in [-1, 1]:
                        control = baseline.copy()
                        control[index] *= 1 + sign * fraction
                        record(
                            vi,
                            control,
                            command,
                            np.zeros(4),
                            "condenser",
                            -1,
                            mag,
                            f"{lens} {sign * fraction:+.2%}",
                        )
            for size, control in zip(curve["image_magnifications"], curve["controls"]):
                record(
                    vi,
                    control / ratio,
                    command,
                    np.zeros(4),
                    "condenser",
                    -1,
                    mag,
                    f"Coupled source image ×{size:g}",
                )
            record(
                vi,
                baseline * np.array([1.0007, 0.9993, 1.001]),
                command,
                np.zeros(4),
                "check",
                -1,
                mag,
                "Combined condenser check",
            )
            record(
                vi,
                baseline,
                command,
                [0.0, protocol["shift_m"], 0.0, 0.0],
                "check",
                -1,
                mag,
                "Y shift check",
            )
            record(
                vi,
                baseline,
                command,
                [0.0, 0.0, 0.0, protocol["tilt_rad"]],
                "check",
                -1,
                mag,
                "Y tilt check",
            )
    keys = [
        "voltage_index",
        "condenser_controls",
        "projector_controls",
        "specimen_kick",
        "purpose",
        "response_group",
        "nominal_magnification",
        "lens_change",
    ]
    np.savez_compressed(
        root / "settings.npz",
        **{key: np.asarray(values) for key, values in zip(keys, zip(*rows))},
    )
    specification = {
        "voltages_v": list(voltages_v),
        "aperture_diameters_um": list(aperture_diameters_um),
        "aperture_edge_width_m": 150e-9,
        "aperture_inserted_for_every_image": True,
        "specimen_region": "vacuum hole; no specimen scattering",
        "length_scale": "already calibrated; specimen-plane shifts and tilts are known in metres/radians",
        "kick_order": ["x_m", "y_m", "theta_x_rad", "theta_y_rad"],
        "kick_location": "calibrated displacement and angle at specimen; realized by two post-aperture deflectors",
        "deflector_gap_fractions": [0.3, 0.7],
        "deflector_positions": "illustrative fractions of the free gap from aperture to mini lens; not measured or fitted",
        "deflector_alignment": "native DoubleDeflector shift/tilt balances aligned by Newton-Raphson at the specimen",
        "steering_assumption": "both deflectors are after the aperture, so steering leaves its illumination unchanged",
        "objective_current": 1.0,
        "nominal_magnification": 40000.0,
        "nominal_magnifications": protocol["nominal_magnifications"],
        "crossover_fraction_bounds": protocol["crossover_fraction_bounds"],
        "condenser_reference_controls": illumination.tolist(),
        "crossover_assumption": "First source image lies in the free gap between CL1 exit and CL3 entrance; this is a model restriction, not measured geometry.",
        "shape_yx": [256, 256],
        "pixel_pitch_m": 110e-6,
        "psf_sigma_pixels": 1.0,
        "background_counts": 3.0,
        "read_noise_counts": 2.0,
        "random_seed": random_seed,
        "nominal_incident_electrons_per_image": protocol[
            "nominal_incident_electrons_per_image"
        ],
        "dose_assumption": "equal incident dose within each aperture pair; fitted independently for every pair",
        "source_model": "round partially coherent Gaussian waist, independent at each voltage",
        "source_parameters_per_voltage": 7,
        "noise_model": "independent Poisson electrons plus calibrated Gaussian read noise",
    }
    (root / "experiment.json").write_text(json.dumps(specification, indent=2) + "\n")
    (root / "scoring_truth.json").write_text(
        json.dumps(
            {"projector_parameters": pp.tolist(), "condenser_parameters": cp.tolist()},
            indent=2,
        )
        + "\n"
    )
    return root


def setting_column(
    condenser_parameters,
    projector_parameters,
    *,
    voltage_v,
    reference_voltage_v,
    source_index,
    condenser_controls,
    projector_controls,
    specimen_kick,
    aperture_index,
    aperture_diameter_um,
    aperture_edge_width_m,
    gap_fractions=(0.3, 0.7),
):
    """Native source-to-detector column for one physical microscope setting.

    Parameters
    ----------
    condenser_parameters, projector_parameters : array_like
        Physical model parameters (31 and 16 values), as saved in
        ``scoring_truth.json`` or decoded from a fit.
    voltage_v, reference_voltage_v : float
        Beam voltage and the voltage at which the parameters are stated.
    source_index : int
        0 for the reference voltage's source, 1 for the second voltage's source.
    condenser_controls, projector_controls : array_like
        Relative lens controls: CL1, CL3, mini; and Objective, IL1, IL2, IL3, PL1.
    specimen_kick : array_like
        Calibrated specimen displacement and angle ``[x, y, theta_x, theta_y]`` in
        metres and radians, realised by a post-aperture double deflector.
    aperture_index : int
        Which of the two fitted aperture centres to use.
    aperture_diameter_um, aperture_edge_width_m : float
        Inserted aperture diameter and its sigmoid edge width.
    gap_fractions : tuple of float, optional
        Illustrative deflector positions as fractions of the aperture-to-mini gap.

    Returns
    -------
    column : tuple
        Native components from the condenser to the detector plane.
    mean, covariance : numpy.ndarray
        Source ray moments (4 and 4x4).
    source_z : float
        Source position relative to the specimen, in metres.
    """
    from .apertures import aperture_centres, effective_parameters
    from .condenser_model import (
        condenser_with_aperture,
        condenser_with_steering,
        source_moments,
    )
    from .projector_model import at_voltage, projector_components

    cp = effective_parameters(
        condenser_parameters, voltage_v, reference_voltage_v, source_index
    )
    pp = at_voltage(projector_parameters, voltage_v, reference_voltage_v)
    column = condenser_with_aperture(
        cp,
        condenser_controls,
        radius_m=aperture_diameter_um * 0.5e-6,
        edge_width_m=aperture_edge_width_m,
        centre_m=aperture_centres(cp, 2)[aperture_index] * 1e-6,
    )
    column = condenser_with_steering(
        column,
        specimen_kick,
        voltage_v=voltage_v,
        gap_fractions=gap_fractions,
    )
    column = (
        *column,
        *projector_components(pp, projector_controls),
    )
    mean, covariance = map(np.asarray, source_moments(cp, voltage_v))
    return column, mean, covariance, -np.exp(np.asarray(cp[:9])).sum()


def _column_source(spec, settings, model, row, aperture):
    """Use the recorded voltage, currents and kicks with native components."""
    vi = int(settings["voltage_index"][row])
    voltage, reference = spec["voltages_v"][vi], spec["voltages_v"][0]
    column, mean, covariance, source_z = setting_column(
        model["condenser_parameters"],
        model["projector_parameters"],
        voltage_v=voltage,
        reference_voltage_v=reference,
        source_index=vi,
        condenser_controls=settings["condenser_controls"][row],
        projector_controls=settings["projector_controls"][row],
        specimen_kick=settings["specimen_kick"][row],
        aperture_index=aperture,
        aperture_diameter_um=spec["aperture_diameters_um"][aperture],
        aperture_edge_width_m=spec["aperture_edge_width_m"],
        gap_fractions=spec.get("deflector_gap_fractions", (0.3, 0.7)),
    )
    return column, mean, covariance, source_z, voltage


@lru_cache(maxsize=4)
def _ray_renderer(shape):
    """Compile native ray transport and weighted detector binning once per shape."""
    import jax
    import jax.numpy as jnp

    from temgym_core.components import Detector, SigmoidAperture
    from temgym_core.run import run_iter

    @jax.jit
    def render(column, source_rays, pixel_pitch):
        weights = jnp.ones_like(source_rays.x)
        for component, rays in run_iter(source_rays, column):
            if isinstance(component, SigmoidAperture):
                # Native Rays have no amplitude; accumulate intensity separately.
                weights *= jnp.exp(2 * component.log_transmission(rays.r_xy))
        detector = Detector(
            z=rays.z, shape=shape, pixel_size=(pixel_pitch, pixel_pitch)
        )
        iy, ix = detector.metres_to_pixels((rays.x, rays.y))
        inside = (iy >= 0) & (iy < shape[0]) & (ix >= 0) & (ix < shape[1])
        index = jnp.where(inside, iy * shape[1] + ix, 0)
        return (
            jnp.bincount(
                index, weights=jnp.where(inside, weights, 0), length=shape[0] * shape[1]
            ).reshape(shape)
            / weights.size
        )

    return render


@lru_cache(maxsize=1)
def _source_samples(power):
    from scipy.special import ndtri
    from scipy.stats import qmc

    # Fixed, evenly distributed samples reduce sampling speckle between settings.
    return ndtri(qmc.Sobol(d=4, scramble=True, seed=24003).random_base2(power))


def render_ray_image(spec, settings, model, row, aperture):
    """Render detector signal from native rays; no aperture diffraction or noise."""
    column, mean, covariance, source_z, voltage = _column_source(
        spec, settings, model, row, aperture
    )
    image = render_rays(
        column,
        mean,
        covariance,
        source_z,
        voltage,
        shape_yx=spec["shape_yx"],
        pixel_pitch_m=spec["pixel_pitch_m"],
        psf_sigma_pixels=spec["psf_sigma_pixels"],
        ray_sample_power=spec.get("ray_sample_power", 18),
        detector_samples_per_pixel=spec.get("detector_samples_per_pixel", 1),
    )
    return image * spec["nominal_incident_electrons_per_image"]


def render_rays(
    column,
    mean,
    covariance,
    source_z,
    voltage_v,
    *,
    shape_yx,
    pixel_pitch_m,
    psf_sigma_pixels,
    ray_sample_power=18,
    detector_samples_per_pixel=1,
):
    """Expected detector signal per incident electron from native rays.

    Parameters
    ----------
    column : tuple
        Native components ending at the detector plane, e.g. from
        :func:`setting_column`.
    mean, covariance : array_like
        Source ray moments; the source is sampled with fixed Sobol points.
    source_z : float
        Source position in metres.
    voltage_v : float
        Beam voltage.
    shape_yx : tuple of int
        Detector rows and columns.
    pixel_pitch_m : float
        Detector pixel pitch.
    psf_sigma_pixels : float
        Gaussian detector point-spread function, applied before pixel integration.
    ray_sample_power : int, optional
        Trace ``2**ray_sample_power`` rays.
    detector_samples_per_pixel : int, optional
        Internal sub-samples per pixel axis before the PSF and binning.

    Returns
    -------
    numpy.ndarray
        Transmitted fraction per pixel (sums to the transmitted fraction on the
        detector); no aperture diffraction and no noise.
    """
    from scipy.signal import fftconvolve

    from temgym_core.ray import Ray

    samples = (
        mean
        + _source_samples(ray_sample_power)
        @ np.linalg.cholesky(covariance).T
    )
    rays = Ray(*samples.T, z=source_z, pathlength=0.0, voltage=voltage_v)
    # Apply the detector PSF before integrating detector pixels. A fine internal
    # grid avoids rounding sparse rays to whole pixels before the blur.
    sampling = detector_samples_per_pixel
    height, width = shape_yx
    image = np.asarray(
        _ray_renderer((height * sampling, width * sampling))(
            column, rays, pixel_pitch_m / sampling
        )
    )
    # Blur only occupied rows/columns plus the complete PSF support. Keeping the
    # crop on pixel boundaries gives the same image without filtering empty space.
    occupied = [np.flatnonzero(image.any(axis=axis)) for axis in (1, 0)]
    result = np.zeros((height, width))
    if all(len(indices) for indices in occupied):
        halo = int(np.ceil(8 * psf_sigma_pixels)) + 1
        bounds = [
            (max(0, indices[0] // sampling - halo),
             min(size, indices[-1] // sampling + halo + 1))
            for indices, size in zip(occupied, (height, width))
        ]
        (y0, y1), (x0, x1) = bounds
        crop = image[y0 * sampling:y1 * sampling, x0 * sampling:x1 * sampling]
        # Separable FFT convolution is fast even on the fine detector grid.
        # Retain eight sigma: four-sigma truncation biases high-dose beam widths.
        sigma = psf_sigma_pixels * sampling
        blurred = crop
        if sigma > 0:
            radius = int(8 * sigma + 0.5)
            kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
            kernel /= kernel.sum()
            blurred = fftconvolve(crop, kernel[:, None], mode="same", axes=0)
            blurred = fftconvolve(blurred, kernel[None, :], mode="same", axes=1)
            blurred = np.maximum(blurred, 0)  # Remove FFT round-off below zero.
        result[y0:y1, x0:x1] = blurred.reshape(
            y1 - y0, sampling, x1 - x0, sampling
        ).sum(axis=(1, 3))
    return result
