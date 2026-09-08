"""Synthetic two-voltage condenser acquisition with fixed objective current."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .apertures import aperture_specification, make_focus_transmission
from .condenser_model import (
    condenser_at_voltage,
    magnetic_strength_ratio,
    projector_at_voltage,
)


def simulate_voltage_experiment(
    *,
    protocol_path,
    projector_views_path,
    output_directory,
    focus_counts_path=None,
    voltages_v=(200000.0, 198000.0),
    view_indices=(0, 1, 2),
    random_seed=13001,
    voltage_offset_v=0.0,
):
    """Recorded controls and calibrated maps -> fresh MAGICAL pixels and counts.

    One 4 micrometre aperture, 39 focus-scan settings per voltage, and an open
    reference for each reading. Source properties deliberately change between
    voltages. Only this generating function reads generating lens parameters.
    No ellipse measurement or objective-current sweep is acquired.
    """
    from .condenser_images import make_image_operator, predict_beam_coordinates
    from .condenser_model import generating_parameters
    from .projector_images import make_magical_template
    from .projector_model import choose_objective_projector_views

    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    views = json.loads(Path(projector_views_path).read_text())
    (folder / "reference_projector.json").write_text(json.dumps(views, indent=2))
    base_controls = np.asarray(protocol["calibration_controls"])
    if "current_controls" in protocol:
        base_scan = np.asarray(protocol["current_controls"])
    else:
        scan_indices = np.concatenate(
            [np.arange(g * 41 + 14, g * 41 + 27) for g in range(3)]
        )
        base_scan = np.load(focus_counts_path)["controls"][scan_indices]
    base = np.r_[generating_parameters(), np.log(0.0025 / (0.056 - 0.0025)), 0.2, -0.15]
    source0 = np.r_[base[0], base[13:19]]
    sources = [source0]
    for index in range(1, len(voltages_v)):
        # Independent waist shifts 0.05 mm; size +3%, divergence -2% per step.
        source = source0.copy()
        source[0] = np.log(np.exp(source0[0]) + index * 0.05e-3)
        source[1:5] += index * np.array([0.02, -0.01, 0.001, -0.001])
        source[5:] += index * np.log([1.03, 0.98])
        sources.append(source)
    rng = np.random.default_rng(random_seed)
    for index, voltage in enumerate(voltages_v):
        target = folder / f"voltage_{index}"
        target.mkdir(exist_ok=True)
        ratio = float(magnetic_strength_ratio(voltage, voltages_v[0]))
        actual_voltage = voltage + (voltage_offset_v if index else 0.0)
        effective = condenser_at_voltage(
            base,
            actual_voltage,
            reference_voltage_v=voltages_v[0],
            source_parameters=sources[index],
        )
        calibration = json.loads(json.dumps(views))
        calibration["views"] = [calibration["views"][i] for i in view_indices]
        p = np.asarray(calibration["calibrated_projector_parameters"])
        p[11] *= ratio
        p[12:16] += np.log(ratio)
        calibration["calibrated_projector_parameters"] = p.tolist()
        try:
            maps, projector_controls = choose_objective_projector_views(
                calibration, 1.0
            )
        except RuntimeError:
            # Follow the calibrated branch in small voltage steps when a direct
            # change gives the underdetermined projector solver a poor start.
            continuation = json.loads(json.dumps(views))
            continuation["views"] = [continuation["views"][i] for i in view_indices]
            for intermediate in np.linspace(
                voltages_v[0], voltage, max(3, int(np.ceil(abs(ratio - 1) / 0.002)) + 1)
            )[1:]:
                step_ratio = float(magnetic_strength_ratio(intermediate, voltages_v[0]))
                adjusted = np.asarray(views["calibrated_projector_parameters"]).copy()
                adjusted[11] *= step_ratio
                adjusted[12:16] += np.log(step_ratio)
                continuation["calibrated_projector_parameters"] = adjusted.tolist()
                maps, projector_controls = choose_objective_projector_views(
                    continuation, 1.0
                )
                for view, command in zip(continuation["views"], projector_controls):
                    view["controls"] = command.tolist()
        actual_maps = np.asarray(
            [
                projector_at_voltage(
                    jnp.asarray(views["calibrated_projector_parameters"]),
                    command,
                    actual_voltage,
                    reference_voltage_v=voltages_v[0],
                )
                for command in projector_controls
            ]
        )
        template, _ = make_magical_template(
            voltage, pitch_m=5e-9, illumination_radius_m=130e-9
        )
        detector = {
            "pixel_pitch_m": 220e-6,
            "shape_yx": [64, 64],
            "read_noise_e": 2.0,
            "psf_sigma_pixels": 0.5,
            "sample_kernel_sigma_m": 2.5e-9,
            "voltage_v": voltage,
        }
        render = make_image_operator(
            projector_matrices=actual_maps, template=template, detector=detector
        )
        controls = base_controls / ratio
        # Instrument setup keeps the illuminated specimen feature visible.
        # At a new voltage an old condenser setting can focus to a sub-nm spot.
        # Adjust only such settings, record the mini-lens command, and do not
        # supply a target beam size as an extra observation to the fitter.
        from scipy.optimize import brentq

        beam_size = jax.jit(
            lambda command, effective=effective: jnp.exp(
                predict_beam_coordinates(effective, command)[4]
            )
        )
        adjusted_settings = []
        for row, control in enumerate(controls):
            size = float(beam_size(control))
            if 10e-9 <= size <= 90e-9:
                continue
            target_size = 30e-9 if size < 10e-9 else 60e-9
            grid = control[2] * np.linspace(0.95, 1.05, 41)

            def size_error(
                mini, control=control, target_size=target_size, beam_size=beam_size
            ):
                return (
                    float(beam_size(jnp.asarray(control).at[2].set(mini))) - target_size
                )

            errors = [size_error(value) for value in grid]
            roots = [
                brentq(size_error, a, b, xtol=1e-12)
                for a, b, fa, fb in zip(grid[:-1], grid[1:], errors[:-1], errors[1:])
                if fa * fb < 0
            ]
            if not roots:
                raise RuntimeError(
                    "Could not keep the specimen feature illuminated in the allowed mini-lens range."
                )
            controls[row, 2] = min(roots, key=lambda value: abs(value - control[2]))
            adjusted_settings.append(row)
        if adjusted_settings:
            print(
                f"{voltage / 1000:g} kV: adjusted mini lens for visible illumination at settings {adjusted_settings}",
                flush=True,
            )
        frames = []
        for control in controls:
            beam = predict_beam_coordinates(effective, control)
            exposure = np.tile(np.log([4e7, 3.0]), len(maps))
            expected = np.asarray(render(np.r_[beam, exposure]))
            frames.append(rng.poisson(expected) + rng.normal(0, 2.0, expected.shape))
        np.savez_compressed(
            target / "images.npz",
            images=frames,
            controls=controls,
            projector_matrices=maps,
            projector_controls=projector_controls,
        )
        np.savez_compressed(target / "template.npz", **template)
        (target / "detector.json").write_text(json.dumps(detector, indent=2))
        spec = aperture_specification()
        spec.update(radii_m=[2e-6], voltage_v=actual_voltage)
        transmit = make_focus_transmission(specification=spec)
        scan = base_scan / ratio
        fractions = np.asarray(jax.vmap(transmit, in_axes=(None, 0))(effective, scan))
        if (
            not np.all(np.isfinite(fractions))
            or fractions.min() < 0
            or fractions.max() > 1.000002
        ):
            raise RuntimeError("Invalid aperture integration.")
        incident = 2e7 * np.exp(rng.normal(0, 0.1, len(scan)))
        np.savez_compressed(
            target / "counts.npz",
            controls=scan,
            transmitted_electrons=rng.poisson(incident[:, None] * fractions),
            reference_electrons=rng.poisson(incident),
        )
        spec["voltage_v"] = voltage
        (target / "aperture.json").write_text(json.dumps(spec, indent=2))
        print(
            f"{voltage / 1000:g} kV: {len(frames) * len(maps)} images, {2 * len(scan)} current readings",
            flush=True,
        )
    (folder / "experiment.json").write_text(
        json.dumps(
            {
                "voltages_v": list(voltages_v),
                "reference_voltage_v": voltages_v[0],
                "view_indices": list(view_indices),
                "random_seed": random_seed,
                "objective_current": 1.0,
                "aperture_diameter_um": 4.0,
                "source_parameters_per_voltage": 7,
                "aperture_centred_by_current_scan": False,
                "calibration_assumption": "Known projector model and voltage scaling; exact in this first synthetic experiment.",
            },
            indent=2,
        )
    )
    (folder / "scoring_truth.json").write_text(
        json.dumps(
            {
                "parameters": np.r_[base, np.asarray(sources[1:]).ravel()].tolist(),
                "voltage_offset_v": voltage_offset_v,
            },
            indent=2,
        )
    )
    return folder


def simulate_voltage_prediction_images(
    *, experiment_directory, protocol_path, random_seed=14001
):
    """Generate six new condenser settings per voltage for frozen prediction checks.

    This is a generator: it may read scoring truth. The analysis and prediction
    checker below do not. New controls are recorded in a separate prediction folder.
    """
    from .condenser_images import make_image_operator, predict_beam_coordinates

    folder = Path(experiment_directory)
    experiment = json.loads((folder / "experiment.json").read_text())
    truth = json.loads((folder / "scoring_truth.json").read_text())
    p = np.asarray(truth["parameters"])
    base_controls = np.asarray(
        json.loads(Path(protocol_path).read_text())["prediction_controls"]
    )
    rng = np.random.default_rng(random_seed)
    calibration = json.loads((folder / "reference_projector.json").read_text())
    for index, voltage in enumerate(experiment["voltages_v"]):
        reference = folder / f"voltage_{index}"
        target = reference / "prediction"
        target.mkdir(exist_ok=True)
        recorded = np.load(reference / "images.npz")
        maps = recorded["projector_matrices"]
        actual_voltage = voltage + (
            truth.get("voltage_offset_v", 0.0) if index else 0.0
        )
        actual_maps = np.asarray(
            [
                projector_at_voltage(
                    jnp.asarray(calibration["calibrated_projector_parameters"]),
                    command,
                    actual_voltage,
                    reference_voltage_v=experiment["reference_voltage_v"],
                )
                for command in recorded["projector_controls"]
            ]
        )
        template = dict(np.load(reference / "template.npz"))
        detector = json.loads((reference / "detector.json").read_text())
        render = make_image_operator(
            projector_matrices=actual_maps, template=template, detector=detector
        )
        effective = condenser_at_voltage(
            p,
            actual_voltage,
            reference_voltage_v=experiment["reference_voltage_v"],
            source_parameters=p[22 + 7 * (index - 1) : 22 + 7 * index]
            if index
            else None,
        )
        controls = base_controls / float(
            magnetic_strength_ratio(voltage, experiment["reference_voltage_v"])
        )
        frames = []
        for control in controls:
            beam = predict_beam_coordinates(effective, control)
            expected = np.asarray(
                render(np.r_[beam, np.tile(np.log([4e7, 3.0]), len(maps))])
            )
            frames.append(rng.poisson(expected) + rng.normal(0, 2.0, expected.shape))
        np.savez_compressed(
            target / "images.npz",
            images=frames,
            controls=controls,
            projector_matrices=maps,
            projector_controls=recorded["projector_controls"],
        )
        np.savez_compressed(target / "template.npz", **template)
        (target / "detector.json").write_text(json.dumps(detector, indent=2))
        # Independent current predictions also test the upstream scale and aperture.
        spec = json.loads((reference / "aperture.json").read_text())
        spec["voltage_v"] = actual_voltage
        transmit = make_focus_transmission(specification=spec)
        current_controls = np.array(
            [[cl1, cl3, 1.0] for cl3 in [0.65, 0.85, 1.15, 1.4] for cl1 in [1.75, 1.95]]
        )
        current_controls /= float(
            magnetic_strength_ratio(voltage, experiment["reference_voltage_v"])
        )
        fractions = np.asarray(
            jax.vmap(transmit, in_axes=(None, 0))(effective, current_controls)
        )[:, 0]
        incident = 2e7 * np.exp(rng.normal(0, 0.1, len(current_controls)))
        np.savez_compressed(
            target / "current_counts.npz",
            controls=current_controls,
            transmitted_electrons=rng.poisson(incident * fractions),
            reference_electrons=rng.poisson(incident),
        )
    return folder
