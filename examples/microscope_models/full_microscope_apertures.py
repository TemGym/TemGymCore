"""Aligned aperture sizes with independent unknown residual centres.

No movement scan is used. Alignment accuracy is a declared experimental
assumption, not a simulated sub-nanometre measurement. All holes are aligned
once at the reference voltage/controls and retain their centres thereafter.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_aperture_experiment import aperture_moments
from condenser_rotation_measurements import (
    make_focus_transmission,
    make_radial_transmission,
)
from condenser_voltage_experiment import condenser_at_voltage


def aperture_centres(parameters, number_of_apertures):
    """Micrometres; first centre is p[20:22], additional centres follow source two."""
    p = jnp.asarray(parameters)
    if number_of_apertures == 1:
        return p[20:22][None, :]
    return jnp.concatenate(
        [p[20:22][None, :], p[29:].reshape(number_of_apertures - 1, 2)]
    )


def effective_parameters(parameters, voltage, reference, source_index):
    p = jnp.asarray(parameters)
    effective = condenser_at_voltage(
        p,
        voltage,
        reference_voltage_v=reference,
        source_parameters=p[22:29] if source_index else None,
    )
    return p.at[:22].set(effective)


def make_aperture_transmission(*, specification):
    if not specification.get("independent_centres", False):
        return make_focus_transmission(specification=specification)
    count = len(specification["radii_m"])
    operators = [
        make_radial_transmission(specification={**specification, "radii_m": [r]})
        for r in specification["radii_m"]
    ]

    def transmit(p, controls):
        relative, variance = aperture_moments(p, controls, "after")
        absolute = relative + p[20:22] * 1e-6
        centres = aperture_centres(p, count) * 1e-6
        return jnp.stack(
            [
                operator(absolute - centre, variance)[0]
                for operator, centre in zip(operators, centres)
            ]
        )

    return jax.jit(transmit)


def current_fractions(transmitted, reference):
    """Ratios and full covariance, including the shared open reference."""
    transmitted, reference = (
        np.asarray(transmitted, dtype=float),
        np.asarray(reference, dtype=float),
    )
    if transmitted.ndim == 1:
        transmitted = transmitted[:, None]
    values = transmitted / reference[:, None]
    covariance = (
        np.asarray([np.diag(row) for row in transmitted])
        / reference[:, None, None] ** 2
        + values[:, :, None] * values[:, None, :] / reference[:, None, None]
    )
    return values, covariance


def alignment_residual(parameters, alignment):
    """Declared alignment accuracy constrains every hole relative to one reference beam."""
    if alignment.get("enabled", True) is False:
        return jnp.empty(0)
    count = alignment.get("number_of_apertures", 1)
    mean, _ = aperture_moments(
        parameters, jnp.asarray(alignment["lens_controls"]), "after"
    )
    offsets = (
        mean + parameters[20:22] * 1e-6 - aperture_centres(parameters, count) * 1e-6
    )
    target = jnp.asarray(alignment["mean_minus_aperture_centre_m"])
    return ((offsets - target) / alignment["isotropic_sigma_m"]).ravel()


def simulate_aligned_apertures(
    *,
    experiment_directory,
    diameters_um=(4.0, 10.0, 20.0),
    centering_sigma_um=0.2,
    random_seed=32001,
):
    """Generator only: replace currents and aperture truth; leave open-aperture pixels intact."""
    root = Path(experiment_directory)
    diameters = np.asarray(diameters_um, dtype=float)
    if (
        diameters.ndim != 1
        or len(diameters) not in (2, 3)
        or np.any(diameters <= 0)
        or len(np.unique(diameters)) != len(diameters)
    ):
        raise ValueError("Supply two or three distinct positive diameters.")
    if centering_sigma_um <= 0:
        raise ValueError("Supply a positive centering uncertainty.")
    experiment = json.loads((root / "experiment.json").read_text())
    truth = json.loads((root / "scoring_truth.json").read_text())
    p = np.asarray(truth["parameters"])[:29].copy()
    controls = np.array([1.75, 0.75, 1.0])
    mean, _ = aperture_moments(jnp.asarray(p), jnp.asarray(controls), "after")
    centre = np.asarray(mean) * 1e6 + p[20:22]
    # Independent alignment errors, drawn before any count noise. Two-hole cases
    # use the same first two errors as three-hole cases at the same seed.
    errors = np.random.default_rng(random_seed).normal(0, centering_sigma_um, (3, 2))[
        : len(diameters)
    ]
    centres = centre + errors
    p[20:22] = centres[0]
    p = np.r_[p, centres[1:].ravel()]
    truth.update(parameters=p.tolist(), aperture_alignment_errors_um=errors.tolist())
    (root / "scoring_truth.json").write_text(json.dumps(truth, indent=2))
    for index, voltage in enumerate(experiment["voltages_v"]):
        folder = root / f"voltage_{index}"
        spec = json.loads((folder / "aperture.json").read_text())
        spec.update(radii_m=(diameters * 0.5e-6).tolist(), independent_centres=True)
        (folder / "aperture.json").write_text(json.dumps(spec, indent=2))
        operator = make_aperture_transmission(specification=spec)
        effective = effective_parameters(
            p, voltage, experiment["reference_voltage_v"], index
        )
        for prediction in [False, True]:
            path = folder / (
                "prediction/current_counts.npz" if prediction else "counts.npz"
            )
            recorded = np.load(path)
            commands = recorded["controls"]
            fractions = np.asarray(
                jax.vmap(operator, in_axes=(None, 0))(effective, commands)
            )
            if (
                np.any(~np.isfinite(fractions))
                or fractions.min() < 0
                or fractions.max() > 1.000002
            ):
                raise RuntimeError("Invalid aperture transmission.")
            # Separate RNG streams make common holes/reference identical between
            # the two- and three-hole experiments, for a controlled comparison.
            seed = random_seed + 100 * index + 1000 * prediction
            rng = np.random.default_rng(seed + 1)
            incident = 2e7 * np.exp(rng.normal(0, 0.1, len(commands)))
            reference = rng.poisson(incident)
            counts = np.stack(
                [
                    np.random.default_rng(seed + 10 + j).poisson(
                        incident * fractions[:, j]
                    )
                    for j in range(len(diameters))
                ],
                axis=1,
            )
            np.savez_compressed(
                path,
                controls=commands,
                transmitted_electrons=counts,
                reference_electrons=reference,
            )
    alignment = root / "alignment"
    alignment.mkdir(exist_ok=True)
    (alignment / "alignment_constraint.json").write_text(
        json.dumps(
            {
                "method": "ordinary_alignment",
                "number_of_apertures": len(diameters),
                "lens_controls": controls.tolist(),
                "mean_minus_aperture_centre_m": [0.0, 0.0],
                "isotropic_sigma_m": centering_sigma_um * 1e-6,
                "current_readings": 0,
                "scope": "Assumed alignment accuracy per axis, not inferred from a movement scan. Independent hole offsets; common unknown axial plane. Centres remain fixed across voltages and current settings.",
            },
            indent=2,
        )
    )
    # A reused acquisition must not retain the old movement-scan observations.
    for name in ["scan.npz", "measurement.json", "scoring_truth.json", "aperture.json"]:
        (alignment / name).unlink(missing_ok=True)
    experiment.update(
        aperture_centred_by_current_scan=False,
        aperture_diameters_um=diameters.tolist(),
        number_of_apertures=len(diameters),
        centering_sigma_um=centering_sigma_um,
        alignment_accuracy_assumed=True,
    )
    (root / "experiment.json").write_text(json.dumps(experiment, indent=2))
    return root


def copy_current_analysis(*, images_directory, analysis_directory):
    """Analyse new currents without repeating independent open-aperture image analysis."""
    import shutil

    source, target = Path(images_directory), Path(analysis_directory)
    experiment = json.loads((source / "experiment.json").read_text())
    # Keep the image-analysis calibration provenance when replacing currents.
    # Generator metadata can still describe the historical exact-projector case.
    measured = json.loads((target / "experiment.json").read_text())
    if "calibration_assumption" in measured:
        experiment["calibration_assumption"] = measured["calibration_assumption"]
    (target / "experiment.json").write_text(json.dumps(experiment, indent=2))
    shutil.copyfile(
        source / "alignment/alignment_constraint.json",
        target / "alignment/alignment_constraint.json",
    )
    for index in [0, 1]:
        folder, out = source / f"voltage_{index}", target / f"voltage_{index}"
        data = np.load(folder / "counts.npz")
        values, covariance = current_fractions(
            data["transmitted_electrons"], data["reference_electrons"]
        )
        np.savez_compressed(
            out / "current_measurements.npz",
            controls=data["controls"],
            values=values,
            covariance=covariance,
        )
        shutil.copyfile(folder / "aperture.json", out / "aperture.json")
