"""Regenerate ``baseline.json`` and ``baseline.npz`` next to this file.

Run from the repository root, on CPU, and only when a numerical change has been
intended and reviewed::

    JAX_PLATFORMS=cpu PYTHONPATH=src:tests python tests/golden/make_baseline.py

``tests/test_golden.py`` recomputes every scenario and compares it with these
files, so a refactor that silently changes physics fails there.
"""
import hashlib
import json
import platform
from pathlib import Path

import jax
import numpy as np
import scipy

import golden_scenario as scenario

HERE = Path(__file__).parent
ROOT = scenario.ROOT

SAVED_RESULTS = {
    "sem_fit": "examples/sem_reflection/output/calibration_gaussian/fit.json",
    "sem_checks": "examples/sem_reflection/output/calibration_gaussian/checks.json",
    "tem_validation": "examples/microscope_models/calibration_results/tilt_y_30_ray_acquisition/validation.json",
    "tem_repeats": "examples/microscope_models/calibration_results/tilt_y_30_ray_acquisition/repeat_validation.json",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_constants():
    """Snapshot the constants the code takes from ``ase.units`` (CODATA 2014)."""
    from ase import units

    names = ("_e", "_me", "_c", "_hplanck", "_hbar", "kg", "C", "s", "J")
    return {name: float(getattr(units, name)) for name in names}


def saved_results():
    """Key numbers from the tracked results of the SEM and TEM calibration threads."""
    load = lambda key: json.loads((ROOT / SAVED_RESULTS[key]).read_text())
    fit, checks = load("sem_fit"), load("sem_checks")
    validation, repeats = load("tem_validation"), load("tem_repeats")
    selected = validation["selected"]
    return {
        "sha256": {key: sha256(ROOT / path) for key, path in SAVED_RESULTS.items()},
        "sem": {
            "parameters": fit["best"]["parameters"],
            "chi_squared": fit["best"]["chi_squared"],
            "rank": fit["rank"],
            "accepted": fit["accepted"],
            "prediction_rms_noise_units": checks["prediction_rms_noise_units"],
            "fit_rms_noise_units": checks["fit_rms_noise_units"],
            "objective_working_distance_mm": checks["objective_working_distance_mm"],
        },
        "tem": {
            "used_images": selected["used_images"],
            "fit_rms": selected["fit_rms"],
            "prediction_rms": selected["prediction_rms"],
            "rank": selected["rank"],
            "source_z": selected["source_z"],
            "PL1_z": selected["PL1_z"],
            "detector_z": selected["detector_z"],
            "repeat_seeds": repeats["seeds"],
            "repeat_accepted": repeats["accepted_realizations"],
            "repeat_geometry": [
                {key: row[key] for key in ("parameter", "rms_error", "empirical_sd",
                                          "maximum_absolute_error", "mean_local_sd")}
                for row in repeats["geometry"]
            ],
        },
    }


def main():
    solution = scenario.focused_microscope()
    planes, at_camera, line_field, patch_field = scenario.propagate_and_render(solution)
    baseline = {
        "provenance": {
            "jax": jax.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "note": "Regenerate only for an intended, reviewed numerical change.",
        },
        "raw_constants": raw_constants(),
        "constants": scenario.constants_scenario(),
        "microscope": scenario.microscope_scenario(solution),
        "planes": planes,
        "beam": scenario.beam_subset(at_camera),
        "fringes": scenario.fringe_numbers(line_field),
        "saved_results": saved_results(),
    }
    (HERE / "baseline.json").write_text(json.dumps(baseline, indent=1) + "\n")
    np.savez_compressed(HERE / "baseline.npz", line_field=line_field, patch_field=patch_field)
    print("wrote", HERE / "baseline.json", "and", HERE / "baseline.npz")


if __name__ == "__main__":
    main()
