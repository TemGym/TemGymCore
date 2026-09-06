"""Compare aligned aperture sizes using the same measured, uncertain projector/beams.

Run after the full notebook. Only currents are regenerated: the image data and
its full calibration covariance are reused unchanged for a controlled test.
"""

import argparse
import json
import shutil
from pathlib import Path

from full_microscope_apertures import copy_current_analysis, simulate_aligned_apertures
from full_microscope_checks import check_microscope
from full_microscope_inference import assemble_full_fit, fit_condenser_block

HERE = Path(__file__).parent


def run_comparison(*, source_directory, output_directory, random_seed=32001):
    source, root = Path(source_directory), Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    report = {}
    cases = [
        ("two", [4.0, 10.0], True),
        ("three", [4.0, 10.0, 20.0], True),
        ("three_no_alignment_constraint", [4.0, 10.0, 20.0], False),
    ]
    for label, diameters, alignment_enabled in cases:
        case = root / label
        if (
            case.resolve() == source.resolve()
            or source.resolve() in case.resolve().parents
        ):
            raise ValueError(
                "Comparison output must be separate from the source acquisition."
            )
        shutil.copytree(source, case, dirs_exist_ok=True)
        for name in [
            "checks.json",
            "condenser/analysis/full_fit.json",
            "condenser/analysis/fit/result.json",
        ]:
            (case / name).unlink(missing_ok=True)
        simulate_aligned_apertures(
            experiment_directory=case / "condenser",
            diameters_um=diameters,
            centering_sigma_um=0.2,
            random_seed=random_seed,
        )
        metadata = json.loads((case / "experiment.json").read_text())
        metadata.update(
            aperture_diameters_um=diameters,
            centering_sigma_um=0.2,
            movement_scan=False,
            calibration_current_readings=78 * (len(diameters) + 1),
            prediction_current_readings=16 * (len(diameters) + 1),
        )
        (case / "experiment.json").write_text(json.dumps(metadata, indent=2))
        analysis = case / "condenser/analysis"
        copy_current_analysis(
            images_directory=case / "condenser", analysis_directory=analysis
        )
        alignment_path = analysis / "alignment/alignment_constraint.json"
        alignment = json.loads(alignment_path.read_text())
        alignment["enabled"] = alignment_enabled
        alignment_path.write_text(json.dumps(alignment, indent=2))
        print("CASE", label, flush=True)
        try:
            condenser = fit_condenser_block(
                analysis_directory=analysis,
                starting_guess_path=HERE / "microscope_condenser_starting_guess.json",
            )
            fitted = assemble_full_fit(
                analysis_directory=analysis,
                condenser_fit=condenser,
                bounds_path=HERE / "projector_position_priors.json",
            )
            checks = check_microscope(
                experiment_directory=case,
                fitted=fitted,
                bounds_path=HERE / "projector_position_priors.json",
            )
            report[label] = {
                "diameters_um": diameters,
                "centering_sigma_um": 0.2,
                "alignment_constraint_enabled": alignment_enabled,
                "fit": condenser,
                "checks": checks,
            }
        except RuntimeError as error:
            report[label] = {"error": str(error)}
        (root / "comparison.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=HERE.parent / "generated/full_microscope_aligned_apertures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE.parent / "generated/full_microscope_aperture_sizes",
    )
    args = parser.parse_args()
    run_comparison(source_directory=args.source, output_directory=args.output)
