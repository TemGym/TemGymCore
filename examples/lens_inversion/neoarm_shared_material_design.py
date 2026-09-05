"""Find a same-column 40k conjugate setting without generating specimen images.

The objective, IL2 and IL3 remain fixed. Only IL1 and PL1 are adjusted inside
the previously declared synthetic +/-0.2 control box. These are model-study
coordinates, not verified hardware commands or fitted NEOARM parameters.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_aperture_capture import ROOT, XY, ANGLE
from neoarm_condenser_endpoint_calibration import digest
from neoarm_condenser_extension_design import load_context, FOLDER
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot
from temgym_core.constants import energy2wavelength
from temgym_core.jax_least_squares import least_squares


def round_canonical_metrics(matrix):
    a = matrix[jnp.ix_(jnp.asarray(XY), jnp.asarray(XY))]
    b = matrix[jnp.ix_(jnp.asarray(XY), jnp.asarray(ANGLE))]
    magnification = jnp.sqrt(jnp.linalg.det(a))
    delta = .5*jnp.trace(jnp.linalg.solve(a, b))
    return jnp.array([magnification, delta, jnp.arctan2(a[1, 0], a[0, 0])])


def neighbour_radius_estimate(delta_m, wavelength_m, spacing_m, width_over_step=1.25,
                              field_envelope_tolerance=1e-16):
    """Analytic admission estimate for identical flat-phase Gaussian beamlets."""
    width = width_over_step*spacing_m
    output_width_over_spacing = np.hypot(width_over_step,
        delta_m*wavelength_m/(np.pi*width*spacing_m))
    return int(np.ceil(np.sqrt(-np.log(field_envelope_tolerance))*output_width_over_spacing))+1


def run(output):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Native map and control-root calculations require GPU float64")
    started = perf_counter()
    sources, computational = provenance_snapshot()
    calibration_path = FOLDER/"condenser_endpoint_calibration.json"
    measured_path = FOLDER/"measured_specimen.json"
    calibration = json.loads(calibration_path.read_text())
    measured = json.loads(measured_path.read_text())
    for report in (calibration, measured):
        verify_provenance_snapshot(report["source_hashes"])
    _, _, _, _, model = load_context()
    p = jnp.asarray(model.fixed)
    target = 40000.
    channels = np.array([model.column.channel_names.index("IL1"), model.column.channel_names.index("PL1")])
    base = jnp.zeros(7)
    matrix_for = jax.jit(lambda v: model.column.matrix(p, base.at[channels].set(v),
        segment="sample_to_detector", coordinates="canonical"))
    residual = jax.jit(lambda v: (round_canonical_metrics(matrix_for(v))[:2]-jnp.array([target, 0.])) /
        jnp.array([target, 1e-6]))
    starts = [np.asarray(u)[channels] for u in calibration["analyser_controls"]]+[np.zeros(2)]
    records = []
    for initial in starts:
        fit = least_squares(residual, initial, bounds=(-.2*np.ones(2), .2*np.ones(2)),
            x_scale=np.array([.1, .1]), ftol=0., xtol=0., gtol=1e-12, max_nfev=80)
        matrix = np.asarray(matrix_for(fit.x))
        metric = np.asarray(round_canonical_metrics(jnp.asarray(matrix)))
        a = matrix[np.ix_(XY, XY)]
        b = matrix[np.ix_(XY, ANGLE)]
        equivalent = np.linalg.solve(a, b)
        jacobian = np.asarray(jax.jacfwd(residual)(fit.x))
        records.append(dict(initial_il1_pl1=initial.tolist(), final_il1_pl1=np.asarray(fit.x).tolist(),
            optimizer_success=bool(fit.success), residual=np.asarray(residual(fit.x)).tolist(),
            magnification=metric[0], effective_defocus_m=metric[1], rotation_rad=metric[2],
            canonical_equivalent_defocus_matrix_m=equivalent.tolist(),
            scaled_constraint_jacobian=jacobian.tolist(), condition_number=float(np.linalg.cond(jacobian)),
            maximum_absolute_B_m=float(np.max(abs(b))),
            controls=np.asarray(base.at[channels].set(fit.x)).tolist(), canonical_matrix=matrix.tolist()))
    if not all(abs(r["magnification"]/target-1) < 1e-10 and
               np.linalg.norm(r["canonical_equivalent_defocus_matrix_m"]) < 1e-12 for r in records):
        raise RuntimeError("The bounded native-map conjugate root did not close")
    selected = records[-1]
    wavelength = float(energy2wavelength(200e3))
    sample = measured["rows"][0]
    spacing = sample["selection"]["selected_spacing_m"]
    beam_spacing = sample["refinements"][-1]["grid_spacing_m"]
    phase_bound = np.pi*wavelength*np.linalg.norm(selected["canonical_equivalent_defocus_matrix_m"], 2)/spacing**2
    neighbours = neighbour_radius_estimate(selected["effective_defocus_m"], wavelength, beam_spacing)
    a = np.asarray(selected["canonical_matrix"])[np.ix_(XY, XY)]
    halo_halfwidth = (128+3)*55e-6
    footprint = halo_halfwidth*np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])@np.linalg.inv(a).T
    for prior in (calibration, measured):
        sources.update(prior["source_hashes"])
    for path in (Path(__file__), calibration_path, measured_path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    verify_provenance_snapshot(sources)
    report = dict(schema="neoarm-same-column-conjugate-material-design-v1", source_hashes=sources,
        computational_input_hashes=computational, backend=jax.default_backend(), x64_enabled=True,
        raw_specimen_images_generated=False, new_noise_or_optical_parameter_fit=False,
        target_magnification=target, free_control_channels=["IL1", "PL1"], fixed_optical_parameters=np.asarray(p).tolist(),
        synthetic_control_bounds=[[-.2, .2], [-.2, .2]], controls=selected["controls"],
        canonical_sample_to_detector=selected["canonical_matrix"], root_records=records,
        objective_fixed=True, exact_same_column_as_condenser_calibration=True,
        specimen_cubic_spacing_m=spacing, specimen_beam_grid_spacing_m=beam_spacing,
        conjugate_intensity_phase_bound_rad=phase_bound, conjugate_intensity_gate_threshold_rad=1e-4,
        predicted_gaussian_neighbour_radius=neighbours, archived_conjugate_neighbour_radius=9,
        detector_mtf_halo_footprint_sample_xy_m=footprint.tolist(),
        same_atlas_can_be_used=True, frozen_crop_coverage_requires_actual_view_check=True,
        proposed_field="Canonical sample-plane field; no mechanical-slope Gaussian is passed to a canonical map without conversion.",
        limitations=["A third material-imaging setting at40k, not reuse of either defocused condenser analyser image.",
            "Only two synthetic controls are solved; no physical optics is fitted or hardware command range verified.",
            "The shared finite-support objective is unchanged; field-tail uncertainty remains outside this conditional design.",
            "A constant canonical phase sample beam has mechanical magnetic circulation; do not call it a mechanically parallel beam.",
            "Pure sample-plane illumination is independent of recovered condenser illumination unless coherent modes are propagated from the source.",
            "Conjugate intensity alone cannot measure pure phase contrast; nominal multilayer amplitude supplies material contrast.",
            "New Gaussian and grid-reference finite-pixel images and fitting-box crop/convergence gates are still required."],
        elapsed_seconds=perf_counter()-started)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(controls=selected["controls"], magnification=selected["magnification"],
        effective_defocus_m=selected["effective_defocus_m"], phase_bound_rad=phase_bound,
        neighbour_radius=neighbours, elapsed_seconds=report["elapsed_seconds"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run(args.output)
