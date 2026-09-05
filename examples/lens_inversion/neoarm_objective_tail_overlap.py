"""Whole-column synthetic objective tails, including controlled-field overlap.

Freeze the saved 40k/80k convergence experiment, then compare the old truncated
field with a smooth continuation from source to detector. Controlled and
objective signed fields are added before canonical propagation. No lens/DAC
parameters are refitted. This tests sensitivity to an unmeasured field model;
it does not promote a Glaser-like tail into a measured NEOARM field profile.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import time
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.axial_field_column import FixedAxialFieldMesh, objective_tail_field
from temgym_core.immersed_objective import SharedObjectiveField, mechanical_to_canonical
from temgym_core.gaussian_image_likelihood import (
    GaussianDetectorOperator, encode_gaussian_image_parameters, gaussian_image_information_factor,
)
from temgym_core.neoarm_calibration import DetectorGeometry, hard_edge_column_from_config
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_shared_objective import SharedObjectiveColumn, shared_imaging_metrics
from neoarm_shared_objective_convergence import compare_images


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/"examples/generated/neoarm_inversion/imaging_recipe/shared_objective_convergence.json"
BODY_INDICES = (-1, 0, -1, 1, -1, 2, -1, -1, -1, -1, 3, -1, 4, -1, 5, -1, 6, -1)
SAMPLE_BOUNDARY = 8


@dataclass(frozen=True, eq=False)
class TailOverlapColumn:
    """Same 34 optical parameters, with a fixed topology following lens edges."""

    base: SharedObjectiveColumn
    mesh: FixedAxialFieldMesh

    @classmethod
    def from_shared(cls, column, parameters, *, steps_per_segment=32, truncate_objective=False):
        # The normalization window remains an amplitude convention, not a
        # physical field stop. Keeping its edges in the mesh makes the
        # truncated comparison equally well aligned with its discontinuities.
        boundaries, _ = cls(column, None).geometry_and_rates(parameters, jnp.zeros(7))
        mesh = FixedAxialFieldMesh.from_boundaries(column.objective, boundaries, BODY_INDICES,
            SAMPLE_BOUNDARY, steps_per_segment=steps_per_segment,
            mesh_centre_m=-.0005, mesh_scale_m=.003, truncate_objective=truncate_objective)
        return cls(column, mesh)

    @property
    def parameter_names(self):
        return self.base.parameter_names

    def geometry_and_rates(self, parameters, controls):
        p, u = jnp.asarray(parameters), jnp.asarray(controls)
        if u.shape != (7,):
            raise ValueError("Expected seven synthetic free-lens controls.")
        edges, rates = [p[0]], []
        for i, (lens, centre, length, bias, slope) in enumerate(self.base._decoded_lenses(p)):
            if i == 3:
                edges.extend((self.base.objective.z_edges_m[0], self.base.sample_z_m,
                              self.base.objective.z_edges_m[-1]))
            edges.extend((centre-length/2, centre+length/2))
            rates.append((bias+slope*(u[lens.channel_index]-lens.reference_dac))/length)
        edges.append(p[1])
        return jnp.stack(edges), jnp.stack(rates)

    def split_maps(self, parameters, controls):
        edges, rates = self.geometry_and_rates(parameters, controls)
        return self.mesh.split_maps(jnp.asarray(parameters)[-4:], edges, rates)

    def trace(self, parameters, controls, ray, *, segment="full", coordinates="mechanical"):
        if coordinates not in {"canonical", "mechanical"}:
            raise ValueError("Declare mechanical or canonical endpoint coordinates.")
        maps = self.split_maps(parameters, controls)
        attributes = {"full": "full", "source_to_sample": "pre", "sample_to_detector": "post"}
        if segment not in attributes:
            raise ValueError("Unknown column segment.")
        attribute = attributes[segment] + ("_mechanical" if coordinates == "mechanical" else "")
        return getattr(maps, attribute)@jnp.asarray(ray)

    def matrix(self, parameters, controls, *, segment="full", coordinates="mechanical"):
        return jax.jacfwd(lambda ray: self.trace(parameters, controls, ray, segment=segment,
            coordinates=coordinates))(jnp.zeros(4, dtype=jnp.asarray(parameters).dtype))

    def matrix_to_plane(self, parameters, controls, z_m):
        """Physical source inputs to canonical state at a plane; positions physical."""
        edges, rates = self.geometry_and_rates(parameters, controls)
        p = jnp.asarray(parameters)[-4:]
        source_k = (objective_tail_field(edges[0], p, self.mesh.normalization_field)
                    *self.mesh.objective_active_segments[0])
        return self.mesh.canonical_map_to_plane(p, edges, rates, z_m)@mechanical_to_canonical(source_k)


def run_audit():
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("This forward-model audit requires GPU float64.")
    started = time.monotonic()
    saved = json.loads(INPUT.read_text())
    paths = [ROOT/name for name in saved["source_hashes"]]
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != saved["source_hashes"][str(path.relative_to(ROOT))]:
            raise ValueError(f"Stale input experiment source: {path}")
    paths.extend((INPUT, Path(__file__), ROOT/"src/temgym_core/axial_field_column.py"))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    parameters = jnp.asarray(saved["fixed_initial_parameters"])
    field = SharedObjectiveField.from_window(-.05, 0., .05, n_slices_per_side=8192)
    config = tomllib.loads((ROOT/"examples/lens_inversion/neoarm_inversion_config.toml").read_text())
    measured = column_with_geometry_anchors(hard_edge_column_from_config(config),
        ROOT/"examples/microscope_models/measure_microscope_geometry.ipynb")
    column = SharedObjectiveColumn.from_hard_edge(measured, objective=field,
                                                 objective_initial=parameters[-4:])
    controls = [np.asarray(row) for row in saved["fixed_controls"]]
    names = ["initialized_40k", "initialized_80k"]
    for q in -np.linspace(.05, np.pi-.05, 9):
        control = controls[0].copy()
        control[6] = (q-float(parameters[28]))/float(parameters[29])
        controls.append(control); names.append(f"PL1_q_{q:.8f}")
    controls = jnp.asarray(np.stack(controls))
    detector = DetectorGeometry(mtf_sigma_px=.65)
    covariance = jnp.asarray(saved["source_covariance_m_rad"])
    kicks = jnp.asarray(saved["calibrated_mechanical_sample_states"])
    dose, read_noise = saved["dose_electrons"], saved["read_noise_electrons"]
    render = jax.jit(GaussianDetectorOperator(detector, quadrature_order=16))
    encode = jax.jit(lambda mean, cov: encode_gaussian_image_parameters(mean, cov, dose, detector))
    information = jax.jit(lambda p: gaussian_image_information_factor(
        p, detector, read_noise_electrons=read_noise, quadrature_order=16))
    xy = jnp.array([0, 2])
    data, variants = {}, {}

    def evaluate(label, model):
        down, full = [jax.jit(jax.vmap(lambda u: model.matrix(parameters, u, segment=segment)))(controls)
                      for segment in ("sample_to_detector", "full")]
        images, encoded, spatial_covariances = [], [], []
        for f, d in zip(full, down):
            spatial = (f@covariance@f.T)[jnp.ix_(xy, xy)]
            spatial_covariances.append(np.asarray(spatial))
            for kick in kicks:
                mean = (d@kick)[xy]
                images.append(render(mean, spatial, dose)); encoded.append(encode(mean, spatial))
        images, encoded = np.asarray(jnp.stack(images)), np.asarray(jnp.stack(encoded))
        metrics = np.asarray(jax.vmap(shared_imaging_metrics)(down))
        data[label] = dict(images=images, image_parameters=encoded)
        variants[label] = dict(mechanical_downstream_matrices=np.asarray(down).tolist(),
            mechanical_full_matrices=np.asarray(full).tolist(),
            metrics_M_rotation_rad_full_BA_row_major=metrics.tolist(),
            spatial_covariance_detector_m2=np.asarray(spatial_covariances).tolist(),
            image_parameters=encoded.tolist(), detected_fraction=(images.sum(axis=(1, 2))/dose).tolist(),
            expected_images_float64_sha256=hashlib.sha256(images.astype('<f8').tobytes()).hexdigest(),
            cf4_intervals=None if not isinstance(model, TailOverlapColumn) else model.mesh.interval_count)
        if isinstance(model, TailOverlapColumn):
            maps = model.split_maps(parameters, controls[0])
            variants[label]["endpoint_signed_k_m_inv"] = [float(maps.source_k_m_inv),
                float(maps.sample_k_m_inv), float(maps.detector_k_m_inv)]
        print(label, "M40/80", metrics[:2, 0].tolist(), "minimum capture", float(images.sum(axis=(1, 2)).min()/dose), flush=True)

    evaluate("archived_truncated_midpoint8192", column)
    for n in (16, 32, 64, 128, 256, 512):
        evaluate(f"full_tail_cf4_{n}", TailOverlapColumn.from_shared(column, parameters, steps_per_segment=n))
    evaluate("truncated_cf4_256", TailOverlapColumn.from_shared(column, parameters,
                                                              steps_per_segment=256, truncate_objective=True))
    reference_label = "full_tail_cf4_512"
    factors = np.stack([np.asarray(information(value)) for value in data[reference_label]["image_parameters"]])
    # Cropped images still have measurable finite-detector observables. Their
    # numerical convergence and their separate capture admission must not be
    # conflated. Never pseudoinvert a rank-deficient image likelihood.
    singular = np.linalg.svd(factors, compute_uv=False)
    resolved = singular[:, -1] > singular[:, 0]*1e-12
    captured = np.asarray(variants[reference_label]["detected_fraction"]) >= .99
    comparisons = {}
    for label, values in data.items():
        candidate = {key: value[resolved] for key, value in values.items()}
        reference = {key: value[resolved] for key, value in data[reference_label].items()}
        if not np.any(resolved):
            raise RuntimeError("No well captured, locally resolved Gaussian probe frames.")
        result = compare_images(candidate, reference, factors[resolved])
        result["resolved_frame_indices"] = np.flatnonzero(resolved).tolist()
        metrics = np.asarray(variants[label]["metrics_M_rotation_rad_full_BA_row_major"])
        ref_metrics = np.asarray(variants[reference_label]["metrics_M_rotation_rad_full_BA_row_major"])
        result["relative_magnification_change"] = (metrics[:, 0]/ref_metrics[:, 0]-1).tolist()
        result["rotation_change_degrees"] = np.rad2deg(np.angle(np.exp(1j*(metrics[:, 1]-ref_metrics[:, 1])))).tolist()
        result["maximum_full_BA_change_m"] = float(np.max(np.abs(metrics[:, 2:]-ref_metrics[:, 2:])))
        comparisons[label] = result
    successive = {}
    for n in (64, 128, 256):
        candidate = {key: value[resolved] for key, value in data[f"full_tail_cf4_{n}"].items()}
        reference = {key: value[resolved] for key, value in data[f"full_tail_cf4_{2*n}"].items()}
        successive[f"{n}_to_{2*n}"] = compare_images(candidate, reference, factors[resolved])
    hashes_after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if hashes_after != hashes:
        raise RuntimeError("An audited source changed during generation; do not save stale evidence.")
    return dict(schema="neoarm-whole-column-objective-tail-overlap-v1", source_hashes=hashes,
        source_hashes_unchanged_during_execution=True, backend=jax.default_backend(),
        devices=[str(device) for device in jax.devices()], jax_version=jax.__version__,
        elapsed_seconds=time.monotonic()-started, fixed_parameters=np.asarray(parameters).tolist(),
        parameter_names=column.parameter_names, fixed_controls=np.asarray(controls).tolist(), state_names=names,
        source_mechanical_covariance_m_rad=np.asarray(covariance).tolist(),
        calibrated_mechanical_sample_states=np.asarray(kicks).tolist(), incident_electrons=dose,
        read_noise_electrons=read_noise, detector_pixel_quadrature_order=16,
        physical_field_amplitude_preserved=True, normalization_reference_window_m=[-.05, .05],
        normalization_reference_slices_per_side=8192, propagation_outside_source_detector=False,
        source_detector_assumed_field_free=False,
        mesh_fixed_tangent_centre_m=-.0005, mesh_fixed_tangent_scale_m=.003,
        reference_variant=reference_label, variants=variants, comparisons_to_full_tail_reference=comparisons,
        successive_refinements_using_final_image_information=successive,
        locally_resolved_frames=resolved.tolist(), well_captured_frames=captured.tolist(),
        excluded_from_observable_comparison=np.flatnonzero(~resolved).tolist(),
        fitted_optical_parameters_changed=False, physical_objective_profile_measured=False,
        validated_over_all_inference_bounds=False, narrow_calibration_probe_frames_validated=False,
        hardware_commands_admitted=False,
        interpretation=["This is sensitivity to an assumed smooth objective continuation, not a measured field.",
            "All signed fields add before k squared; hard-edge geometry is differentiated through explicit moving boundaries.",
            "Source and detector mechanical conversions include their small nonzero objective field.",
            "Gaussian covariances at input are physical mechanical covariances; maps convert them internally.",
            "No optical parameter or DAC command is refitted to conceal changed field support.",
            "Wide states outside capture/resolution admission are retained as diagnostics, not validated measurements.",
            "The full-tail prescription remains a conditional synthetic model even after numerical convergence."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Saved", args.output, flush=True)


if __name__ == '__main__':
    main()
