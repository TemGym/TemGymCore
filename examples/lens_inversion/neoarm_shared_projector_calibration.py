"""Four physical projectors in the same conditional shared-objective column.

This adapter uses native seven-channel controls, never the affine bridge or
posterior from the earlier independent-half-objective experiment. Four lens
lengths, four physical gaps and eight excitation coordinates are free. The
shared objective field, specimen shear and detector axial anchor are explicit
conditional inputs, not measurements of the real instrument.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_aperture_capture import ROOT, XY
from neoarm_condenser_endpoint_calibration import GROUPED, STEPS, PIXEL_M, digest
from neoarm_condenser_extension_design import load_context, FOLDER
from neoarm_coupled_projector_fit import grouped_stage
from neoarm_last_lens_tomography import mechanical_shear
from neoarm_projector_branch_experiment import pure_source, signed_inputs, approximate_response_covariance
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot
from temgym_core.microscope_inversion import drift_matrix

LENS_NAMES = ("PL1", "IL3", "IL2", "IL1")
PARAMETER_NAMES = tuple(f"{name}.{field}" for name in LENS_NAMES
    for field in ("log_length_m", "log_exit_gap_m", "q_bias", "q_slope_per_native_dac"))


class SharedProjectorModel:
    """Interleaved canonical matrices; grouped mechanical signed responses."""

    def __init__(self, column, fixed_parameters, *, steps=STEPS, pixel_m=PIXEL_M):
        self.column = column
        self.fixed = np.asarray(fixed_parameters).copy()
        self.steps, self.pixel_m = np.asarray(steps), float(pixel_m)
        if self.fixed.shape != (34,) or self.steps.shape != (4,) or min(self.steps) <= 0 or self.pixel_m <= 0:
            raise ValueError("Expected the shared 34-coordinate column and positive physical calibration scales")
        self.channels = np.array([column.channel_names.index(name) for name in LENS_NAMES])
        self.references = np.array([column.lenses[6-i].reference_dac for i in range(4)])
        split = column.objective.split_maps(jnp.asarray(self.fixed[-4:]))
        self.post = jnp.asarray(split.post)
        self.kappa_per_m = float(split.sample_k_m_inv)
        self.shear = mechanical_shear(self.kappa_per_m)
        self.objective_exit_m = float(column.objective.z_edges_m[-1])
        self.detector_z_m = float(self.fixed[1])

    def stage_coordinates(self, native_parameters):
        p = np.asarray(native_parameters)
        stages = []
        next_entrance = p[1]
        for i in (6, 5, 4, 3):
            z, log_length, bias, slope = p[2+4*i:6+4*i]
            length = np.exp(log_length)
            gap = next_entrance-z-length/2
            if gap <= 0:
                raise ValueError("A physical field-free exit gap must be positive")
            stages.extend((log_length, np.log(gap), bias, slope))
            next_entrance = z-length/2
        return np.asarray(stages)

    def native_parameters(self, stages):
        """Move all four lens bodies consistently with their lengths/gaps."""
        stages = jnp.asarray(stages).reshape(4, 4)
        p = jnp.asarray(self.fixed)
        next_entrance = self.detector_z_m
        for i in range(4):
            length, gap = jnp.exp(stages[i, :2])
            centre = next_entrance-gap-length/2
            p = p.at[2+4*(6-i):6+4*(6-i)].set(jnp.r_[centre, stages[i, 0], stages[i, 2:]])
            next_entrance = centre-length/2
        return p

    def objective_to_il1_gap(self, stages):
        return self.detector_z_m-self.objective_exit_m-jnp.sum(jnp.exp(jnp.asarray(stages).reshape(4, 4)[:, :2]))

    def matrix(self, stages, native_controls, *, coordinates="canonical"):
        p = jnp.asarray(stages)
        control = jnp.asarray(native_controls)
        if p.shape != (16,) or control.shape != (7,):
            raise ValueError("Sixteen physical stage coordinates and seven native controls required")
        prefix = drift_matrix(self.objective_to_il1_gap(p))@self.post
        grouped = prefix[jnp.ix_(jnp.asarray(GROUPED), jnp.asarray(GROUPED))]
        for i in (3, 2, 1, 0):
            grouped = grouped_stage(p[4*i:4*i+4], control[self.channels[i]]-self.references[i])@grouped
        if coordinates == "mechanical":
            grouped = grouped@self.shear
        elif coordinates != "canonical":
            raise ValueError("Declare canonical or mechanical sample coordinates")
        return grouped[jnp.ix_(jnp.asarray(GROUPED), jnp.asarray(GROUPED))]

    def predict(self, stages, native_controls):
        matrices = jax.vmap(lambda u: self.matrix(stages, u, coordinates="mechanical"))(native_controls)
        return matrices[:, XY][:, :, GROUPED]*jnp.asarray(self.steps)/self.pixel_m

    def admissible(self, stages):
        p = np.asarray(stages)
        if p.shape != (16,) or not np.isfinite(p).all():
            return False
        shape = p.reshape(4, 4)
        return bool(np.all(np.exp(shape[:, 0]) > 1e-4) and np.all(np.exp(shape[:, 1]) > 1e-4)
            and self.objective_to_il1_gap(p) > 1e-4)


def capture_lower_bound(means_m, covariance_m2, *, pixel_m=PIXEL_M, half_detector_pixels=128):
    """Conservative Gaussian-tail union bound, used only for design screening."""
    from jax.scipy.special import ndtr
    sigma = jnp.sqrt(jnp.diagonal(covariance_m2, axis1=-2, axis2=-1)/pixel_m**2+.65**2)
    mean = means_m/pixel_m
    tails = ndtr((-half_detector_pixels-mean)/sigma[:, None, :])+ndtr((mean-half_detector_pixels)/sigma[:, None, :])
    return 1-jnp.sum(tails, axis=-1)


def local_information(jacobian, covariance):
    white = np.linalg.solve(np.linalg.cholesky(covariance), jacobian).reshape(-1, 16)
    norms = np.linalg.norm(white, axis=0)
    _, singular, right = np.linalg.svd(white/norms, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    result = dict(rank=rank, normalized_singular_values=singular.tolist())
    if rank == 16:
        factor = (right.T/singular)/norms[:, None]
        result.update(standard_errors=np.linalg.norm(factor, axis=1).reshape(4, 4).tolist(),
            length_gap_standard_errors_percent=(100*np.linalg.norm(factor, axis=1).reshape(4, 4)[:, :2]).tolist())
    return result


def design(output, *, control_half_width=.35, training_count=12, heldout_count=3):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("Native optics and design require GPU float64")
    started = perf_counter()
    sources, computational = provenance_snapshot()
    _, _, _, aperture, context = load_context()
    model = SharedProjectorModel(context.column, context.fixed)
    stages = model.stage_coordinates(context.fixed)
    # These are declared model-design hypotheses, not an inference initializer.
    offsets = np.array(list(itertools.product((-.04, .04), repeat=8)))
    perturbation = np.zeros((4, 16))
    perturbation[1:, np.array([0, 1, 4, 5, 8, 9, 12, 13])] = offsets[[10, 70, 180]]
    perturbation[1:, np.array([2, 3, 6, 7, 10, 11, 14, 15])] = np.array([
        [-.04, .025, .03, -.02, -.02, .03, .04, -.02],
        [.025, -.02, -.035, .025, .04, -.025, -.03, .02],
        [-.02, -.03, .04, .02, .025, .03, -.04, -.02]])
    moving_stages = stages+perturbation
    if not all(model.admissible(p) for p in moving_stages):
        raise ValueError("Moving-geometry contract test leaves the field-free domain")
    probes = np.array([[0., 0., 0., .2, -.1, .15, -.2],
        [0., 0., 0., -.15, .2, -.25, .3], np.zeros(7)])
    adapter = np.asarray(jax.jit(jax.vmap(lambda p: jax.vmap(lambda u: model.matrix(p, u))(probes)))(moving_stages))
    native = np.asarray(jax.jit(jax.vmap(lambda p: jax.vmap(lambda u:
        model.column.matrix(model.native_parameters(p), u, segment="sample_to_detector", coordinates="canonical"))(probes)))(moving_stages))
    closure = np.linalg.norm(adapter-native, axis=(-2, -1))/np.linalg.norm(native, axis=(-2, -1))
    if np.max(closure) > 1e-10:
        raise ValueError("Moving four-projector geometry fails native canonical-map closure")
    native_mechanical = np.asarray(jax.jit(jax.vmap(lambda u: model.column.matrix(context.fixed,
        u, segment="sample_to_detector", coordinates="mechanical")))(probes))
    response_closure = float(np.max(abs(np.asarray(model.predict(stages, probes))-
        native_mechanical[:, XY][:, :, GROUPED]*STEPS/PIXEL_M)))
    grid = np.array(list(itertools.product((-control_half_width, 0., control_half_width), repeat=4)))
    controls = np.zeros((len(grid), 7)); controls[:, model.channels] = grid
    source = pure_source()
    matrices = np.asarray(jax.jit(jax.vmap(lambda u: model.matrix(stages, u)))(controls))
    covariance = matrices[:, XY]@source.total_covariance@matrices[:, XY].swapaxes(-1, -2)
    canonical_inputs = np.asarray(signed_inputs(STEPS))@np.asarray(model.shear).T
    means = np.einsum("nai,fi->nfa", matrices[:, XY][:, :, GROUPED], canonical_inputs)
    widths = np.sqrt(np.linalg.eigvalsh(covariance))/PIXEL_M
    capture = np.asarray(capture_lower_bound(means, covariance))
    admitted = (widths.min(axis=1) >= .35)&(capture.min(axis=1) >= .999)
    candidate_indices = np.flatnonzero(admitted)
    response_cov = approximate_response_covariance(covariance, PIXEL_M, 2e6)
    jacobian = np.asarray(jax.jit(jax.jacfwd(model.predict))(stages, controls)).reshape(len(controls), 8, 16)
    white = np.linalg.solve(np.linalg.cholesky(response_cov), jacobian)
    scales = np.tile([.1, .1, .2, .2], 4)
    factors = white*scales
    selected = []
    information = np.eye(16)*1e-6
    for _ in range(training_count):
        inverse = np.linalg.inv(information)
        gains = [np.linalg.slogdet(np.eye(8)+factors[i]@inverse@factors[i].T)[1]
                 if i not in selected else -np.inf for i in candidate_indices]
        chosen = int(candidate_indices[np.argmax(gains)])
        selected.append(chosen); information += factors[chosen].T@factors[chosen]
    # Geometric interior controls reserve new states before images/noise.
    held_controls = .5*(controls[selected[:heldout_count]]+controls[selected[heldout_count:2*heldout_count]])
    if np.unique(held_controls, axis=0).shape[0] != heldout_count or np.min(np.linalg.norm(
        held_controls[:, None]-controls[selected][None], axis=-1)) < 1e-12:
        raise ValueError("Reserved interpolation states must differ from all training states")
    all_controls = np.vstack((controls[selected], held_controls))
    for path in [Path(__file__), *[ROOT/p for p in aperture["report_input_hashes"]]]:
        sources[str(path.relative_to(ROOT))] = digest(path)
    verify_provenance_snapshot(sources)
    report = dict(schema="neoarm-shared-physical-projector-design-v1", source_hashes=sources,
        computational_input_hashes=computational, backend=jax.default_backend(), x64_enabled=True,
        parameter_names=PARAMETER_NAMES, objective_and_detector_anchor_fixed_conditionally=True,
        native_control_order=list(model.column.channel_names), native_stage_order=list(LENS_NAMES),
        control_half_width=control_half_width, affine_bridge_used=False, previous_projector_posterior_used=False,
        native_moving_geometry_relative_matrix_errors=closure.tolist(),
        native_mechanical_response_closure_pixels=response_closure,
        moving_contract_stage_coordinates=moving_stages.tolist(), moving_contract_controls=probes.tolist(),
        declared_design_stage_coordinates=stages.tolist(), generating_column_parameters=context.fixed.tolist(),
        candidate_count=len(controls), admitted_candidate_count=len(candidate_indices),
        training_controls=controls[selected].tolist(), heldout_controls=held_controls.tolist(),
        controls=all_controls.tolist(), training_mask=([True]*training_count+[False]*heldout_count),
        heldout_mask=([False]*training_count+[True]*heldout_count),
        selected_minimum_optical_rms_pixels=float(widths[selected].min()),
        selected_maximum_optical_rms_pixels=float(widths[selected].max()),
        selected_minimum_capture_lower_bound=float(capture[selected].min()),
        approximate_training_information=local_information(jacobian[selected], response_cov[selected]),
        source_covariance_canonical=source.total_covariance.tolist(), input_steps_grouped=STEPS.tolist(),
        detector_pixel_m=PIXEL_M, planned_incident_electrons_per_frame=2e6,
        planned_frames_per_state=9, planned_frame_count=len(all_controls)*9,
        frozen_before_new_images_and_noise=True, raw_images_generated=False,
        old_or_new_heldout_observations_used=False, inference_initializer_uses_generating_parameters=False,
        limitations=["Fixed objective postfield/sample shear and detector z are conditional, not identified by this16-coordinate model.",
            "Design uses the declared approximate instrument baseline; it is not a blind recovery or a hardware control recommendation.",
            "Predicted information uses full-camera centroid noise; actual finite-pixel image covariance must replace it.",
            "Pure canonical sample-plane Gaussian and exact mechanical input/pixel calibration are conditional.",
            "Native-map closure includes moving geometry, but fresh image likelihood and recovery are not yet demonstrated."],
        elapsed_seconds=perf_counter()-started)
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(admitted=len(candidate_indices), native_closure=float(closure.max()),
        response_closure=response_closure, information=report["approximate_training_information"],
        elapsed_seconds=report["elapsed_seconds"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); jax.config.update("jax_enable_x64", True)
    design(args.output)
