"""Joint seven-lens recovery in one fixed shared-objective column.

This combines original image-derived data, not fitted posteriors: 102
condenser/aperture coordinates, 16 endpoint calibration coordinates and the
new four-projector responses. The unchanged upstream astigmatic preparation,
objective field, aperture geometry and external scales remain conditional.
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

from neoarm_aperture_capture import ROOT, XY
from neoarm_condenser_cold_joint_endpoint import select_cold_seed
from neoarm_condenser_covariance_fit import VisibleSourceModel, geometry_gaps
from neoarm_condenser_endpoint_calibration import GROUPED, digest
from neoarm_condenser_endpoint_recovery import (
    JointEndpointModel, joint_training_residual, calibration_training, prediction_check, verify,
)
from neoarm_condenser_extension_design import load_context, covariance_factor, FOLDER
from neoarm_condenser_extension_recovery import selected_data as condenser_data
from neoarm_condenser_geometry_profile_diagnostic import GEOMETRY_INDICES
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_condenser_scaled_cl1_profile import scaled_profiled_least_squares
from neoarm_condenser_cl1_profile import source_diagnostic
from neoarm_shared_projector_calibration import SharedProjectorModel
from neoarm_shared_projector_recovery import (
    GEOMETRY, selected_data as projector_data, make_residual, first_branch_admissible,
)


class PhysicalEndpointRows:
    """Measured analyser rows derived from the same four physical projectors."""
    def __init__(self, projectors, analyser_controls):
        self.projectors = projectors
        self.controls = jnp.asarray(analyser_controls)
        self.pixel_m = projectors.pixel_m
        if self.controls.shape != (2, 7):
            raise ValueError("The condenser experiment requires two seven-control analysers.")

    def canonical_rows(self, parameters):
        matrices = jax.vmap(lambda u: self.projectors.matrix(parameters, u))(self.controls)
        return matrices[:, XY][:, :, GROUPED]

    def response(self, parameters):
        return self.projectors.predict(parameters, self.controls)


class SharedColumnModel(JointEndpointModel):
    def __init__(self, quotient, endpoints, prefield, projector_controls):
        super().__init__(quotient, endpoints, prefield)
        self.projector_controls = np.asarray(projector_controls)

    def admissible(self, parameters):
        p = np.asarray(parameters)
        return bool(p.shape == (39,) and np.isfinite(p).all()
            and self.quotient.admissible(p[:23])
            and first_branch_admissible(self.endpoints.projectors, p[23:], self.projector_controls))


def shared_residual(joint, old, aperture, fresh, calibration, projectors):
    original = joint_training_residual(joint, old, aperture, fresh, calibration)
    downstream = make_residual(joint.endpoints.projectors, projectors)
    # Exactly one radius prior remains at the end for transparent accounting.
    def residual(p):
        base = original(p)
        return jnp.r_[base[:-1], downstream(p[23:]), base[-1]]
    return residual


def source_coordinates(chart):
    outer = np.r_[GEOMETRY_INDICES, 23+GEOMETRY]
    order = np.r_[outer, np.setdiff1d(np.arange(39), outer)]
    inverse = np.argsort(order)
    def encode(p):
        return jnp.r_[chart.encode(p[:23]), p[23:]][order]
    def decode(q):
        p = q[inverse]
        return jnp.r_[chart.decode(p[:23]), p[23:]]
    return encode, decode, len(outer)


def run(output):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("The shared-column fit requires GPU float64.")
    start = perf_counter()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reports, sources = {}, {}
    for name in ("condenser_moment_initialization", "condenser_endpoint_calibration",
                 "condenser_extension_images", "shared_projector_images", "shared_projector_recovery"):
        path = FOLDER/(name+".json")
        report = json.loads(path.read_text()); verify(report)
        reports[name] = report; sources.update(report["source_hashes"])
        sources[str(path.relative_to(ROOT))] = digest(path)
    def archive(name, selector):
        report = reports[name]; path = ROOT/report["arrays_path"]
        if digest(path) != report["arrays_sha256"]:
            raise ValueError("Image-derived inference archive changed.")
        sources[str(path.relative_to(ROOT))] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            return selector(data)
    calibration = archive("condenser_endpoint_calibration", calibration_training)
    fresh = archive("condenser_extension_images", lambda a: condenser_data(a, True))
    projector_training = archive("shared_projector_images", lambda a: projector_data(a, True))
    cond_initial, cond_selection = select_cold_seed(reports["condenser_moment_initialization"])
    fit_report = reports["shared_projector_recovery"]
    if not fit_report["accepted_stationary_fit"]:
        raise ValueError("An accepted training-only projector initialization is required.")
    projector_initial = np.asarray(fit_report["rows"][fit_report["best_index"]]["parameters"])
    _, old, _, aperture, conditional = load_context()
    quotient = VisibleSourceModel(conditional)
    projectors = SharedProjectorModel(conditional.column, conditional.fixed)
    endpoint = PhysicalEndpointRows(projectors, reports["condenser_endpoint_calibration"]["analyser_controls"])
    split = conditional.column.objective.split_maps(jnp.asarray(conditional.fixed[-4:]))
    joint = SharedColumnModel(quotient, endpoint, split.pre,
        np.vstack((projector_training[0], np.asarray(endpoint.controls))))
    benchmark = reports["condenser_moment_initialization"]
    chart = ReferenceSourceChart(quotient, benchmark["reference_control"], benchmark["reference_width_m"])
    encode, decode, outer_count = source_coordinates(chart)
    initial = np.r_[cond_initial, projector_initial]
    if not joint.admissible(initial):
        raise ValueError("Separate training fits do not compose into the declared physical domain.")
    residual = shared_residual(joint, old, aperture, fresh, calibration, projector_training)
    # Only a forward-contract check uses the fixed generating column here.
    fixed_stages = projectors.stage_coordinates(conditional.fixed)
    closure_parameters = np.r_[cond_initial, fixed_stages]
    expected, _ = quotient.observables(cond_initial, fresh[0], jnp.zeros((0, 7)))
    closure = float(np.max(abs(np.asarray(joint.beam_observables(closure_parameters, fresh[0]))-expected)))
    if closure > 1e-8:
        raise RuntimeError("Native shared-column and condenser maps fail closure.")
    for path in (Path(__file__), ROOT/"tests/test_neoarm_shared_column_recovery.py"):
        sources[str(path.relative_to(ROOT))] = digest(path)
    print("shared39 fitting", len(np.asarray(residual(initial))), "residuals", flush=True)
    fit = scaled_profiled_least_squares(lambda q: residual(decode(q)), np.asarray(encode(initial)),
        lambda q: joint.admissible(np.asarray(decode(q))), scale=1e-3, n_outer=outer_count,
        max_iterations=20, max_inner_iterations=30, gradient_tolerance=1e-6)
    p = np.asarray(decode(fit["parameters"]))
    r = np.asarray(jax.jit(residual)(p)); jac = np.asarray(jax.jit(jax.jacfwd(residual))(p))
    norms = np.linalg.norm(jac, axis=0)
    singular = np.linalg.svd(jac/norms, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    factor = covariance_factor(jac) if rank == 39 else None
    gradient = float(np.max(abs((jac/norms).T@r)))
    full_gradient = None if factor is None else float(np.max(abs((jac@factor).T@r)))
    accepted = bool(fit["success"] and rank == 39 and gradient < 1e-6
        and full_gradient < 1e-6 and joint.admissible(p))
    expected_count = 119+projector_training[1].size
    if r.shape != (expected_count,):
        raise RuntimeError("Original observations and the single radius prior must occur exactly once.")
    report = dict(schema="neoarm-shared-seven-lens-recovery-v1", source_hashes=sources,
        backend=jax.default_backend(), all_28_physical_lens_coordinates_free=True,
        free_parameter_count=39, outer_geometry_count=outer_count, training_data_coordinate_count=len(r)-1,
        condenser_data_coordinate_count=102, endpoint_data_coordinate_count=16,
        projector_data_coordinate_count=projector_training[1].size, aperture_radius_prior_count=1,
        initial_parameters=initial.tolist(), parameters=p.tolist(),
        training_data_chi_square=float(r[:-1]@r[:-1]), radius_prior_chi_square=float(r[-1]**2),
        total_chi_square=float(r@r), normalized_gradient=gradient,
        information_whitened_gradient=full_gradient, numerical_rank=rank,
        accepted_stationary_fit=accepted, optimizer_success=fit["success"], termination=fit["reason"],
        history=fit["history"], initial_nuisance_history=fit.get("initial_nuisance_history", []),
        original_data_enter_once=True, initialization_posteriors_used_as_priors=False,
        condenser_seed=cond_selection, projector_seed_selected_index=fit_report["best_index"],
        canonical_condenser_forward_closure=closure,
        source=source_diagnostic(p[:23]), condenser_geometry_gaps_m=geometry_gaps(conditional, quotient.decode(p[:23])[0]).tolist(),
        objective_to_il1_gap_m=float(projectors.objective_to_il1_gap(p[23:])),
        heldout_data_used_in_fit=False, full_column_hardware_acceptance=False,
        limitations=["First-order conditional seven-lens fit, not nonlinear-field or hardware validity.",
            "The unknown internal-stigmator model is not used; stable anisotropy upstream of CL1 is assumed.",
            "Objective field, source-reference gauge, aperture geometry, sample input scales and detector are conditional.",
            "Only the declared first projector excitation branch and one composition of separate training seeds are tested.",
            "Material images are independently validated in this column but do not enter this optical response likelihood.",
            "Held-out measurements were inspected in the separate fits; joint checks are reused diagnostics."])
    if accepted:
        covariance = factor@factor.T
        report["parameter_covariance"] = covariance.tolist()
        report["projector_length_gap_standard_errors_percent"] = (100*np.sqrt(np.diag(covariance)[23:]).reshape(4, 4)[:, :2]).tolist()
        optical_factor = np.asarray(jax.jacfwd(lambda x: quotient.decode(x[:23])[0])(p))@factor
        report["condenser_length_standard_errors_percent"] = (100*np.linalg.norm(optical_factor[[1, 5, 9]], axis=1)).tolist()
        held = archive("shared_projector_images", lambda a: projector_data(a, False))
        check = make_residual(projectors, held)
        report["reused_projector_checks"] = prediction_check(lambda x: check(x[23:]), p, covariance)
    report["elapsed_seconds"] = perf_counter()-start
    verify(report)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("shared39", accepted, report["total_chi_square"], gradient, full_gradient, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); jax.config.update("jax_enable_x64", True)
    run(args.output)
