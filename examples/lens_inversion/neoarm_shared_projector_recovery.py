"""Training-only physical projector fit in the condenser's shared-field column.

The first-order objective postfield, specimen magnetic shear, detector axial
anchor and input/pixel calibration are conditional. No old projector posterior
or affine control bridge is loaded. Geometry seeds use the original diagram;
excitation seeds are generic first-branch guesses, not generating values.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import tomllib

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_aperture_capture import ROOT
from neoarm_condenser_endpoint_calibration import digest
from neoarm_condenser_extension_design import load_context, covariance_factor, FOLDER
from neoarm_condenser_scaled_cl1_profile import scaled_profiled_least_squares
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_shared_objective_audit import verify_provenance_snapshot
from neoarm_shared_projector_calibration import SharedProjectorModel, LENS_NAMES, PARAMETER_NAMES
from temgym_core.jax_least_squares import least_squares
from temgym_core.neoarm_calibration import hard_edge_column_from_config

GEOMETRY = np.array([0, 1, 4, 5, 8, 9, 12, 13])
EXCITATION = np.array([2, 3, 6, 7, 10, 11, 14, 15])
ORDER = np.r_[GEOMETRY, EXCITATION]
INVERSE = np.argsort(ORDER)


def selected_data(archive, training=True):
    """Only selected noisy responses/covariances enter the optical likelihood."""
    train, check = np.asarray(archive["training_mask"]), np.asarray(archive["heldout_mask"])
    if train.dtype != bool or check.dtype != bool or np.any(train & check) or not np.all(train | check):
        raise ValueError("Frozen disjoint exhaustive Boolean roles are required")
    role = train if training else check
    controls = np.asarray(archive["native_controls"])[role]
    response, covariance = np.asarray(archive["response"])[role], np.asarray(archive["response_covariance"])[role]
    if controls.shape != (len(controls), 7) or response.shape != (len(controls), 2, 4) or covariance.shape != (len(controls), 8, 8):
        raise ValueError("Native seven-channel controls and complete eight-coordinate covariance required")
    if not all(np.isfinite(x).all() for x in (controls, response, covariance)):
        raise ValueError("Finite image-derived data required")
    np.linalg.cholesky(covariance)
    return controls, response, covariance


def diagram_geometry_seed(column, sign):
    """No generating optics, previous fit or measured response is an argument."""
    if sign not in (-1, 1):
        raise ValueError("Two prescribed, explicitly displaced diagram starts")
    values, next_entrance = [], column.detector_z_m
    by_name = {lens.name: lens for lens in column.lenses}
    for name in LENS_NAMES:
        lens = by_name[name]
        gap = next_entrance-lens.z_center_m-lens.length_m/2
        if gap <= 0:
            raise ValueError("The diagram must have ordered nonoverlapping projector bodies")
        values.extend((np.log(lens.length_m), np.log(gap), -1.5 if name == "PL1" else 1.5,
            -.5 if name == "PL1" else .5))
        next_entrance = lens.z_center_m-lens.length_m/2
    p = np.asarray(values)
    p[GEOMETRY] += sign*np.array([.05, -.05, -.04, .04, .06, -.06, -.03, .03])
    return p


def first_branch_admissible(model, p, controls):
    if not model.admissible(p):
        return False
    stages = np.asarray(p).reshape(4, 4)
    length_gap = np.exp(stages[:, :2])
    if np.any(length_gap[:, 0] < .001) or np.any(length_gap[:, 0] > .08) or np.any(length_gap[:, 1] > .65):
        return False
    signs = np.array([-1., 1., 1., 1.])
    q = stages[:, 2]+stages[:, 3]*(np.asarray(controls)[:, model.channels]-model.references)
    return bool(np.all(q*signs > .05) and np.all(q*signs < np.pi-.05)
        and np.all(stages[:, 3]*signs > .01) and np.all(abs(stages[:, 3]) < 3.))


def make_residual(model, data):
    controls, observed, covariance = data
    lower = jnp.asarray(np.linalg.cholesky(covariance))
    controls, observed = jnp.asarray(controls), jnp.asarray(observed)
    return lambda p: jnp.linalg.solve(lower,
        (model.predict(p, controls)-observed).reshape(-1, 8, 1)).reshape(-1)


def fit_one(model, data, initial, *, max_outer_iterations=20):
    residual = make_residual(model, data)
    admissible = lambda p: first_branch_admissible(model, p, data[0])
    if not admissible(initial):
        raise ValueError("The prescribed diagram seed leaves the declared geometry/branch domain")
    # Staging is initialization only; all16 coordinates are subsequently free.
    qloss = lambda q: residual(jnp.asarray(initial).at[jnp.asarray(EXCITATION)].set(q))
    lower = np.tile([.05, .01], 4); upper = np.tile([np.pi-.05, 3.], 4)
    lower[:2], upper[:2] = -upper[:2].copy(), -lower[:2].copy()
    initializer = least_squares(qloss, initial[EXCITATION], bounds=(lower, upper), x_scale=.2,
        max_nfev=220, max_iter=300, ftol=1e-12, xtol=0., gtol=1e-7, initial_damping=1e-6)
    initialized = initial.copy(); initialized[EXCITATION] = np.asarray(initializer.x)
    record = dict(initial_parameters=initial.tolist(), excitation_initialization_parameters=initialized.tolist(),
        excitation_initialization_chi_square=float(2*initializer.cost),
        excitation_initialization_success=bool(initializer.success), excitation_initialization_nfev=initializer.nfev)
    if not admissible(initialized):
        return dict(**record, accepted=False, termination="excitation initialization leaves physical branch", parameters=initialized.tolist())
    decode = lambda p: jnp.asarray(p)[jnp.asarray(INVERSE)]
    fit = scaled_profiled_least_squares(lambda q: residual(decode(q)), initialized[ORDER],
        lambda q: admissible(np.asarray(decode(q))), scale=1e-3, n_outer=8,
        max_iterations=max_outer_iterations, max_inner_iterations=30, gradient_tolerance=1e-6)
    p = np.asarray(decode(fit["parameters"]))
    r = np.asarray(jax.jit(residual)(p)); jac = np.asarray(jax.jit(jax.jacfwd(residual))(p))
    norms = np.linalg.norm(jac, axis=0)
    singular = np.linalg.svd(jac/norms, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    factor = covariance_factor(jac) if rank == 16 else None
    gradient = float(np.max(abs((jac/norms).T@r)))
    full_gradient = None if factor is None else float(np.max(abs((jac@factor).T@r)))
    hessian_eigenvalues = None
    if factor is not None:
        hessian = np.asarray(jax.jit(jax.hessian(lambda q: .5*jnp.sum(residual(q)**2)))(p))
        whitened = factor.T@hessian@factor
        hessian_eigenvalues = np.linalg.eigvalsh(.5*(whitened+whitened.T))
    accepted = bool(fit["success"] and admissible(p) and rank == 16 and gradient < 1e-6
        and full_gradient < 1e-6 and hessian_eigenvalues.min() > 0)
    return dict(**record, parameters=p.tolist(), accepted=accepted, optimizer_success=fit["success"],
        termination=fit["reason"], training_chi_square=float(r@r), original_normalized_gradient=gradient,
        original_information_whitened_gradient=full_gradient, numerical_rank=rank,
        normalized_singular_values=singular.tolist(), parameter_covariance=None if factor is None else (factor@factor.T).tolist(),
        information_whitened_hessian_eigenvalues=None if factor is None else hessian_eigenvalues.tolist(),
        history=fit["history"], initial_nuisance_history=fit["initial_nuisance_history"],
        all_sixteen_coordinates_free=True, objective_to_il1_entrance_gap_m=float(model.objective_to_il1_gap(p)),
        unit_conversion=fit["unit_conversion"])


def run(output, *, max_outer_iterations=20):
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        raise RuntimeError("All optical prediction and differentiation requires GPU float64")
    if not 1 <= max_outer_iterations <= 20:
        raise ValueError("This pilot permits at most20 outer iterations per prescribed start")
    started = perf_counter(); path = FOLDER/"shared_projector_images.json"
    acquisition = json.loads(path.read_text()); verify_provenance_snapshot(acquisition["source_hashes"])
    archive_path = ROOT/acquisition["arrays_path"]
    if not acquisition["all_images_accepted"] or digest(archive_path) != acquisition["arrays_sha256"]:
        raise ValueError("Unchanged admitted physical Gaussian images are required")
    with np.load(archive_path, allow_pickle=False) as archive:
        training = selected_data(archive, True)
    _, _, _, _, context = load_context()
    model = SharedProjectorModel(context.column, context.fixed)
    config_path = ROOT/"examples/lens_inversion/neoarm_inversion_config.toml"
    diagram_path = ROOT/"examples/microscope_models/measure_microscope_geometry.ipynb"
    diagram = column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(config_path.read_text())), diagram_path)
    sources = dict(acquisition["source_hashes"])
    for source in (Path(__file__), ROOT/"tests/test_neoarm_shared_projector_recovery.py", path, archive_path,
                   config_path, diagram_path, ROOT/"src/temgym_core/profiled_least_squares.py",
                   ROOT/"src/temgym_core/jax_least_squares.py", ROOT/"examples/lens_inversion/neoarm_condenser_scaled_cl1_profile.py"):
        sources[str(source.relative_to(ROOT))] = digest(source)
    rows = []
    for sign in (-1, 1):
        row = fit_one(model, training, diagram_geometry_seed(diagram, sign), max_outer_iterations=max_outer_iterations)
        row["prescribed_diagram_displacement_sign"] = sign; rows.append(row)
        print("shared projector fit", sign, row["accepted"], row.get("training_chi_square"), row["termination"], flush=True)
    candidates = [i for i, row in enumerate(rows) if row["accepted"]]
    best_index = min(candidates, key=lambda i: rows[i]["training_chi_square"]) if candidates else None
    # Selection is complete before any held-out response or truth file is read.
    diagnostics = None
    if best_index is not None:
        best = rows[best_index]; p = np.asarray(best["parameters"]); pcov = np.asarray(best["parameter_covariance"])
        with np.load(archive_path, allow_pickle=False) as archive:
            heldout = selected_data(archive, False)
        residual = make_residual(model, heldout)
        held_r = np.asarray(jax.jit(residual)(p)); held_j = np.asarray(jax.jit(jax.jacfwd(residual))(p))
        predictive = np.eye(len(held_r))+held_j@pcov@held_j.T
        truth_path = ROOT/acquisition["truth_path"]
        if digest(truth_path) != acquisition["truth_sha256"]:
            raise ValueError("Post-selection generating diagnostic archive changed")
        with np.load(truth_path, allow_pickle=False) as truth:
            generating = np.asarray(truth["stage_coordinates"])
            noiseless = np.asarray(truth["expected_response"])
        true_r = np.asarray(jax.jit(make_residual(model, training))(generating))
        all_controls = np.vstack((training[0], heldout[0]))
        closure = np.asarray(jax.jit(model.predict)(generating, all_controls))-noiseless
        se = np.sqrt(np.diag(pcov))
        diagnostics = dict(heldout_coordinate_count=len(held_r), heldout_raw_chi_square=float(held_r@held_r),
            heldout_linear_predictive_chi_square=float(held_r@np.linalg.solve(predictive, held_r)),
            maximum_heldout_response_error_pixels=float(np.max(abs(np.asarray(model.predict(p, heldout[0]))-heldout[1]))),
            generating_training_chi_square=float(true_r@true_r), native_noiseless_truth_closure_pixels=float(np.max(abs(closure))),
            generating_parameters=generating.tolist(), length_gap_errors_percent=(100*np.expm1((p-generating).reshape(4, 4)[:, :2])).tolist(),
            length_gap_standard_errors_percent=(100*se.reshape(4, 4)[:, :2]).tolist(),
            parameter_error_local_standard_deviations=((p-generating)/se).tolist(),
            truth_and_heldout_loaded_after_training_selection=True)
        sources[str(truth_path.relative_to(ROOT))] = digest(truth_path)
    verify_provenance_snapshot(sources)
    report = dict(schema="neoarm-shared-physical-projector-recovery-v1", source_hashes=sources,
        acquisition_path=str(path.relative_to(ROOT)), acquisition_sha256=digest(path),
        arrays_path=str(archive_path.relative_to(ROOT)), arrays_sha256=digest(archive_path),
        backend=jax.default_backend(), x64_enabled=True, parameter_names=PARAMETER_NAMES,
        training_response_coordinate_count=training[1].size, free_parameter_count=16, prior_coordinate_count=0,
        prescribed_start_count=2, accepted_start_count=len(candidates), rows=rows, best_index=best_index,
        accepted_stationary_fit=best_index is not None, post_selection_diagnostics=diagnostics,
        initialization_uses_generating_parameters=False, old_projector_posterior_used=False, affine_bridge_used=False,
        heldout_data_used_for_initialization_or_selection=False, fixed_objective_and_detector_anchor_conditional=True,
        broad_branch_uniqueness_proved=False, full_column_recovery=False, elapsed_seconds=perf_counter()-started,
        limitations=["Fixed shared objective postfield, specimen shear and detector z are not inferred here or calibrated on hardware.",
            "Only the signed first excitation branch is searched; alternative branches and objective uncertainty require separate profiling.",
            "The original diagram geometry also generated this synthetic fixture; both actual starts are explicitly displaced and generic q seeds do not use the generator.",
            "Calibrated mechanical input/pixel scales and detector PSF are conditional; no extra prior or imported old projector posterior is used.",
            "These are sixteen individual projector coordinates, not yet jointly fitted with condenser optics or a free objective field."])
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-outer-iterations", type=int, default=20)
    args = parser.parse_args(); jax.config.update("jax_enable_x64", True)
    run(args.output, max_outer_iterations=args.max_outer_iterations)
