"""Conditional condenser fit with the observable source-covariance quotient.

Reuses the frozen noisy-image experiment; family 14 is now withheld in both
modalities. A physical covariance completion is profiled, not observed. The
objective/downstream calibration and aperture plane remain conditional inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.constants import energy2wavelength
from temgym_core.gaussian_image_likelihood import validate_phase_space_covariance
from neoarm_aperture_capture import ROOT, XY, load_cases
from neoarm_condenser_image_recovery import (
    CondenserModel, covariance_observable, linear_source_initializer, rotation,
    finite_json, read_verified_aperture, DEFAULT_APERTURE, OMEGA,
)
from neoarm_shared_objective_audit import provenance_snapshot, verify_provenance_snapshot

DEFAULT_INPUT = ROOT/"examples/generated/neoarm_inversion/imaging_recipe/condenser_image_recovery.json"
EPSILON = float(energy2wavelength(200e3))/(4*np.pi)
SOURCE_SCALE = np.array([10e-9, EPSILON/10e-9]*2)
INVISIBLE = np.zeros((4, 4))
INVISIBLE[0, 3] = INVISIBLE[3, 0] = 1.
INVISIBLE[1, 2] = INVISIBLE[2, 1] = -1.


def visible_basis():
    result = []
    for i in range(4):
        b = np.zeros((4, 4)); b[i, i] = 1.; result.append(b)
    for i, j in ((0, 1), (2, 3), (0, 2), (1, 3)):
        b = np.zeros((4, 4)); b[i, j] = b[j, i] = 1/np.sqrt(2); result.append(b)
    b = np.zeros((4, 4))
    b[0, 3] = b[3, 0] = b[1, 2] = b[2, 1] = .5
    return np.asarray(result+[b])


BASIS = visible_basis()


def visible_coordinates(covariance):
    return np.einsum("kij,ij->k", BASIS, np.asarray(covariance)/np.outer(SOURCE_SCALE, SOURCE_SCALE))


def visible_covariance(coordinates):
    return jnp.einsum("k,kij->ij", coordinates, jnp.asarray(BASIS))*jnp.outer(SOURCE_SCALE, SOURCE_SCALE)


def quantum_completion(coordinates, *, interval=False):
    """Profile the unobserved cross covariance on the complete quantum cone.

    The smallest Hermitian eigenvalue is concave in the one invisible scalar.
    Its maximum-margin representative is solely a numerical gauge choice.
    """
    c = np.einsum("k,kij->ij", coordinates, BASIS)
    if not np.all(np.isfinite(c)) or np.any(np.diag(c) <= 0):
        return dict(feasible=False, margin=-np.inf)
    s = c[0, 3]
    a, b = np.sqrt(c[0, 0]*c[3, 3]), np.sqrt(c[1, 1]*c[2, 2])
    lo, hi = max(-s-a, s-b), min(-s+a, s+b)
    if hi < lo:
        return dict(feasible=False, margin=-np.inf)
    def margin(g):
        return float(np.linalg.eigvalsh(c+g*INVISIBLE+1j*OMEGA)[0])
    left, right = lo, hi
    ratio = (np.sqrt(5)-1)/2
    u, v = right-ratio*(right-left), left+ratio*(right-left)
    fu, fv = margin(u), margin(v)
    for _ in range(55):
        if fu < fv:
            left, u, fu = u, v, fv
            v = left+ratio*(right-left); fv = margin(v)
        else:
            right, v, fv = v, u, fu
            u = right-ratio*(right-left); fu = margin(u)
    g = (left+right)/2; value = margin(g)
    result = dict(feasible=value >= -1e-10, margin=value, representative=g,
        representative_covariance=(c+g*INVISIBLE)*np.outer(SOURCE_SCALE, SOURCE_SCALE))
    if interval and result["feasible"]:
        edges = []
        for outer in (lo, hi):
            inner = g
            for _ in range(65):
                middle = (inner+outer)/2
                if margin(middle) >= 0: inner = middle
                else: outer = middle
            edges.append(inner)
        result["feasible_interval"] = edges
    return result


def geometry_gaps(model, optical):
    lenses = np.asarray(optical).reshape(3, 4)
    z, length = lenses[:, 0], np.exp(lenses[:, 1])
    entrance, leave = z-length/2, z+length/2
    return np.array([entrance[0]-model.source_z_m, entrance[1]-leave[0],
        model.aperture_z_m-leave[1], entrance[2]-model.aperture_z_m,
        model.objective_entrance_m-leave[2]])


class VisibleSourceModel:
    def __init__(self, model):
        self.model = model

    def decode(self, p):
        physical = (jnp.asarray(self.model.base)+jnp.asarray(self.model.scales)*p[:12]).reshape(3, 4)
        # Strength divided by sqrt(length) is the leading focusing variable.
        # This is an invertible chart of the same hard-edge physical model,
        # avoiding the artificial curved q~sqrt(L) valley of direct coordinates.
        physical = physical.at[:, 2:4].multiply(jnp.exp(.05*p[:12].reshape(3, 4)[:, 1, None]))
        return (physical.reshape(-1),
                visible_covariance(p[12:21]), p[21], p[22])

    def optical_coordinates(self, optical):
        value = np.asarray(optical).reshape(3, 4).copy()
        base = self.model.base.reshape(3, 4)
        value[:, 2:4] *= np.exp(-(value[:, 1]-base[:, 1])/2)[:, None]
        return ((value.reshape(-1)-self.model.base)/self.model.scales)

    def observables(self, p, beam_controls, aperture_controls):
        optical, covariance, angle, log_radius_scale = self.decode(p)
        def beam(control):
            maps = self.model.maps(optical, control)
            output = maps@covariance@jnp.swapaxes(maps, -1, -2)
            return jax.vmap(covariance_observable)(output[:, XY][:, :, XY]/(55e-6)**2)
        def aperture(control):
            m = self.model.condenser_matrix(optical, control, to_aperture=True)
            c = m@covariance@m.T
            r = rotation(angle)
            return covariance_observable(r@c[jnp.ix_(jnp.asarray(XY), jnp.asarray(XY))]@r.T)-jnp.array([log_radius_scale, log_radius_scale, 0.])
        return jax.vmap(beam)(jnp.asarray(beam_controls)), jax.vmap(aperture)(jnp.asarray(aperture_controls))

    def admissible(self, p):
        if not np.all(np.isfinite(p)) or np.any(np.abs(p[:12]) > np.tile([1., 4., 1., 3.], 3)):
            return False
        optical = np.asarray(self.decode(p)[0])
        return bool(np.min(geometry_gaps(self.model, optical)) > 1e-9 and quantum_completion(p[12:21])["feasible"])


def svd_gauss_newton(residual, initial, admissible, *, max_iterations=100, gradient_tolerance=1e-6):
    """Full-column SVD trust-region Gauss--Newton and feasible backtracking.

    No singular direction is dropped. A line-search or rank failure is a
    reported non-convergence, never an xtol success.
    """
    fun = jax.jit(residual); jac = jax.jit(jax.jacfwd(residual))
    curvature = jax.jit(lambda x, d: jax.jvp(
        lambda y: jax.jvp(residual, (y,), (d,))[1], (x,), (d,))[1])
    p = np.asarray(initial).copy()
    if not admissible(p): raise ValueError("Initializer outside physical geometry/source domain")
    r = np.asarray(fun(p)); history = []; evaluations = 1
    native = np.r_[np.tile([.02, .1, .05, .1], 3), [.1, 1., .1, 1., .1, .1, .1, 1., .1], .01, .005]
    if len(p) != len(native): native = np.ones_like(p)
    trust_radius = 1.
    success = False; reason = "iteration budget exhausted"
    for iteration in range(max_iterations):
        j = np.asarray(jac(p))
        norms = np.maximum(np.linalg.norm(j, axis=0), 1e-12)
        scaled = j/norms
        left, singular, right = np.linalg.svd(scaled, full_matrices=False)
        gradient = float(np.max(np.abs(scaled.T@r)))
        record = dict(iteration=iteration, chi_square=float(r@r), normalized_gradient=gradient,
                      singular_min=float(singular[-1]), singular_max=float(singular[0]))
        history.append(record)
        if gradient < gradient_tolerance:
            success = True; reason = "normalized gradient tolerance"; break
        if singular[-1] < singular[0]*1e-13:
            reason = "additional numerical rank deficiency; no direction truncated"; break
        # Solve the complete trust-region subproblem in declared native
        # coordinates. Tikhonov damping here is a step constraint, not a prior.
        u, s, v = np.linalg.svd(j*native, full_matrices=False)
        projected = u.T@r
        def trust_step(damping):
            return -(v.T@((s*projected)/(s*s+damping)))
        natural_step = trust_step(0.)
        damping = 0.
        if np.linalg.norm(natural_step) > trust_radius:
            low, high = 0., max(s[0]**2, 1.)
            for _ in range(80):
                mid = (low+high)/2
                if np.linalg.norm(trust_step(mid)) > trust_radius: low = mid
                else: high = mid
            damping = high; natural_step = trust_step(damping)
        step = natural_step*native
        second = np.asarray(curvature(p, step))
        acceleration = -(v.T@((s*(u.T@second))/(s*s+damping)))
        curvature_ratio = np.linalg.norm(acceleration)/max(np.linalg.norm(natural_step), 1e-300)
        if curvature_ratio < 1.:
            step += .5*acceleration*native
        record["geodesic_acceleration_ratio"] = float(curvature_ratio)
        directional = float(r@j@step)
        accepted = False
        for power in range(35):
            alpha = 2.**(-power); candidate = p+alpha*step
            if not admissible(candidate): continue
            trial = np.asarray(fun(candidate)); evaluations += 1
            if np.all(np.isfinite(trial)) and trial@trial <= r@r+2e-4*alpha*directional:
                predicted = -(2*alpha*directional+alpha**2*np.sum((j@step)**2))
                quality = float((r@r-trial@trial)/max(predicted, 1e-300))
                accepted = True; p = candidate; r = trial
                record.update(step_fraction=alpha, trust_radius=trust_radius, step_quality=quality, damping=damping)
                if alpha < .5 or quality < .25: trust_radius *= .25
                elif quality > .75 and np.linalg.norm(natural_step) > .8*trust_radius: trust_radius *= 2
                break
        if iteration % 10 == 0:
            print(json.dumps(record), flush=True)
        if not accepted:
            reason = "physical feasible line search failed"; break
    j = np.asarray(jac(p)); norms = np.maximum(np.linalg.norm(j, axis=0), 1e-12)
    gradient = float(np.max(np.abs((j/norms).T@r)))
    if gradient < gradient_tolerance: success = True; reason = "normalized gradient tolerance"
    return dict(parameters=p, residual=r, jacobian=j, success=success, reason=reason,
                normalized_gradient=gradient, evaluations=evaluations, history=history)


def verified_input(path):
    report = json.loads(path.read_text())
    for section in ("source_hashes", "computational_input_hashes"):
        for name, expected in report.get(section, {}).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Stale input {name}")
    array_path = Path(report["arrays_path"])
    if not array_path.is_absolute(): array_path = ROOT/array_path
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != report["arrays_sha256"]:
        raise ValueError("Frozen image array digest mismatch")
    return report, dict(np.load(array_path))


def information_diagnostics(training_jacobian, heldout_jacobian, physical_jacobian):
    """Local conditioning only, including nuisance and the common radius prior."""
    j, h = np.asarray(training_jacobian), np.asarray(heldout_jacobian)
    norms = np.maximum(np.linalg.norm(j, axis=0), 1e-12)
    _, normalized_singular, normalized_right = np.linalg.svd(j/norms, full_matrices=False)
    _, s, v = np.linalg.svd(j, full_matrices=False)
    covariance = (v.T/s**2)@v
    physical_covariance = physical_jacobian@covariance@physical_jacobian.T
    nuisance_left, nuisance_singular, _ = np.linalg.svd(j[:, 12:], full_matrices=False)
    nuisance_rank = int(np.sum(nuisance_singular > nuisance_singular[0]*1e-12))
    profiled = j[:, :12]-nuisance_left[:, :nuisance_rank]@(nuisance_left[:, :nuisance_rank].T@j[:, :12])
    profiled_singular = np.linalg.svd(profiled/np.maximum(np.linalg.norm(j[:, :12], axis=0), 1e-12), compute_uv=False)
    weak_direction = normalized_right[-1]/norms/normalized_singular[-1]
    prediction = h@covariance@h.T
    groups = dict(beam_family6=(0, 6), beam_family7=(6, 12), beam_family14=(12, 18), aperture_family14=(18, 21))
    return dict(full_column_normalized_singular_values=normalized_singular.tolist(),
        full_column_normalized_condition_number=float(normalized_singular[0]/normalized_singular[-1]),
        numerical_rank_at_relative_1e_10=int(np.sum(normalized_singular > normalized_singular[0]*1e-10)),
        nuisance_rank=nuisance_rank, optical_profiled_column_normalized_singular_values=profiled_singular.tolist(),
        physical_parameter_standard_errors=np.sqrt(np.maximum(np.diag(physical_covariance), 0.)).tolist(),
        weakest_full_normalized_direction=normalized_right[-1].tolist(),
        weakest_direction_one_linear_training_sigma_optical_change=(physical_jacobian@weak_direction).tolist(),
        weakest_direction_one_linear_training_sigma_heldout_norm=float(np.linalg.norm(h@weak_direction)),
        heldout_prediction_standard_errors_in_measurement_sigma=np.sqrt(np.maximum(np.diag(prediction), 0.)).tolist(),
        heldout_family_predicted_parameter_uncertainty_rms_sigma={name: float(np.sqrt(np.trace(prediction[a:b, a:b])/(b-a))) for name, (a, b) in groups.items()},
        interpretation="Local linear diagnostic, not a nonlinear profile, global uniqueness proof, or accepted recovery uncertainty. Large one-sigma directions may leave the valid geometry box.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--starts", type=int, default=1, choices=(1, 3))
    args = parser.parse_args()
    args.input = args.input.resolve(); args.output = args.output.resolve()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu": raise RuntimeError("JAX forward and fitting require GPU")
    started = perf_counter()
    old, arrays = verified_input(args.input)
    aperture = read_verified_aperture(DEFAULT_APERTURE)
    information_path = next(ROOT/n for n in aperture["report_input_hashes"] if "information_narrow_probe" in n)
    information = json.loads(information_path.read_text())
    _, source_truth, column, truth = load_cases(information_path)
    model = CondenserModel(column, truth, np.asarray(information["projection_states"]["controls"])[[0, 2]])
    quotient = VisibleSourceModel(model)
    sources, inputs = provenance_snapshot()
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    controls = arrays["controls"]; observed = arrays["beam_observables"]; covariance = arrays["beam_observable_covariances"]
    apcontrols = np.asarray([row["controls"] for row in aperture["records"]])
    apobserved = arrays["aperture_observables"]; apcovariance = arrays["aperture_observable_covariances"]
    def residual_factory(beam_slice, ap_slice, prior):
        bc, ac = controls[beam_slice], apcontrols[ap_slice]
        bo, ao = observed[beam_slice], apobserved[ap_slice]
        br = jnp.asarray(np.linalg.cholesky(covariance[beam_slice]))
        ar = jnp.asarray(np.linalg.cholesky(apcovariance[ap_slice]))
        def residual(p):
            bp, ap = quotient.observables(p, bc, ac)
            data = jnp.r_[jnp.linalg.solve(br, (bp-bo)[..., None]).reshape(-1),
                          jnp.linalg.solve(ar, (ap-ao)[..., None]).reshape(-1)]
            return jnp.r_[data, p[22]/.005] if prior else data
        return residual
    residual = residual_factory(slice(0, 8), slice(0, 2), True)
    heldout = jax.jit(residual_factory(slice(8, None), slice(2, None), False))
    records = []; saved = {}
    original_base = model.base.copy()
    for index in range(args.starts):
        if index:
            sign = 1 if index == 1 else -1
            seed = original_base.copy().reshape(3, 4)
            seed[:, 0] += sign*np.array([1e-3, -.3e-3, .15e-3])
            dl = sign*np.array([.08, -.06, .10])
            seed[:, 1] += dl; seed[:, 2:4] *= np.exp(dl[:, None]/2)
            model.base = seed.reshape(-1)
        initialized, init_meta = linear_source_initializer(model, controls[:8], observed[:8], covariance[:8],
            apcontrols[:2], apobserved[:2], apcovariance[:2])
        optical_seed, covariance_seed, angle_seed = model.decode(initialized)
        model.base = original_base.copy()
        initial = np.r_[quotient.optical_coordinates(optical_seed),
            visible_coordinates(covariance_seed), float(angle_seed), 0.]
        fit = svd_gauss_newton(residual, initial, quotient.admissible, max_iterations=args.max_iterations)
        p = fit["parameters"]; optical, _, angle, radius = quotient.decode(p)
        completion = quantum_completion(p[12:21], interval=True)
        h = np.asarray(heldout(p)); j = fit["jacobian"]
        _, singular, right = np.linalg.svd(j, full_matrices=False)
        parameter_covariance = (right.T/singular**2)@right
        hj = np.asarray(jax.jit(jax.jacfwd(heldout))(p))
        predictive = np.eye(len(h))+hj@parameter_covariance@hj.T
        predictive_chi = float(h@np.linalg.solve(predictive, h))
        physical_jac = np.asarray(jax.jacfwd(lambda q: quotient.decode(q)[0])(p))
        physical_covariance = physical_jac@parameter_covariance@physical_jac.T
        local_information = information_diagnostics(j, hj, physical_jac)
        record = dict(start=index, initialization=init_meta, initial_parameters=initial.tolist(),
            parameters=p.tolist(), success=fit["success"], reason=fit["reason"], history=fit["history"],
            normalized_gradient=fit["normalized_gradient"], evaluations=fit["evaluations"],
            training_data_chi_square=float(fit["residual"][:-1]@fit["residual"][:-1]),
            radius_prior_chi_square=float((p[22]/.005)**2), heldout_chi_square=float(h@h),
            heldout_predictive_chi_square=predictive_chi,
            optical_parameters=np.asarray(optical).tolist(), optical_errors=(np.asarray(optical)-truth[2:14]).reshape(3, 4).tolist(),
            physical_parameter_standard_errors=np.sqrt(np.diag(physical_covariance)).tolist(),
            quantum_margin=completion["margin"], invisible_covariance_feasible_interval=completion.get("feasible_interval"),
            invisible_covariance_representative=completion.get("representative"),
            geometry_gaps_m=geometry_gaps(model, optical).tolist(), common_aperture_rotation_rad=float(angle),
            common_radius_scale=float(np.exp(radius)),
            local_information=local_information,
            heldout_point_prediction_chi_square_z=float((h@h-len(h))/np.sqrt(2*len(h))),
            accepted=bool(fit["success"] and abs((h@h-len(h))/np.sqrt(2*len(h))) < 5 and completion["feasible"]))
        records.append(record)
        saved[f"start{index}_parameters"] = p
        saved[f"start{index}_joint_covariance"] = parameter_covariance
        saved[f"start{index}_training_jacobian"] = j
        saved[f"start{index}_heldout_jacobian"] = hj
        saved[f"start{index}_physical_source_covariance"] = completion["representative_covariance"]
        print(json.dumps({k: record[k] for k in ("start", "success", "reason", "training_data_chi_square", "heldout_chi_square", "heldout_predictive_chi_square", "normalized_gradient")}), flush=True)
    collection_path = next(ROOT/n for n in aperture["report_input_hashes"] if "aperture_collection.json" in n)
    collection = json.loads(collection_path.read_text())
    am = np.asarray(collection["aperture_collection"]["aperture_to_detector_matrices"][0])[np.ix_(XY, XY)]
    angle_truth = np.arctan2(am[1, 0], am[0, 0])
    p_truth = np.r_[quotient.optical_coordinates(truth[2:14]), visible_coordinates(source_truth), angle_truth, 0.]
    truth_residual = np.asarray(jax.jit(residual)(p_truth)); truth_holdout = np.asarray(heldout(p_truth))
    truth_jacobian = np.asarray(jax.jit(jax.jacfwd(residual))(p_truth))
    truth_heldout_jacobian = np.asarray(jax.jit(jax.jacfwd(heldout))(p_truth))
    truth_physical_jacobian = np.asarray(jax.jacfwd(lambda p: quotient.decode(p)[0])(p_truth))
    truth_information = information_diagnostics(truth_jacobian, truth_heldout_jacobian, truth_physical_jacobian)
    saved.update(truth_training_jacobian=truth_jacobian, truth_heldout_jacobian=truth_heldout_jacobian)
    verify_provenance_snapshot(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array_path = args.output.with_suffix(".npz"); np.savez_compressed(array_path, **saved)
    report = dict(schema="neoarm-condenser-visible-source-fit-v1", source_hashes=sources,
        computational_input_hashes=inputs, input_report_path=str(args.input.relative_to(ROOT)),
        input_report_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        frozen_image_array_sha256=old["arrays_sha256"], arrays_path=str(array_path.relative_to(ROOT)),
        arrays_sha256=hashlib.sha256(array_path.read_bytes()).hexdigest(), backend=jax.default_backend(),
        elapsed_seconds=perf_counter()-started, records=records,
        training_families=[0, 1, 2, 3, 4, 5, 12, 13], training_aperture_families=[0, 2],
        heldout_families=[6, 7, 14], heldout_aperture_families=[14], training_data_coordinates=54, heldout_data_coordinates=21,
        truth_training_chi_square=float(truth_residual@truth_residual), truth_heldout_chi_square=float(truth_holdout@truth_holdout),
        truth_local_information=truth_information,
        source_visible_coordinates=9, exact_invisible_source_covariance_family_profiled=True,
        source_reference_plane_is_coordinate_gauge=True, source_reference_plane_m=model.source_z_m,
        common_radius_scale_prior_sigma=.005, aperture_rotation_free=True, independent_angle_tag=False,
        fixed_conditional_objective_and_downstream_calibration=True,
        aperture_plane_fixed_conditional=True, aperture_exact_conjugacy_metrology_conditional=True,
        physically_unchanged_astigmatic_input_preparation_unconfirmed=True,
        reused_noisy_dataset_new_whole_state_holdout_split=True,
        approximate_diagram_base_coincides_with_synthetic_truth=True,
        nontrivially_offset_starts_demonstrated=bool(args.starts > 1 and all(r["accepted"] for r in records)),
        complete_condenser_recovery_accepted=all(r["accepted"] for r in records),
        fit_chart="Three blocks [centre, log-length, bias/sqrt(length), slope/sqrt(length)] in declared scaled coordinates; nine visible covariance combinations; aperture rotation; common log radius scale.",
        weak_direction_followup="Add independent CL1/CL3 excitation measurements that are actually collected on camera, and allocate informative narrow-probe states to both training and a new untouched holdout. Do not silently make the existing held-out family a training datum.",
        global_branch_uniqueness_demonstrated=False, hardware_admitted=False,
        limitations=["Conditional objective/downstream maps and aperture-plane position; their calibration uncertainty is not yet propagated.",
            "Fixed but unknown astigmatic input upstream of CL1 is a conditional preparation, not a known stigmator location.",
            "Aperture metrology is conditional on conjugacy/finite-B certification and a sufficiently clear specimen vacuum path.",
            "Radius uncertainty is common across all aperture data; source scale/direction completion is not silently fixed.",
            "Local held-out fit and multiple starts do not certify global excitation-branch uniqueness."])
    args.output.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False)+"\n")
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
