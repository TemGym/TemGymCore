"""Explicit coupled-likelihood diagnostics and trust-region recovery.

Run from the repository with ``.venv/bin/python examples/lens_inversion/
neoarm_recovery.py --problem ... --output ...``. The problem is a numeric,
checksum-verified snapshot, not an executable notebook or pickled function.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import json
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize, LinearConstraint

from neoarm_problem_io import load_problem
from temgym_core.condenser_astigmatic_adapter import make_astigmatic_full_column_adapter
from temgym_core.condenser_astigmatic_likelihood import make_astigmatic_condenser_likelihood
from temgym_core.condenser_astigmatic_holdout import (
    make_astigmatic_held_out_split, astigmatic_training_residual,
    astigmatic_held_out_residual,
)
from temgym_core.neoarm_calibration import hard_edge_column_from_config
from temgym_core.physical_observables import make_calibration_aware_physical_predictor, CentroidDifference
from temgym_core.physical_nuisance import (
    GlobalObservationLayout, SharedLatentNuisanceLikelihood,
    evaluate_held_out_shared_likelihood,
)


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def probe_calibration(config, metadata):
    """Physical Gaussian preparation recorded by the synthetic renderer.

    This is a separately measured probe calibration on an instrument, not a
    covariance inferred from commanded position/tilt uncertainty.
    """
    from temgym_core.constants import energy2wavelength
    waist = metadata["vacuum_probe_waist_m"]
    wavelength = float(energy2wavelength(config["beam"]["voltage_v"]))
    covariance = np.diag([waist**2/4, (wavelength/(2*np.pi*waist))**2]*2)
    x, y = config["sample"]["pose_translation_xy_m"]
    return np.asarray([x, 0., y, 0.]), covariance


def check_saved_probe_centroids(root):
    """Independent closure against the actual GPU-rendered image arrays."""
    import tomllib
    from neoarm_workflow import load_saved_acquisition
    from temgym_core.detector_moments import finite_gaussian_centroid

    root = Path(root)
    config_path = root / "examples/lens_inversion/neoarm_inversion_config.toml"
    config = tomllib.loads(config_path.read_text())
    generated = root / "examples/generated/neoarm_inversion"
    run = load_saved_acquisition(generated / "physical_checkpoint_native_imaging_v5_gpu", config_path)
    metadata = json.loads((generated / "physical_checkpoint_150k_256px_693x693_v11_gpu/manifest.json").read_text())["payload"]["render_metadata"]
    mean, covariance = probe_calibration(config, metadata)
    if not np.array_equal(metadata["parameters"], run.metadata["parameters"]):
        raise ValueError("Probe calibration and acquisition column differ.")
    column = hard_edge_column_from_config(config)
    yy, xx = np.indices(run.detector.shape_yx, dtype=float)
    rows = []
    for request in run.requests:
        if request.measurement not in {"perforation", "position_basis", "angle_basis", "basis_negative", "basis_orthogonal"}:
            continue
        pixels = np.asarray(run.image(request.frame_id), dtype=float)
        measured = np.array([(pixels*xx).sum(), (pixels*yy).sum()])/pixels.sum()
        matrix = np.asarray(column.matrix(run.metadata["parameters"], request.raw_dac_values,
                                          segment="sample_to_detector"))[[0, 2], :]
        transported_mean = matrix @ (mean + request.sample_state.as_array())
        transported_covariance = matrix @ covariance @ matrix.T
        finite = np.asarray(finite_gaussian_centroid(transported_mean, transported_covariance, run.detector))
        physical = run.detector.affine_xy @ (transported_mean-run.detector.centre_xy_m)
        unbounded = physical/np.asarray(run.detector.pixel_size_yx_m[::-1]) + (np.asarray(run.detector.shape_yx[::-1])-1)/2
        rows.append({"frame_id":request.frame_id, "state":request.optical_state_label,
                     "finite_error_xy_px":(finite-measured).tolist(),
                     "unbounded_error_xy_px":(unbounded-measured).tolist()})
    errors = np.asarray([row["finite_error_xy_px"] for row in rows])
    return {"frame_count":len(rows), "manifest_sha256":run.checkpoint.manifest_sha256,
            "finite_rms_px":rms(errors), "finite_maximum_px":float(np.max(np.abs(errors))),
            "unbounded_rms_px":rms([row["unbounded_error_xy_px"] for row in rows]),
            "rows":rows}


@dataclass
class CoupledProblem:
    snapshot: dict
    root: Path
    finite_detector: bool = True
    probes_only: bool = False

    def __post_init__(self):
        s = self.snapshot
        self.column = hard_edge_column_from_config(s["config"])
        self.truth_optical = np.asarray(s["parameters"])
        specs = s["specs"]
        if self.finite_detector or self.probes_only:
            manifest = self.root / "examples/generated/neoarm_inversion/physical_checkpoint_150k_256px_693x693_v11_gpu/manifest.json"
            saved = json.loads(manifest.read_text())["payload"]["render_metadata"]
            if not np.array_equal(saved["parameters"], self.truth_optical):
                raise ValueError("Probe calibration and likelihood belong to different synthetic columns.")
            mean, covariance = probe_calibration(s["config"], saved)
            specs = tuple(replace(spec, parts=tuple(
                replace(part, finite_detector_sample_covariance=covariance,
                        finite_detector_reference_mean=mean)
                if isinstance(part, CentroidDifference) else part
                for part in spec.parts)) for spec in specs)
        self.specs = specs
        if self.probes_only:
            from neoarm_probe_likelihood import make_probe_likelihood
            self.image = make_probe_likelihood(s, self.root, self.column, mean, covariance)
        else:
            priors, predictor = make_calibration_aware_physical_predictor(
                self.column, self.truth_optical, s["detector"], specs, dataset=s["dataset"])
            self.image = SharedLatentNuisanceLikelihood(
                GlobalObservationLayout.from_dataset(s["dataset"]),
                self.column.n_parameters, predictor, calibration_priors=priors)
        self.condenser = make_astigmatic_condenser_likelihood(
            self.root / "examples/lens_inversion/neoarm_condenser_astigmatic_design.json",
            observations=s["condenser_observations"], repository_root=self.root)
        self.adapter = make_astigmatic_full_column_adapter(self.condenser, self.column)
        self.split = make_astigmatic_held_out_split(self.condenser)
        self.n_base = self.image.n_joint_parameters
        self.truth = np.r_[self.image.join_parameters(self.truth_optical),
                           self.adapter.outer_contract.reference]
        self.scales = np.r_[s["optical_scales"], np.ones(self.image.n_nuisance_latents),
                            self.adapter.outer_contract.scales]
        self.names = (tuple(self.column.parameter_names) + self.image.calibration_latent_names
                      + self.adapter.outer_contract.names)
        self.residual = jax.jit(self._residual)
        self.jacobian = jax.jit(jax.jacfwd(self._residual))

    def native(self, joint):
        return self.adapter.likelihood_parameters_from_parts(joint[:36], joint[self.n_base:])

    def _residual(self, joint):
        return jnp.concatenate((
            self.image.joint_residual(joint[:self.n_base]),
            astigmatic_training_residual(self.condenser, self.native(joint), self.split)))

    def diagnostics(self, joint):
        image_data = np.asarray(self.image.data_residual(joint[:self.n_base]))
        condenser = np.asarray(astigmatic_training_residual(
            self.condenser, self.native(joint), self.split))
        held_condenser = np.asarray(astigmatic_held_out_residual(
            self.condenser, self.native(joint), self.split))
        if self.probes_only:
            held_image = np.asarray(self.image.held_residual(joint[:self.n_base]))
            held_summary = {"rms_sigma":rms(held_image), "maximum_sigma":float(np.max(np.abs(held_image)))}
        else:
            held_image = evaluate_held_out_shared_likelihood(
                self.image, self.snapshot["dataset"], joint[:self.n_base])
            held_summary = {"rms_sigma":held_image.weighted_rms_sigma,
                            "maximum_sigma":held_image.maximum_absolute_whitened_residual,
                            "block_indices":list(held_image.held_out_block_indices)}
        ranked = np.argsort(np.abs(image_data))[-12:][::-1]
        return {
            "image_training_rms_sigma": rms(image_data),
            "condenser_training_rms_sigma": rms(condenser),
            "condenser_held_out_rms_sigma": rms(held_condenser),
            "image_held_out": held_summary,
            "worst_image_rows": [{"key":self.image.layout.observation_keys[i],
                                  "residual_sigma":float(image_data[i])} for i in ranked],
        }

    def bounds(self):
        lower, upper = self.column.default_bounds(
            position_half_width_m=3e-3, length_ratio=(.4, 2.5),
            q_half_width=.8, slope_half_width=.8)
        # Preserve the fixed C-aperture plane within the CL3/C-mini gap.
        from temgym_core.critical_line_calibration import (
            HardEdgeInternalPlaneConstraint, restrict_hard_edge_internal_planes_bounds)
        certificates = restrict_hard_edge_internal_planes_bounds(
            self.column, (lower, upper),
            (HardEdgeInternalPlaneConstraint(self.condenser.branch_experiment.aperture_z_m,
                                              "CL3", "C-mini", .0005),),
            reference_parameters=self.truth_optical)
        lower, upper = certificates[0].bounds
        return (np.r_[lower, np.full(self.image.n_nuisance_latents, -5.),
                      self.adapter.outer_contract.lower],
                np.r_[upper, np.full(self.image.n_nuisance_latents, 5.),
                      self.adapter.outer_contract.upper])


def trust_region_fit(problem, initial, *, free_indices=None, max_nfev=1000, verbose=0):
    """Solve the raw whitened least-squares problem, without a log-cost transform.

    All declared nuisance parameters remain explicit. No practical-null basis
    at a bad starting point is frozen into the optimization.
    """
    initial = np.asarray(initial, dtype=float)
    indices = np.arange(initial.size) if free_indices is None else np.asarray(free_indices)
    scale = problem.scales[indices]
    lower, upper = problem.bounds()
    lower = (lower[indices]-initial[indices])/scale
    upper = (upper[indices]-initial[indices])/scale

    def decode(x):
        values = initial.copy()
        values[indices] += scale * x
        return values

    def residual(x):
        return np.asarray(problem.residual(jnp.asarray(decode(x))))

    def jacobian(x):
        return np.asarray(problem.jacobian(jnp.asarray(decode(x))))[:, indices]*scale

    started = time.perf_counter()
    result = least_squares(
        residual, np.zeros(indices.size), jac=jacobian, bounds=(lower, upper),
        method="trf", x_scale="jac", ftol=1e-11, xtol=1e-12, gtol=1e-8,
        max_nfev=max_nfev, verbose=verbose)
    return decode(result.x), {
        "success": bool(result.success), "message": str(result.message),
        "nfev": result.nfev, "cost":float(result.cost),
        "optimality":float(result.optimality), "elapsed_s":time.perf_counter()-started,
        "residual_rms":rms(result.fun),
    }


def initialize_from_probe_shapes(problem, initial, *, max_nfev=1000):
    """Fit image-reconstructed signed position/angle maps before fine likelihood.

    Unbounded Gaussian centres retain information when a beam is broader than
    the detector. Signed calibrated states identify the complex A/B responses.
    No synthetic optical truth enters these
    targets, and held-out exposures are excluded. Weights here condition a
    coarse initialization; they are not experimental uncertainty estimates.
    """
    if not problem.probes_only:
        raise ValueError("Probe-shape initialization requires the probe benchmark.")
    from types import SimpleNamespace
    image = problem.image
    controls, targets, weights = [], [], []
    states = sorted({image.requests[i].optical_state_index for i in image.train_indices})
    for state in states:
        selected = [i for i in image.train_indices if image.requests[i].optical_state_index == state]
        mean = np.asarray(image.means)[selected]
        x, tx, y, ty = (mean/np.array([1e-9, 1e-6, 1e-9, 1e-6])).T
        design = np.stack((np.column_stack((x, -y, tx, -ty)),
                           np.column_stack((y, x, ty, tx))), axis=1).reshape(-1, 4)
        observed = image.gaussian_estimates[selected, :2]
        centre_whitener = np.linalg.inv(np.linalg.cholesky(image.gaussian_estimate_covariances[selected, :2, :2]))
        design = np.einsum("nij,njk->nik", centre_whitener, design.reshape(-1, 2, 4)).reshape(-1, 4)
        observed = np.einsum("nij,nj->ni", centre_whitener, observed).reshape(-1)
        coefficients, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
        if rank != 4:
            raise ValueError("Calibrated probe states do not span signed A/B responses.")
        scale = np.array([1e6, 1e6, 1e3, 1e3])
        target = coefficients*scale
        covariance = np.linalg.inv(design.T@design)*np.outer(scale, scale)
        covariance += np.diag(np.array([.005*np.linalg.norm(target[:2])]*2 + [.1]*2)**2)
        controls.append(image.requests[selected[0]].raw_dac_values)
        targets.append(target)
        weights.append(np.linalg.inv(np.linalg.cholesky(covariance)))
    observed, weights = jnp.asarray(targets), jnp.asarray(weights)

    def one(parameters, dacs):
        matrix = problem.column.matrix(parameters[:36], dacs, segment="sample_to_detector")[[0, 2], :]
        return jnp.stack((matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1]))

    batch = jax.vmap(one, in_axes=(None, 0))

    def residual(parameters):
        predicted = batch(parameters, jnp.asarray(controls))
        return jnp.einsum("nij,nj->ni", weights, predicted-observed).reshape(-1)

    proxy = SimpleNamespace(scales=problem.scales, bounds=problem.bounds,
                            residual=jax.jit(residual), jacobian=jax.jit(jax.jacfwd(residual)))
    upstream = set(problem.adapter.target_physical_indices + problem.adapter.upstream_physical_indices)
    downstream = [i for i in range(36) if i not in upstream]
    excitation = [i for i in downstream if ".q_" in problem.names[i]]
    initialized, excitation_fit = trust_region_fit(proxy, initial, free_indices=excitation, max_nfev=max_nfev)
    fitted, full_fit = trust_region_fit(proxy, initialized, free_indices=downstream, max_nfev=max_nfev)
    full_fit["excitation_stage"] = excitation_fit
    return fitted, full_fit


def polish_in_likelihood_coordinates(problem, initial, *, maxiter=1000):
    """Whiten parameter sensitivity near a statistically consistent solution.

    All parameter directions are retained, including weak/null directions;
    singular values below one are capped for scaling, not removed. Original
    physical bounds become exact linear constraints in this local chart.
    The minimized objective is the raw whitened chi-square, never log(cost).
    """
    initial = np.asarray(initial)
    jacobian = np.asarray(problem.jacobian(initial))*problem.scales
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    transform = problem.scales[:, None]*right.T/np.maximum(singular, 1.)
    lower, upper = problem.bounds()
    constraint = LinearConstraint(transform/problem.scales[:, None],
                                   (lower-initial)/problem.scales, (upper-initial)/problem.scales)

    def objective(coordinates):
        parameters = initial+transform@coordinates
        residual = np.asarray(problem.residual(parameters))
        jacobian = np.asarray(problem.jacobian(parameters))@transform
        return .5*float(residual@residual), jacobian.T@residual

    started = time.perf_counter()
    fit = minimize(objective, np.zeros(len(initial)), jac=True, method="SLSQP",
                   constraints=(constraint,), options={"maxiter":maxiter, "ftol":1e-9})
    fitted = initial+transform@fit.x
    residual = np.asarray(problem.residual(fitted))
    return fitted, {"success":bool(fit.success), "message":str(fit.message),
                    "iterations":int(fit.nit), "nfev":int(fit.nfev),
                    "cost":float(fit.fun), "residual_rms":rms(residual),
                    "elapsed_s":time.perf_counter()-started,
                    "singular_values":singular.tolist(), "directions_removed":0,
                    "scaled_maximum_bound_violation":float(max(0., np.max(np.maximum(lower-fitted, fitted-upper)/problem.scales)))}


def adjacent_excitation_starts(problem, reference, *, orders=(-1, 1)):
    """Explicit adjacent hard-edge phase hypotheses, without clipping seeds.

    q -> q+n*pi and L -> L*(q+n*pi)/q preserves a lens-body map
    at one anchor only. Drifts and all other DAC states are evaluated by the
    full likelihood. This enumerates one-lens neighbours, not all combinations.
    """
    reference = np.asarray(reference)
    lower, upper = problem.bounds()
    anchor = np.median(np.asarray(problem.image.dacs)[problem.image.train_indices], axis=0)
    names = list(problem.column.parameter_names)
    candidates = []
    for lens in problem.column.lenses:
        if lens.channel_index is None:
            continue
        bias = names.index(lens.name+".q_bias")
        slope = names.index(lens.name+".q_slope_per_dac")
        length = names.index(lens.name+".log_length_m")
        dac = anchor[lens.channel_index]-lens.reference_dac
        q = reference[bias]+reference[slope]*dac
        if abs(dac) < 1e-12 or abs(q) < 1e-12:
            continue
        for order in orders:
            ratio = (q+order*np.pi)/q
            if ratio <= 0:
                continue
            candidate = reference.copy()
            candidate[slope] += order*np.pi/dac
            candidate[length] += np.log(ratio)
            if np.any(candidate < lower) or np.any(candidate > upper):
                continue
            candidates.append(({"lens":lens.name, "order":int(order),
                                "anchor_raw_dac":float(anchor[lens.channel_index]),
                                "anchor_q":float(q), "alias_q":float(q+order*np.pi)}, candidate))
    return candidates


def recovery_diagnostics(problem, fitted):
    """Local rank and physical errors; null-contaminated marginals are flagged."""
    fitted = np.asarray(fitted)
    jacobian = np.asarray(problem.jacobian(fitted))
    nuisance = jacobian[:, 36:]*problem.scales[36:]
    u, singular_nuisance, _ = np.linalg.svd(nuisance, full_matrices=False)
    nuisance_rank = int(np.sum(singular_nuisance > singular_nuisance[0]*1e-12))
    optical = jacobian[:, :36]*problem.scales[:36]
    profiled = optical-u[:, :nuisance_rank]@(u[:, :nuisance_rank].T@optical)
    _, singular, right = np.linalg.svd(profiled, full_matrices=False)
    numerical_threshold = np.finfo(float).eps*max(profiled.shape)*singular[0]
    rank = int(np.sum(singular > numerical_threshold))
    null_fraction = np.linalg.norm(right[rank:], axis=0)
    records = []
    for index, name in enumerate(problem.names[:36]):
        truth, estimate = problem.truth[index], fitted[index]
        if ".log_length_m" in name:
            truth, estimate = np.exp(truth), np.exp(estimate)
        records.append({"name":name, "truth":float(truth), "estimate":float(estimate),
                        "relative_error":None if truth == 0 else float((estimate-truth)/abs(truth)),
                        "absolute_error":float(estimate-truth),
                        "null_projection":float(null_fraction[index])})
    return {"profiled_optical_singular_values":singular.tolist(),
            "numerical_optical_rank":rank,
            "numerical_singular_value_threshold":float(numerical_threshold),
            "relative_threshold_rank_1e_10":int(np.sum(singular > singular[0]*1e-10)),
            "noise_resolved_scaled_directions_s_gt_1":int(np.sum(singular > 1)),
            "n_optical_parameters":36, "parameter_errors":records,
            "rank_scope":"local; nuisance-profiled; not a global uniqueness proof",
            "null_vectors_scaled_coordinates":right[rank:].tolist()}


def source_plane_gauge_family(problem, reference, *, offsets_m=(-.002, -.001, 0., .001, .002)):
    """An explicit equivalent family: move source plane and transport its covariance.

    Unknown entrance moments absorb the free-space plane choice. Lens fields,
    astigmatic orientation and detector predictions remain unchanged.
    """
    reference = np.asarray(reference)
    chart = problem.condenser.branch_experiment.entrance_covariance_chart
    covariance = np.asarray(chart.covariance(reference[problem.n_base:problem.n_base+10]))
    inverse = np.linalg.inv(chart.cholesky_factor)
    baseline = np.asarray(problem.residual(reference))
    lower, upper = problem.bounds()
    records = []
    for offset in offsets_m:
        transport = np.eye(4)
        transport[0, 1] = transport[2, 3] = offset
        whitened = inverse@transport@covariance@transport.T@inverse.T
        eigenvalues, eigenvectors = np.linalg.eigh(whitened)
        generator = (eigenvectors*np.log(eigenvalues))@eigenvectors.T
        candidate = reference.copy()
        candidate[0] += offset
        candidate[problem.n_base:problem.n_base+10] = np.einsum("ij,kij->k", generator, chart.symmetric_basis)
        if np.any(candidate < lower) or np.any(candidate > upper):
            continue
        residual = np.asarray(problem.residual(candidate))
        records.append({"offset_m":float(offset), "source_z_m":float(candidate[0]),
                        "delta_cost":float(.5*(residual@residual-baseline@baseline)),
                        "maximum_residual_change_sigma":float(np.max(np.abs(residual-baseline))),
                        "parameters":candidate.tolist()})
    return records


def held_out_optical_errors(problem, fitted):
    """Compare held-out optical predictions with generating truth (not fit data)."""
    image = problem.image
    predicted = np.asarray(image.predict(fitted[:36], image.dacs, image.means))
    truth = np.asarray(image.predict(problem.truth[:36], image.dacs, image.means))
    magnification, rotation, diameter = [], [], []
    for index in image.held_indices:
        control = image.dacs[index]
        maps = [np.asarray(problem.column.matrix(p, control, segment="sample_to_detector"))[np.ix_([0, 2], [0, 2])]
                for p in (problem.truth[:36], fitted[:36])]
        magnification.append(np.sqrt(abs(np.linalg.det(maps[1])/np.linalg.det(maps[0])))-1)
        angles = [np.arctan2(m[1, 0]-m[0, 1], m[0, 0]+m[1, 1]) for m in maps]
        rotation.append(np.rad2deg(np.angle(np.exp(1j*(angles[1]-angles[0])))))
        variances = []
        for moments in (truth[index], predicted[index]):
            mean = moments[:2]
            covariance = np.array([[moments[2], moments[3]], [moments[3], moments[4]]])-np.outer(mean, mean)
            variances.append(np.linalg.eigvalsh(covariance)[-1])
        diameter.append(np.sqrt(variances[1]/variances[0])-1)
    return {"maximum_magnification_relative_error":float(np.max(np.abs(magnification))),
            "maximum_rotation_error_deg":float(np.max(np.abs(rotation))),
            "maximum_centroid_error_px":float(np.max(np.abs(predicted[image.held_indices, :2]-truth[image.held_indices, :2]))),
            "maximum_detector_beam_diameter_relative_error":float(np.max(np.abs(diameter))),
            "scope":"held-out probe states; no claim for specimen/Bragg estimator closure"}


def audit_adjacent_branches(problem, reference, *, max_nfev=1000, output=None):
    """Challenge the selected local solution with explicitly named neighbours.

    A failed optimizer is an unresolved hypothesis, not proof of exclusion.
    Approach-direction metadata are never treated as measured continuation.
    """
    base_cost = .5*float(np.sum(np.asarray(problem.residual(reference))**2))
    report = {"base_cost":base_cost, "scope":"one-lens ±pi anchor hypotheses only",
              "instrument_branch_anchored":False, "candidates":[]}
    for label, initial in adjacent_excitation_starts(problem, reference):
        fitted, stage = initialize_from_probe_shapes(problem, initial, max_nfev=3000)
        fitted, fit = trust_region_fit(problem, fitted, max_nfev=max_nfev)
        polish = None
        if fit["residual_rms"] < 3:
            fitted, polish = polish_in_likelihood_coordinates(problem, fitted, maxiter=500)
        terminal = polish or fit
        consistent = bool(terminal["success"] and terminal["residual_rms"] < 2)
        record = {"hypothesis":label, "stage":stage, "fit":fit, "polish":polish,
                  "fitted_parameters":fitted.tolist(), "noise_consistent":consistent,
                  "competitive":bool(consistent and terminal["cost"] <= base_cost+5),
                  "status":"resolved_local_fit" if terminal["success"] else "unresolved_optimization"}
        report["candidates"].append(record)
        if output is not None:
            write_report(output, report)
        print("Branch challenge", label, "RMS", terminal["residual_rms"], record["status"], flush=True)
    report["competitive_alternatives"] = sum(item["competitive"] for item in report["candidates"])
    report["unresolved_hypotheses"] = sum(item["status"] == "unresolved_optimization" for item in report["candidates"])
    report["global_uniqueness_established"] = False
    if output is not None:
        write_report(output, report)
    return report


def write_report(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--legacy-linear-centroids", action="store_true")
    parser.add_argument("--probes-only", action="store_true",
                        help="Conditional probe-image + astigmatic/aperture benchmark, excluding material estimators")
    parser.add_argument("--initialize-shapes", action="store_true")
    parser.add_argument("--initial-amplitude", type=float, default=.02,
                        help="Declared synthetic starting perturbation in stored parameter scales")
    parser.add_argument("--initial-parameters", type=Path,
                        help="JSON array of 36 optical or all joint starting parameters")
    parser.add_argument("--polish-rounds", type=int, default=3)
    parser.add_argument("--branch-audit-output", type=Path,
                        help="Optional explicit adjacent-branch challenge; adds several minutes")
    parser.add_argument("--max-nfev", type=int, default=2000)
    args = parser.parse_args()
    if not np.isfinite(args.initial_amplitude) or args.max_nfev < 1 or args.polish_rounds < 0:
        parser.error("Use a finite initial amplitude, positive evaluation count, and nonnegative polish rounds.")
    root = Path(__file__).resolve().parents[2]
    problem = CoupledProblem(load_problem(args.problem), root, not args.legacy_linear_centroids, args.probes_only)
    print("Loaded coupled problem", len(problem.truth), "parameters", flush=True)
    report = {"scope":"conditional probe-image plus separate astigmatic/aperture likelihood" if args.probes_only
                      else "archived material/probe/diffraction plus condenser likelihood",
              "truth":problem.diagnostics(problem.truth), "instrument_accepted":False,
              "full_plan_accepted":False, "jax_backend":jax.default_backend(),
              "devices":[str(device) for device in jax.devices()],
              "problem_sha256":json.loads(args.problem.read_text())["sha256"]}
    report["condenser_image_measurement_contract_validated"] = False
    report["condenser_measurement_warning"] = (
        "Archived summaries include offset-minus-sum(q) as an independent astigmatic orientation. "
        "That is not the transported image-ellipse angle for the released general entrance covariance. "
        "Synthetic closure of this stipulated likelihood is not physical condenser recovery. "
        "See neoarm_imaging_information_audit.py and NEOARM_IMAGING_RECIPE.md.")
    if args.probes_only:
        report["probe_provenance"] = problem.image.provenance
    write_report(args.output, report)
    print(json.dumps(report["truth"], indent=2), flush=True)
    if args.fit:
        if report["truth"]["image_training_rms_sigma"] > 2 or report["truth"]["condenser_training_rms_sigma"] > 2:
            raise RuntimeError("Synthetic truth fails likelihood closure; optimizer has not been started.")
        initial = problem.truth.copy()
        initial[:36] = problem.column.nominal_parameters
        if args.initial_parameters:
            supplied = np.asarray(json.loads(args.initial_parameters.read_text()), dtype=float)
            if supplied.shape not in {(36,), initial.shape}:
                raise ValueError("Starting array must contain 36 optical or all joint parameters.")
            initial[:len(supplied)] = supplied
        else:
            initial[:36] += problem.scales[:36]*args.initial_amplitude*np.sin((np.arange(36)+1)*np.sqrt(2))
        if not np.all(np.isfinite(initial)) or np.any(initial < problem.bounds()[0]) or np.any(initial > problem.bounds()[1]):
            raise ValueError("Starting parameters violate the physical bounds; no silent clipping.")
        report["initial_parameters"] = initial.tolist()
        report["initial"] = problem.diagnostics(initial)
        write_report(args.output, report)
        if args.initialize_shapes or args.probes_only:
            initial, stage = initialize_from_probe_shapes(problem, initial, max_nfev=args.max_nfev)
            report["shape_initialization"] = stage
            report["initialized_parameters"] = initial.tolist()
            write_report(args.output, report)
            print("Shape initialization", stage, flush=True)
        fitted, fit = trust_region_fit(problem, initial, max_nfev=args.max_nfev, verbose=1)
        report.update(fit=fit, fitted_parameters=fitted.tolist(), final=problem.diagnostics(fitted))
        write_report(args.output, report)
        print(json.dumps(fit), flush=True)
        report["polish"] = []
        if fit["residual_rms"] < 3:
            for _ in range(args.polish_rounds):
                fitted, polished = polish_in_likelihood_coordinates(problem, fitted, maxiter=500)
                report["polish"].append({**polished, "parameters":fitted.tolist()})
                report.update(fitted_parameters=fitted.tolist(), final=problem.diagnostics(fitted))
                write_report(args.output, report)
                print("Sensitivity-scaled polish", {k:v for k,v in polished.items() if k != "singular_values"}, flush=True)
                if polished["success"]:
                    break
        if args.probes_only:
            report["recovery"] = recovery_diagnostics(problem, fitted)
            report["source_plane_gauge_family"] = source_plane_gauge_family(problem, fitted)
            report["held_out_optical_errors"] = held_out_optical_errors(problem, fitted)
            diagnostics = report["final"]
            last = report["polish"][-1] if report["polish"] else fit
            errors = report["held_out_optical_errors"]
            report["conditional_probe_condenser_recovery_passed"] = bool(last["success"] and
                max(diagnostics["image_training_rms_sigma"], diagnostics["condenser_training_rms_sigma"],
                    diagnostics["image_held_out"]["rms_sigma"], diagnostics["condenser_held_out_rms_sigma"]) < 2 and
                errors["maximum_magnification_relative_error"] < .005 and
                errors["maximum_rotation_error_deg"] < .5 and
                errors["maximum_centroid_error_px"] < .5 and
                errors["maximum_detector_beam_diameter_relative_error"] < .02 and
                last.get("scaled_maximum_bound_violation", 0.) < 1e-8)
        write_report(args.output, report)
        if args.branch_audit_output:
            if not report.get("conditional_probe_condenser_recovery_passed", False):
                raise RuntimeError("No noise-consistent recovered branch is available to challenge.")
            audit_adjacent_branches(problem, fitted, output=args.branch_audit_output)


if __name__ == "__main__":
    main()
