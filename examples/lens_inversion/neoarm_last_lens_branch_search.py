"""Variable-projection PL1 initialization and explicit excitation alternatives.

The sixteen upstream response coefficients are linear once log(d/L), q bias
and q slope are specified. Eliminate them by QR in the *whitened image-response
metric*, then search only three nonlinear coordinates. No upstream generating
matrix or generating geometry is supplied to a start. The saved example uses
conditional response data; this is not a claim about achievable image precision.

Finite multistart searches do not prove global uniqueness. Failed candidates
remain unresolved. Successful poor fits record tested incompatible solutions,
not a global lower bound on every solution in their excitation interval.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from neoarm_last_lens_tomography import predict_response_parameters, geometry_coordinates
from temgym_core.jax_least_squares import least_squares


def linear_response_design(nonlinear, controls):
    """Return (setting, flattened response, coefficient) in row-major H order."""
    nonlinear, controls = jnp.asarray(nonlinear), jnp.asarray(controls)
    values = jax.vmap(lambda w: predict_response_parameters(
        jnp.concatenate((nonlinear, w)), controls))(jnp.eye(16))
    return values.reshape(16, controls.size, 8).transpose(1, 2, 0)


class ProfiledResponse:
    """QR elimination retaining heteroscedastic, correlated response errors.

    Cholesky factors have shape (N,8,8), matching ``H.reshape(N,8)``. Different
    settings are independent here; shared calibration errors must be released
    separately, or encoded in a future full cross-setting covariance adapter.
    """

    def __init__(self, controls, observed, response_cholesky):
        t, y, chol = map(np.asarray, (controls, observed, response_cholesky))
        if t.ndim != 1 or len(t) < 4 or y.shape != (len(t), 2, 4):
            raise ValueError("At least four controls and (N,2,4) signed responses are required.")
        if chol.shape != (len(t), 8, 8):
            raise ValueError("Response Cholesky factors must have shape (N,8,8).")
        if not all(np.all(np.isfinite(a)) for a in (t, y, chol)):
            raise ValueError("Responses, controls and covariance factors must be finite.")
        if np.any(np.diagonal(chol, axis1=1, axis2=2) <= 0) or not np.allclose(chol, np.tril(chol)):
            raise ValueError("Covariance factors must be lower triangular with positive diagonal.")
        self.controls, self.observed, self.cholesky = map(jnp.asarray, (t, y, chol))
        self.whitened_observed = self.whiten(self.observed.reshape(len(t), 8, 1)).reshape(-1)

    def whiten(self, values):
        return jax.vmap(lambda lower, value: solve_triangular(lower, value, lower=True))(
            self.cholesky, values)

    def solve(self, nonlinear):
        design = self.whiten(linear_response_design(nonlinear, self.controls)).reshape(-1, 16)
        # Scaling aids the QR solve without changing the physical coefficients.
        scales = jnp.linalg.norm(design, axis=0)
        orthogonal, triangular = jnp.linalg.qr(design/scales, mode="reduced")
        scaled_coefficients = solve_triangular(
            triangular, orthogonal.T@self.whitened_observed, lower=False)
        coefficients = scaled_coefficients/scales
        return coefficients, design@coefficients-self.whitened_observed

    def residual(self, nonlinear):
        return self.solve(nonlinear)[1]


def search_response_branches(problem, *, q_centres=None, ratios=(4., 16.),
                             slope_seeds=(-2.4, -.8, .8, 2.4), max_nfev=250):
    """Search declared q-bias intervals; keep every start and failed termination.

    The default 64 starts span eight pi-wide bias intervals, both slope signs,
    and two positive geometry ratios. These are synthetic phase coordinates,
    not authorised lens-current/DAC ranges. Bias bounds label hypotheses and
    do not grant the true branch to the optimiser.
    """
    if q_centres is None:
        q_centres = np.pi*(np.arange(-4, 4)+.5)
    results = []
    residual = jax.jit(problem.residual)
    for centre in q_centres:
        for ratio in ratios:
            for slope in slope_seeds:
                if ratio <= 0 or slope == 0:
                    raise ValueError("Positive ratio and nonzero slope seeds required.")
                lower_slope, upper_slope = (-4*np.pi, -.05) if slope < 0 else (.05, 4*np.pi)
                initial = np.array([np.log(ratio), centre, slope])
                lower = np.array([np.log(.2), centre-np.pi/2, lower_slope])
                upper = np.array([np.log(200.), centre+np.pi/2, upper_slope])
                result = least_squares(residual, initial, bounds=(lower, upper),
                    x_scale=np.ones(3), max_nfev=max_nfev, max_iter=500,
                    ftol=1e-11, xtol=1e-11, gtol=1e-7)
                coefficients, fitted_residual = problem.solve(result.x)
                full = np.r_[np.asarray(result.x), np.asarray(coefficients)]
                chi_square = float(jnp.sum(fitted_residual**2))
                determinant_sign = float(np.linalg.slogdet(full[3:].reshape(4, 4))[0])
                active = np.flatnonzero(np.minimum(np.abs(np.asarray(result.x)-lower),
                                                   np.abs(np.asarray(result.x)-upper)) < 1e-6)
                results.append(dict(initial=initial.tolist(), lower=lower.tolist(), upper=upper.tolist(),
                    q_bias_interval=[float(lower[1]), float(upper[1])], slope_sign=int(np.sign(slope)),
                    optimizer_success=result.success, termination=result.status, nfev=result.nfev,
                    parameters=full.tolist(), chi_square=chi_square,
                    residual_rms_sigma=float(np.sqrt(chi_square/problem.observed.size)),
                    positive_coefficient_volume=determinant_sign > 0,
                    active_nonlinear_bounds=active.tolist(),
                    status="tested_stationary_solution" if result.success else "unresolved_optimizer_failure"))
    return results


def distinct_solution(candidate, reference, *, phase_threshold=.1, log_ratio_threshold=.05):
    """Include geometric alternatives even when their excitation laws agree."""
    first, second = np.asarray(candidate[:3]), np.asarray(reference[:3])
    return bool(abs(first[0]-second[0]) > log_ratio_threshold or
                np.max(np.abs(first[1:]-second[1:])) > phase_threshold)


def run_audit(input_path, output_path, *, max_nfev=250, image_report_path=None):
    started = perf_counter()
    root = Path(__file__).resolve().parents[2]
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    reference = json.loads(input_path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, checksum in reference["source_hashes"].items():
        if digest(root/name) != checksum:
            raise ValueError(f"Stale conditional response input: {name}")
    paths = {Path(__file__), input_path, Path(__file__).with_name("neoarm_last_lens_tomography.py"),
             root/"src/temgym_core/jax_least_squares.py"}
    paths.update(root/name for name in reference["source_hashes"])
    inputs = reference["inputs"]
    t = np.asarray(inputs["normalized_dac"])
    observed = np.asarray(inputs["observed_response_pixels"])
    sigma = inputs["response_noise_standard_deviation_pixels"]
    cholesky = np.broadcast_to(sigma*np.eye(8), (len(t), 8, 8)).copy()
    image_report = None
    if image_report_path is not None:
        image_report_path = Path(image_report_path).resolve()
        image_report = json.loads(image_report_path.read_text())
        for name, checksum in image_report["source_hashes"].items():
            if digest(root/name) != checksum:
                raise ValueError(f"Stale image-response source: {name}")
        arrays_path = root/image_report["arrays_path"]
        if digest(arrays_path) != image_report["arrays_sha256"]:
            raise ValueError("Image-response NPZ checksum mismatch.")
        if not all(row["passed"] for row in image_report["independent_gaussian_checks"]):
            raise ValueError("Independent Gaussian image validation must pass before branch fitting.")
        paths.update((image_report_path, arrays_path))
        paths.update(root/name for name in image_report["source_hashes"])
        with np.load(arrays_path, allow_pickle=False) as saved:
            t, observed = saved["accepted_normalized_dac"], saved["response"]
            cholesky = np.linalg.cholesky(saved["response_covariance"])
    hashes = {str(path.relative_to(root)): digest(path) for path in sorted(paths)}
    problem = ProfiledResponse(t, observed, cholesky)
    rows = search_response_branches(problem, max_nfev=max_nfev)
    eligible = [row for row in rows if row["optimizer_success"] and row["positive_coefficient_volume"]]
    best = min(eligible, key=lambda row: row["chi_square"]) if eligible else None
    # Generating truth is accessed only for the *post-search* recovery check.
    truth = jnp.asarray(inputs["truth_parameters"])
    basis = jnp.asarray(inputs["input_basis_steps_grouped"])
    pixel = inputs["detector_pixel_m"]
    expected = predict_response_parameters(truth, t)
    truth_noiseless = ProfiledResponse(t, expected, cholesky)
    closure = float(jnp.max(jnp.abs(truth_noiseless.residual(truth[:3]))))
    diagnostic = {}
    if best is not None:
        recovered = jnp.asarray(best["parameters"])
        geometry_error = np.asarray(geometry_coordinates(recovered, basis, pixel)-
                                    geometry_coordinates(truth, basis, pixel))
        full_residual = lambda p: problem.whiten(
            (predict_response_parameters(p, t)-problem.observed).reshape(len(t), 8, 1)).reshape(-1)
        jacobian = np.asarray(jax.jacfwd(full_residual)(recovered))
        norms = np.linalg.norm(jacobian, axis=0)
        _, singular, right = np.linalg.svd(jacobian/norms, full_matrices=False)
        factor = (right.T/singular)/norms[:, None]
        g_jac = np.asarray(jax.jacfwd(geometry_coordinates)(recovered, basis, pixel))
        geometry_sigma = np.sqrt(np.sum((g_jac@factor)**2, axis=1))
        diagnostic = dict(length_gap_errors_percent=(100*np.expm1(geometry_error[:2])).tolist(),
            length_gap_errors_sigma=(geometry_error[:2]/geometry_sigma[:2]).tolist(),
            length_gap_standard_errors_percent=(100*geometry_sigma[:2]).tolist(),
            fitted_q_law=np.asarray(recovered[1:3]).tolist(),
            numerical_full_parameter_rank=int(np.sum(singular > singular[0]*1e-10)))
    successful_rivals = [row for row in eligible if best is not None and
                         distinct_solution(row["parameters"], best["parameters"])]
    for row in rows:
        geometry = np.asarray(geometry_coordinates(jnp.asarray(row["parameters"]), basis, pixel))
        row["fitted_length_m"], row["fitted_detector_gap_m"] = np.exp(geometry[:2]).tolist()
    report = dict(schema="neoarm-last-lens-branch-search-v1", source_hashes=hashes,
        input_path=str(input_path.relative_to(root)), input_sha256=digest(input_path),
        backend=jax.default_backend(), jax_version=jax.__version__, x64_enabled=jax.config.x64_enabled,
        start_count=len(rows), rows=rows, best=best, recovery_diagnostic=diagnostic,
        noiseless_truth_closure_max_abs_sigma=closure,
        converged_start_count=sum(row["optimizer_success"] for row in rows),
        unresolved_start_count=sum(not row["optimizer_success"] for row in rows),
        nearest_tested_stationary_alternative_delta_chi_square=(
            min(row["chi_square"] for row in successful_rivals)-best["chi_square"] if successful_rivals else None),
        all_PL1_response_parameters_released=True, generating_upstream_used_for_initialization=False,
        nonlinear_search_coordinates=["log_gap_over_length", "q_bias", "q_slope"],
        linearly_profiled_coefficient_count=16,
        global_branch_uniqueness_proved=False,
        synthetic_image_derived_response_covariance=image_report is not None,
        image_report_path=str(image_report_path.relative_to(root)) if image_report_path is not None else None,
        image_report_sha256=digest(image_report_path) if image_report_path is not None else None,
        response_cholesky=cholesky.tolist(), accepted_normalized_dac=t.tolist(),
        practical_image_precision_demonstrated=False,
        assumptions=[("Image-fitted signed responses use their heterogeneous full per-setting covariance."
                      if image_report is not None else
                      "Conditional response data have independent 0.0001-pixel Gaussian uncertainty."),
            "Sixteen upstream coefficients are QR-fitted at every trial, not fixed to their generating values.",
            "The same likelihood is used across all declared sign/bias alternatives.",
            "Both input planes and detector pitch are calibrated exactly in this branch-search test.",
            "A local minimum with poor residual is not a global lower bound on that entire phase interval.",
            "Unconverged starts remain unresolved; omitted branches are not excluded.",
            "This model assumes a hard-edge last lens and affine excitation over the scan."])
    report["local_recovery_check_passed"] = bool(best is not None and closure < 1e-6 and
        diagnostic["numerical_full_parameter_rank"] == 19 and
        max(abs(value) for value in diagnostic["length_gap_errors_sigma"]) < 3)
    if any(digest(root/path) != checksum for path, checksum in hashes.items()):
        raise RuntimeError("A source or input changed during the branch search.")
    report["source_hashes_unchanged_during_execution"] = True
    report["elapsed_seconds"] = perf_counter()-started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=250)
    parser.add_argument("--image-report", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    report = run_audit(args.input, args.output, max_nfev=args.max_nfev, image_report_path=args.image_report)
    print(json.dumps({key: report[key] for key in ("local_recovery_check_passed", "start_count",
        "converged_start_count", "unresolved_start_count", "recovery_diagnostic",
        "nearest_tested_stationary_alternative_delta_chi_square", "elapsed_seconds")}, indent=2))
