"""Conditional geometry-constrained profile of the 19-coordinate PL1 relaxation.

This retains the image-derived correlated response likelihood. A fixed-cofactor
chart enforces positive calibrated four-volume while fitting the total last-lens
span explicitly. The default span cap of 0.9 m is a conditional gross geometry
allowance, NOT independently verified NEOARM metrology. No generating upstream
map or generating length is supplied to an initializer.

The coefficient model is still a relaxation: positive volume does not enforce
magnetic symplecticity or round optics. Competing minima reported here are not
automatically physically realizable electron-optical instruments.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from neoarm_last_lens_branch_search import ProfiledResponse, distinct_solution
from neoarm_last_lens_tomography import predict_response_parameters, geometry_coordinates, J4
from temgym_core.jax_least_squares import least_squares


ROOT = Path(__file__).resolve().parents[2]
BRANCH_REPORT = ROOT/"examples/generated/neoarm_inversion/imaging_recipe/last_lens_image_branches.json"


def _det3(matrix):
    return jnp.dot(matrix[0], jnp.cross(matrix[1], matrix[2]))


def _det4(matrix):
    # Polynomial form remains differentiable when the pivot-zero matrix is
    # singular; do not differentiate a determinant via an inverse there.
    return sum((-1.)**column*matrix[0, column]*_det3(
        matrix[1:, jnp.array([i for i in range(4) if i != column])]) for column in range(4))


@dataclass(frozen=True)
class VolumeChart:
    """19 coordinates: log span, log s, q0, q', and fifteen W entries.

    One omitted W entry is solved using a fixed nonzero cofactor. Charts are
    chosen once per start, never switched within JIT or a least-squares loss.
    A singular cofactor produces NaNs and an unresolved fit, not a clipped
    determinant or a silently changed chart.
    """

    pivot: int
    calibrated_column_product: float

    def __post_init__(self):
        if not isinstance(self.pivot, (int, np.integer)) or not 0 <= self.pivot < 16:
            raise ValueError("Choose a row-major pivot index in 0..15.")
        if not np.isfinite(self.calibrated_column_product) or self.calibrated_column_product <= 0:
            raise ValueError("The four calibrated input-step/pixel factors need positive product.")

    @property
    def free_indices(self):
        return tuple(index for index in range(16) if index != self.pivot)

    def cofactor(self, matrix):
        row, column = divmod(self.pivot, 4)
        rows, columns = jnp.array([i for i in range(4) if i != row]), jnp.array([i for i in range(4) if i != column])
        minor = matrix[jnp.ix_(rows, columns)]
        value = (-1.)**(row+column)*_det3(minor)
        scale = jnp.prod(jnp.linalg.norm(minor, axis=1))
        return value, scale

    @classmethod
    def from_response_parameters(cls, parameters, input_steps, pixel_m):
        matrix = np.asarray(parameters)[3:].reshape(4, 4)
        cofactors = np.array([[(-1.)**(r+c)*np.linalg.det(np.delete(np.delete(matrix, r, 0), c, 1))
                               for c in range(4)] for r in range(4)])
        pivot = int(np.argmax(np.abs(cofactors)))
        if not np.isfinite(cofactors.ravel()[pivot]) or cofactors.ravel()[pivot] == 0:
            raise ValueError("No nonsingular cofactor chart exists at this seed.")
        return cls(pivot, float(np.prod(np.asarray(input_steps)/pixel_m)))

    def encode(self, parameters, *, span_m=None):
        p = np.asarray(parameters)
        if p.shape != (19,) or not np.all(np.isfinite(p)):
            raise ValueError("Expected nineteen finite response coordinates.")
        sign, logdet = np.linalg.slogdet(p[3:].reshape(4, 4))
        if sign <= 0:
            raise ValueError("A positive-volume seed is required.")
        gap = np.exp(.5*(logdet-np.log(self.calibrated_column_product)))
        span = gap*(1+np.exp(-p[0])) if span_m is None else float(span_m)
        if not np.isfinite(span) or span <= 0:
            raise ValueError("The seed span must be positive and finite.")
        return np.r_[np.log(span), p[:3], p[3:][list(self.free_indices)]]

    def decode(self, coordinates):
        theta = jnp.asarray(coordinates)
        if theta.shape != (19,):
            raise ValueError("The unfixed volume chart has nineteen coordinates.")
        span, ratio = jnp.exp(theta[0]), jnp.exp(theta[1])
        gap = span*jax.nn.sigmoid(theta[1])
        target = gap**2*self.calibrated_column_product
        matrix = jnp.zeros(16, dtype=theta.dtype).at[jnp.asarray(self.free_indices)].set(theta[4:]).reshape(4, 4)
        cofactor, scale = self.cofactor(matrix)
        # Determinant is affine in the pivot. Evaluation at zero has the same
        # fixed cofactor and avoids solving a nonlinear scalar equation.
        determinant_at_zero = _det4(matrix)
        valid = (jnp.all(jnp.isfinite(theta)) & jnp.isfinite(span) & jnp.isfinite(ratio)
                 & (span > 0) & (ratio > 0) & jnp.isfinite(cofactor)
                 & (scale > 0) & (jnp.abs(cofactor) > 32*jnp.finfo(theta.dtype).eps*scale))
        pivot_value = (target-determinant_at_zero)/jnp.where(valid, cofactor, jnp.nan)
        matrix = matrix.reshape(-1).at[self.pivot].set(pivot_value)
        return jnp.concatenate((theta[1:4], matrix))


def magnetic_twoform_diagnostic(parameters, input_steps, pixel_m):
    """Geometry and deviation from F.T J F = T(kappa).T J T(kappa).

    This raw defect has no attached significance test. Measurement noise must
    be propagated or a physical upstream model refitted before rejecting it.
    """
    p = np.asarray(parameters)
    geometry = np.asarray(geometry_coordinates(jnp.asarray(p), jnp.asarray(input_steps), pixel_m))
    length, gap, ratio = np.exp(geometry[:3])
    w = p[3:].reshape(2, 2, 4)/(np.asarray(input_steps)/pixel_m)
    upstream = np.concatenate((-w[0]/ratio, w[1]/length))
    omega = upstream.T@np.asarray(J4)@upstream
    kappa = (omega[1, 0]-omega[0, 1])/4
    target = np.asarray(J4).copy()
    target[:2, :2] = 2*kappa*np.array([[0., -1.], [1., 0.]])
    ell = float(np.sqrt(np.prod(np.asarray(input_steps)[:2])/np.prod(np.asarray(input_steps)[2:])))
    scale = np.diag([np.sqrt(ell)]*2+[1/np.sqrt(ell)]*2)
    defect = scale.T@(omega-target)@scale
    return dict(length_m=float(length), detector_gap_m=float(gap), total_span_m=float(length+gap),
        reconstructed_upstream_mechanical=upstream.tolist(), coefficient_volume_sign=float(np.linalg.slogdet(p[3:].reshape(4, 4))[0]),
        upstream_determinant=float(np.linalg.det(upstream)), inferred_kappa_m_inv=float(kappa),
        sample_pullback_twoform=omega.tolist(), canonical_balance_length_m=ell,
        balanced_twoform_defect=defect.tolist(), maximum_balanced_twoform_defect=float(np.max(np.abs(defect))),
        magnetic_symplectic_upstream_enforced=False, physical_realizability_established=False)


def fit_chart(problem, seed, chart, *, span_bounds_m, ratio_bounds, q_bounds, slope_bounds,
              fixed_ratio=None, max_nfev=250):
    initial = chart.encode(seed)
    initial[0] = min(initial[0], np.log(.95*span_bounds_m[1]))
    # Preserve a data-derived feasible span exactly; only inadmissible seeds
    # are moved into the predeclared gross box, then the pivot is recomputed.
    lower = np.r_[np.log(span_bounds_m[0]), np.log(ratio_bounds[0]), q_bounds[0], slope_bounds[0], np.full(15, -1e6)]
    upper = np.r_[np.log(span_bounds_m[1]), np.log(ratio_bounds[1]), q_bounds[1], slope_bounds[1], np.full(15, 1e6)]
    if fixed_ratio is not None:
        initial[1] = np.log(fixed_ratio)
        active = np.r_[0, np.arange(2, 19)]
        expand = lambda value: jnp.zeros(19).at[1].set(np.log(fixed_ratio)).at[jnp.asarray(active)].set(value)
        initial, lower, upper = initial[active], lower[active], upper[active]
    else:
        expand = lambda value: value
    residual = lambda value: problem.whiten((predict_response_parameters(chart.decode(expand(value)), problem.controls)
                                            -problem.observed).reshape(len(problem.controls), 8, 1)).reshape(-1)
    jac = np.asarray(jax.jacfwd(residual)(jnp.asarray(initial)))
    norms = np.linalg.norm(jac, axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        return dict(optimizer_success=False, status='unresolved_initial_chart', fixed_ratio=fixed_ratio,
                    pivot=chart.pivot, initial_coordinates=initial.tolist())
    fit = least_squares(residual, initial, bounds=(lower, upper), x_scale=1/norms,
        max_nfev=max_nfev, max_iter=500, ftol=1e-11, xtol=1e-11, gtol=1e-7)
    theta = expand(fit.x)
    response = np.asarray(chart.decode(theta))
    cofactor, cofactor_scale = chart.cofactor(jnp.asarray(response[3:].reshape(4, 4)))
    gap = float(jnp.exp(theta[0])*jax.nn.sigmoid(theta[1]))
    target = gap**2*chart.calibrated_column_product
    relative_volume_error = float(np.linalg.det(response[3:].reshape(4, 4))/target-1)
    finite = bool(np.all(np.isfinite(response)) and abs(relative_volume_error) < 1e-8)
    return dict(optimizer_success=bool(fit.success and finite), termination=fit.status,
        status='tested_local_minimum' if fit.success and finite else 'unresolved_optimizer_or_chart',
        free_coordinate_count=len(initial), fixed_ratio=fixed_ratio, pivot=chart.pivot,
        initial_coordinates=initial.tolist(), fitted_chart_coordinates=np.asarray(theta).tolist(),
        parameters=response.tolist(), chi_square=float(np.sum(np.asarray(fit.residual)**2)),
        nfev=fit.nfev, coefficient_volume_relative_error=relative_volume_error,
        final_relative_cofactor=float(abs(cofactor)/cofactor_scale),
        active_bounds=np.flatnonzero(np.minimum(abs(np.asarray(fit.x)-lower), abs(np.asarray(fit.x)-upper)) < 1e-6).tolist())


def run_audit(output, *, maximum_span_m=.9, profile_points=25, max_nfev=250):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Run the differentiable response model on GPU float64.')
    if not np.isfinite(maximum_span_m) or maximum_span_m <= .001 or profile_points < 3:
        raise ValueError('A positive gross span cap and at least three profile points are required.')
    started = perf_counter()
    branch = json.loads(BRANCH_REPORT.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, checksum in branch['source_hashes'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Stale branch input: {name}')
    paths = [ROOT/name for name in branch['source_hashes']]+[BRANCH_REPORT, Path(__file__)]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    image = json.loads((ROOT/branch['image_report_path']).read_text())
    calibration = json.loads((ROOT/branch['input_path']).read_text())['inputs']
    input_steps, pixel = np.asarray(calibration['input_basis_steps_grouped']), calibration['detector_pixel_m']
    with np.load(ROOT/image['arrays_path'], allow_pickle=False) as saved:
        problem = ProfiledResponse(saved['accepted_normalized_dac'], saved['response'],
                                   np.linalg.cholesky(saved['response_covariance']))
    best = branch['best']
    rivals = [row for row in branch['rows'] if row['optimizer_success'] and row['positive_coefficient_volume']
              and distinct_solution(row['parameters'], best['parameters'])]
    rival = min(rivals, key=lambda row: row['chi_square'])
    seeds = [np.asarray(best['parameters']), np.asarray(rival['parameters'])]
    charts = [VolumeChart.from_response_parameters(seed, input_steps, pixel) for seed in seeds]
    q_bounds = tuple(best['q_bias_interval'])
    slope_bounds = tuple((best['lower'][2], best['upper'][2]))
    arguments = dict(span_bounds_m=(.001, maximum_span_m), ratio_bounds=(.7, 40.),
                     q_bounds=q_bounds, slope_bounds=slope_bounds, max_nfev=max_nfev)
    free = [fit_chart(problem, seed, chart, **arguments) for seed, chart in zip(seeds, charts)]
    ratios = np.unique(np.r_[np.geomspace(.7, 40., profile_points), np.exp(seeds[0][0]), np.exp(seeds[1][0])])
    rows = []
    for ratio in ratios:
        fits = []
        for index, (seed, chart) in enumerate(zip(seeds, charts)):
            fit = fit_chart(problem, seed, chart, fixed_ratio=float(ratio), **arguments)
            fit['seed_family'] = ('best_unconstrained', 'closest_unconstrained_rival')[index]
            if 'parameters' in fit:
                fit['geometry_and_twoform'] = magnetic_twoform_diagnostic(fit['parameters'], input_steps, pixel)
            fits.append(fit)
        successes = [fit for fit in fits if fit['optimizer_success']]
        selected = min(successes, key=lambda row: row['chi_square']) if successes else None
        rows.append(dict(ratio=float(ratio), fits=fits, best_converged=selected,
                         all_starts_converged=all(fit['optimizer_success'] for fit in fits)))
        print('ratio', ratio, 'chi2', None if selected is None else selected['chi_square'],
              'converged', len(successes), '/2', flush=True)
    for fit in free:
        if 'parameters' in fit:
            fit['geometry_and_twoform'] = magnetic_twoform_diagnostic(fit['parameters'], input_steps, pixel)
    if any(digest(ROOT/name) != checksum for name, checksum in hashes.items()):
        raise RuntimeError('Sources changed during constrained profiling.')
    report = dict(schema='neoarm-last-lens-conditional-geometry-profile-v1', source_hashes=hashes,
        source_hashes_unchanged_during_execution=True, backend=jax.default_backend(), jax_version=jax.__version__,
        x64_enabled=True, elapsed_seconds=perf_counter()-started,
        conditional_span_bounds_m=list(arguments['span_bounds_m']), geometry_allowance_independently_verified=False,
        ratio_bounds=list(arguments['ratio_bounds']), q_bias_bounds=list(q_bounds), q_slope_bounds=list(slope_bounds),
        calibrated_input_steps_grouped=input_steps.tolist(), detector_pixel_m=pixel,
        observed_response_count=int(problem.observed.size), reference_unconstrained_chi_square=best['chi_square'],
        unconstrained_seed_diagnostics=[magnetic_twoform_diagnostic(seed, input_steps, pixel) for seed in seeds],
        free_ratio_fits=free, fixed_ratio_profiles=rows, generating_optics_used_for_initialization=False,
        unrestricted_coordinate_count=19, fixed_ratio_coordinate_count=18,
        magnetic_symplectic_upstream_enforced=False, round_upstream_enforced=False,
        physical_alternative_proved=False, global_uniqueness_proved=False,
        assumptions=['The 0.9 m default cap is a conditional gross allowance, not measured instrument geometry.',
            'The exact same correlated Gaussian response likelihood is used; calibration is fixed in this test.',
            'The full 19-coordinate positive-volume model is retained; fixed-s profiles have 18 free coordinates.',
            'Determinant constraints invalidate unconstrained QR filtering, so all remaining coefficients are refitted.',
            'The upstream matrix is a volume-preserving relaxation, not yet magnetic-symplectic or round.',
            'Two starting families and finite profile sampling do not certify global minima or exclude omitted branches.',
            'Failed points remain unresolved; no likelihood cutoff or Wilks significance is assumed.'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-span-m', type=float, default=.9)
    parser.add_argument('--profile-points', type=int, default=25)
    parser.add_argument('--max-nfev', type=int, default=250)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    result = run_audit(args.output, maximum_span_m=args.maximum_span_m,
                       profile_points=args.profile_points, max_nfev=args.max_nfev)
    print('Saved', args.output, 'in', result['elapsed_seconds'], 'seconds', flush=True)
