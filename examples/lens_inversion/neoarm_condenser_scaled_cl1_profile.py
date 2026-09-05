"""Retry the same four CL1 profiles with a larger numerical trust-step unit.

Multiplying EVERY supplied residual by one positive constant preserves the
likelihood minimizer and relative data/prior weights. It changes the existing
solver's unit-information step cap: scale1e-3 permits1000 original noise units
per step instead of one. All public costs, gradients, residuals and Jacobians
are converted back; original-unit stationarity is independently checked.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import FunctionType

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from temgym_core.profiled_least_squares import profiled_least_squares
import neoarm_condenser_cl1_profile as experiment
from neoarm_aperture_capture import ROOT


COST_KEYS = frozenset(('chi_square', 'chi_square_change'))
GRADIENT_KEYS = frozenset(('normalized_gradient', 'normalized_nuisance_gradient',
    'whitened_nuisance_gradient', 'whitened_profile_gradient'))


def original_unit_history(value, scale):
    """Convert nested trial and inner histories, not only the final metric.

    Information-whitened Hessian eigenvalues, damping shifts, fractional steps
    and whitening closure errors are invariant under uniform residual scaling.
    """
    if isinstance(value, list):
        return [original_unit_history(item, scale) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in COST_KEYS:
            result[key] = float(item)/(scale*scale)
        elif key in GRADIENT_KEYS:
            result[key] = float(item)/scale
        else:
            result[key] = original_unit_history(item, scale)
    return result


def scaled_profiled_least_squares(residual, initial, admissible, *, scale=1e-3,
                                 gradient_tolerance=1e-6, **options):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('One finite positive uniform residual scale is required.')
    if not np.isfinite(gradient_tolerance) or not 0 < gradient_tolerance < 1:
        raise ValueError('The original gradient tolerance must lie between zero and one.')
    internal_tolerance = gradient_tolerance*scale
    if not 0 < internal_tolerance < 1:
        raise ValueError('The scaled gradient tolerance is outside the core solver contract.')
    scaled = lambda p: scale*residual(p)
    fit = profiled_least_squares(scaled, initial, admissible,
        gradient_tolerance=internal_tolerance, **options)
    internal_success = bool(fit['success'])
    internal_gradient = float(fit['normalized_gradient'])
    predicted_r = np.asarray(fit['residual'])/scale
    predicted_j = np.asarray(fit['jacobian'])/scale
    p = fit['parameters']
    # Recompute, rather than trusting a converted convergence flag. The
    # experiment also checks its original23-coordinate conditional gradient.
    original_r = np.asarray(jax.jit(residual)(p))
    original_j = np.asarray(jax.jit(jax.jacfwd(residual))(p))
    norms = np.linalg.norm(original_j, axis=0)
    gradient = float(np.max(abs((original_j/norms).T@original_r)))
    residual_closure = float(np.max(abs(original_r-predicted_r)))
    derivative_closure = float(np.linalg.norm(original_j-predicted_j)/max(1., np.linalg.norm(original_j)))
    finite = bool(np.isfinite(original_r).all() and np.isfinite(original_j).all() and np.all(norms > 0))
    accepted = bool(internal_success and finite and gradient < gradient_tolerance
                    and admissible(p) and residual_closure < 1e-5 and derivative_closure < 1e-8)
    fit['success'] = accepted
    if internal_success and not accepted:
        fit['reason'] = 'original-unit stationarity, admissibility or scaling closure failed'
    fit['residual'], fit['jacobian'], fit['normalized_gradient'] = original_r, original_j, gradient
    if fit.get('profile_jacobian') is not None:
        fit['profile_jacobian'] = np.asarray(fit['profile_jacobian'])/scale
    fit['history'] = original_unit_history(fit['history'], scale)
    fit['initial_nuisance_history'] = original_unit_history(fit['initial_nuisance_history'], scale)
    fit['unit_conversion'] = dict(uniform_residual_scale=float(scale),
        complete_residual_including_prior_scaled=True, original_gradient_tolerance=gradient_tolerance,
        internal_gradient_tolerance=internal_tolerance, internal_solver_success=internal_success,
        internal_normalized_gradient=internal_gradient, original_normalized_gradient=gradient,
        residual_conversion_closure_original_units=residual_closure,
        jacobian_conversion_relative_closure=derivative_closure,
        original_information_step_cap=1/scale,
        public_history_chi_square_and_gradients_in_original_units=True,
        information_whitened_Hessian_and_damping_invariant=True)
    return fit


def with_private_solver(function, solver):
    """Inject a callback into a private global namespace; mutate no old module."""
    namespace = dict(function.__globals__)
    namespace['profiled_least_squares'] = solver
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


def run(output, *, scale=1e-3):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('The profile retry requires GPU float64.')
    adapter_inputs = (Path(__file__), ROOT/'tests/test_neoarm_condenser_scaled_cl1_profile.py',
                      ROOT/'src/temgym_core/profiled_least_squares.py')
    adapter_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in adapter_inputs}
    conversions = []
    def solver(*args, **kwargs):
        fit = scaled_profiled_least_squares(*args, scale=scale, **kwargs)
        conversions.append(fit['unit_conversion'])
        return fit
    private_run = with_private_solver(experiment.run, solver)
    report = private_run(output, max_outer_iterations=20, max_inner_iterations=30)
    if len(conversions) != 4:
        raise RuntimeError('This retry must preserve exactly four optimizer calls and targets.')
    report['schema'] = 'neoarm-condenser-CL1-centre-scaled-step-profile-v1'
    report['optimizer_adapter'] = str(Path(__file__).relative_to(ROOT))
    report['uniform_residual_scale'] = scale
    report['solver_conversion_checks'] = conversions
    report['all_history_costs_and_gradients_in_original_units'] = True
    report['changed_relative_likelihood_weights'] = False
    report['scaled_entire_residual_including_radius_prior'] = True
    report['original_information_step_cap'] = 1/scale
    report['existing_module_globals_modified'] = False
    report['source_hashes'].update(adapter_hashes)
    for name, checksum in report['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'A bound input changed during the retry: {name}')
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scale', type=float, default=1e-3)
    args = parser.parse_args()
    run(args.output, scale=args.scale)
