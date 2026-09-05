"""Reproduce one failed condenser step; never run an optical fitting campaign."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_extension_recovery import selected_data, beam_residual
from neoarm_condenser_extension_design import load_context, training_residual, FOLDER
from neoarm_condenser_covariance_fit import VisibleSourceModel, quantum_completion
from neoarm_aperture_capture import ROOT
from temgym_core.feasible_least_squares import _guard_acceleration, svd_gauss_newton


NATIVE_SCALES = np.r_[np.tile([.02, .1, .05, .1], 3),
    [.1, 1., .1, 1., .1, .1, .1, 1., .1], .01, .005]


def audit(output):
    if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
        raise RuntimeError('Use GPU float64 for the actual saved-step diagnostic.')
    report_path = FOLDER/'condenser_extension_recovery.json'
    original = json.loads(report_path.read_text())
    row = next(r for r in original['rows'] if r['initializer'] == 'old_training_iterate')
    if row['optimizer_success'] or row['termination'] != 'physical feasible line search failed':
        raise ValueError('This diagnostic requires the preserved original failed checkpoint.')
    old, arrays, repair, aperture, model = load_context()
    quotient = VisibleSourceModel(model)
    arrays_path = ROOT/original['arrays_path']
    with np.load(arrays_path, allow_pickle=False) as data:
        fresh = selected_data(data, True)
    old_loss = training_residual(quotient, arrays, aperture)
    new_loss = beam_residual(quotient, fresh)
    residual = jax.jit(lambda p: jnp.r_[old_loss(p), new_loss(p)])
    p = np.asarray(row['parameters'])
    r = np.asarray(residual(p))
    if not np.isclose(r@r, row['total_chi_square'], rtol=1e-12, atol=1e-8):
        raise RuntimeError('The reconstructed likelihood differs from the saved checkpoint.')
    jac = np.asarray(jax.jit(jax.jacfwd(residual))(p))
    previous = row['history'][-2]
    radius = previous['trust_radius']
    if previous['step_fraction'] < .5 or previous['step_quality'] < .25:
        radius *= .25
    elif previous['step_quality'] > .75:
        if previous['damping'] <= 0:
            raise ValueError('Cannot infer the old radius-growth test from an undamped saved step.')
        radius *= 2
    left, singular, right = np.linalg.svd(jac*NATIVE_SCALES, full_matrices=False)
    projected = left.T@r
    step_at = lambda damping: -(right.T@((singular*projected)/(singular**2+damping)))
    damping = 0.
    if np.linalg.norm(step_at(0.)) > radius:
        low, high = 0., max(singular[0]**2, 1.)
        for _ in range(80):
            middle = (low+high)/2
            if np.linalg.norm(step_at(middle)) > radius: low = middle
            else: high = middle
        damping = high
    natural = step_at(damping)
    base = natural*NATIVE_SCALES
    second = np.asarray(jax.jit(lambda value, direction: jax.jvp(
        lambda x: jax.jvp(residual, (x,), (direction,))[1], (value,), (direction,))[1])(p, base))
    acceleration = -(right.T@((singular*(left.T@second))/(singular**2+damping)))*NATIVE_SCALES
    ratio = float(np.linalg.norm(acceleration/NATIVE_SCALES)/np.linalg.norm(natural))
    corrected = base+.5*acceleration if ratio < 1. else base
    protected, guard, _ = _guard_acceleration(jac, r, base, acceleration, NATIVE_SCALES, radius)
    scans = []
    for name, direction in [('base_damped_gauss_newton', base), ('old_geodesic', corrected)]:
        directional = float(r@jac@direction)
        trials = []
        for power in range(35):
            alpha = 2.**(-power)
            candidate = p+alpha*direction
            feasible = quotient.admissible(candidate)
            record = dict(halvings=power, step_fraction=alpha, feasible=feasible)
            if feasible:
                trial = np.asarray(residual(candidate))
                change = float(trial@trial-r@r)
                armijo = 2e-4*alpha*directional
                record.update(chi_square_change=change, armijo_required_change=armijo,
                              accepted_by_old_line_search=bool(np.isfinite(trial).all() and change <= armijo))
            trials.append(record)
        scans.append(dict(direction=name, step=direction.tolist(), directional_derivative=directional,
            full_step_predicted_reduction=float(-2*directional-np.sum((jac@direction)**2)), trials=trials))
    check = svd_gauss_newton(residual, p, quotient.admissible, max_iterations=1, native_scales=NATIVE_SCALES)
    paths = [Path(__file__), report_path, arrays_path,
        ROOT/'src/temgym_core/feasible_least_squares.py', ROOT/'tests/test_feasible_least_squares.py',
        ROOT/'examples/lens_inversion/neoarm_condenser_extension_recovery.py',
        ROOT/'examples/lens_inversion/neoarm_condenser_extension_design.py',
        ROOT/'examples/lens_inversion/neoarm_condenser_covariance_fit.py']
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(schema='neoarm-condenser-line-search-diagnostic-v1',
        source_hashes={str(path.relative_to(ROOT)): digest(path) for path in paths},
        source_scope='Direct diagnostic, original failed checkpoint, selected-image input and solver sources/tests.',
        backend=jax.default_backend(), x64_enabled=True, parameters=p.tolist(), native_scales=NATIVE_SCALES.tolist(),
        original_chi_square=float(r@r), original_normalized_gradient=row['normalized_gradient'],
        quantum_margin=quantum_completion(p[12:21])['margin'], reconstructed_trust_radius=radius,
        reconstructed_damping=float(damping), natural_step_norm=float(np.linalg.norm(natural)),
        geodesic_acceleration_ratio=ratio, guarded_acceleration_decision=guard,
        guarded_direction_equals_base=bool(np.array_equal(protected, base)), line_scans=scans,
        safeguarded_one_iteration=dict(chi_square=float(check['residual']@check['residual']),
            success=check['success'], termination=check['reason'], history=check['history']),
        nonlinear_campaign_repeated=False, truth_parameters_used=False,
        findings=[
            'The old accepted geodesic correction changes a descent direction into an ascent direction despite its small parameter-space norm.',
            'A feasible Armijo-decreasing base Gauss--Newton step exists at the same saved state.',
            'The new guard rejects non-descent/nonpositive model reductions, retries the base direction, and never computes trust quality from a nonpositive denominator.',
            'This diagnoses a numerical failure; one improving step does not establish successful optical recovery.'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    result = audit(args.output)
    print(result['guarded_acceleration_decision'], result['original_chi_square'],
          result['safeguarded_one_iteration']['chi_square'], flush=True)
