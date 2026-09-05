"""Retry the unchanged condenser likelihood in a fixed-reference source chart.

The acquisition plane, controls and numerical width are frozen from the
previous training-only rough-geometry initializer. No source measurement or
additional likelihood prior is introduced. Original-coordinate stationarity
is checked again after every transformed fit.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np

from neoarm_condenser_extension_design import load_context, FOLDER
from neoarm_condenser_extension_recovery import run
from neoarm_condenser_covariance_fit import VisibleSourceModel
from neoarm_condenser_reference_source import ReferenceSourceChart
from neoarm_condenser_newton_refinement import refine_training
from temgym_core.profiled_least_squares import profiled_least_squares


def fixed_reference_chart():
    """Choose units from a data-only source seed, never from generator values."""
    _, arrays, _, _, model = load_context()
    quotient = VisibleSourceModel(model)
    prior = json.loads((FOLDER/"condenser_guarded_recovery.json").read_text())
    initial = next(row["initial_parameters"] for row in prior["rows"]
                   if row["initializer"] == "rough_geometry_data_source")
    control = arrays["controls"][0]
    optical, source, _, _ = quotient.decode(jnp.asarray(initial))
    matrix = model.condenser_matrix(optical, jnp.asarray(control))
    covariance = np.asarray(matrix@source@matrix.T)
    width = float((covariance[0, 0]*covariance[2, 2])**.25)
    return ReferenceSourceChart(quotient, control, width)


def source_reference_solver(residual, initial, admissible, *, max_iterations=30,
                            method="newton", max_inner_iterations=30):
    """Fit through an invertible chart; verify the original statistical problem."""
    chart = fixed_reference_chart()
    start = np.asarray(chart.encode(initial))
    function = jax.jit(residual)
    transformed = lambda p: residual(chart.decode(p))
    closure = float(np.max(abs(np.asarray(transformed(start))-np.asarray(function(initial)))))
    if not np.isfinite(closure) or closure > 1e-4:
        raise ValueError("The reference chart must preserve the complete likelihood.")
    domain = lambda p: admissible(np.asarray(chart.decode(p)))
    if method == "newton":
        fit = refine_training(transformed, start, domain, max_iterations=max_iterations)
    elif method == "profiled":
        fit = profiled_least_squares(transformed, start, domain, n_outer=12,
            max_iterations=max_iterations, max_inner_iterations=max_inner_iterations)
    else:
        raise ValueError("Select newton or profiled source-reference fitting.")
    result = np.asarray(chart.decode(fit["parameters"]))
    r = np.asarray(function(result))
    jacobian = np.asarray(jax.jit(jax.jacfwd(residual))(result))
    normalized = float(np.max(abs((jacobian/np.linalg.norm(jacobian, axis=0)).T@r)))
    final_closure = float(np.max(abs(r-np.asarray(fit["residual"]))))
    success = bool(fit["success"] and normalized < 1e-6 and admissible(result)
                   and final_closure < 1e-4)
    reason = fit["reason"] if success or not fit["success"] else "original-coordinate stationarity or closure failed"
    history = [dict(source_reference_control=np.asarray(chart.reference_control).tolist(),
                    source_reference_width_m=chart.reference_width_m,
                    coordinate_change_only=True, initial_residual_closure_sigma=closure,
                    final_residual_closure_sigma=final_closure,
                    transformed_solver=method, transformed_success=bool(fit["success"]))]
    history.extend(fit["history"])
    # Count successful trial evaluations for the diagnostic Newton adapter;
    # the profiled implementation already records its complete evaluation count.
    evaluations = fit.get("evaluations", len(fit["history"])+sum(
        sum(bool(t.get("feasible")) for t in row.get("line_search", [])) for row in fit["history"]))
    return dict(parameters=result, residual=r, jacobian=jacobian, success=success,
                reason=reason, normalized_gradient=normalized,
                evaluations=evaluations, history=history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("newton", "profiled"), default="newton")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-inner-iterations", type=int, default=30)
    args = parser.parse_args()
    run(args.inference, args.output, max_iterations=args.max_iterations,
        solver=source_reference_solver,
        solver_options=dict(method=args.method, max_inner_iterations=args.max_inner_iterations),
        include_rough_data_start=True)
