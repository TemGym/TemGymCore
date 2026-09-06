"""Search alternative fits and profile physical geometry, with explicit failures.

Profile coordinates use lengths in mm so all requested geometry constraints
are linear. Positivity, aperture ordering and engineering bounds are enforced
as constraints, not as extra likelihood terms or soft priors.
"""

import json
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_simplification_design import QUANTITIES, physical_geometry
from scipy.linalg import null_space
from scipy.optimize import LinearConstraint, least_squares, minimize
from scipy.stats import chi2


def to_coordinates(parameters):
    p = np.asarray(parameters)
    x = p.copy()
    x[:9] = np.exp(p[:9]) * 1000
    x[19] = float(jax.nn.sigmoid(p[19])) * x[2]
    return x


def from_coordinates(coordinates):
    x = jnp.asarray(coordinates)
    return (
        x.at[:9].set(jnp.log(x[:9] / 1000)).at[19].set(jnp.log(x[19] / (x[2] - x[19])))
    )


def coordinate_geometry(coordinates):
    """Exactly linear physical quantities in the transformed coordinates."""
    x = jnp.asarray(coordinates)
    gaps, lengths = x[:5], x[5:9]
    source = -jnp.sum(x[:9])
    centres = (
        source
        + jnp.cumsum(gaps[:-1])
        + jnp.r_[0.0, jnp.cumsum(lengths[:-1])]
        + lengths / 2
    )
    aperture = centres[1] + lengths[1] / 2 + x[19]
    return jnp.r_[source, centres, lengths, aperture, x[20:22]]


def coordinate_bounds(lower, upper):
    lo, hi = np.asarray(lower).copy(), np.asarray(upper).copy()
    lo[:9], hi[:9] = np.exp(lo[:9]) * 1000, np.exp(hi[:9]) * 1000
    fraction_lo, fraction_hi = (
        float(jax.nn.sigmoid(lower[19])),
        float(jax.nn.sigmoid(upper[19])),
    )
    lo[19], hi[19] = fraction_lo * lo[2], fraction_hi * hi[2]
    matrix = np.eye(len(lo))
    aperture = np.zeros((2, len(lo)))
    aperture[0, [19, 2]] = [1.0, -fraction_lo]
    aperture[1, [19, 2]] = [-1.0, fraction_hi]
    return (
        lo,
        hi,
        np.r_[matrix, aperture],
        np.r_[lo, 0.0, 0.0],
        np.r_[hi, np.inf, np.inf],
    )


def search_alternatives(
    *,
    residual,
    starting_parameters,
    bounds_lower,
    bounds_upper,
    output_directory,
    max_evaluations=2000,
):
    """Fit every supplied valid guess and retain every result, including failures."""
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    derivative = jax.jit(jax.jacfwd(residual))
    lower, upper = np.asarray(bounds_lower), np.asarray(bounds_upper)
    records = []
    for index, guess in enumerate(starting_parameters):
        guess = np.asarray(guess)
        if np.any(guess <= lower) or np.any(guess >= upper):
            raise ValueError(
                "Starting guesses must be strictly inside declared physical bounds."
            )
        fit = least_squares(
            lambda p: np.asarray(residual(p)),
            guess,
            jac=lambda p: np.asarray(derivative(p)),
            bounds=(lower, upper),
            max_nfev=max_evaluations,
            x_scale="jac",
            ftol=1e-9,
            xtol=1e-11,
            gtol=1e-5,
        )
        record = {
            "index": index,
            "starting_parameters": guess.tolist(),
            "parameters": fit.x.tolist(),
            "geometry": np.asarray(physical_geometry(fit.x)).tolist(),
            "success": bool(fit.success),
            "message": str(fit.message),
            "evaluations": int(fit.nfev),
            "chi_squared": float(2 * fit.cost),
            "residual_rms": float(np.sqrt(np.mean(fit.fun**2))),
            "active_bounds": np.flatnonzero(fit.active_mask).tolist(),
        }
        records.append(record)
        (folder / f"start_{index:03d}.json").write_text(json.dumps(record, indent=2))
        print(
            f"Start {index + 1}: RMS {record['residual_rms']:.4g}, converged {fit.success}",
            flush=True,
        )
    converged = [r for r in records if r["success"]]
    best = min(converged or records, key=lambda r: r["chi_squared"])
    jac = np.asarray(derivative(np.asarray(best["parameters"])))
    singular = np.linalg.svd(
        jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-30), compute_uv=False
    )
    rank = int(np.sum(singular > singular[0] * 1e-9))
    _, raw_singular, vectors = np.linalg.svd(jac, full_matrices=False)
    covariance = (
        ((vectors.T / raw_singular) @ (vectors / raw_singular[:, None]))
        if rank == len(lower)
        else None
    )
    threshold = float(chi2.ppf(0.99, max(len(jac) - rank, 1)))
    delta = float(chi2.ppf(0.95, rank))
    for r in records:
        r["acceptable"] = (
            r["success"]
            and r["chi_squared"] <= threshold
            and r["chi_squared"] <= best["chi_squared"] + delta
        )
    result = {
        "best_fit": best,
        "starts": records,
        "rank": rank,
        "parameter_count": len(lower),
        "covariance": covariance.tolist() if covariance is not None else None,
        "bounds_lower": lower.tolist(),
        "bounds_upper": upper.tolist(),
        "acceptance": {
            "goodness_of_fit_limit": threshold,
            "delta_chi_squared_limit": delta,
            "scope": "Provisional chi-squared approximations conditional on the noise/model; validate with complete noise repetitions.",
        },
    }
    (folder / "search.json").write_text(json.dumps(result, indent=2))
    return result


@lru_cache(maxsize=3)
def _prepare_profile(residual, parameter_values, bounds_lower, bounds_upper):
    """Share compiled derivatives and coordinate scaling between all quantities."""
    best = np.asarray(parameter_values)
    centre = to_coordinates(best)
    lower, upper = np.asarray(bounds_lower), np.asarray(bounds_upper)
    lo, hi, constraints, constraint_lo, constraint_hi = coordinate_bounds(lower, upper)
    geometry_matrix = np.asarray(jax.jacfwd(coordinate_geometry)(centre))
    fraction_lo, fraction_hi = jax.nn.sigmoid(lower[19]), jax.nn.sigmoid(upper[19])

    def safe_parameters(x):
        # Guard trial evaluations outside the feasible set. The final result
        # must satisfy the actual linear constraints; clipping is not accepted
        # as a way of satisfying a requested physical target.
        x = jnp.clip(x, jnp.asarray(lo), jnp.asarray(hi))
        x = x.at[19].set(jnp.clip(x[19], fraction_lo * x[2], fraction_hi * x[2]))
        return from_coordinates(x)

    fn = jax.jit(lambda x: residual(safe_parameters(x)))
    jacobian = jax.jit(jax.jacfwd(fn))
    scale = np.maximum(np.abs(centre), 0.1)
    _, singular, vt = np.linalg.svd(
        np.asarray(jacobian(centre)) * scale, full_matrices=False
    )
    transform = scale[:, None] * vt.T / np.maximum(singular, 1e-3)
    matrix = constraints @ transform
    normalizers = np.maximum(np.linalg.norm(matrix, axis=1), 1e-20)
    inequalities = LinearConstraint(
        matrix / normalizers[:, None],
        (constraint_lo - constraints @ centre) / normalizers,
        (constraint_hi - constraints @ centre) / normalizers,
    )
    return (
        centre,
        constraints,
        constraint_lo,
        constraint_hi,
        geometry_matrix,
        fn,
        jacobian,
        transform,
        inequalities,
    )


def profile_physical_quantity(
    *,
    residual,
    fitted_parameters,
    bounds_lower,
    bounds_upper,
    quantity,
    targets,
    starting_parameters,
    output_directory,
    max_iterations=500,
):
    """Fix one physical quantity, refit all others from each supplied start.

    The SVD changes optimization coordinates only. It is not a prior. A profile
    point is marked valid only after checking its equality and all inequalities.
    """
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    index = QUANTITIES.index(quantity)
    best = np.asarray(fitted_parameters)
    (
        centre,
        constraints,
        constraint_lo,
        constraint_hi,
        geometry_matrix,
        fn,
        jacobian,
        transform,
        inequalities,
    ) = _prepare_profile(
        residual, tuple(best), tuple(bounds_lower), tuple(bounds_upper)
    )
    row = geometry_matrix[index]
    target_row = row @ transform
    target_scale = np.linalg.norm(target_row)
    null = null_space(target_row[None] / target_scale)
    free_transform = transform @ null
    reference_cost = float(np.sum(np.asarray(residual(best)) ** 2))
    points = []
    for target in targets:
        equality = LinearConstraint(
            target_row[None] / target_scale,
            [(target - row @ centre) / target_scale],
            [(target - row @ centre) / target_scale],
        )
        fixed = target_row * (target - row @ centre) / np.dot(target_row, target_row)
        fixed_centre = centre + transform @ fixed
        trials = []
        for start_index, start in enumerate(starting_parameters):
            initial = np.linalg.solve(transform, to_coordinates(start) - centre)
            initial += (
                target_row
                * (target - row @ centre - target_row @ initial)
                / np.dot(target_row, target_row)
            )

            def objective(v):
                x = centre + transform @ v
                errors = np.asarray(fn(x))
                gradient = (np.asarray(jacobian(x)) @ transform).T @ errors
                return 0.5 * np.dot(errors, errors), gradient

            # Eliminate the equality exactly for the fast Gauss-Newton step.
            # Bounds are checked explicitly; an infeasible or unfinished result
            # is passed to the solver with all linear inequalities enforced.
            least = least_squares(
                lambda w, fixed_centre=fixed_centre: np.asarray(
                    fn(fixed_centre + free_transform @ w)
                ),
                null.T @ (initial - fixed),
                jac=lambda w, fixed_centre=fixed_centre: (
                    np.asarray(jacobian(fixed_centre + free_transform @ w))
                    @ free_transform
                ),
                max_nfev=max_iterations,
                ftol=1e-9,
                xtol=1e-10,
                gtol=1e-5,
                x_scale="jac",
            )
            candidate = fixed + null @ least.x
            x = centre + transform @ candidate
            values = constraints @ x
            violation = max(
                float(np.max(constraint_lo - values)),
                float(np.max(values - constraint_hi)),
                0.0,
            )
            solver, steps = "equality-eliminated least squares", int(least.nfev)
            if least.success and violation < 1e-8:
                fit = least
            else:
                fit = minimize(
                    objective,
                    candidate if violation < 1e-8 else initial,
                    jac=True,
                    method="SLSQP",
                    constraints=[inequalities, equality],
                    options={"maxiter": max_iterations, "ftol": 1e-7},
                )
                x = centre + transform @ fit.x
                solver += "; constrained fallback"
                steps += int(fit.nit)
            values = constraints @ x
            violation = max(
                float(np.max(constraint_lo - values)),
                float(np.max(values - constraint_hi)),
                0.0,
            )
            target_error = float(abs(row @ x - target))
            feasible = (
                violation < 1e-8
                and target_error < 1e-8
                and np.all(x[:9] > 0)
                and 0 < x[19] < x[2]
            )
            parameters = np.asarray(from_coordinates(x)) if feasible else None
            cost = (
                float(np.sum(np.asarray(residual(parameters)) ** 2))
                if feasible
                else None
            )
            trials.append(
                {
                    "start_index": start_index,
                    "starting_parameters": np.asarray(start).tolist(),
                    "success": bool(fit.success and feasible),
                    "message": str(fit.message),
                    "iterations_or_evaluations": steps,
                    "solver": solver,
                    "constraint_violation": violation,
                    "target_error": target_error,
                    "parameters": parameters.tolist()
                    if parameters is not None
                    else None,
                    "chi_squared": cost,
                    "delta_chi_squared": cost - reference_cost
                    if cost is not None
                    else None,
                }
            )
        successful = [r for r in trials if r["success"]]
        chosen = min(successful, key=lambda r: r["chi_squared"]) if successful else None
        point = {
            "target": float(target),
            "trials": trials,
            "best": chosen,
            "resolved": bool(successful),
        }
        points.append(point)
        (folder / f"{quantity}.json").write_text(
            json.dumps(
                {
                    "quantity": quantity,
                    "points": points,
                    "scope": "All other parameters refitted from multiple starts; failed points do not bound uncertainty.",
                },
                indent=2,
            )
        )
        print(
            f"Profile {quantity} = {target:.6g}: {len(successful)}/{len(trials)} starts converged",
            flush=True,
        )
    return points


def physical_quantity(parameters, quantity):
    """A shared geometry quantity or the second voltage's independent source waist."""
    p = jnp.asarray(parameters)
    if quantity == "source_z_second_voltage_mm":
        return physical_geometry(p)[0] + 1000 * (jnp.exp(p[0]) - jnp.exp(p[22]))
    return physical_geometry(p)[QUANTITIES.index(quantity)]


@lru_cache(maxsize=32)
def _native_profile_functions(
    residual, parameter_values, lower_values, upper_values, quantity
):
    best, lower, upper = (
        np.asarray(parameter_values),
        np.asarray(lower_values),
        np.asarray(upper_values),
    )
    index = (
        12 if quantity == "source_z_second_voltage_mm" else QUANTITIES.index(quantity)
    )
    fixed_index = (
        22
        if index == 12
        else 0
        if index == 0
        else 4
        if index < 5 or index == 9
        else index
        if index < 9
        else index + 10
    )
    free = np.delete(np.arange(len(best)), fixed_index)

    def unpack(values, target):
        p = jnp.asarray(best).at[free].set(values)
        if index < 5 or index in (9, 12):
            drift = (
                jnp.exp(p[fixed_index])
                - (target - physical_quantity(p, quantity)) / 1000
            )
            value = jnp.log(
                jnp.clip(
                    drift, jnp.exp(lower[fixed_index]), jnp.exp(upper[fixed_index])
                )
            )
        elif index < 9:
            value = jnp.log(target / 1000)
        else:
            value = target
        return p.at[fixed_index].set(value)

    fn = jax.jit(lambda values, target: residual(unpack(values, target)))
    return free, unpack, fn, jax.jit(jax.jacfwd(fn, argnums=0))


def profile_native_quantity(
    *,
    residual,
    fitted_parameters,
    bounds_lower,
    bounds_upper,
    quantity,
    targets,
    starting_parameters,
    output_directory,
    max_evaluations=2000,
):
    """Profile by solving one geometry constraint explicitly in native parameters.

    Source z determines its entrance drift; a lens/aperture z determines the
    final specimen drift. A thickness or lateral centre fixes that parameter
    directly. All other parameters retain their ordinary bounded-fit coordinates.
    Trial drift clipping keeps the model physical, but a clipped result is never
    accepted unless it also satisfies the requested physical target exactly.
    """
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    best = np.asarray(fitted_parameters)
    lower, upper = np.asarray(bounds_lower), np.asarray(bounds_upper)
    free, unpack, fn, jac = _native_profile_functions(
        residual, tuple(best), tuple(lower), tuple(upper), quantity
    )
    reference_cost = float(np.sum(np.asarray(residual(best)) ** 2))
    points = []
    for target in targets:
        trials = []
        for start_index, start in enumerate(starting_parameters):
            start = np.asarray(start)
            if np.any(start < lower) or np.any(start > upper):
                raise ValueError("Profile starts must satisfy the physical bounds.")
            fit = least_squares(
                lambda v, target=target: np.asarray(fn(v, target)),
                start[free],
                jac=lambda v, target=target: np.asarray(jac(v, target)),
                bounds=(lower[free], upper[free]),
                x_scale="jac",
                max_nfev=max_evaluations,
                ftol=1e-9,
                xtol=1e-11,
                gtol=1e-5,
            )
            parameters = np.asarray(unpack(fit.x, target))
            target_error = abs(float(physical_quantity(parameters, quantity)) - target)
            feasible = (
                target_error < 1e-8
                and np.all(parameters >= lower - 1e-12)
                and np.all(parameters <= upper + 1e-12)
            )
            cost = float(np.sum(np.asarray(residual(parameters)) ** 2))
            trials.append(
                {
                    "start_index": start_index,
                    "starting_parameters": start.tolist(),
                    "parameters": parameters.tolist(),
                    "success": bool(fit.success and feasible),
                    "message": str(fit.message),
                    "target_error": float(target_error),
                    "chi_squared": cost,
                    "delta_chi_squared": cost - reference_cost,
                    "iterations_or_evaluations": int(fit.nfev),
                    "solver": "bounded native parameters; explicit geometry constraint",
                }
            )
        successful = [r for r in trials if r["success"]]
        points.append(
            {
                "target": float(target),
                "trials": trials,
                "best": min(successful, key=lambda r: r["chi_squared"])
                if successful
                else None,
                "resolved": bool(successful),
            }
        )
        (folder / f"{quantity}.json").write_text(
            json.dumps(
                {
                    "quantity": quantity,
                    "points": points,
                    "scope": "All other parameters refitted in native coordinates; unresolved targets do not bound uncertainty.",
                },
                indent=2,
            )
        )
        print(
            f"Native profile {quantity} = {target:.6g}: {len(successful)}/{len(trials)} starts converged",
            flush=True,
        )
    return points
