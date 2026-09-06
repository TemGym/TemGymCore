"""Fit measured specimen beams and expose the two exact condenser ambiguities."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from condenser_calibration_model import (
    equivalent_parameters,
    initial_parameters,
    matrix5,
    named_parameters,
    specimen_moments,
)
from scipy.optimize import least_squares


def beam_to_moments(beam):
    size2 = jnp.exp(2 * beam[4])
    h = beam[5] * 1e4
    spread2 = jnp.exp(2 * beam[6])
    return jnp.r_[
        beam[:4], size2 / 1e-14, h * size2 / 1e-10, (h * h * size2 + spread2) / 1e-6
    ]


def model_moments(parameters, controls):
    mean, cov = specimen_moments(parameters, controls)
    return jnp.r_[
        mean[:2] / 1e-7,
        mean[2:] / 1e-3,
        cov[0, 0] / 1e-14,
        cov[0, 2] / 1e-10,
        cov[2, 2] / 1e-6,
    ]


def fit_condenser(
    *,
    measurements_path,
    output_directory,
    number_of_starts=12,
    random_seed=293,
    max_evaluations=5000,
    starting_parameters=None,
):
    """Fit every identifiable direction and retain the exact two-dimensional family.

    To solve a singular problem numerically, one representative uses the
    initial source-to-CL1 entrance drift and initial pre-field strength. These
    are declared coordinate choices, not measurements or asserted truths.
    Equivalent family members release both of them without changing the data.
    No soft prior is used. Reported physical values must retain this qualifier.
    """
    if not jax.config.x64_enabled:
        raise ValueError(
            "Enable JAX 64-bit precision before fitting: jax.config.update('jax_enable_x64', True)"
        )
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    data = np.load(measurements_path)
    beam_values, beam_covariances = data["values"], data["covariance"]
    convert = jax.jit(jax.vmap(beam_to_moments))
    derivative = jax.jit(jax.vmap(jax.jacfwd(beam_to_moments)))
    values = np.asarray(convert(beam_values))
    transforms = np.asarray(derivative(beam_values))
    covariance = transforms @ beam_covariances @ transforms.transpose(0, 2, 1)
    whitening = jnp.asarray(np.linalg.inv(np.linalg.cholesky(covariance)))
    controls = jnp.asarray(data["controls"])
    predict = jax.jit(jax.vmap(model_moments, in_axes=(None, 0)))

    def residual(p):
        return jnp.einsum(
            "nij,nj->ni", whitening, predict(p, controls) - values
        ).ravel()

    residual = jax.jit(residual)
    jacobian = jax.jit(jax.jacfwd(residual))
    p0 = (
        initial_parameters()
        if starting_parameters is None
        else np.asarray(starting_parameters, dtype=float)
    )
    lower = p0.copy()
    upper = p0.copy()
    lower[:9] -= np.log(2.5)
    upper[:9] += np.log(2.5)
    lower[9:13] -= np.log(1.8)
    upper[9:13] += np.log(1.8)
    lower[17:] = np.log([1e-9, 10e-6])
    upper[17:] = np.log([80e-9, 300e-6])
    free = np.array([i for i in range(13) if i not in (0, 12)])
    # For fixed optics the source mean and its two variances enter linearly.
    # Solve those six coefficients exactly within each optical iteration.
    # Source variances are constrained to the declared physical bounds.
    scales = jnp.array([1e-7, 1e-7, 1e-3, 1e-3])

    def source_design(p, command):
        transfer = matrix5(p, command)[:4, :4]
        design = (
            jnp.zeros((7, 6))
            .at[:4, :4]
            .set(
                transfer
                * jnp.array([1e-8, 1e-8, 1e-4, 1e-4])[None, :]
                / scales[:, None]
            )
        )
        for column, (part, factor) in enumerate(
            [(transfer[:, :2], 1e-16), (transfer[:, 2:], 1e-8)]
        ):
            cov = part @ part.T * factor
            design = design.at[4:, 4 + column].set(
                jnp.array([cov[0, 0] / 1e-14, cov[0, 2] / 1e-10, cov[2, 2] / 1e-6])
            )
        return design

    target = jnp.einsum("nij,nj->ni", whitening, values).ravel()

    def profile(v):
        p = jnp.asarray(p0).at[free].set(v)
        design = jax.vmap(source_design, in_axes=(None, 0))(p, controls)
        design = jnp.einsum("nij,njk->nik", whitening, design).reshape(-1, 6)
        # Eliminate the four unrestricted source means first. Then enumerate
        # the interior/edges/corners of a two-variable bounded least squares.
        mean_design, variance_design = design[:, :4], design[:, 4:]
        q, r = jnp.linalg.qr(mean_design, mode="reduced")
        projected_design = variance_design - q @ (q.T @ variance_design)
        projected_target = target - q @ (q.T @ target)
        lo = jnp.exp(2 * jnp.asarray(lower[17:])) / jnp.array([1e-16, 1e-8])
        hi = jnp.exp(2 * jnp.asarray(upper[17:])) / jnp.array([1e-16, 1e-8])
        normal = projected_design.T @ projected_design
        rhs = projected_design.T @ projected_target
        candidates = [jnp.linalg.solve(normal, rhs)]
        for index in (0, 1):
            other = 1 - index
            for boundary in (lo[index], hi[index]):
                value = jnp.clip(
                    (rhs[other] - normal[other, index] * boundary)
                    / normal[other, other],
                    lo[other],
                    hi[other],
                )
                candidates.append(
                    jnp.zeros(2).at[index].set(boundary).at[other].set(value)
                )
        candidates = jnp.stack(candidates)
        costs = jnp.sum(
            (candidates @ projected_design.T - projected_target) ** 2, axis=1
        )
        valid = jnp.all((candidates >= lo) & (candidates <= hi), axis=1)
        variances = candidates[jnp.argmin(jnp.where(valid, costs, jnp.inf))]
        means = jnp.linalg.solve(r, q.T @ (target - variance_design @ variances))
        coefficients = jnp.r_[means, variances]
        p = (
            p.at[13:17]
            .set(means)
            .at[17:]
            .set(0.5 * jnp.log(variances * jnp.array([1e-16, 1e-8])))
        )
        return p, design @ coefficients - target

    profile = jax.jit(profile)
    reduced = jax.jit(lambda v: profile(v)[1])
    reduced_jacobian = jax.jit(jax.jacfwd(reduced))
    rng = np.random.default_rng(random_seed)
    records = []
    for index in range(number_of_starts):
        start = p0.copy()
        if index:
            start[:13] += rng.normal(0, 0.07, 13)
            start[13:17] += rng.normal(0, 0.02, 4)
            start[17:] += rng.normal(0, 0.15, 2)
        fit = least_squares(
            lambda v: np.asarray(reduced(v)),
            start[free],
            jac=lambda v: np.asarray(reduced_jacobian(v)),
            bounds=(lower[free], upper[free]),
            x_scale="jac",
            max_nfev=max_evaluations,
            ftol=1e-7,
            xtol=1e-10,
            gtol=1e-5,
        )
        p = np.asarray(profile(fit.x)[0])
        rms = float(np.sqrt(np.mean(fit.fun**2)))
        record = {
            "index": index,
            "success": bool(fit.success),
            "message": str(fit.message),
            "data_rms": rms,
            "chi_squared": float(2 * fit.cost),
            "parameters": p.tolist(),
            "model": named_parameters(p),
            "evaluations": int(fit.nfev),
            "active_parameters": free[np.flatnonzero(fit.active_mask)].tolist(),
        }
        records.append(record)
        (folder / f"start_{index:02d}.json").write_text(json.dumps(record, indent=2))
        print(
            f"Condenser start {index + 1}/{number_of_starts}: RMS {rms:.3f}, success={fit.success}",
            flush=True,
        )
    acceptable = [r for r in records if r["success"] and r["data_rms"] < 1.5]
    if not acceptable:
        raise RuntimeError("No acceptable condenser fit; inspect the saved starts.")
    best = min(acceptable, key=lambda r: r["chi_squared"])
    p = np.asarray(best["parameters"])
    _u, s, _vt = np.linalg.svd(jacobian(p), full_matrices=False)
    family = []
    for scale, q in [(1.0, -np.exp(p[12])), (0.8, -0.35), (1.2, -0.6), (1.3, -0.6)]:
        alt = equivalent_parameters(p, upstream_scale=scale, objective_strength=q)
        difference = np.asarray(predict(alt, controls) - predict(p, controls))
        family.append(
            {
                "upstream_scale": scale,
                "objective_strength": q,
                "parameters": alt.tolist(),
                "model": named_parameters(alt),
                "maximum_normalized_moment_change": float(np.max(abs(difference))),
            }
        )
    report = {
        "best_fit": best,
        "starts": records,
        "acceptable_starts": len(acceptable),
        "number_of_starts": len(records),
        "agreeing_starts": sum(
            r["chi_squared"] <= best["chi_squared"] + 1.0 for r in acceptable
        ),
        "agreement_definition": "Data chi-squared within 1 of the best; this does not assert identical parameters",
        "singular_values": s.tolist(),
        "data_rank": int(np.sum(s > s[0] * 1e-10)),
        "numerical_representative": {
            "source_to_CL1_drift_m": float(np.exp(p0[0])),
            "objective_prefield_strength": float(-np.exp(p0[12])),
            "meaning": "Coordinate choices only; both remain physically unknown",
        },
        "equivalent_models": family,
        "use_soft_priors": False,
        "image_count": int(len(controls) * 5),
        "optical_parameter_count": 13,
        "source_parameter_count": 6,
        "optimizer_settings": {
            "max_evaluations": max_evaluations,
            "ftol": 1e-7,
            "xtol": 1e-10,
            "gtol": 1e-5,
        },
        "bounds": {
            "optical_lower": lower[:13].tolist(),
            "optical_upper": upper[:13].tolist(),
            "source_width_lower": lower[17:].tolist(),
            "source_width_upper": upper[17:].tolist(),
            "source_means": "Unbounded linear fit",
        },
    }
    (folder / "fit_summary.json").write_text(json.dumps(report, indent=2))
    return report
