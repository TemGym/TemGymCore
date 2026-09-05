"""Two-analyser reconstruction of an upstream ray map, including calibration error.

Two calibrated last-lens settings measure all four rows of the upstream map
using position measurements alone. This is an initialization/diagnostic step,
not a substitute for a joint optical fit. The shared analyser uncertainty
correlates every reconstructed setting and must not be counted independently.
Coordinates are grouped ``[x, y, theta_x, theta_y]``; input positions are metres,
angles radians, and measured central-difference responses are detector pixels.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .microscope_inversion import drift_matrix, hard_edge_solenoid_matrix


def analyser_trace(ray, parameters, control):
    """Trace through a hard-edge lens and detector gap in grouped coordinates.

    ``parameters`` are ``[log(length_m), log(gap_m), q_bias, q_slope]``.
    The control is a declared local scan coordinate, not a hardware command.
    """
    length, gap = jnp.exp(parameters[:2])
    q = parameters[2] + parameters[3] * control
    order = jnp.array([0, 2, 1, 3])
    interleaved = drift_matrix(gap) @ hard_edge_solenoid_matrix(q, length)
    return (interleaved @ ray[order])[order]


def analyser_rows(parameters, control):
    """Position Jacobian of one traced input ray, not numerical ray differences."""
    return jax.jacfwd(analyser_trace, argnums=0)(
        jnp.zeros(4, dtype=jnp.asarray(parameters).dtype), parameters, control
    )[:2]


def pair_matrix(parameters, controls):
    """Stack position rows from two analyser states into a square observation map."""
    return jax.vmap(analyser_rows, in_axes=(None, 0))(parameters, controls).reshape(4, 4)


def predict_response_pair(upstream, parameters, controls, input_steps, pixel_m):
    """Predict two signed 2×4 response images for an arbitrary upstream map."""
    return (pair_matrix(parameters, controls) @ upstream *
            (jnp.asarray(input_steps) / pixel_m)[None, :]).reshape(2, 2, 4)


def reconstruct_response_pair(responses, parameters, controls, input_steps, pixel_m):
    """Invert one nondegenerate pair; use the checked scan adapter for admission."""
    physical = jnp.asarray(responses).reshape(4, 4) * (
        pixel_m / jnp.asarray(input_steps))[None, :]
    return jnp.linalg.solve(pair_matrix(parameters, controls), physical)


def _covariance(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size, size) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite ({size},{size}) covariance.")
    scale = max(float(np.max(np.abs(value))), np.finfo(float).tiny)
    if not np.allclose(value, value.T, atol=1e-12*scale, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric.")
    value = (value + value.T) / 2
    if np.linalg.eigvalsh(value).min() < -1e-10*scale:
        raise ValueError(f"{name} must be positive semidefinite.")
    return value


@dataclass(frozen=True)
class ReconstructedMapScan:
    maps: np.ndarray
    measurement_covariance: np.ndarray
    analyser_jacobian: np.ndarray
    shared_analyser_covariance: np.ndarray
    response_analyser_cross_term: np.ndarray
    joint_covariance: np.ndarray
    balanced_analyser_condition_number: float


def reconstruct_map_scan(responses, parameters, controls, input_steps, pixel_m,
                         response_covariance, analyser_covariance, *,
                         calibration_data_independent,
                         response_analyser_cross_covariance=None,
                         balance_length_m=.01, maximum_condition_number=1e6):
    """Recover maps with a joint delta-method covariance across all settings.

    Responses have shape ``(N,2,2,4)``. Their covariance is a full ``16N×16N``
    matrix in that flattening order, permitting shared references and repeated
    exposures. Analyser calibration covariance is ``4×4`` in the declared
    parameter coordinates. If calibration reuses response data, a ``16N×4``
    cross-covariance is required; independence cannot be silently assumed.

    The returned uncertainty is first order, not a nonlinear posterior or a
    guarantee of identifiability. Input/pixel calibration uncertainties are
    not included here; release those shared inputs in the subsequent joint fit.
    """
    data, p, t, steps = map(np.asarray, (responses, parameters, controls, input_steps))
    if data.ndim != 4 or data.shape[1:] != (2, 2, 4) or len(data) == 0:
        raise ValueError("Responses must have nonempty shape (N,2,2,4).")
    if p.shape != (4,) or t.shape != (2,) or steps.shape != (4,):
        raise ValueError("Expected four analyser parameters, two controls, four input steps.")
    if not all(np.all(np.isfinite(a)) for a in (data, p, t, steps)):
        raise ValueError("Responses and physical inputs must be finite.")
    if (np.any(steps <= 0) or not np.isfinite(pixel_m) or pixel_m <= 0 or
            not np.isfinite(balance_length_m) or balance_length_m <= 0 or
            not np.isfinite(maximum_condition_number) or maximum_condition_number <= 1):
        raise ValueError("Positive physical scales and a finite condition limit >1 are required.")
    matrix = np.asarray(pair_matrix(jnp.asarray(p), jnp.asarray(t)))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Analyser matrix must be finite at the declared parameters.")
    balanced = matrix * np.array([1., 1., 1/balance_length_m, 1/balance_length_m])
    condition = float(np.linalg.cond(balanced))
    if not np.isfinite(condition) or condition > maximum_condition_number:
        raise ValueError("Analyser pair is singular or too poorly conditioned at the declared scale.")
    count = data.size
    c_data = _covariance(response_covariance, count, "Response covariance")
    c_cal = _covariance(analyser_covariance, 4, "Analyser covariance")
    if response_analyser_cross_covariance is None:
        if not calibration_data_independent:
            raise ValueError("Reused calibration data require response/analyser cross-covariance.")
        cross = np.zeros((count, 4))
    else:
        cross = np.asarray(response_analyser_cross_covariance)
        if cross.shape != (count, 4) or not np.all(np.isfinite(cross)):
            raise ValueError("Response/analyser cross-covariance must have shape (16N,4).")
        if calibration_data_independent and np.any(cross != 0):
            raise ValueError("Independent calibration cannot have nonzero cross-covariance.")
    _covariance(np.block([[c_data, cross], [cross.T, c_cal]]), count+4,
                "Joint input covariance")

    def recover(flat_data, cal):
        return jax.vmap(reconstruct_response_pair, in_axes=(0, None, None, None, None))(
            flat_data.reshape(data.shape), cal, jnp.asarray(t), jnp.asarray(steps), pixel_m
        ).reshape(-1)

    flat_data, cal = jnp.asarray(data.ravel()), jnp.asarray(p)
    maps = np.asarray(recover(flat_data, cal)).reshape(-1, 4, 4)
    j_data, j_cal = map(np.asarray, jax.jacfwd(recover, argnums=(0, 1))(flat_data, cal))
    measurement = j_data @ c_data @ j_data.T
    shared = j_cal @ c_cal @ j_cal.T
    cross_term = j_data @ cross @ j_cal.T
    cross_term = cross_term + cross_term.T
    joint = measurement + shared + cross_term
    return ReconstructedMapScan(maps, measurement, j_cal, shared, cross_term,
                                (joint+joint.T)/2, condition)
