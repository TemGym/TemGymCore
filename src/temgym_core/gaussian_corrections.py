"""Diagnostics for the local quadratic approximation of Gaussian actions.

These helpers sample an optical component over the support of each Gaussian
beam and compare the sampled action with the quadratic Taylor model used by
the Gaussian propagation method.  They do not modify the beam; the reported
width scale is guidance for refining an input Gaussian decomposition.
"""

from __future__ import annotations

import inspect
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .gaussian import GaussianBeam, taylor_expand


class QuadraticResidual(NamedTuple):
    """Residual of a quadratic Taylor model over one Gaussian support."""

    max_abs: jnp.ndarray
    rms: jnp.ndarray
    points: jnp.ndarray
    residuals: jnp.ndarray


class GaussianActionSamplingDiagnostic(NamedTuple):
    """Per-beam phase and log-amplitude sampling diagnostics."""

    needs_refinement: jnp.ndarray
    phase_max_abs: jnp.ndarray
    phase_rms: jnp.ndarray
    amplitude_max_abs: jnp.ndarray
    amplitude_rms: jnp.ndarray
    phase_width_scale: jnp.ndarray
    amplitude_width_scale: jnp.ndarray
    recommended_width_scale: jnp.ndarray


def _as_single_gaussian(ray: GaussianBeam) -> GaussianBeam:
    if jnp.asarray(ray.r_xy).ndim > 1:
        raise ValueError("Expected a single Gaussian beam, got a batched bundle.")
    return ray


def _as_gaussian_bundle(ray: GaussianBeam) -> GaussianBeam:
    return ray.to_vector()


def _gaussian_bundle_size(ray: GaussianBeam) -> int:
    return int(jnp.asarray(ray.x).shape[0])


def _gaussian_bundle_d_xy(ray: GaussianBeam) -> jnp.ndarray:
    dx, dy = jnp.broadcast_arrays(jnp.asarray(ray.dx), jnp.asarray(ray.dy))
    return jnp.stack((dx, dy), axis=-1)


def _symmetrize_real(matrix: jnp.ndarray) -> jnp.ndarray:
    matrix = jnp.real(matrix)
    return 0.5 * (matrix + matrix.T)


def _symmetric_eigh_2x2(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Analytic eigendecomposition for a real symmetric 2 by 2 matrix."""

    matrix = _symmetrize_real(matrix)
    a = matrix[0, 0]
    b = matrix[0, 1]
    d = matrix[1, 1]
    delta = jnp.sqrt((a - d) ** 2 + 4.0 * b * b)
    eigvals = jnp.array(
        (0.5 * (a + d - delta), 0.5 * (a + d + delta)),
        dtype=matrix.dtype,
    )
    theta = 0.5 * jnp.atan2(2.0 * b, a - d)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    eigvec_small = jnp.array((-sin_t, cos_t), dtype=matrix.dtype)
    eigvec_large = jnp.array((cos_t, sin_t), dtype=matrix.dtype)
    return eigvals, jnp.stack((eigvec_small, eigvec_large), axis=1)


def gaussian_envelope_precision(ray: GaussianBeam) -> jnp.ndarray:
    """Return the real precision matrix of a single beam's field envelope."""

    ray = _as_single_gaussian(ray)
    return ray.k * _symmetrize_real(jnp.imag(ray.Q_inv))


def _support_frame(ray: GaussianBeam, *, floor: float = 1e-300):
    precision = gaussian_envelope_precision(ray)
    eigvals, eigvecs = _symmetric_eigh_2x2(precision)
    eigvals = jnp.maximum(eigvals, floor)
    radii = jnp.sqrt(2.0 / eigvals)
    return precision, eigvals, eigvecs, radii


def gaussian_1e_radii(ray: GaussianBeam, *, floor: float = 1e-300) -> jnp.ndarray:
    """Return principal-axis radii where the field envelope falls to ``1/e``."""

    _, _, _, radii = _support_frame(ray, floor=floor)
    return radii


def _default_support_directions(dtype):
    directions = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=dtype,
    )
    return directions / jnp.linalg.norm(directions, axis=1, keepdims=True)


def gaussian_support_offsets(
    ray: GaussianBeam,
    *,
    support_radius: float = 1.0,
    directions: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return offsets sampling the elliptical support of one Gaussian beam."""

    ray = _as_single_gaussian(ray)
    _, _, eigvecs, radii = _support_frame(ray)
    radii = support_radius * radii
    if directions is None:
        directions = _default_support_directions(radii.dtype)
    directions = jnp.asarray(directions, dtype=radii.dtype)
    return (directions * radii[None, :]) @ eigvecs.T


def _bundle_support_offsets(
    ray: GaussianBeam,
    *,
    support_radius: float = 1.0,
    directions: jnp.ndarray | None = None,
) -> jnp.ndarray:
    ray = _as_gaussian_bundle(ray)
    precision = ray.k[..., None, None] * jnp.imag(ray.Q_inv)
    eigvals, eigvecs = jax.vmap(_symmetric_eigh_2x2)(precision)
    eigvals = jnp.maximum(eigvals, 1e-300)
    radii = support_radius * jnp.sqrt(2.0 / eigvals)
    if directions is None:
        directions = _default_support_directions(radii.dtype)
    directions = jnp.asarray(directions, dtype=radii.dtype)
    scaled = directions[None, :, :] * radii[:, None, :]
    return jnp.einsum("ndi,nji->ndj", scaled, eigvecs)


def quadratic_residual_on_beam(
    fn: Callable[..., jnp.ndarray],
    ray: GaussianBeam,
    *args,
    scale: float | jnp.ndarray | None = None,
    support_radius: float = 1.0,
    directions: jnp.ndarray | None = None,
    complex_valued: bool = False,
) -> QuadraticResidual:
    """Measure a function's departure from its local quadratic model."""

    ray = _as_single_gaussian(ray)
    xy_ref = ray.r_xy
    offsets = gaussian_support_offsets(
        ray,
        support_radius=support_radius,
        directions=directions,
    )
    points = xy_ref[None, :] + offsets
    if complex_valued:
        d0, d1, d2 = taylor_expand(fn, xy_ref, *args)
    else:

        def scalar_fn(xy, *fn_args):
            return jnp.real(fn(xy, *fn_args))

        d0 = scalar_fn(xy_ref, *args)
        d1 = jax.grad(scalar_fn)(xy_ref, *args)
        d2 = jax.hessian(scalar_fn)(xy_ref, *args)

    if scale is None:
        scale = 1.0

    def residual_at(offset, point):
        model = d0 + jnp.dot(d1, offset) + 0.5 * jnp.dot(offset, d2 @ offset)
        return scale * (fn(point, *args) - model)

    residuals = jax.vmap(residual_at)(offsets, points)
    abs_residuals = jnp.abs(residuals)
    return QuadraticResidual(
        max_abs=jnp.max(abs_residuals),
        rms=jnp.sqrt(jnp.mean(abs_residuals**2)),
        points=points,
        residuals=residuals,
    )


def estimate_width_scale_for_residual(
    residual: float | jnp.ndarray,
    *,
    target: float = 0.25,
    order: int = 3,
    min_scale: float = 0.1,
    max_scale: float = 1.0,
) -> jnp.ndarray:
    """Estimate a beam-width reduction that brings a residual below target."""

    residual = jnp.asarray(residual)
    safe = jnp.maximum(residual, 1e-300)
    scale = (target / safe) ** (1.0 / order)
    scale = jnp.where(residual <= target, max_scale, scale)
    return jnp.clip(scale, min_scale, max_scale)


def _zero_bundle_metrics(ray: GaussianBeam):
    n = _gaussian_bundle_size(_as_gaussian_bundle(ray))
    dtype = jnp.result_type(jnp.asarray(ray.x), jnp.float32)
    zeros = jnp.zeros((n,), dtype=dtype)
    ones = jnp.ones((n,), dtype=dtype)
    return zeros, zeros, ones


def _bundle_quadratic_residuals(
    fn: Callable[..., jnp.ndarray],
    ray: GaussianBeam,
    *,
    fn_args: tuple[jnp.ndarray, ...] = (),
    scale: jnp.ndarray | float = 1.0,
    support_radius: float = 1.0,
    directions: jnp.ndarray | None = None,
    target: float = 0.25,
    min_width_scale: float = 0.1,
):
    ray = _as_gaussian_bundle(ray)
    x, y = jnp.broadcast_arrays(jnp.asarray(ray.x), jnp.asarray(ray.y))
    xy_ref = jnp.stack((x, y), axis=-1)
    offsets = _bundle_support_offsets(
        ray,
        support_radius=support_radius,
        directions=directions,
    )
    points = xy_ref[:, None, :] + offsets

    def scalar_fn(xy, *args):
        return jnp.real(fn(xy, *args))

    d0 = jax.vmap(scalar_fn)(xy_ref, *fn_args)
    d1 = jax.vmap(jax.grad(scalar_fn))(xy_ref, *fn_args)
    d2 = jax.vmap(jax.hessian(scalar_fn))(xy_ref, *fn_args)
    model = (
        d0[:, None]
        + jnp.einsum("ni,ndi->nd", d1, offsets)
        + 0.5 * jnp.einsum("ndi,nij,ndj->nd", offsets, d2, offsets)
    )
    scale = jnp.asarray(scale)
    if scale.ndim == 0:
        scale = jnp.broadcast_to(scale, (xy_ref.shape[0],))

    def fn_on_beam(beam_points, *beam_args):
        return jax.vmap(lambda point: fn(point, *beam_args))(beam_points)

    fn_points = jax.vmap(fn_on_beam)(points, *fn_args)
    residuals = scale[:, None] * (fn_points - model)
    abs_residuals = jnp.abs(residuals)
    max_abs = jnp.max(abs_residuals, axis=1)
    rms = jnp.sqrt(jnp.mean(abs_residuals**2, axis=1))
    width_scale = estimate_width_scale_for_residual(
        max_abs,
        target=target,
        min_scale=min_width_scale,
    )
    return max_abs, rms, width_scale


_SLOPE_PARAMETER_NAMES = frozenset(
    {"dxy", "d_xy", "slope", "slopes", "theta", "direction"}
)


def _legacy_phase_arguments(component, ray: GaussianBeam) -> tuple[jnp.ndarray, ...]:
    """Resolve beam-derived arguments required by a legacy ``phase_shift``."""

    phase_shift = component.phase_shift
    signature = inspect.signature(phase_shift)
    xy_probe = jnp.zeros((2,))
    try:
        signature.bind(xy_probe)
        return ()
    except TypeError:
        pass

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        parameter
        for parameter in positional[1:]
        if parameter.default is inspect.Parameter.empty
    ]

    def beam_argument(name: str):
        normalized = name.lower()
        if normalized in _SLOPE_PARAMETER_NAMES:
            return _gaussian_bundle_d_xy(ray)
        if normalized in {"voltage", "energy", "energy_ev"}:
            return jnp.asarray(ray.voltage)
        if normalized in {"z", "axial_position"}:
            return jnp.asarray(ray.z)
        if normalized in {"sigma", "interaction_constant"}:
            return jnp.asarray(ray.sigma)
        if normalized in {"k", "wavenumber"}:
            return jnp.asarray(ray.k)
        if normalized in {"wavelength", "lam", "lambda_"}:
            return jnp.asarray(ray.wavelength)
        raise KeyError(name)

    try:
        arguments = tuple(beam_argument(parameter.name) for parameter in required)
        signature.bind(xy_probe, *(argument[0] for argument in arguments))
    except (KeyError, TypeError) as resolution_error:
        names = ", ".join(parameter.name for parameter in required)
        raise TypeError(
            "Cannot derive legacy phase_shift arguments "
            f"({names}) from GaussianBeam fields. Define pathlength_shift or "
            "pathlength_shift_with_slope to make the diagnostic contract explicit."
        ) from resolution_error
    return arguments


def _pathlength_action(component, ray: GaussianBeam):
    """Return an action callable and per-beam arguments in compatibility order."""

    action = getattr(component, "pathlength_shift_with_slope", None)
    if callable(action):
        return action, (_gaussian_bundle_d_xy(ray),)

    action = getattr(component, "pathlength_shift", None)
    if callable(action):
        return action, ()

    action = getattr(component, "phase_shift", None)
    if callable(action):
        return action, _legacy_phase_arguments(component, ray)

    return None, ()


def diagnose_gaussian_action_sampling(
    component,
    ray: GaussianBeam,
    *,
    phase_tol: float = 0.25,
    amplitude_tol: float = 0.05,
    support_radius: float = 1.0,
    directions: jnp.ndarray | None = None,
    min_width_scale: float = 0.1,
    log_floor: float = -50.0,
    check_phase: bool = True,
    check_amplitude: bool = True,
) -> GaussianActionSamplingDiagnostic:
    """Estimate whether a component is locally resolved by each Gaussian.

    The component's pathlength action and log-transmission are compared with
    their local quadratic Taylor models over every beam's support. Phase
    residuals are returned in radians; amplitude residuals are in natural
    log-amplitude units.

    Pathlength action names are resolved in this order:
    ``pathlength_shift_with_slope``, ``pathlength_shift``, then the local
    legacy name ``phase_shift``.
    """

    ray = _as_gaussian_bundle(ray)
    if check_phase:
        action, action_args = _pathlength_action(component, ray)
    else:
        action, action_args = None, ()

    if action is not None:
        phase_max_abs, phase_rms, phase_width_scale = _bundle_quadratic_residuals(
            action,
            ray,
            fn_args=action_args,
            scale=ray.k,
            support_radius=support_radius,
            directions=directions,
            target=phase_tol,
            min_width_scale=min_width_scale,
        )
    else:
        phase_max_abs, phase_rms, phase_width_scale = _zero_bundle_metrics(ray)

    log_transmission_fn = getattr(component, "log_transmission", None)
    if check_amplitude and callable(log_transmission_fn):

        def log_transmission(xy):
            return jnp.real(jnp.logaddexp(log_transmission_fn(xy), log_floor))

        amplitude_max_abs, amplitude_rms, amplitude_width_scale = (
            _bundle_quadratic_residuals(
                log_transmission,
                ray,
                support_radius=support_radius,
                directions=directions,
                target=amplitude_tol,
                min_width_scale=min_width_scale,
            )
        )
    else:
        amplitude_max_abs, amplitude_rms, amplitude_width_scale = _zero_bundle_metrics(ray)

    needs_refinement = (phase_max_abs > phase_tol) | (amplitude_max_abs > amplitude_tol)
    recommended_width_scale = jnp.minimum(phase_width_scale, amplitude_width_scale)
    return GaussianActionSamplingDiagnostic(
        needs_refinement=needs_refinement,
        phase_max_abs=phase_max_abs,
        phase_rms=phase_rms,
        amplitude_max_abs=amplitude_max_abs,
        amplitude_rms=amplitude_rms,
        phase_width_scale=phase_width_scale,
        amplitude_width_scale=amplitude_width_scale,
        recommended_width_scale=recommended_width_scale,
    )


__all__ = [
    "GaussianActionSamplingDiagnostic",
    "QuadraticResidual",
    "diagnose_gaussian_action_sampling",
    "estimate_width_scale_for_residual",
    "gaussian_1e_radii",
    "gaussian_envelope_precision",
    "gaussian_support_offsets",
    "quadratic_residual_on_beam",
]
