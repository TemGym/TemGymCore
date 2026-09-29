"""Optical components for rays and coherent Gaussian beams.

Read Component for dispatch, then Lens for a simple physical example.
The file groups planes, lenses, steering, phase/amplitude elements and specimens.
GaussianActionComponent shares the local optical-path/transmission update.

Use component(beam) at its plane or run_to_end(beam, components) for a column.
Positions use the beam length unit; slopes are radians. Parameters annotated
Degrees use degrees; parameters ending in _rad use radians. See each component
for finite-body propagation and any ray/Gaussian support limitations.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax_dataclasses as jdc
from interpax import Interpolator2D, Interpolator3D
from jax import lax
from jax.nn import softplus

from . import CoordsXY, Degrees, ScaleYX, ShapeYX
from .gaussian import (
    LENGTH,
    FreeSpacePropagator,
    GaussianBeam,
    _solve_2x2,
    _pupil_action_to_spatial,
    apply_action_delta,
    taylor_expand,
)
from .aberrations import (
    KrivanekCoeffs,
    SeidelCoeffs,
    Seidel_object_pos_aperture_pos,
    W_krivanek,
    grad_W_krivanek,
    zernike_noll_cartesian,
)
from .constants import (
    compute_Rc_from_voltage,
    effective_accelerating_potential,
    energy2wavelength,
)
from .grid import Grid
from .ray import Ray
from .tree_utils import HasParamsMixin


# Component interface and Gaussian action

class Component(HasParamsMixin):
    """A component transforms a Ray or GaussianBeam at its current plane.

    Call component(beam); dispatch uses beam.ray_family. The runner handles
    free-space travel to component.z. A finite component handles its own body
    propagation. Unsupported beam families raise NotImplementedError.
    """

    def __call__(self, ray: Ray) -> Ray:
        family = getattr(ray, "ray_family", "ray")
        if family == "gaussian":
            return self._call_gaussian(ray)
        if family == "ray":
            return self._call_ray(ray)
        raise TypeError(
            f"Unsupported ray family '{family}' for {type(self).__name__}."
        )

    def _call_ray(self, ray: Ray) -> Ray:
        raise NotImplementedError(
            f"{type(self).__name__} is not implemented for ray_family='ray'."
        )

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        raise NotImplementedError(
            f"{type(self).__name__} is not implemented for ray_family='gaussian'."
        )


def _call_single_or_batched_gaussian(ray: GaussianBeam, single_call):
    if jnp.asarray(ray.r_xy).ndim > 1:
        return jax.vmap(single_call)(ray.to_vector())
    return single_call(ray)


def _default_action_terms(xy_ref: jnp.ndarray):
    return (
        jnp.zeros((), dtype=xy_ref.dtype),
        jnp.zeros_like(xy_ref),
        jnp.zeros((2, 2), dtype=xy_ref.dtype),
    )


def _value_grad_hessian(fn, x: jnp.ndarray):
    value, grad = jax.value_and_grad(fn)(x)
    return value, grad, jax.hessian(fn)(x)


def _uses_default_log_transmission(component: "GaussianActionComponent") -> bool:
    return type(component).log_transmission is GaussianActionComponent.log_transmission


def _min_eigval_2x2(M: jnp.ndarray) -> jnp.ndarray:
    """Minimum eigenvalue of a real symmetric 2×2 matrix."""
    tr = M[..., 0, 0] + M[..., 1, 1]
    # Difference form avoids cancellation at a round Gaussian. At an exact
    # repeated eigenvalue choose the symmetric subgradient (half the trace).
    # Guard the sqrt input itself: masking sqrt(0) afterwards still produces
    # 0 * inf in autodiff, even when the envelope regularizer is inactive.
    disc = (M[..., 0, 0] - M[..., 1, 1]) ** 2 + (M[..., 0, 1] + M[..., 1, 0]) ** 2
    distinct = disc > 0
    root = jnp.sqrt(jnp.where(distinct, disc, 1.))
    return 0.5 * (tr - jnp.where(distinct, root, 0.))


def _regularize_L2k(
    L2k: jnp.ndarray,
    ImQ: jnp.ndarray,
    min_frac: float = 0.1,
) -> jnp.ndarray:
    """Ensure ``ImQ - L2k`` stays positive definite.

    If the minimum eigenvalue of ``ImQ - L2k`` would drop below
    ``min_frac * min_eigval(ImQ)``, a diagonal shift is subtracted
    from *L2k* to bring it back within bounds.
    """
    M = ImQ - L2k
    min_eig_M = _min_eigval_2x2(M)
    min_eig_Q = _min_eigval_2x2(ImQ)
    required = min_frac * min_eig_Q
    shift = jnp.maximum(required - min_eig_M, 0.0)
    return L2k - shift[..., None, None] * jnp.eye(2, dtype=L2k.dtype)


def _apply_gaussian_action(ray, path_terms, log_amplitude_terms=None):
    """Apply local value/gradient/Hessian terms from AD or specimen interpolation."""
    dS0, dS1, dS2 = path_terms
    if log_amplitude_terms is not None:
        log_value, log_gradient, log_hessian = log_amplitude_terms
        k = ray.k
        im_q = jnp.imag(ray.Q_inv)
        # Limit the inferred shift toward higher transmission, then keep the
        # updated Gaussian envelope decaying. Both specimen paths use this rule.
        estimated_shift = _solve_2x2(im_q, log_gradient) / k
        scale = 1.0 / (1.0 + jnp.dot(log_gradient, estimated_shift))
        log_gradient = log_gradient * scale
        log_hessian = log_hessian * scale
        log_hessian_over_k = _regularize_L2k(log_hessian / k, im_q)
        dS0 = dS0 - 1j * (log_value / k)
        dS1 = dS1 - 1j * (log_gradient / k)
        dS2 = dS2 - 1j * log_hessian_over_k

    r_xy, d_xy, amplitude, pathlength, q_new = apply_action_delta(
        ray, dS0=dS0, dS1=dS1, dS2=dS2
    )
    return ray.derive(
        x=r_xy[..., 0], y=r_xy[..., 1],
        dx=d_xy[..., 0], dy=d_xy[..., 1], z=ray.z,
        amplitude=amplitude, pathlength=pathlength, Q_inv=q_new,
    )


class GaussianActionComponent(Component):
    """Base for thin elements described by optical path and field transmission.

    Despite its name, phase_shift(xy) returns optical path, in the beam's
    length unit. log_transmission(xy) returns log field amplitude. Together:

        field multiplier = exp(i k path + log_amplitude)
        complex action  = path - i log_amplitude / k

    Values, gradients and Hessians update amplitude, beam centre and shape.
    The shared action helper limits absorption-driven recentering and keeps
    the Gaussian envelope decaying. Subclasses can provide exact quadratic
    updates or interpolated derivatives instead of automatic differentiation.
    """

    def phase_shift(self, xy: jnp.ndarray):
        return 0.0

    def log_transmission(self, xy: jnp.ndarray):
        return 0.0

    def complex_action(self, xy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy) - 1j * (L / k)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        return _call_single_or_batched_gaussian(ray, self._call_gaussian_single)

    def _call_gaussian_single(self, ray: GaussianBeam) -> GaussianBeam:
        xy = ray.r_xy
        if type(self).phase_shift is GaussianActionComponent.phase_shift:
            path_terms = _default_action_terms(xy)
        else:
            path_terms = _value_grad_hessian(lambda p: jnp.real(self.phase_shift(p)), xy)

        log_terms = None
        if not _uses_default_log_transmission(self):
            def log_transmission(p):
                return jnp.real(jnp.logaddexp(self.log_transmission(p), -50.0))

            log_terms = _value_grad_hessian(log_transmission, xy)
        return _apply_gaussian_action(ray, path_terms, log_terms)


def _matvec(matrix: jnp.ndarray, vector: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...j->...i", matrix, vector)


def _matmat(matrix: jnp.ndarray, tensor: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...jk->...ik", matrix, tensor)


def _rotate_quadratic_form(rotation: jnp.ndarray, q_inv: jnp.ndarray) -> jnp.ndarray:
    """Rotate Q consistently with r' = R r: Q' = R Q R.T."""
    return jnp.einsum("...ij,...jk,...lk->...il", rotation, q_inv, rotation)


# Planes and detectors

@jdc.pytree_dataclass
class Plane(Component):
    """A marker plane; propagation to z is performed by the runner."""

    z: float

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray


@jdc.pytree_dataclass
class ScanGrid(Component, Grid):
    """Scan-plane geometry; calling it leaves the beam unchanged."""

    z: float
    pixel_size: ScaleYX
    shape: ShapeYX
    rotation: Degrees = 0.0
    centre: CoordsXY = (0.0, 0)
    flip_y: bool = False

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray


@jdc.pytree_dataclass
class Detector(Component, Grid):
    """Detector geometry. Calling it marks a plane; rendering is separate."""

    z: float
    pixel_size: ScaleYX
    shape: ShapeYX
    rotation: Degrees = 0.0
    centre: CoordsXY = (0.0, 0)
    flip_y: bool = False

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray


# Lenses and matrix transport

@jdc.pytree_dataclass
class Lens(GaussianActionComponent):
    """Thin round lens; focal_length, x0 and y0 use the beam length unit."""

    z: float
    focal_length: float
    x0: float = 0.0
    y0: float = 0.0

    def _call_ray(self, ray: Ray):
        f = self.focal_length
        x = ray.x - self.x0 * ray._one
        y = ray.y - self.y0 * ray._one
        dx, dy = ray.dx, ray.dy
        new_dx = -x / f + dx
        new_dy = -y / f + dy
        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)
        return ray.derive(dx=new_dx, dy=new_dy, pathlength=pathlength)

    def _call_gaussian(self, ray: GaussianBeam):
        if (
            type(self).phase_shift is not Lens.phase_shift
            or not _uses_default_log_transmission(self)
        ):
            return GaussianActionComponent._call_gaussian(self, ray)

        x = ray.x - self.x0 * ray._one
        y = ray.y - self.y0 * ray._one
        focal_length = jnp.asarray(self.focal_length, dtype=jnp.float64)
        q_delta = (-1.0 / focal_length)[..., None, None] * jnp.eye(
            2, dtype=jnp.asarray(ray.Q_inv).dtype
        )
        return ray.derive(
            dx=ray.dx - x / focal_length,
            dy=ray.dy - y / focal_length,
            pathlength=ray.pathlength
            - (x * x + y * y) / (2.0 * focal_length),
            Q_inv=ray.Q_inv + q_delta,
        )

    def phase_shift(self, xy: jnp.ndarray):  # type: ignore # noqa
        x, y = xy[..., 0] - self.x0, xy[..., 1] - self.y0
        rho2 = x * x + y * y
        return -0.5 * rho2 / self.focal_length


@jdc.pytree_dataclass
class Stigmator(GaussianActionComponent):
    """Anisotropic thin lens with independent focal lengths in x and y."""

    z: float
    focal_length_x: float
    focal_length_y: float
    x0: float = 0.0
    y0: float = 0.0

    def _call_ray(self, ray: Ray):
        fx = self.focal_length_x
        fy = self.focal_length_y
        x = ray.x - self.x0 * ray._one
        y = ray.y - self.y0 * ray._one
        dx, dy = ray.dx, ray.dy

        new_dx = -x / fx + dx
        new_dy = -y / fy + dy
        pathlength = ray.pathlength - (x**2) / (2 * fx) - (y**2) / (2 * fy)
        return ray.derive(dx=new_dx, dy=new_dy, pathlength=pathlength)

    def _call_gaussian(self, ray: GaussianBeam):
        if (
            type(self).phase_shift is not Stigmator.phase_shift
            or not _uses_default_log_transmission(self)
        ):
            return GaussianActionComponent._call_gaussian(self, ray)

        x = ray.x - self.x0 * ray._one
        y = ray.y - self.y0 * ray._one
        focal_length_x = jnp.asarray(self.focal_length_x, dtype=jnp.float64)
        focal_length_y = jnp.asarray(self.focal_length_y, dtype=jnp.float64)
        powers = jnp.stack((-1.0 / focal_length_x, -1.0 / focal_length_y), axis=-1)
        q_delta = powers[..., :, None].astype(jnp.asarray(ray.Q_inv).dtype) * jnp.eye(
            2, dtype=jnp.asarray(ray.Q_inv).dtype
        )
        return ray.derive(
            dx=ray.dx - x / focal_length_x,
            dy=ray.dy - y / focal_length_y,
            pathlength=ray.pathlength
            - x * x / (2.0 * focal_length_x)
            - y * y / (2.0 * focal_length_y),
            Q_inv=ray.Q_inv + q_delta,
        )

    def phase_shift(self, xy: jnp.ndarray):
        x = xy[..., 0] - self.x0
        y = xy[..., 1] - self.y0
        return -0.5 * (x * x / self.focal_length_x + y * y / self.focal_length_y)


@jdc.pytree_dataclass
class KrivanekLens(Lens):
    """Thin lens with Krivanek aberrations.

    The Gaussian path expands a local optical action. The legacy ray path
    instead corrects slopes about the origin and replaces pathlength with
    the aberration value; it does not accumulate the ideal-lens optical path.
    """

    coeffs: KrivanekCoeffs = dataclasses.field(default_factory=KrivanekCoeffs)
    axis_eps: float = 1e-24

    def _call_gaussian(self, ray: GaussianBeam):
        # The aberration needs the generic local-action expansion rather than
        # Lens's exact quadratic fast path.
        return GaussianActionComponent._call_gaussian(self, ray)

    def _call_ray(self, ray: Ray):
        f = self.focal_length
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        ideal_dx = -x / f + dx
        ideal_dy = -y / f + dy

        alpha = jnp.hypot(ideal_dx, ideal_dy)
        phi = jnp.arctan2(ideal_dy, ideal_dx)

        dWx, dWy = grad_W_krivanek(ideal_dx, ideal_dy, self.coeffs)
        dux, duy = -dWx / f, -dWy / f

        aber_dx = ideal_dx + dux
        aber_dy = ideal_dy + duy
        pathlength = W_krivanek(alpha, phi, self.coeffs)

        return ray.derive(dx=aber_dx, dy=aber_dy, pathlength=pathlength)

    def phase_shift(self, xy: jnp.ndarray):
        x = xy[..., 0] - self.x0
        y = xy[..., 1] - self.y0
        f = self.focal_length
        rho2 = x * x + y * y

        def _with_aberrations(_):
            rho = jnp.sqrt(rho2)
            phi = jnp.arctan2(y, x)
            alpha = rho / f
            return -0.5 * rho2 / f - W_krivanek(alpha, phi, self.coeffs)

        def _on_axis(_):
            # Quadratic terms retain a nonzero Hessian at the origin. Avoid
            # polar-coordinate derivatives here without dropping astigmatism.
            astigmatism = self.coeffs.C12 * (
                (x*x-y*y)*jnp.cos(2*self.coeffs.phi12)
                + 2*x*y*jnp.sin(2*self.coeffs.phi12)
            )
            return -0.5 * rho2 / f - 0.5 * (self.coeffs.C10*rho2 + astigmatism) / f**2

        return lax.cond(rho2 > self.axis_eps, _with_aberrations, _on_axis, operand=None)


@jdc.pytree_dataclass
class AberratedLensKrivanek(KrivanekLens):
    def __post_init__(self):
        warnings.warn(
            "`AberratedLensKrivanek` is deprecated; use `KrivanekLens`.",
            DeprecationWarning,
            stacklevel=2,
        )


@jdc.pytree_dataclass
class SeidelLens(Lens):
    """Local Gaussian approximation to the Seidel object-to-pupil kernel.

    Position, slope, and object_plane_dist determine the reference object
    coordinate internally. A joint quadratic expansion about the aberrated
    central ray is integrated analytically, retaining its positive decay
    curvature explicitly. The kernel uses the Fresnel amplitude and
    exp(-ik W); field-dependent pupil apodization is not included.

    The central-ray solve assumes a smooth branch near the ideal ray, and
    beam widths must resolve the omitted higher-order terms. ``object_xy``
    optionally selects the prescribed point-object pupil phase instead.
    """

    object_plane_dist: float = 0.0
    coeffs: SeidelCoeffs = dataclasses.field(default_factory=SeidelCoeffs)
    object_xy: tuple[float, float] | None = None

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "SeidelLens is not implemented for singular rays. Use ray_family='gaussian'."
        )

    def log_transmission(self, xy):
        return 0.0

    def _seidel_coefficients(self):
        return self.coeffs

    def _phase_shift_at_object(self, xy, object_xy):
        xy = jnp.asarray(xy)
        x_a, y_a = xy[..., 0], xy[..., 1]
        rho2 = (x_a - self.x0) ** 2 + (y_a - self.y0) ** 2
        aberration = Seidel_object_pos_aperture_pos(
            object_xy[..., 0], object_xy[..., 1], x_a, y_a,
            self._seidel_coefficients(),
        )
        return -0.5 * rho2 / self.focal_length - aberration

    def phase_shift(self, xy, dxy):
        object_xy = (
            jnp.asarray(self.object_xy)
            if self.object_xy is not None
            else jnp.asarray(xy) - self.object_plane_dist * jnp.asarray(dxy)
        )
        return self._phase_shift_at_object(xy, object_xy)

    def complex_action(self, xy: jnp.ndarray, dxy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy, dxy) - 1j * (L / k)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        return _call_single_or_batched_gaussian(ray, self._call_gaussian_single)

    def _call_gaussian_single(self, ray: GaussianBeam) -> GaussianBeam:
        if self.object_xy is None:
            return self._joint_gaussian(ray)
        else:
            def action_at_object(xy):
                log_amplitude = jnp.logaddexp(self.log_transmission(xy), -50.0)
                return self._phase_shift_at_object(
                    xy, jnp.asarray(self.object_xy)
                ) - 1j * log_amplitude / ray.k

            terms = taylor_expand(action_at_object, ray.r_xy)

        r_xy, d_xy, amplitude, pathlength, Q_new = apply_action_delta(ray, *terms)
        return ray.derive(
            x=r_xy[..., 0], y=r_xy[..., 1], dx=d_xy[..., 0], dy=d_xy[..., 1],
            z=ray.z, amplitude=amplitude, pathlength=pathlength, Q_inv=Q_new,
        )

    def _joint_gaussian(self, ray: GaussianBeam):
        distance = self.object_plane_dist
        # This is the same reparameterization as W(aperture, slope). Object
        # coordinates are internal integration variables, not extra inputs.
        object_xy = ray.r_xy - distance * ray.d_xy

        def phase(endpoints):
            return self._phase_shift_at_object(endpoints[2:], endpoints[:2])

        def central_ray_step(_, aperture_xy):
            endpoints = jnp.concatenate((object_xy, aperture_xy))
            gradient = jax.grad(phase)(endpoints)
            hessian = jax.hessian(phase)(endpoints)
            residual = aperture_xy - ray.r_xy - distance * gradient[:2]
            jacobian = jnp.eye(2) - distance * hessian[:2, 2:]
            return aperture_xy - jnp.linalg.solve(jacobian, residual)

        aperture_xy = lax.fori_loop(0, 6, central_ray_step, ray.r_xy)

        endpoints = jnp.concatenate((object_xy, aperture_xy))
        value, gradient, hessian = map(jnp.real, taylor_expand(phase, endpoints))
        go, ga = gradient[:2], gradient[2:]
        hoo, hoa = hessian[:2, :2], hessian[:2, 2:]
        eye = jnp.eye(2, dtype=ray.Q_inv.dtype)
        t = eye - distance * ray.Q_inv
        jacobian = eye - distance * hoa

        # For a real joint action, the output imaginary curvature is a
        # congruence of the incoming positive imaginary curvature. Computing
        # it as Im(Q + delta_Q) loses that positivity near a distortion fold.
        # U = inv(I + d*t*Hoo); Im(Q_out) = J.T U Im(Q) U.H J.
        mapped = jnp.linalg.solve((eye + distance*t @ hoo).conj().T, jacobian)
        imaginary_q = jnp.real(mapped.conj().T @ jnp.imag(ray.Q_inv) @ mapped)
        prefactor, _, delta_q = _pupil_action_to_spatial(
            ray, distance, 0., jnp.zeros(4), hessian)
        real_q = jnp.real(ray.Q_inv + delta_q)
        q = real_q + 1j*imaginary_q
        q = .5*(q+q.T)

        # Evaluate at the stationary centre rather than re-expanding about
        # the old pupil position and cancelling large imaginary path terms.
        # This also works at d=0: the slope includes both endpoint gradients.
        output = ray.derive(
            x=aperture_xy[0], y=aperture_xy[1],
            dx=ray.dx+go[0]+ga[0], dy=ray.dy+go[1]+ga[1],
            amplitude=ray.amplitude*jnp.exp(-ray.k*jnp.imag(prefactor)),
            pathlength=ray.pathlength+value+distance*(ray.d_xy @ go+.5*(go @ go))
            +jnp.real(prefactor), Q_inv=q,
        )
        # Pupil transmission is a spatial action after the real joint kernel.
        # Retain it for subclasses; the native Seidel/DistortedLens is lossless.
        terms = taylor_expand(lambda xy: -1j*jnp.logaddexp(
            self.log_transmission(xy), -50.)/ray.k, output.r_xy)
        xy, slope, amplitude, pathlength, curvature = apply_action_delta(output, *terms)
        return output.derive(x=xy[0], y=xy[1], dx=slope[0], dy=slope[1],
                             amplitude=amplitude, pathlength=pathlength, Q_inv=curvature)


@jdc.pytree_dataclass
class DistortedLens(SeidelLens):
    """Radial and tangential terms of the Seidel wave aberration."""

    IsoDist: float = 0.0
    AnisoDist: float = 0.0

    def _seidel_coefficients(self):
        return SeidelCoeffs(E=self.IsoDist, e=self.AnisoDist)


@jdc.pytree_dataclass(kw_only=True)
class ABCDTransfer(Component):
    """Linear 4D map with 2×2 A, B, C, D blocks in [x, y, dx, dy] order.

    Transports positions, slopes and Gaussian curvature. This primitive does
    not add optical path or the coherent amplitude factor; physical wrappers
    such as HardEdgeSolenoid supply them.
    """

    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.zeros((2, 2), dtype=jnp.float64)
    )
    D: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.eye(2, dtype=jnp.float64)
    )
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        A = jnp.asarray(self.A, dtype=jnp.float64)
        B = jnp.asarray(self.B, dtype=jnp.float64)
        C = jnp.asarray(self.C, dtype=jnp.float64)
        D = jnp.asarray(self.D, dtype=jnp.float64)
        r_xy_new = _matvec(A, ray.r_xy) + _matvec(B, ray.d_xy)
        d_xy_new = _matvec(C, ray.r_xy) + _matvec(D, ray.d_xy)

        return ray.derive(
            x=r_xy_new[..., 0],
            y=r_xy_new[..., 1],
            dx=d_xy_new[..., 0],
            dy=d_xy_new[..., 1],
        )

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        A = jnp.asarray(self.A, dtype=jnp.float64)
        B = jnp.asarray(self.B, dtype=jnp.float64)
        C = jnp.asarray(self.C, dtype=jnp.float64)
        D = jnp.asarray(self.D, dtype=jnp.float64)

        r_xy_new = _matvec(A, ray.r_xy) + _matvec(B, ray.d_xy)
        d_xy_new = _matvec(C, ray.r_xy) + _matvec(D, ray.d_xy)

        Q = ray.Q_inv
        denom = A + _matmat(B, Q)
        numer = C + _matmat(D, Q)
        def complex_denominator(_):
            # Solve the transposed system for numer @ inv(denom), retaining
            # any leading bundle axes.
            return jnp.swapaxes(jnp.linalg.solve(
                jnp.swapaxes(denom, -1, -2),
                jnp.swapaxes(numer, -1, -2)), -1, -2)

        def real_denominator(_):
            # With B=0, a complex solve can contaminate tiny decay curvature
            # with roundoff from the real curvature after a Seidel fold.
            real = jnp.swapaxes(jnp.linalg.solve(
                jnp.swapaxes(A, -1, -2),
                jnp.swapaxes(jnp.real(numer), -1, -2)), -1, -2)
            imag = jnp.swapaxes(jnp.linalg.solve(
                jnp.swapaxes(A, -1, -2),
                jnp.swapaxes(_matmat(D, jnp.imag(Q)), -1, -2)), -1, -2)
            return real + 1j*imag

        Q_new = jax.lax.cond(jnp.all(B == 0), real_denominator,
                             complex_denominator, operand=None)

        return ray.derive(
            x=r_xy_new[..., 0],
            y=r_xy_new[..., 1],
            dx=d_xy_new[..., 0],
            dy=d_xy_new[..., 1],
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass
class ThickLens(Component):
    """Ray-only principal-plane approximation, with separate object/image planes."""

    z_po: float
    z_pi: float
    focal_length: float

    def _call_ray(self, ray: Ray):
        f = self.focal_length
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy
        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)
        new_z = ray.z - (self.z_po - self.z_pi)

        return ray.derive(dx=new_dx, dy=new_dy, pathlength=pathlength, z=new_z)

    def _call_gaussian(self, ray: GaussianBeam):
        raise NotImplementedError(
            "ThickLens is not implemented for gaussian beams."
        )

    @property
    def z(self):
        return self.z_po


@jdc.pytree_dataclass(kw_only=True)
class HardEdgeSolenoid(Component):
    """Entrance at ``z``, body length in metres, signed Larmor integral ``q``.

    Includes both edges and propagation through the body. Rotation is -q;
    q is proportional to current / relativistic momentum. This ideal uniform
    field is distinct from the principal-plane ``ThickLens`` approximation.
    Coherent transport includes the quadratic action and Gouy factor. The
    coherent branch covers the first radial half-period, abs(q) < pi;
    longer fields must be split into shorter bodies to continue the phase.
    """

    z: float
    length: float
    q: float
    include_axial_path: jdc.Static[bool] = True

    @property
    def matrix(self):
        q, length = jnp.asarray(self.q), jnp.asarray(self.length)
        cosine, sine = jnp.cos(q), jnp.sin(q)
        rotation = jnp.array([[cosine, sine], [-sine, cosine]])
        drift = length * jnp.sinc(q / jnp.pi)
        power = -q * sine / length
        return jnp.block([
            [cosine * rotation, drift * rotation],
            [power * rotation, cosine * rotation],
        ])

    def _call_ray(self, ray):
        m = self.matrix
        out = ABCDTransfer(A=m[:2, :2], B=m[:2, 2:],
                           C=m[2:, :2], D=m[2:, 2:])(ray)
        return out.derive(z=ray.z + self.length,
                          pathlength=ray.pathlength + self.length)

    def _call_gaussian(self, ray):
        m = self.matrix
        out = ABCDTransfer(A=m[:2, :2], B=m[:2, 2:],
                           C=m[2:, :2], D=m[2:, 2:])(ray)
        # The solenoid is symplectic: preserve its positive decay curvature
        # by congruence. Extracting Im(Q) from the complex quotient can turn
        # nearly collapsed Seidel beamlets into growing Gaussians.
        transport = m[:2, :2] + m[:2, 2:] @ ray.Q_inv
        inverse = jnp.linalg.solve(transport, jnp.broadcast_to(
            jnp.eye(2, dtype=transport.dtype), transport.shape))
        decay = jnp.real(jnp.swapaxes(inverse.conj(), -1, -2)
                         @ jnp.imag(ray.Q_inv) @ inverse)
        curvature = jnp.real(out.Q_inv) + 1j*decay
        out = out.derive(Q_inv=.5*(curvature+jnp.swapaxes(curvature, -1, -2)))
        # Remove the real rotation from A+BQ before following its phase.
        # For |q|<pi, b>0 and both modal factors lie in the upper half-plane.
        b = self.length * jnp.sinc(self.q / jnp.pi)
        denominator = jnp.cos(self.q) * jnp.eye(2) + b * ray.Q_inv
        determinant = jnp.linalg.det(denominator)
        root = jnp.sqrt(determinant)
        root = jnp.where(jnp.imag(determinant) < 0., -root, root)
        # Homogeneous quadratic canonical action, including magnetic rotation.
        action = .5 * (jnp.sum(out.r_xy * out.d_xy, axis=-1)
                       - jnp.sum(ray.r_xy * ray.d_xy, axis=-1))
        valid = (jnp.abs(self.q) < jnp.pi) & (self.length > 0)
        return out.derive(
            z=ray.z + self.length,
            pathlength=ray.pathlength + (self.length if self.include_axial_path else 0.) + action,
            amplitude=jnp.where(valid, ray.amplitude / root, jnp.nan + 0j),
        )


def solenoid_column(gaps, lengths, strengths, *, source_z=0.):
    """Build lenses and a final plane; gaps run before/between/after bodies."""
    if len(gaps) != len(lengths) + 1 or len(strengths) != len(lengths):
        raise ValueError("Supply N lengths/strengths and N+1 free-space gaps.")
    entrance = source_z
    components = []
    for i in range(len(lengths)):
        entrance = entrance + gaps[i]
        components.append(HardEdgeSolenoid(z=entrance, length=lengths[i], q=strengths[i]))
        entrance = entrance + lengths[i]
    components.append(Plane(z=entrance + gaps[-1]))
    return tuple(components)


@jdc.pytree_dataclass
class RotatingLens(Component):
    """A rotation in degrees followed by a thin round lens."""

    z: float
    focal_length: float
    rotation: Degrees
    x0: float = 0.0
    y0: float = 0.0

    def _call_ray(self, ray: Ray):
        rotated_ray = Rotator(z=self.z, angle=self.rotation)(ray)
        return Lens(z=self.z, focal_length=self.focal_length, x0=self.x0, y0=self.y0)(
            rotated_ray
        )

    def _call_gaussian(self, ray: GaussianBeam):
        return self._call_ray(ray)


@jdc.pytree_dataclass
class ElectromagneticLens(Component):
    """Current-controlled focus, optional body drift, then magnetic rotation.

    Focal length varies inversely with excitation squared; rotation is linear
    in excitation, with relativistic voltage factors. Nonzero Tc represents
    the body as two half-strength lenses separated by a drift.

    Existing round-ray behaviour uses the optical axis even when x0/y0 are
    set. Gaussian and stigmator paths use those offsets. This distinction is
    retained here; it needs a separate physics correction and validation.
    """

    z: float
    turns: float
    current: float
    Gc: float  # Voltage / (ampere-turns squared × beam length unit)
    Tc: float = 0.0  # Beam length unit / ampere-turn
    x0: float = 0.0
    y0: float = 0.0
    stigmator_strength_x: float = 0.0
    stigmator_strength_y: float = 0.0
    stigmator_eps: float = 1e-6

    @property
    def excitation(self) -> float:
        return self.turns * self.current

    @property
    def I0(self) -> float:
        """Backward-compatible alias for ampere-turn excitation."""
        return self.excitation

    def focal_length(self, voltage: float) -> float:
        V_star = effective_accelerating_potential(voltage)
        denom = jnp.asarray(self.Gc) * jnp.asarray(self.excitation) ** 2
        return jnp.where(denom == 0.0, jnp.inf, V_star / denom)

    def focal_length_at(self, voltage: float) -> float:
        """Convenience alias for focal_length(voltage)."""
        return self.focal_length(voltage)

    def rotation_angle(self, voltage: float) -> float:
        return compute_Rc_from_voltage(voltage) * self.excitation

    @property
    def thickness(self) -> float:
        return jnp.maximum(self.Tc, 0.0) * jnp.abs(self.excitation)

    def phase_shift(self, xy: jnp.ndarray, voltage: float) -> float:
        x, y = xy[..., 0] - self.x0, xy[..., 1] - self.y0
        rho2 = x * x + y * y
        return -0.5 * rho2 / self.focal_length(voltage)

    def _effective_stigmator_strengths(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        min_strength = -1.0 + jnp.abs(self.stigmator_eps)
        sx = jnp.clip(jnp.asarray(self.stigmator_strength_x), min_strength, jnp.inf)
        sy = jnp.clip(jnp.asarray(self.stigmator_strength_y), min_strength, jnp.inf)
        return sx, sy

    def _stigmator_is_active(self) -> jnp.ndarray:
        sx, sy = self._effective_stigmator_strengths()
        eps = jnp.asarray(self.stigmator_eps)
        return jnp.logical_or(jnp.abs(sx) > eps, jnp.abs(sy) > eps)

    def _stigmator_focal_lengths(self, voltage: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        f = self.focal_length(voltage)
        sx, sy = self._effective_stigmator_strengths()
        return f / (1.0 + sx), f / (1.0 + sy)

    def _apply_thin_stigmator(self, ray: Ray | GaussianBeam, fx, fy):
        stig = Stigmator(
            z=self.z,
            focal_length_x=fx,
            focal_length_y=fy,
            x0=self.x0,
            y0=self.y0,
        )
        return stig(ray)

    def _apply_thin_lens_ray(self, ray: Ray, focal_length: float) -> Ray:
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / focal_length + dx
        new_dy = -y / focal_length + dy
        pathlength = ray.pathlength - (x**2 + y**2) / (2.0 * focal_length)

        return ray.derive(
            dx=new_dx,
            dy=new_dy,
            pathlength=pathlength,
        )

    def _apply_rotation_ray(self, ray: Ray, angle: float) -> Ray:
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)

        rot_x = cos_a * x - sin_a * y
        rot_y = sin_a * x + cos_a * y
        rot_dx = cos_a * dx - sin_a * dy
        rot_dy = sin_a * dx + cos_a * dy

        return ray.derive(
            x=rot_x,
            y=rot_y,
            dx=rot_dx,
            dy=rot_dy,
        )

    def _focus(self, ray, thin_lens, propagate):
        """Apply one thin kick, or two half-strength kicks around a body drift."""
        focal_length = self.focal_length(ray.voltage)
        fx, fy = self._stigmator_focal_lengths(ray.voltage)
        thickness = self.thickness

        def kick(beam, focal_scale):
            return lax.cond(
                self._stigmator_is_active(),
                lambda b: self._apply_thin_stigmator(b, focal_scale * fx, focal_scale * fy),
                lambda b: thin_lens(b, focal_scale * focal_length),
                beam,
            )

        def through_body(_):
            out = kick(ray, 2.0)
            out = propagate(out, thickness)
            return kick(out, 2.0)

        return lax.cond(thickness > 0.0, through_body, lambda _: kick(ray, 1.0), None)

    def _call_ray(self, ray: Ray) -> Ray:
        from .propagator import FreeSpaceParaxial

        out = self._focus(ray, self._apply_thin_lens_ray, FreeSpaceParaxial())
        return self._apply_rotation_ray(out, self.rotation_angle(ray.voltage))

    def _apply_thin_lens_gaussian(self, ray, focal_length):
        return Lens(
            z=self.z, focal_length=focal_length, x0=self.x0, y0=self.y0
        )(ray)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        out = self._focus(ray, self._apply_thin_lens_gaussian, FreeSpacePropagator())

        angle = self.rotation_angle(ray.voltage)
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        R = jnp.stack(
            (
                jnp.stack((cos_a, -sin_a), axis=-1),
                jnp.stack((sin_a, cos_a), axis=-1),
            ),
            axis=-2,
        ).astype(jnp.float64)

        r_xy_rot = _matvec(R, out.r_xy)
        d_xy_rot = _matvec(R, out.d_xy)
        Q_rot = _rotate_quadratic_form(R, out.Q_inv)

        return out.derive(
            x=r_xy_rot[..., 0],
            y=r_xy_rot[..., 1],
            dx=d_xy_rot[..., 0],
            dy=d_xy_rot[..., 1],
            Q_inv=Q_rot,
        )


@jdc.pytree_dataclass(kw_only=True)
class FourierTransform:
    """Explicit drift(f) → lens(f) → drift(f) for rays or Gaussians."""

    f: float | jnp.ndarray
    x0: float = 0.0
    y0: float = 0.0

    def __call__(self, ray: Ray) -> Ray:
        family = getattr(ray, "ray_family", "ray")

        if family == "gaussian":
            fs = FreeSpacePropagator()
        else:
            from .propagator import FreeSpaceParaxial

            fs = FreeSpaceParaxial()

        ray = fs(ray, self.f)
        lens = Lens(z=ray.z, focal_length=self.f, x0=self.x0, y0=self.y0)
        ray = lens(ray)
        ray = fs(ray, self.f)
        return ray


# Steering, rotation and scanning

@jdc.pytree_dataclass
class Deflector(Component):
    """Add a slope kick and its linear optical path; Gaussian shape is unchanged."""

    z: float
    def_x: float
    def_y: float

    def _call_ray(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x * ray._one,
            dy=dy + self.def_y * ray._one,
            pathlength=ray.pathlength + self.def_x * x + self.def_y * y,
        )

    def _call_gaussian(self, ray: GaussianBeam):
        return self._call_ray(ray)


@jdc.pytree_dataclass
class DoubleDeflector(Component):
    """Two deflector kicks separated by an internal free-space spacing.

    User-facing commands (shift/tilt) are combined with hardware calibration
    ratios to produce the two physical kicks in each axis.
    ``include_axial_path=False`` omits the common straight-path phase, as in
    the Gaussian free-space propagator.
    """

    z: float
    spacing: float

    shift_x: float = 0.0
    shift_y: float = 0.0
    tilt_x: float = 0.0
    tilt_y: float = 0.0

    shift_balance_x: float = 1.0
    shift_balance_y: float = 1.0
    tilt_balance_x: float = 1.0
    tilt_balance_y: float = 1.0
    include_axial_path: jdc.Static[bool] = True

    @property
    def z_second(self) -> float:
        return self.z + self.spacing

    @property
    def def1_x(self):
        return self.shift_x + self.tilt_x

    @property
    def def1_y(self):
        return self.shift_y + self.tilt_y

    @property
    def def2_x(self):
        return (
            -self.shift_balance_x * self.shift_x
            -self.tilt_balance_x * self.tilt_x
        )

    @property
    def def2_y(self):
        return (
            -self.shift_balance_y * self.shift_y
            -self.tilt_balance_y * self.tilt_y
        )

    @property
    def deflectors(self):
        """The two physical kicks, including the shift/tilt balance mixing."""
        return (
            Deflector(self.z, self.def1_x, self.def1_y),
            Deflector(self.z_second, self.def2_x, self.def2_y),
        )

    def _call_ray(self, ray: Ray):
        from .propagator import FreeSpaceParaxial

        first, second = self.deflectors
        ray = FreeSpaceParaxial()(first(ray), self.spacing)
        if not self.include_axial_path:
            ray = ray.derive(pathlength=ray.pathlength - self.spacing)
        return second(ray)

    def _call_gaussian(self, ray: GaussianBeam):
        first, second = self.deflectors
        ray = FreeSpacePropagator(include_axial_path=self.include_axial_path)(
            first(ray), self.spacing
        )
        return second(ray)


@jdc.pytree_dataclass
class Rotator(Component):
    z: float
    angle: Degrees

    def _rotation(self):
        angle = jnp.deg2rad(self.angle)
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        return cos_a, sin_a

    def _call_ray(self, ray: Ray):
        cos_a, sin_a = self._rotation()
        new_x = ray.x * cos_a - ray.y * sin_a
        new_y = ray.x * sin_a + ray.y * cos_a
        new_dx = ray.dx * cos_a - ray.dy * sin_a
        new_dy = ray.dx * sin_a + ray.dy * cos_a
        return ray.derive(x=new_x, y=new_y, dx=new_dx, dy=new_dy)

    def _call_gaussian(self, ray: GaussianBeam):
        cos_a, sin_a = self._rotation()
        R = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=jnp.float64)
        r_xy_rot = _matvec(R, ray.r_xy)
        d_xy_rot = _matvec(R, ray.d_xy)
        Q_rot = _rotate_quadratic_form(R, ray.Q_inv)

        return ray.derive(
            x=r_xy_rot[..., 0],
            y=r_xy_rot[..., 1],
            dx=d_xy_rot[..., 0],
            dy=d_xy_rot[..., 1],
            Q_inv=Q_rot,
        )


@jdc.pytree_dataclass
class DeflectionBiprism(Component):
    z: float
    offset: float = 0.0
    rotation: Degrees = 0.0
    def_x: float = 0.0
    side: int = 1

    def _call_ray(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x * ray._one * jnp.sign(ray.x),
            dy=dy,
            pathlength=ray.pathlength + dx * x + dy * y,
        )

    def _call_gaussian(self, ray: GaussianBeam):
        return self._call_ray(ray)


@jdc.pytree_dataclass
class Biprism(DeflectionBiprism):
    def __post_init__(self):
        warnings.warn(
            "`components.Biprism` is deprecated and will remain as an alias for "
            "`DeflectionBiprism`. Prefer `DeflectionBiprism` for clarity.",
            DeprecationWarning,
            stacklevel=2,
        )


class DescanError(NamedTuple):
    """Affine position/slope response and offsets used by Descanner."""

    pxo_pxi: float = 0.0
    pxo_pyi: float = 0.0
    pyo_pxi: float = 0.0
    pyo_pyi: float = 0.0
    sxo_pxi: float = 0.0
    sxo_pyi: float = 0.0
    syo_pxi: float = 0.0
    syo_pyi: float = 0.0
    offpxi: float = 0.0
    offpyi: float = 0.0
    offsxi: float = 0.0
    offsyi: float = 0.0

    def as_array(self) -> jnp.ndarray:
        return jnp.array(self)

    def as_matrix(self) -> jnp.ndarray:
        return jnp.array(
            [
                [self.pxo_pxi, self.pxo_pyi, 0.0, 0.0, self.offpxi],
                [self.pyo_pxi, self.pyo_pyi, 0.0, 0.0, self.offpyi],
                [self.sxo_pxi, self.sxo_pyi, 0.0, 0.0, self.offsxi],
                [self.syo_pxi, self.syo_pyi, 0.0, 0.0, self.offsyi],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )


@jdc.pytree_dataclass
class Scanner(Component):
    """Add calibrated position and slope offsets; Gaussian shape is unchanged."""

    z: float
    scan_pos_x: float
    scan_pos_y: float
    scan_tilt_x: float = 0.0
    scan_tilt_y: float = 0.0

    def _call_ray(self, ray: Ray):
        return ray.derive(
            x=ray.x + self.scan_pos_x * ray._one,
            y=ray.y + self.scan_pos_y * ray._one,
            dx=ray.dx + self.scan_tilt_x * ray._one,
            dy=ray.dy + self.scan_tilt_y * ray._one,
        )

    def _call_gaussian(self, ray: GaussianBeam):
        return self._call_ray(ray)


@jdc.pytree_dataclass
class Descanner(Component):
    """Undo scan offsets, allowing the affine errors in descan_error."""

    z: float
    scan_pos_x: float
    scan_pos_y: float
    scan_tilt_x: float = 0.0
    scan_tilt_y: float = 0.0
    descan_error: DescanError = DescanError()

    def _offsets(self):
        de = self.descan_error
        sp_x, sp_y = self.scan_pos_x, self.scan_pos_y
        st_x, st_y = self.scan_tilt_x, self.scan_tilt_y
        return (
            sp_x * de.pxo_pxi + sp_y * de.pxo_pyi + de.offpxi - sp_x,
            sp_x * de.pyo_pxi + sp_y * de.pyo_pyi + de.offpyi - sp_y,
            sp_x * de.sxo_pxi + sp_y * de.sxo_pyi + de.offsxi - st_x,
            sp_x * de.syo_pxi + sp_y * de.syo_pyi + de.offsyi - st_y,
        )

    def _call_ray(self, ray: Ray):
        ox, oy, odx, ody = self._offsets()
        return ray.derive(
            x=ray.x + ox * ray._one,
            y=ray.y + oy * ray._one,
            dx=ray.dx + odx * ray._one,
            dy=ray.dy + ody * ray._one,
        )

    def _call_gaussian(self, ray: GaussianBeam):
        return self._call_ray(ray)


# Apertures and prescribed phase/amplitude

@jdc.pytree_dataclass(kw_only=True)
class SigmoidAperture(GaussianActionComponent):
    """Smooth field-amplitude transmission for Gaussians; rays pass through."""

    radius: float = 1.0
    edge_width: float = 0.5
    sharpness: float = 1.0
    t_low: float = 0.0
    t_high: float = 1.0
    x0: float = 0.0
    y0: float = 0.0
    eps: float = 1e-15
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        # Rays carry no amplitude in this model, so aperture transmission
        # cannot attenuate them; treat as geometric pass-through.
        return ray

    def phase_shift(self, xy):
        return 0.0

    def log_transmission(self, xy):
        x, y = xy[..., 0] - self.x0, xy[..., 1] - self.y0
        rho = jnp.sqrt(x * x + y * y + self.eps * self.eps) - self.eps
        w = jnp.maximum(jnp.abs(self.edge_width), self.eps)
        s = jnn.sigmoid(self.sharpness * (rho - self.radius) / w)
        t = self.t_high - (self.t_high - self.t_low) * s
        t_clamped = jnp.clip(t, self.eps, None)
        return jnp.log(t_clamped)


@jdc.pytree_dataclass(kw_only=True)
class PhaseBiprism(GaussianActionComponent):
    strength: float
    width: float
    length: float | None = None
    theta: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    sharpness: float = 50.0
    eps: float = 1e-15
    z: float = 0.0

    def _uv(self, xy: jnp.ndarray):
        x, y = xy[..., 0], xy[..., 1]
        xr, yr = x - self.x0, y - self.y0
        c, s = jnp.cos(self.theta), jnp.sin(self.theta)
        u = c * xr + s * yr
        v = -s * xr + c * yr
        return u, v

    def phase_shift(self, xy: jnp.ndarray):
        u, _ = self._uv(xy)
        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)
        return -self.strength * au

    def log_transmission(self, xy: jnp.ndarray):
        u, v = self._uv(xy)

        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)
        tx = self.sharpness * (au - hu)
        logA_u = -softplus(-tx)
        if self.length is None:
            # Avoid tracing the "with_length" path when length is None.
            # lax.cond traces both branches and would attempt jnp.asarray(None).
            logA_v = jnp.asarray(0.0)
        else:
            length = jnp.asarray(self.length)
            hv = 0.5 * length
            eps_v = self.eps * hu
            av = jnp.sqrt(v * v + eps_v * eps_v)
            ty = self.sharpness * (av - hv)
            logA_v = -softplus(-ty)
        return logA_u + logA_v


@jdc.pytree_dataclass(kw_only=True)
class ConstantPhaseShift(GaussianActionComponent):
    constant_phase_shift: float
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        return self.constant_phase_shift


@jdc.pytree_dataclass(kw_only=True)
class LinearPhaseShift(GaussianActionComponent):
    linear_phase_shift: jnp.ndarray
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        return jnp.dot(self.linear_phase_shift, xy)


@jdc.pytree_dataclass(kw_only=True)
class QuadraticPhaseShift(GaussianActionComponent):
    quadratic_phase_shift: jnp.ndarray
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        return 0.5 * xy @ self.quadratic_phase_shift @ xy


@jdc.pytree_dataclass(kw_only=True)
class ZernikeAberration(GaussianActionComponent):
    """Thin pathlength aberration expressed in unnormalised Noll modes."""

    coefficients: Mapping[int, float]
    radius: float
    x0: float = 0.0
    y0: float = 0.0
    outside: float = 0.0
    z: float = 0.0

    def phase_shift(self, xy: jnp.ndarray):
        x = (xy[..., 0] - self.x0) / self.radius
        y = (xy[..., 1] - self.y0) / self.radius
        out = jnp.zeros_like(x, dtype=jnp.float64)
        for j, coefficient in self.coefficients.items():
            out = out + coefficient * zernike_noll_cartesian(
                int(j),
                x,
                y,
                outside=self.outside,
            )
        return out


@jdc.pytree_dataclass(kw_only=True)
class ConstantAmplitudeShift(GaussianActionComponent):
    amplitude: float
    z: float = 0.0

    def log_transmission(self, xy: jnp.ndarray):
        return jnp.log(self.amplitude)


@jdc.pytree_dataclass(kw_only=True)
class LinearAmplitudeShift(GaussianActionComponent):
    linear_log_amplitude: jnp.ndarray | None = None
    linear_amplitude: jnp.ndarray | None = None
    z: float = 0.0

    def log_transmission(self, xy: jnp.ndarray):
        coeffs = self.linear_log_amplitude
        if coeffs is None:
            coeffs = self.linear_amplitude
        if coeffs is None:
            raise ValueError("LinearAmplitudeShift requires linear_log_amplitude.")
        return jnp.dot(coeffs, xy)


@jdc.pytree_dataclass(kw_only=True)
class QuadraticAmplitudeShift(GaussianActionComponent):
    quadratic_log_amplitude: jnp.ndarray
    z: float = 0.0

    def log_transmission(self, xy: jnp.ndarray):
        return 0.5 * xy @ self.quadratic_log_amplitude @ xy


# Analytical specimen profiles

@jdc.pytree_dataclass(kw_only=True)
class MagneticPhaseSample(GaussianActionComponent):
    strength: float
    width: float
    height: float
    x0: float = 0.0
    y0: float = 0.0
    theta: float = 0.0
    edge_sharpness: float = 5e6
    modulation_strength: float = 0.3
    skew_strength: float = 0.2
    radial_strength: float = 0.15
    eps: float = 1e-9
    z: float = 0.0

    def _local_coords(self, xy):
        x = xy[..., 0] - self.x0
        y = xy[..., 1] - self.y0
        c = jnp.cos(self.theta)
        s = jnp.sin(self.theta)
        u = c * x + s * y
        v = -s * x + c * y
        return u, v

    def _soft_indicator(self, coord, half_extent):
        sharp = self.edge_sharpness
        pos = jax.nn.sigmoid(sharp * (coord + half_extent))
        neg = jax.nn.sigmoid(sharp * (coord - half_extent))
        plateau = jax.nn.sigmoid(sharp * half_extent) - jax.nn.sigmoid(-sharp * half_extent)
        plateau = jnp.maximum(plateau, 1e-9)
        return (pos - neg) / plateau

    def phase_shift(self, xy):
        u, v = self._local_coords(xy)

        hx = 0.5 * self.width
        hy = 0.5 * self.height

        mask = self._soft_indicator(u, hx) * self._soft_indicator(v, hy)

        u_norm = u / (hx + self.eps)
        v_norm = v / (hy + self.eps)
        radial = jnp.sqrt(u_norm * u_norm + v_norm * v_norm + self.eps)

        texture = jnp.sin(jnp.pi * u_norm) * jnp.cos(jnp.pi * v_norm)
        skew = u_norm * v_norm
        radial_term = radial - 0.5

        profile = (
            1.0
            + self.modulation_strength * texture
            + self.skew_strength * skew
            + self.radial_strength * radial_term
        )

        return self.strength * mask * profile


@jdc.pytree_dataclass(kw_only=True)
class RandomPhaseSample(GaussianActionComponent):
    strength: float
    width: float
    height: float
    x0: float = 0.0
    y0: float = 0.0
    theta: float = 0.0
    edge_sharpness: float = 5e6
    correlation_length: float = 2e-9
    eps: float = 1e-9
    z: float = 0.0

    def _local_coords(self, xy):
        x = xy[..., 0] - self.x0
        y = xy[..., 1] - self.y0
        c = jnp.cos(self.theta)
        s = jnp.sin(self.theta)
        u = c * x + s * y
        v = -s * x + c * y
        return u, v

    def _soft_indicator(self, coord, half_extent):
        sharp = self.edge_sharpness
        pos = jax.nn.sigmoid(sharp * (coord + half_extent))
        neg = jax.nn.sigmoid(sharp * (coord - half_extent))
        plateau = jax.nn.sigmoid(sharp * half_extent) - jax.nn.sigmoid(-sharp * half_extent)
        plateau = jnp.maximum(plateau, 1e-9)
        return (pos - neg) / plateau

    def _hash(self, i, j):
        return jnp.mod(jnp.sin(127.1 * i + 311.7 * j) * 43758.5453, 1.0)

    def _value_noise(self, x, y):
        xi = jnp.floor(x)
        yi = jnp.floor(y)
        xf = x - xi
        yf = y - yi

        n00 = self._hash(xi, yi)
        n10 = self._hash(xi + 1, yi)
        n01 = self._hash(xi, yi + 1)
        n11 = self._hash(xi + 1, yi + 1)

        def fade(t):
            return t * t * (3.0 - 2.0 * t)

        u = fade(xf)
        v = fade(yf)

        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return nx0 + v * (nx1 - nx0)

    def phase_shift(self, xy):
        u, v = self._local_coords(xy)

        hx = 0.5 * self.width
        hy = 0.5 * self.height
        mask = self._soft_indicator(u, hx) * self._soft_indicator(v, hy)

        corr = self.correlation_length + self.eps
        xn = u / corr
        yn = v / corr

        raw = self._value_noise(xn, yn)
        noise = (raw - 0.5) * 2.0

        raw2 = self._value_noise(xn * 2.0, yn * 2.0)
        noise2 = (raw2 - 0.5) * 2.0
        combined = 0.7 * noise + 0.3 * noise2

        return self.strength * mask * combined


# Specimens

@jdc.pytree_dataclass(kw_only=True)
class BraggSample(Component):
    """Thin periodic transmission represented by explicit Fourier orders.

    The field multiplier is sum(c_m exp(i q_m . (xy - origin_xy))). Each
    incident Gaussian produces one child per retained order, with amplitude
    c_m and slope increment q_m/k; the phase at its centre is preserved.
    Coordinates, z and periods use the beam's length unit; q_xy uses radians
    per that unit. Coefficients describe transmission at the chosen energy,
    not a potential that is automatically rescaled with beam voltage.

    Use run_to_end on a scalar beam or a directly batched bundle. This sample
    changes the beam count: N inputs and M orders produce N*M outputs. A
    separate vmap over inputs would retain an extra parent-beam axis.
    This is an infinite periodic projected specimen, not dynamical diffraction.

    angular_response optionally supplies a complex field multiplier as a
    function of grazing angle and voltage. surface_normal_xyz is the outward
    surface normal in the incoming beam frame (unit length is recommended).
    Ordinary __call__ still uses forward/folded-frame thin-grating geometry;
    it does not itself reflect a 3D surface. A reflection adapter can instead
    use coefficients_at with its physical incident angles. Angular response
    is constant within each Gaussian packet: use a narrow angular basis when
    the response varies across the illumination.
    """

    q_xy: jnp.ndarray
    coefficients: jnp.ndarray
    z: float = 0.0
    origin_xy: tuple = (0.0, 0.0)
    angular_response: Any = None
    surface_normal_xyz: tuple = (0.0, 0.0, -1.0)

    def coefficients_at(self, grazing_angle_rad, voltage):
        """Order field coefficients with an optional complex angular response.

        Returns shape (..., M), without renormalizing retained orders. Surface
        reflection adapters should call this with physical incidence angles.
        The ordinary __call__ retains the thin forward/folded-frame geometry;
        an angular response alone does not reverse the propagation direction.
        """
        angles = jnp.asarray(grazing_angle_rad)
        c = jnp.broadcast_to(jnp.asarray(self.coefficients), angles.shape + (len(self.coefficients),))
        if self.angular_response is not None:
            c = c*self.angular_response.amplitude(angles, voltage)[...,None]
        return c

    def order_action(self, grazing_angle_rad, voltage, k):
        """Complex order action: exp(i*k*action) equals coefficients_at.

        An exactly zero order has infinite attenuation; direct coefficient
        multiplication in propagation preserves exact zeros without log(0).
        """
        return -1j*jnp.log(self.coefficients_at(grazing_angle_rad, voltage).astype(complex))/jnp.asarray(k)[...,None]

    @classmethod
    def from_unit_cell(
        cls, transmission_yx, *, periods_xy, orders_hk, z=0.0,
        origin_xy=(0.0, 0.0),
    ):
        """Fourier-expand a rectangular unit cell of complex field transmission.

        T[j,i] is sampled at (i*period_x/nx, j*period_y/ny) relative to
        origin_xy, without duplicate endpoints. Supply physical transmission,
        e.g. exp(1j*phase_radians), in image (y,x) order. orders_hk is a fixed
        (M,2) list of integer reciprocal indices, including (0,0) if desired.

        Construction runs on the host; subsequent propagation supports JAX.
        Refine the unit-cell grid and retained orders independently. Truncated
        coefficients are never renormalized: omitted orders remain omitted.
        A grid's Nyquist limit alone does not establish convergence.
        """
        transmission = np.asarray(transmission_yx, dtype=np.complex128)
        periods = np.asarray(periods_xy, dtype=float)
        orders = np.asarray(orders_hk)
        origin = np.asarray(origin_xy, dtype=float)
        if transmission.ndim != 2 or min(transmission.shape) < 2:
            raise ValueError("Supply a 2D unit-cell transmission with at least two samples per axis.")
        if not np.isfinite(transmission).all():
            raise ValueError("Unit-cell transmission must be finite.")
        if periods.shape != (2,) or not np.isfinite(periods).all() or np.any(periods <= 0):
            raise ValueError("periods_xy must contain two positive finite periods.")
        if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(z):
            raise ValueError("Supply a finite origin_xy pair and z position.")
        if orders.ndim != 2 or orders.shape[1] != 2 or len(orders) == 0:
            raise ValueError("orders_hk must be a nonempty (M,2) list of integer orders.")
        if not np.isfinite(orders).all() or not np.equal(orders, np.round(orders)).all():
            raise ValueError("orders_hk must contain integers.")
        ny, nx = transmission.shape
        if np.any(2 * np.abs(orders) >= np.array([nx, ny])):
            raise ValueError("Retained orders must lie strictly inside the unit-cell Nyquist limits.")
        orders = orders.astype(int)
        if len(np.unique(orders, axis=0)) != len(orders):
            raise ValueError("orders_hk must contain unique orders.")
        spectrum = np.fft.fft2(transmission) / transmission.size
        coefficients = spectrum[orders[:, 1] % ny, orders[:, 0] % nx]
        return cls(
            q_xy=jnp.asarray(2 * np.pi * orders / periods),
            coefficients=jnp.asarray(coefficients), z=z,
            origin_xy=jnp.asarray(origin),
        )

    def evaluate_complex(self, xy, *, grazing_angle_rad=None, voltage=None):
        """Evaluate the retained field multiplier at specimen coordinates.

        Angle and voltage are required when angular_response is enabled.
        """
        coefficients = jnp.asarray(self.coefficients)
        if self.angular_response is not None:
            if grazing_angle_rad is None or voltage is None:
                raise ValueError("Supply grazing_angle_rad and voltage for an angular response.")
            coefficients = self.coefficients_at(grazing_angle_rad, voltage)
        phase = (jnp.asarray(xy) - jnp.asarray(self.origin_xy)) @ jnp.asarray(self.q_xy).T
        return jnp.sum(coefficients * jnp.exp(1j * phase), axis=-1)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        ray = ray.to_vector()
        q = jnp.asarray(self.q_xy)
        coefficients = jnp.asarray(self.coefficients)
        if q.ndim != 2 or q.shape[1] != 2 or q.shape[0] == 0:
            raise ValueError("q_xy must have shape (M,2) with at least one order.")
        if coefficients.shape != (q.shape[0],):
            raise ValueError("Supply one complex coefficient per q_xy order.")
        phase = (ray.r_xy - jnp.asarray(self.origin_xy)) @ q.T
        if self.angular_response is not None:
            direction = jnp.column_stack((ray.dx, ray.dy, jnp.ones_like(ray.dx)))
            direction /= jnp.linalg.norm(direction, axis=-1, keepdims=True)
            normal = jnp.asarray(self.surface_normal_xyz)
            normal = normal/jnp.linalg.norm(normal)
            grazing = jnp.arcsin(jnp.clip(-direction @ normal, -1., 1.))
            coefficients = self.coefficients_at(grazing, ray.voltage)
        amplitude = ray.amplitude[:, None] * coefficients * jnp.exp(1j * phase)
        slopes = ray.d_xy[:, None, :] + q[None, :, :] / ray.k[:, None, None]
        # to_vector broadcasts shared scalar fields before repeating parents.
        children = jax.tree.map(lambda value: jnp.repeat(value, q.shape[0], axis=0), ray)
        return children.derive(
            dx=slopes[..., 0].ravel(), dy=slopes[..., 1].ravel(),
            amplitude=amplitude.ravel(),
        )


class ScalarActionJet2D(NamedTuple):
    value: jnp.ndarray
    gradient_xy: jnp.ndarray
    hessian_xy: jnp.ndarray


class SampleActionJet2D(NamedTuple):
    pathlength: ScalarActionJet2D
    log_amplitude: ScalarActionJet2D


@jdc.pytree_dataclass(kw_only=True)
class InterpolatedSample2D(GaussianActionComponent):
    """A thin specimen described by interpolated amplitude and phase.

    Prefer ``from_action_maps`` for physical optical path and log field
    amplitude maps in image (y, x) order. It uses the C2 cubic interpolant
    and explicit spatial derivatives needed by Gaussian propagation.
    ``from_array`` retains the original complex-transmission API (x, y order).
    """

    interpolator: Interpolator2D
    pathlength_per_radian: float = 1.0
    method: jdc.Static[str] = "catmull-rom"
    z: float = 0.0
    action_maps: jdc.Static[bool] = False
    centre_xy_m: tuple = (0.0, 0.0)
    rotation_rad: float = 0.0
    tilt_xy_rad: tuple = (0.0, 0.0)
    domain_bounds_xyxy_m: jdc.Static[tuple | None] = None
    boundary: jdc.Static[str] = "vacuum"
    log_floor: jdc.Static[float] = -50.0
    periodic_origin_xy_m: jdc.Static[tuple] = (0.0, 0.0)

    @classmethod
    def from_action_maps(
        cls,
        *,
        pathlength_yx_m,
        log_amplitude_yx,
        x_coords_m,
        y_coords_m,
        z=0.0,
        centre_xy_m=(0.0, 0.0),
        rotation_rad=0.0,
        tilt_xy_rad=(0.0, 0.0),
        boundary="vacuum",
        log_floor=-50.0,
    ):
        """Build a fixed specimen from two real maps and pixel centres in metres.

        Both maps have shape (len(y_coords_m), len(x_coords_m)). Coordinates
        must be uniformly spaced with at least four knots per axis. Keep the
        illuminated field inside the map, or choose a suitable boundary mode.
        Construction runs on the host; evaluation and propagation support JAX.
        """
        path = np.asarray(pathlength_yx_m, dtype=np.float64)
        log_amplitude = np.asarray(log_amplitude_yx, dtype=np.float64)
        x_coords = np.asarray(x_coords_m, dtype=np.float64)
        y_coords = np.asarray(y_coords_m, dtype=np.float64)
        if path.ndim != 2 or path.shape != log_amplitude.shape:
            raise ValueError("Supply equal-shape 2D pathlength and log-amplitude maps.")
        if path.shape != (y_coords.size, x_coords.size):
            raise ValueError("Maps must use (y, x) image order.")
        for coords in (x_coords, y_coords):
            if coords.ndim != 1 or coords.size < 4:
                raise ValueError("C2 cubic interpolation needs four knots per axis.")
            spacing = np.diff(coords)
            if not np.all(spacing > 0) or not np.allclose(
                spacing, spacing[0], rtol=1e-9, atol=0
            ):
                raise ValueError("Coordinates must be increasing and uniformly spaced.")
        if not all(
            np.isfinite(a).all() for a in (path, log_amplitude, x_coords, y_coords)
        ):
            raise ValueError("Maps and coordinates must be finite.")

        def finite_pair(name, value):
            pair = np.asarray(value, dtype=np.float64)
            if pair.shape != (2,) or not np.isfinite(pair).all():
                raise ValueError(f"{name} must contain two finite values.")
            return pair

        if boundary == "clamp":
            boundary = "constant"
        if boundary not in ("vacuum", "constant", "periodic", "nan"):
            raise ValueError(
                "boundary must be 'vacuum', 'constant', 'periodic', or 'nan'."
            )
        centre = np.asarray(finite_pair("centre_xy_m", centre_xy_m), dtype=np.float64)
        if not np.isfinite(rotation_rad) or not np.isfinite(z):
            raise ValueError("rotation_rad and z must be finite.")
        tilt = np.asarray(finite_pair("tilt_xy_rad", tilt_xy_rad), dtype=np.float64)
        if np.any(np.abs(tilt) >= 0.5 * np.pi - 1e-6):
            raise ValueError("tilt_xy_rad must keep the projected sample plane nonsingular.")
        if not np.isfinite(log_floor):
            raise ValueError("log_floor must be finite.")
        # Interpax's first two data axes are x then y.  Public storage remains
        # conventional yx and this is the only conversion point.
        action_xyc = np.stack(
            (path.T, log_amplitude.T), axis=-1
        )
        extrap = np.zeros((2,), dtype=np.float64) if boundary == "vacuum" else False
        domain_bounds = (
            float(x_coords[0]),
            float(x_coords[-1]),
            float(y_coords[0]),
            float(y_coords[-1]),
        )
        if boundary == "constant":
            # Queries are clamped before evaluation; True prevents a spurious
            # NaN at floating-point roundoff just outside the final knot.
            extrap = True
        periodic_origin = (0.0, 0.0)
        derivative_kwargs: dict[str, np.ndarray] = {}
        if boundary == "periodic":
            # Append the wrapped seam knot and solve periodic cubic splines for
            # the cached first/mixed derivatives.  Interpax's default cubic2
            # derivative solve is not periodic; padding it merely makes the
            # seam error smaller, not C2.  These tensor-product derivatives
            # make the two limiting bicubic patches agree through order two.
            dy, dx = y_coords[1] - y_coords[0], x_coords[1] - x_coords[0]
            period_x = x_coords.size * dx
            period_y = y_coords.size * dy
            x_coords = np.concatenate(
                (x_coords, [x_coords[0] + period_x])
            )
            y_coords = np.concatenate(
                (y_coords, [y_coords[0] + period_y])
            )
            action_xyc = np.concatenate((action_xyc, action_xyc[:1]), axis=0)
            action_xyc = np.concatenate((action_xyc, action_xyc[:, :1]), axis=1)
            fx = CubicSpline(
                x_coords, action_xyc, axis=0, bc_type="periodic"
            )(x_coords, 1)
            fy = CubicSpline(
                y_coords, action_xyc, axis=1, bc_type="periodic"
            )(y_coords, 1)
            fxy = CubicSpline(
                y_coords, fx, axis=1, bc_type="periodic"
            )(y_coords, 1)
            derivative_kwargs = {"fx": fx, "fy": fy, "fxy": fxy}
            # Public wrapping uses the conventional pixel-edge extent.  That
            # interval has the same period as the spline and places its seam
            # halfway between the final and first stored pixel centres.
            domain_bounds = (
                float(x_coords[0] - dx / 2), float(x_coords[-2] + dx / 2),
                float(y_coords[0] - dy / 2), float(y_coords[-2] + dy / 2),
            )
            periodic_origin = (float(x_coords[0]), float(y_coords[0]))
            extrap = True
        interpolator = Interpolator2D(
            x=x_coords,
            y=y_coords,
            f=action_xyc,
            method="cubic2",
            extrap=extrap,
            **derivative_kwargs,
        )
        return cls(
            interpolator=interpolator,
            centre_xy_m=jnp.asarray(centre),
            rotation_rad=jnp.asarray(rotation_rad, dtype=jnp.float64),
            tilt_xy_rad=jnp.asarray(tilt, dtype=jnp.float64),
            domain_bounds_xyxy_m=domain_bounds,
            z=float(z),
            boundary=boundary,
            action_maps=True,
            method="cubic2",
            log_floor=float(log_floor),
            periodic_origin_xy_m=periodic_origin,
        )

    @property
    def rotation_world_from_local(self) -> jnp.ndarray:
        c = jnp.cos(self.rotation_rad)
        s = jnp.sin(self.rotation_rad)
        return jnp.asarray(((c, -s), (s, c)), dtype=jnp.float64)

    @property
    def projection_world_from_local(self) -> jnp.ndarray:
        """Orthographic xy projection of the tilted physical sample plane."""

        tx, ty = self.tilt_xy_rad
        cx, sx = jnp.cos(tx), jnp.sin(tx)
        cy, sy = jnp.cos(ty), jnp.sin(ty)
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        one = jnp.asarray(1.0, dtype=jnp.float64)
        rotation_x = jnp.asarray(
            ((one, zero, zero), (zero, cx, -sx), (zero, sx, cx))
        )
        rotation_y = jnp.asarray(
            ((cy, zero, sy), (zero, one, zero), (-sy, zero, cy))
        )
        c = jnp.cos(self.rotation_rad)
        s = jnp.sin(self.rotation_rad)
        rotation_z = jnp.asarray(
            ((c, -s, zero), (s, c, zero), (zero, zero, one))
        )
        return (rotation_z @ rotation_y @ rotation_x)[:2, :2]

    @property
    def projected_path_scale(self) -> jnp.ndarray:
        """Normal-thickness multiplier for the tilted projected action."""

        tx, ty = self.tilt_xy_rad
        return 1.0 / jnp.abs(jnp.cos(tx) * jnp.cos(ty))

    def world_to_local(self, xy_world: jnp.ndarray) -> jnp.ndarray:
        xy_world = jnp.asarray(xy_world, dtype=jnp.float64)
        if xy_world.shape[-1] != 2:
            raise ValueError("xy_world must have final dimension two.")
        local_from_world = jnp.linalg.inv(self.projection_world_from_local)
        return jnp.einsum(
            "ij,...j->...i", local_from_world, xy_world - self.centre_xy_m
        )

    def _local_action_jet(self, xy_local: jnp.ndarray):
        x = xy_local[..., 0]
        y = xy_local[..., 1]
        query_shape = x.shape
        x_min, x_max, y_min, y_max = self.domain_bounds_xyxy_m

        if self.boundary == "constant":
            inside_x = (x >= x_min) & (x <= x_max)
            inside_y = (y >= y_min) & (y <= y_max)
            xq = jnp.clip(x, x_min, x_max)
            yq = jnp.clip(y, y_min, y_max)
        elif self.boundary == "periodic":
            inside_x = jnp.ones_like(x, dtype=bool)
            inside_y = jnp.ones_like(y, dtype=bool)
            origin_x, origin_y = self.periodic_origin_xy_m
            xq = origin_x + jnp.mod(x - origin_x, x_max - x_min)
            yq = origin_y + jnp.mod(y - origin_y, y_max - y_min)
        else:
            inside_x = jnp.ones_like(x, dtype=bool)
            inside_y = jnp.ones_like(y, dtype=bool)
            xq, yq = x, y

        # Interpax 0.3.x documents arbitrary query shapes, but its bicubic
        # channel broadcasting is reliable for a one-dimensional query axis.
        # Flatten and restore here so image-shaped and batched calls have the
        # same public contract.
        xq_flat = jnp.ravel(xq)
        yq_flat = jnp.ravel(yq)

        def evaluate(dx: int, dy: int):
            values = self.interpolator(xq_flat, yq_flat, dx=dx, dy=dy)
            return values.reshape(query_shape + (2,))

        value = evaluate(0, 0)
        fx = evaluate(1, 0)
        fy = evaluate(0, 1)
        fxx = evaluate(2, 0)
        fxy = evaluate(1, 1)
        fyy = evaluate(0, 2)

        gradient = jnp.stack((fx, fy), axis=-1)
        hessian = jnp.stack(
            (
                jnp.stack((fxx, fxy), axis=-1),
                jnp.stack((fxy, fyy), axis=-1),
            ),
            axis=-2,
        )
        if self.boundary == "constant":
            derivative_mask = jnp.stack((inside_x, inside_y), axis=-1)
            gradient = gradient * derivative_mask[..., None, :]
            hessian = hessian * (
                derivative_mask[..., :, None] * derivative_mask[..., None, :]
            )[..., None, :, :]
        return value, gradient, hessian

    def action_jet(self, xy_world: jnp.ndarray) -> SampleActionJet2D:
        """Evaluate both actions and their explicit world-coordinate jets."""

        if not self.action_maps:
            def jet(point):
                return SampleActionJet2D(
                    ScalarActionJet2D(*_value_grad_hessian(self.phase_shift, point)),
                    ScalarActionJet2D(*_value_grad_hessian(self.log_transmission, point)),
                )
            xy = jnp.asarray(xy_world)
            if xy.ndim == 1:
                return jet(xy)
            flat = jax.vmap(jet)(xy.reshape(-1, 2))
            return jax.tree.map(lambda a: a.reshape(xy.shape[:-1] + a.shape[1:]), flat)
        local = self.world_to_local(xy_world)
        value, local_gradient, local_hessian = self._local_action_jet(local)
        local_from_world = jnp.linalg.inv(self.projection_world_from_local)
        action_scale = self.projected_path_scale
        value = value * action_scale
        gradient = action_scale * jnp.einsum(
            "ji,...cj->...ci", local_from_world, local_gradient
        )
        hessian = jnp.einsum(
            "ji,...cjk,kl->...cil",
            local_from_world,
            local_hessian,
            local_from_world,
        ) * action_scale
        return SampleActionJet2D(
            pathlength=ScalarActionJet2D(
                value[..., 0], gradient[..., 0, :], hessian[..., 0, :, :]
            ),
            log_amplitude=ScalarActionJet2D(
                value[..., 1], gradient[..., 1, :], hessian[..., 1, :, :]
            ),
        )

    def evaluate_actions(self, xy_world: jnp.ndarray) -> jnp.ndarray:
        """Return ``[..., (pathlength_m, log_field_amplitude)]``."""

        jet = self.action_jet(xy_world)
        return jnp.stack((jet.pathlength.value, jet.log_amplitude.value), axis=-1)

    def phase_shift(self, xy):
        if self.action_maps:
            return self.action_jet(xy).pathlength.value
        return jnp.angle(self.evaluate_complex(xy)) * self.pathlength_per_radian

    def log_transmission(self, xy):
        if self.action_maps:
            return self.action_jet(xy).log_amplitude.value
        value = self.evaluate_complex(xy)
        return 0.5 * jnp.log(jnp.real(value)**2 + jnp.imag(value)**2 + 1e-30)

    def _call_ray(self, ray: Ray):
        return ray

    @staticmethod
    def _floored_log_amplitude_jet(jet: ScalarActionJet2D, floor: float):
        # logaddexp(L, floor) and its analytic first/second derivatives.
        weight = jax.nn.sigmoid(jet.value - floor)
        value = jnp.logaddexp(jet.value, floor)
        gradient = weight[..., None] * jet.gradient_xy
        hessian = (
            weight[..., None, None] * jet.hessian_xy
            + (weight * (1.0 - weight))[..., None, None]
            * jnp.einsum("...i,...j->...ij", jet.gradient_xy, jet.gradient_xy)
        )
        return ScalarActionJet2D(value, gradient, hessian)

    def _call_gaussian_single(self, ray: GaussianBeam) -> GaussianBeam:
        if not self.action_maps:
            return super()._call_gaussian_single(ray)
        jet = self.action_jet(ray.r_xy)
        log_jet = self._floored_log_amplitude_jet(jet.log_amplitude, self.log_floor)
        return _apply_gaussian_action(ray, jet.pathlength, log_jet)

    @classmethod
    def from_array(
        cls,
        sample,
        x_coords,
        y_coords,
        extrap=1.0,
        z=0.0,
        method="cubic",
        *,
        voltage: float | None = None,
        wavelength: float | None = None,
        wavelength_unit: str = "m",
    ):
        """Create a sample from a complex amplitude transmission.

        Provide ``voltage`` (electron optics) or ``wavelength`` when the
        complex argument is a physical phase in radians. With neither, the
        argument is interpreted as a pathlength for compatibility with older
        TemGymCore notebooks. An explicit ``wavelength`` is expressed in
        ``wavelength_unit``.
        """
        if voltage is not None and wavelength is not None:
            raise ValueError("Provide at most one of `voltage` or `wavelength`.")
        if wavelength_unit not in LENGTH:
            raise ValueError(
                f"Unknown wavelength unit {wavelength_unit!r}; "
                f"expected one of {tuple(LENGTH)}."
            )
        if voltage is not None:
            wavelength = energy2wavelength(voltage) / LENGTH[wavelength_unit]
        pathlength_per_radian = (
            1.0 if wavelength is None else wavelength / (2.0 * jnp.pi)
        )
        interpolator = Interpolator2D(
            x=x_coords,
            y=y_coords,
            f=sample,
            method=method,
            extrap=extrap,
        )
        return cls(
            z=z,
            interpolator=interpolator,
            method=method,
            pathlength_per_radian=pathlength_per_radian,
        )

    def evaluate_complex(self, xy, *, wavelength_m=None):
        """Field transmission; action maps require the electron wavelength."""
        if not self.action_maps:
            return self.interpolator(xy[..., 0], xy[..., 1])
        if wavelength_m is None:
            raise ValueError("Supply wavelength_m to convert optical path to phase.")
        actions = self.evaluate_actions(xy)
        return jnp.exp(actions[..., 1] + 2j * jnp.pi * actions[..., 0] / wavelength_m)


def sample_interpolant(
    sample,
    x_coords,
    y_coords,
    *,
    z: float = 0.0,
    method: str = "cubic",
    extrap: float = 1.0,
    voltage: float | None = None,
    wavelength: float | None = None,
    wavelength_unit: str = "m",
) -> InterpolatedSample2D:
    """Create an interpolated sample component from a complex sample array.

    This is a convenience wrapper around ``InterpolatedSample2D.from_array``
    for notebook and pipeline code that expects a ``sample_interpolant`` API.
    """
    return InterpolatedSample2D.from_array(
        sample=sample,
        x_coords=x_coords,
        y_coords=y_coords,
        extrap=extrap,
        z=z,
        method=method,
        voltage=voltage,
        wavelength=wavelength,
        wavelength_unit=wavelength_unit,
    )


@jdc.pytree_dataclass(kw_only=True)
class InterpolatedFields3D(Component):
    """Interpolated field container; evaluate(xyz) returns the stored fields.

    Beam interaction is not implemented: calling this component currently
    returns the beam unchanged.
    """

    interpolator: Interpolator3D
    method: jdc.Static[str] = "catmull-rom"
    z: float = 0.0

    @classmethod
    def from_array(cls, fields, x_coords, y_coords, z_coords, *, method="cubic"):
        interpolator = Interpolator3D(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            f=fields,
            method=method,
            extrap=0.0,
        )
        return cls(interpolator=interpolator, method=method)

    def evaluate(self, xyz):
        x, y, z = xyz
        return self.interpolator(x, y, z)

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray


__all__ = [
    "Component",
    "GaussianActionComponent",
    "DescanError",
    "Plane",
    "Lens",
    "Stigmator",
    "KrivanekLens",
    "AberratedLensKrivanek",
    "SeidelLens",
    "DistortedLens",
    "ScanGrid",
    "Scanner",
    "Descanner",
    "Detector",
    "ThickLens",
    "HardEdgeSolenoid",
    "solenoid_column",
    "Deflector",
    "DoubleDeflector",
    "Rotator",
    "DeflectionBiprism",
    "PhaseBiprism",
    "Biprism",
    "RotatingLens",
    "ElectromagneticLens",
    "ABCDTransfer",
    "SigmoidAperture",
    "ConstantPhaseShift",
    "LinearPhaseShift",
    "QuadraticPhaseShift",
    "ZernikeAberration",
    "ConstantAmplitudeShift",
    "LinearAmplitudeShift",
    "QuadraticAmplitudeShift",
    "MagneticPhaseSample",
    "RandomPhaseSample",
    "BraggSample",
    "InterpolatedSample2D",
    "sample_interpolant",
    "InterpolatedFields3D",
    "FourierTransform",
]
