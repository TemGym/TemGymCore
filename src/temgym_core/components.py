from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from collections.abc import Mapping
from typing import Any, Dict, NamedTuple

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
    apply_action_delta,
    taylor_expand,
)
from .aberrations import (
    KrivanekCoeffs,
    SeidelCoeffs,
    Seidel_aperture_pos_aperture_slope,
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
from .potential import potential_smoothed
from .ray import Ray
from .tree_utils import HasParamsMixin


class Component(HasParamsMixin):
    """Unified base component for paraxial rays and gaussian beams."""

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


def _is_batched_gaussian(ray: GaussianBeam) -> bool:
    return jnp.asarray(ray.r_xy).ndim > 1


def _call_single_or_batched_gaussian(ray: GaussianBeam, single_call):
    if _is_batched_gaussian(ray):
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


def _uses_default_phase_shift(component: "GaussianActionComponent") -> bool:
    return type(component).phase_shift is GaussianActionComponent.phase_shift


def _uses_default_log_transmission(component: "GaussianActionComponent") -> bool:
    return type(component).log_transmission is GaussianActionComponent.log_transmission


def _matvec(matrix: jnp.ndarray, vector: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...j->...i", matrix, vector)


def _matmat(matrix: jnp.ndarray, tensor: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...jk->...ik", matrix, tensor)


def _rotate_quadratic_form(rotation: jnp.ndarray, q_inv: jnp.ndarray) -> jnp.ndarray:
    """Rotate Q consistently with r' = R r: Q' = R Q R.T."""
    return jnp.einsum("...ij,...jk,...lk->...il", rotation, q_inv, rotation)


class GaussianActionComponent(Component):
    """Mixin for components defined by an action (phase + log-transmission).

    Phase and log-transmission are Taylor-expanded *separately* to second
    order, then combined analytically into complex action coefficients:

        dS = phi - i * L / k

    This captures all physical effects of the transmission profile:

    * **Zeroth order** – amplitude scaling at the beam centre.
    * **First order** – sub-picometer beam-centre shift toward higher
      transmission.
    * **Second order** – beam-width modification from the curvature of
      the absorption profile (narrowing near absorbing edges).

    Separating the expansions avoids numerical issues that arise when
    autodiff is applied to non-smooth formulations of ``log_transmission``
    (e.g. ``abs`` → ``log``).  A safety regularisation on Im(Q') prevents
    degenerate cases where the Hessian of log-transmission would make the
    Gaussian envelope non-decaying.
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
        xy_ref = ray.r_xy
        k = ray.k

        if _uses_default_phase_shift(self):
            dS0, dS1, dS2 = _default_action_terms(xy_ref)
        else:
            def phase_scalar(xy):
                return jnp.real(self.phase_shift(xy))

            dS0, dS1, dS2 = _value_grad_hessian(phase_scalar, xy_ref)

        if not _uses_default_log_transmission(self):
            def logt_scalar(xy):
                return jnp.real(jnp.logaddexp(self.log_transmission(xy), -50.0))

            L0, L1, L2 = _value_grad_hessian(logt_scalar, xy_ref)

            # Suppress unreliable higher-order amplitude terms when the
            # implied recentering becomes comparable with the beam width.
            ImQ = jnp.imag(ray.Q_inv)
            dx_est = _solve_2x2(ImQ, L1) / k
            ratio = jnp.dot(L1, dx_est)
            scale = 1.0 / (1.0 + ratio)
            L1_eff = L1 * scale
            L2_eff = L2 * scale
            L2k = _regularize_L2k(L2_eff / k, ImQ)
            dS0 = dS0 - 1j * (L0 / k)
            dS1 = dS1 - 1j * (L1_eff / k)
            dS2 = dS2 - 1j * L2k

        r_xy, d_xy, amplitude, pathlength, Q_new = apply_action_delta(
            ray, dS0=dS0, dS1=dS1, dS2=dS2
        )

        return ray.derive(
            x=r_xy[..., 0],
            y=r_xy[..., 1],
            dx=d_xy[..., 0],
            dy=d_xy[..., 1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=Q_new,
        )


class DescanError(NamedTuple):
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
class Plane(Component):
    z: float

    def _call_ray(self, ray: Ray):
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        return ray


@jdc.pytree_dataclass
class Lens(GaussianActionComponent):
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
    coeffs: Dict = dataclasses.field(default_factory=dict)
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
            return -0.5 * rho2 / f

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
    object_plane_dist: float = 0.0
    coeffs: SeidelCoeffs = dataclasses.field(default_factory=SeidelCoeffs)
    object_xy: tuple[float, float] | None = None

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "SeidelLens is not implemented for singular rays. Use ray_family='gaussian'."
        )

    def log_transmission(self, xy):
        return 0.0

    def phase_shift(self, xy, dxy):
        xy = jnp.asarray(xy)
        x_a, y_a = xy[..., 0], xy[..., 1]
        rho2 = (x_a - self.x0) ** 2 + (y_a - self.y0) ** 2
        if self.object_xy is not None:
            aberration = Seidel_object_pos_aperture_pos(
                self.object_xy[0],
                self.object_xy[1],
                x_a,
                y_a,
                self.coeffs,
            )
        else:
            dxy = jnp.asarray(dxy)
            aberration = Seidel_aperture_pos_aperture_slope(
                x_a,
                y_a,
                dxy[..., 0],
                dxy[..., 1],
                self.object_plane_dist,
                self.coeffs,
            )
        return -0.5 * rho2 / self.focal_length - aberration

    def complex_action(self, xy: jnp.ndarray, dxy: jnp.ndarray, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy, dxy) - 1j * (L / k)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        return _call_single_or_batched_gaussian(ray, self._call_gaussian_single)

    def _call_gaussian_single(self, ray: GaussianBeam) -> GaussianBeam:
        dS0, dS1, dS2 = taylor_expand(
            self.complex_action,
            ray.r_xy,
            ray.d_xy,
            ray.k,
        )
        r_xy, d_xy, amplitude, pathlength, Q_new = apply_action_delta(
            ray,
            dS0=dS0,
            dS1=dS1,
            dS2=dS2,
        )
        return ray.derive(
            x=r_xy[..., 0],
            y=r_xy[..., 1],
            dx=d_xy[..., 0],
            dy=d_xy[..., 1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass
class DistortedLens(SeidelLens):
    IsoDist: float = 0.0
    AnisoDist: float = 0.0

    def phase_shift(self, xy, dxy):
        xy = jnp.asarray(xy)
        dxy = jnp.asarray(dxy)
        x_a, y_a = xy[..., 0], xy[..., 1]
        x_ap, y_ap = dxy[..., 0], dxy[..., 1]

        rho2 = x_a * x_a + y_a * y_a
        coeffs = SeidelCoeffs(E=self.IsoDist, e=self.AnisoDist)
        return -0.5 * rho2 / self.focal_length - Seidel_aperture_pos_aperture_slope(
            x_a,
            y_a,
            x_ap,
            y_ap,
            self.object_plane_dist,
            coeffs,
        )


@jdc.pytree_dataclass
class ScanGrid(Component, Grid):
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
class Scanner(Component):
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
        return ray.derive(
            x=ray.x + self.scan_pos_x * ray._one,
            y=ray.y + self.scan_pos_y * ray._one,
            dx=ray.dx + self.scan_tilt_x * ray._one,
            dy=ray.dy + self.scan_tilt_y * ray._one,
        )


@jdc.pytree_dataclass
class Descanner(Component):
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
        ox, oy, odx, ody = self._offsets()
        return ray.derive(
            x=ray.x + ox * ray._one,
            y=ray.y + oy * ray._one,
            dx=ray.dx + odx * ray._one,
            dy=ray.dy + ody * ray._one,
        )


@jdc.pytree_dataclass
class Detector(Component, Grid):
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
class ThickLens(Component):
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
class Deflector(Component):
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
        """Direct slope modification for deflector with implicit action.
        
        A deflector applies a transverse momentum kick: dx += def_x, dy += def_y
        The pathlength change arises from the phase shift: def_x * x + def_y * y
        
        For Gaussian beams, we apply this directly without using taylor_expand,
        as the implicit action (linear phase) can cause numerical issues.
        When def_x or def_y are non-zero (tilted), the action derivatives
        should be handled explicitly via the phase_shift method.
        """
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        new_dx = dx + self.def_x * ray._one
        new_dy = dy + self.def_y * ray._one
        
        # For a pure slope change with no amplitude loss,
        # the Q_inv (inverse waist parameter) is unchanged
        # but pathlength picks up the deflection-induced phase
        return ray.derive(
            dx=new_dx,
            dy=new_dy,
            pathlength=ray.pathlength + self.def_x * x + self.def_y * y,
        )


@jdc.pytree_dataclass
class DoubleDeflector(Component):
    """Two deflector kicks separated by an internal free-space spacing.

    User-facing commands (shift/tilt) are combined with hardware calibration
    ratios to produce the two physical kicks in each axis.
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

    @staticmethod
    def _apply_kick(ray: Ray | GaussianBeam, def_x, def_y):
        """Apply a deflection kick to a ray or beam.
        
        A transverse kick (def_x, def_y) imparts phase shift: def_x * x + def_y * y
        """
        return ray.derive(
            dx=ray.dx + def_x * ray._one,
            dy=ray.dy + def_y * ray._one,
            pathlength=ray.pathlength + def_x * ray.x + def_y * ray.y,
        )

    def _call_ray(self, ray: Ray):
        from .propagator import FreeSpaceParaxial

        ray = self._apply_kick(ray, self.def1_x, self.def1_y)
        ray = FreeSpaceParaxial()(ray, self.spacing)
        ray = self._apply_kick(ray, self.def2_x, self.def2_y)
        return ray

    def _call_gaussian(self, ray: GaussianBeam):
        ray = self._apply_kick(ray, self.def1_x, self.def1_y)
        ray = FreeSpacePropagator()(ray, self.spacing)
        ray = self._apply_kick(ray, self.def2_x, self.def2_y)
        return ray


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
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        return ray.derive(
            dx=dx + self.def_x * ray._one * jnp.sign(ray.x),
            dy=dy,
            pathlength=ray.pathlength + dx * x + dy * y,
        )


@jdc.pytree_dataclass
class Biprism(DeflectionBiprism):
    def __post_init__(self):
        warnings.warn(
            "`components.Biprism` is deprecated and will remain as an alias for "
            "`DeflectionBiprism`. Prefer `DeflectionBiprism` for clarity.",
            DeprecationWarning,
            stacklevel=2,
        )


@jdc.pytree_dataclass
class RotatingLens(Component):
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
        rotated_ray = Rotator(z=self.z, angle=self.rotation)(ray)
        return Lens(z=self.z, focal_length=self.focal_length, x0=self.x0, y0=self.y0)(
            rotated_ray
        )


@jdc.pytree_dataclass
class ElectromagneticLens(Component):
    z: float
    turns: float
    current: float
    Gc: float  # Pure geometry constant [1/(AT²·V·m)]
    Tc: float = 0.0  # Thickness constant
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

    def _call_ray(self, ray: Ray) -> Ray:
        from .propagator import FreeSpaceParaxial

        voltage = ray.voltage
        f = self.focal_length(voltage)
        D = self.thickness
        fx, fy = self._stigmator_focal_lengths(voltage)

        def _stigmator_path(_):
            def _thin(_):
                return self._apply_thin_stigmator(ray, fx, fy)

            def _thick(_):
                out = self._apply_thin_stigmator(ray, 2.0 * fx, 2.0 * fy)
                out = FreeSpaceParaxial()(out, D)
                return self._apply_thin_stigmator(out, 2.0 * fx, 2.0 * fy)

            return lax.cond(D > 0.0, _thick, _thin, operand=None)

        def _lens_path(_):
            def _thin(_):
                return self._apply_thin_lens_ray(ray, f)

            def _thick(_):
                out = self._apply_thin_lens_ray(ray, 2.0 * f)
                out = FreeSpaceParaxial()(out, D)
                return self._apply_thin_lens_ray(out, 2.0 * f)

            return lax.cond(D > 0.0, _thick, _thin, operand=None)

        out = lax.cond(self._stigmator_is_active(), _stigmator_path, _lens_path, operand=None)
        return self._apply_rotation_ray(out, self.rotation_angle(voltage))

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        from .gaussian import FreeSpacePropagator

        voltage = ray.voltage
        f = self.focal_length(voltage)
        D = self.thickness
        fx, fy = self._stigmator_focal_lengths(voltage)

        def _stigmator_path(_):
            def _thin(_):
                return self._apply_thin_stigmator(ray, fx, fy)

            def _thick(_):
                out = self._apply_thin_stigmator(ray, 2.0 * fx, 2.0 * fy)
                out = FreeSpacePropagator()(out, D)
                return self._apply_thin_stigmator(out, 2.0 * fx, 2.0 * fy)

            return lax.cond(D > 0.0, _thick, _thin, operand=None)

        def _lens_path(_):
            def _thin(_):
                thin_lens = Lens(
                    z=self.z,
                    focal_length=f,
                    x0=self.x0,
                    y0=self.y0,
                )
                return thin_lens(ray)

            def _thick(_):
                half_lens = Lens(
                    z=self.z,
                    focal_length=2.0 * f,
                    x0=self.x0,
                    y0=self.y0,
                )
                out = half_lens(ray)
                out = FreeSpacePropagator()(out, D)
                return half_lens(out)

            return lax.cond(D > 0.0, _thick, _thin, operand=None)

        out = lax.cond(self._stigmator_is_active(), _stigmator_path, _lens_path, operand=None)

        angle = self.rotation_angle(voltage)
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
class ABCDTransfer(Component):
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
        # Solve the transposed system to evaluate numer @ inv(denom) without
        # explicitly forming an inverse; swapaxes keeps bundle axes intact.
        Q_new = jnp.swapaxes(
            jnp.linalg.solve(
                jnp.swapaxes(denom, -1, -2),
                jnp.swapaxes(numer, -1, -2),
            ),
            -1,
            -2,
        )

        return ray.derive(
            x=r_xy_new[..., 0],
            y=r_xy_new[..., 1],
            dx=d_xy_new[..., 0],
            dy=d_xy_new[..., 1],
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass(kw_only=True)
class SigmoidAperture(GaussianActionComponent):
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

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "PhaseBiprism is only implemented for gaussian beams."
        )

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

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "ConstantPhaseShift is only implemented for gaussian beams."
        )

    def phase_shift(self, xy: jnp.ndarray):
        return self.constant_phase_shift


@jdc.pytree_dataclass(kw_only=True)
class LinearPhaseShift(GaussianActionComponent):
    linear_phase_shift: jnp.ndarray
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "LinearPhaseShift is only implemented for gaussian beams."
        )

    def phase_shift(self, xy: jnp.ndarray):
        return jnp.dot(self.linear_phase_shift, xy)


@jdc.pytree_dataclass(kw_only=True)
class QuadraticPhaseShift(GaussianActionComponent):
    quadratic_phase_shift: jnp.ndarray
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "QuadraticPhaseShift is only implemented for gaussian beams."
        )

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

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "ZernikeAberration is only implemented for gaussian beams."
        )

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

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "ConstantAmplitudeShift is only implemented for gaussian beams."
        )

    def log_transmission(self, xy: jnp.ndarray):
        return jnp.log(self.amplitude)


@jdc.pytree_dataclass(kw_only=True)
class LinearAmplitudeShift(GaussianActionComponent):
    linear_log_amplitude: jnp.ndarray | None = None
    linear_amplitude: jnp.ndarray | None = None
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "LinearAmplitudeShift is only implemented for gaussian beams."
        )

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

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "QuadraticAmplitudeShift is only implemented for gaussian beams."
        )

    def log_transmission(self, xy: jnp.ndarray):
        return 0.5 * xy @ self.quadratic_log_amplitude @ xy


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

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "MagneticPhaseSample is only implemented for gaussian beams."
        )

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

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "RandomPhaseSample is only implemented for gaussian beams."
        )

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
        k = ray.k

        # Match GaussianActionComponent's absorption recentering and Q safety
        # while using the spline's explicit derivatives rather than nested AD.
        im_q = jnp.imag(ray.Q_inv)
        dx_est = _solve_2x2(im_q, log_jet.gradient_xy) / k
        ratio = jnp.dot(log_jet.gradient_xy, dx_est)
        scale = 1.0 / (1.0 + ratio)
        log_gradient = log_jet.gradient_xy * scale
        log_hessian = log_jet.hessian_xy * scale
        log_hessian_over_k = _regularize_L2k(log_hessian / k, im_q)

        d_s0 = jet.pathlength.value - 1j * (log_jet.value / k)
        d_s1 = jet.pathlength.gradient_xy - 1j * (log_gradient / k)
        d_s2 = jet.pathlength.hessian_xy - 1j * log_hessian_over_k
        r_xy, d_xy, amplitude, pathlength, q_new = apply_action_delta(
            ray, dS0=d_s0, dS1=d_s1, dS2=d_s2
        )
        return ray.derive(
            x=r_xy[..., 0],
            y=r_xy[..., 1],
            dx=d_xy[..., 0],
            dy=d_xy[..., 1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=q_new,
        )

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        if jnp.asarray(ray.r_xy).ndim > 1:
            return jax.vmap(self._call_gaussian_single)(ray.to_vector())
        return self._call_gaussian_single(ray)

    def sampling_diagnostic(self, ray: GaussianBeam, **kwargs):
        """Run the repository's local quadratic-action adequacy diagnostic."""

        from .gaussian_corrections import diagnose_gaussian_action_sampling

        return diagnose_gaussian_action_sampling(self, ray, **kwargs)

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


@jdc.pytree_dataclass(kw_only=True)
class AtomicPotential(Component):
    atom_xyz: jnp.ndarray
    element_params: jnp.ndarray
    cutoff_radius: float
    z: float = 0.0

    def _call_ray(self, ray: Ray):
        raise NotImplementedError(
            "AtomicPotential is only implemented for gaussian beams."
        )

    def log_transmission(self, xy):
        return 0.0

    def phase_shift(self, xy: jnp.ndarray, z, sigma, k: float) -> complex:
        x, y = xy[..., 0], xy[..., 1]
        r = jnp.sqrt(
            (x - self.atom_xyz[0]) ** 2
            + (y - self.atom_xyz[1]) ** 2
            + (z - self.atom_xyz[2]) ** 2
        )
        V = potential_smoothed(r, self.element_params, self.cutoff_radius)
        interaction_constant = sigma / k
        return -interaction_constant * V

    def complex_action(self, xy: jnp.ndarray, z: jnp.ndarray, sigma, k: float) -> complex:
        logA = self.log_transmission(xy)
        L = jnp.logaddexp(logA, -50)
        return self.phase_shift(xy, z, sigma, k) - 1j * (L / k)

    def _call_gaussian(self, ray: GaussianBeam) -> GaussianBeam:
        return _call_single_or_batched_gaussian(ray, self._call_gaussian_single)

    def _call_gaussian_single(self, ray: GaussianBeam) -> GaussianBeam:
        dS0, dS1, dS2 = taylor_expand(
            self.complex_action,
            ray.r_xy,
            ray.z,
            ray.sigma,
            ray.k,
        )

        r_xy, d_xy, amplitude, pathlength, Q_new = apply_action_delta(
            ray,
            dS0=dS0,
            dS1=dS1,
            dS2=dS2,
        )

        return ray.derive(
            x=r_xy[..., 0],
            y=r_xy[..., 1],
            dx=d_xy[..., 0],
            dy=d_xy[..., 1],
            z=ray.z,
            amplitude=amplitude,
            pathlength=pathlength,
            Q_inv=Q_new,
        )


@jdc.pytree_dataclass(kw_only=True)
class FourierTransform:
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


__all__ = [
    "Component",
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
    "InterpolatedSample2D",
    "sample_interpolant",
    "InterpolatedFields3D",
    "AtomicPotential",
    "FourierTransform",
]
