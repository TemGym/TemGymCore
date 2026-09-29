"""Lenses, matrix transport, solenoids and electromagnetic lenses."""


from __future__ import annotations

import dataclasses
import warnings


import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import lax

from .. import Degrees
from ..gaussian import (
    FreeSpacePropagator,
    GaussianBeam,
    _pupil_action_to_spatial,
    apply_action_delta,
    taylor_expand,
)
from ..aberrations import (
    KrivanekCoeffs,
    SeidelCoeffs,
    Seidel_object_pos_aperture_pos,
    W_krivanek,
)
from ..constants import (
    compute_Rc_from_voltage,
    effective_accelerating_potential,
)
from ..ray import Ray



from ._base import (
    Component,
    GaussianActionComponent,
    _call_single_or_batched_gaussian,
    _matmat,
    _matvec,
    _rotate_quadratic_form,
    _uses_default_log_transmission,
)
from .planes import Plane
from .deflection import Rotator


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

    Rays and Gaussian beamlets take the same local optical action, ``phase_shift``: the ideal
    lens path plus the aberration function of the aperture position.
    """

    coeffs: KrivanekCoeffs = dataclasses.field(default_factory=KrivanekCoeffs)
    axis_eps: float = 1e-24

    def _call_gaussian(self, ray: GaussianBeam):
        # The aberration needs the generic local-action expansion rather than
        # Lens's exact quadratic fast path.
        return GaussianActionComponent._call_gaussian(self, ray)

    def _call_ray(self, ray: Ray):
        return GaussianActionComponent._call_ray(self, ray)

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
    Rays and Gaussian beamlets accumulate the same quadratic action; coherent
    transport adds the Gouy factor. The coherent branch covers the first radial
    half-period, abs(q) < pi; longer fields must be split into shorter bodies to
    continue the phase.
    """

    z: float
    length: float
    q: float

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

    @staticmethod
    def _action(ray_in, ray_out):
        """Homogeneous quadratic canonical action, including magnetic rotation."""
        return .5 * (jnp.sum(ray_out.r_xy * ray_out.d_xy, axis=-1)
                     - jnp.sum(ray_in.r_xy * ray_in.d_xy, axis=-1))

    def _call_ray(self, ray):
        m = self.matrix
        out = ABCDTransfer(A=m[:2, :2], B=m[:2, 2:],
                           C=m[2:, :2], D=m[2:, 2:])(ray)
        return out.derive(z=ray.z + self.length,
                          pathlength=ray.pathlength + self._action(ray, out))

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
        valid = (jnp.abs(self.q) < jnp.pi) & (self.length > 0)
        return out.derive(
            z=ray.z + self.length,
            pathlength=ray.pathlength + self._action(ray, out),
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
        from ..propagator import FreeSpaceParaxial

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
            from ..propagator import FreeSpaceParaxial

            fs = FreeSpaceParaxial()

        ray = fs(ray, self.f)
        lens = Lens(z=ray.z, focal_length=self.f, x0=self.x0, y0=self.y0)
        ray = lens(ray)
        ray = fs(ray, self.f)
        return ray
