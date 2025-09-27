import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax.nn import softplus
import interpax as interp
from typing import Sequence, Callable, Any, Generator, Union, NamedTuple
from temgym_core.components import Detector
from temgym_core.gaussian import GaussianRayBeta
from temgym_core.ray import Ray
from inspect import signature


def grad_complex_action(component, xy, k):
    """Gradient of complex_action w.r.t xy for a single XY. Returns complex gradient (2,)."""
    def vec(xy):
        z = component.complex_action(xy, k)
        return jnp.stack([jnp.real(z), jnp.imag(z)])  # (2,)

    # jacobian(vec) -> (2,2)
    J = jax.jacobian(vec)(xy)
    g_phi = J[0, :]
    g_imag = J[1, :]
    return g_phi + 1j * g_imag


def hess_complex_action(component, xy, k):
    """Symmetric Hessian of complex_action w.r.t xy for a single XY. Returns complex Hessian (2,2)."""
    def vec(xy):
        z = component.complex_action(xy, k)
        return jnp.stack([jnp.real(z), jnp.imag(z)])

    # H shape (out=2, in=2, in=2)
    H = jax.jacfwd(jax.jacrev(vec))(xy)
    H_phi = H[0, :, :]
    H_imag = H[1, :, :]
    Hc = H_phi + 1j * H_imag
    # ensure symmetry (numerical)
    return 0.5 * (Hc + jnp.swapaxes(Hc, -1, -2))


@jdc.pytree_dataclass
class Component:
    z: float

    def opl_shift(self, xy):
        raise NotImplementedError

    def transmission(self, xy):
        raise NotImplementedError

    def complex_action(self, xy, k):
        # default: compose from the clear channels (keeps k argument for elements that need it)
        opd = self.opl_shift(xy)
        t = self.transmission(xy)
        L = -jnp.log(t + 1e-16)  # attenuation length (meters), t in [0,1]
        return opd + 1j * (L / k)

    def __call__(self, ray: GaussianRayBeta):
        # Expect a single ray (no internal vmapping). Users should vmap this __call__ outside.
        r = jnp.asarray(ray.r_xy)  # shape (2,)
        k = ray.k

        # ΔS and its derivatives (meters, complex) for a single XY
        dS0 = self.complex_action(r, k)                     # complex scalar
        dS1 = grad_complex_action(self, r, k)  # complex (2,)
        dS2 = hess_complex_action(self, r, k)  # complex (2,2)

        S_out = jdc.replace(
            ray.S,
            const=ray.S.const + dS0,
            lin=ray.S.lin + dS1,
            quad=ray.S.quad + dS2,
        )

        # −ikS convention: kick from +Re(∇ΔS) only
        dxy_out = ray.d_xy + jnp.real(dS1)

        dx_new = dxy_out[0]
        dy_new = dxy_out[1]

        return ray.derive(S=S_out, dx=dx_new, dy=dy_new, z=self.z)


@jdc.pytree_dataclass
class Lens(Component):
    focal_length: float  # f > 0 focusing, f < 0 defocusing

    def opl_shift(self, xy):
        x, y = xy[0], xy[1]
        rho2 = x * x + y * y
        # Pure phase (real) ΔS; amplitude unchanged
        return -0.5 * rho2 / self.focal_length

    def transmission(self, xy):
        return 0.0  # no amplitude change


@jdc.pytree_dataclass
class SigmoidAperture(Component):
    radius: float
    sharpness: float = 50.0     # 1/m
    t_outside: float = 0.1
    eps: float = 1e-12

    # Clear channel: pure attenuator → no phase
    def opl_shift(self, xy):
        return 0.0

    def transmission(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        r = jnp.sqrt(x*x + y*y + self.eps)
        s = jax.nn.sigmoid(self.sharpness * (r - self.radius))  # 0 in → 1 out
        return (1.0 - s) + s * self.t_outside                   # 1 inside, t_out outside


@jdc.pytree_dataclass
class AberratedLens(Component):
    focal_length: float
    C_sph: float = 0.0
    C_coma_x: float = 0.0
    C_coma_y: float = 0.0

    def complex_action(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        rho2 = x*x + y*y

        return (
            -0.5 * rho2 / self.focal_length
            + self.C_sph * (rho2**2)
            + self.C_coma_x * (x**3 + x*y**2)
            + self.C_coma_y * (y**3 + y*x**2)
        ) + 0.0j


@jdc.pytree_dataclass
class SigmoidAperture(Component):
    """
    Circular, smooth-rolloff aperture (length-native).
    Pure attenuation: ΔS = -i * L_a(x,y)   [meters]
    """
    radius: float              # aperture radius [m]
    sharpness: float = 50.0    # edge steepness (1/m scaled into t)
    L_outside: float = 0.0     # attenuation length far outside [m]
    eps: float = 1e-12

    def complex_action(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        r = jnp.sqrt(x*x + y*y + self.eps)

        # Smooth edge: inside -> 0, outside -> 1
        t = self.sharpness * (r - self.radius)
        s = jax.nn.sigmoid(t)            # 0 inside, →1 outside (C∞-smooth)
        L_a = self.L_outside * s          # meters

        # Pure amplitude: imaginary negative ΔS; produces no ray kick
        return -1j * L_a


@jdc.pytree_dataclass
class SigmoidSquareAperture(Component):
    """Smooth square aperture with soft roll-off.

    This is the rectangular analogue of ``SigmoidAperture`` (circular).
    It produces a pure amplitude mask A(x,y) ≈ 1 inside the box
    |x| ≤ half_width_x and |y| ≤ half_width_y, and decays smoothly to 0
    outside with an exponential-like tail controlled by ``sharpness``.

    Returned complex action: ψ(x,y) = φ(x,y) - i ℓ(x,y) with φ=0 and
    ℓ(x,y) = log A(x,y) ≤ 0 so the element only affects amplitude while
    remaining fully differentiable for gradient/Hessian extraction.
    """
    half_width_x: float
    half_width_y: float | None = None
    sharpness: float = 50.0
    eps: float = 1e-12

    def complex_action(self, xy):
        x, y = xy[..., 0], xy[..., 1]

        # Allow square via defaulting y half-width to x half-width
        hx = self.half_width_x
        hy = self.half_width_y if self.half_width_y is not None else hx

        # Smooth absolute value for differentiable gradients/Hessians
        ax = jnp.sqrt(x * x + self.eps)
        ay = jnp.sqrt(y * y + self.eps)

        # Signed distance-like terms to each pair of parallel edges
        # Negative inside, positive outside.
        tx = self.sharpness * (ax - hx)
        ty = self.sharpness * (ay - hy)

        # Soft roll-off on each axis; product in amplitude => sum in log-space
        logA = -softplus(tx) - softplus(ty)

        # Pure amplitude mask: ψ = φ - iℓ, here φ=0, ℓ = logA (≤ 0)
        return -1j * logA


@jdc.pytree_dataclass
class Potential(Component):
    V: jnp.ndarray          # shape (Ny, Nx), y is rows, x is cols
    sx: float               # pixel size in x (same units as x queries)
    sy: float               # pixel size in y
    x0: float               # physical x of pixel index 0
    y0: float               # physical y of pixel index 0
    method: str = 'catmull-rom'
    order: int = 3

    # Cache the compiled interpolator; exclude from pytree so JIT treats it as static
    _interp: jdc.Static[interp.Interpolator2D] = None

    def __post_init__(self):
        Ny, Nx = self.V.shape

        # Physical knot positions (Å) for each pixel center
        x_phys = self.x0 + self.sx * jnp.arange(Nx)  # (Nx,)
        y_phys = self.y0 + self.sy * jnp.arange(Ny)  # (Ny,)

        # interpax.Interpolator2D expects f shaped (Nx, Ny, ...): transpose image
        f = self.V.T  # (Nx, Ny)

        # Optional: map your legacy 'order' to a method
        method = self.method
        if self.method is None:
            method = 'linear' if self.order == 1 else 'cubic2'

        # NOTE: set extrap to a value if you’d rather not get NaNs outside the grid.
        object.__setattr__(
            self,
            "_interp",
            interp.Interpolator2D(x_phys, y_phys, f, method=method, extrap=[0.0, 0.0]),
        )

    def __call__(self, ray: GaussianRayBeta):

        # (B,2) positions
        r_xy = ray.r_xy
        r_xy = _as_batch(r_xy)

        # Per-ray scalar factor
        sk = (ray.sigma / ray.k)

        # Scalar OPL term (B,)
        opl = sk * self.V_interp(r_xy)

        # Gradient (B,2) — broadcast scalar over last axis
        grad_real = sk[..., None] * self.grad_V_interp(r_xy)

        # Hessian (B,2,2) — broadcast scalar over the last two axes
        Hess_real = sk[..., None, None] * self.hess_V_interp(r_xy)

        # promote to complex (imaginary parts are zero)
        grad_opl = grad_real.astype(jnp.complex128)
        Hess_opl = Hess_real.astype(jnp.complex128)
        out = apply_thin_element_from_complex_opl(ray, opl, grad_opl, Hess_opl)

        return out.derive(z=self.z)

    def V_interp(self, xy):
        xq = xy[..., 0]
        yq = xy[..., 1]
        shp = xq.shape
        Vq = self._interp(jnp.ravel(xq), jnp.ravel(yq)).reshape(shp)
        return Vq

    def grad_V_interp(self, xy):
        xq = xy[..., 0]
        yq = xy[..., 1]

        shp = xq.shape
        dVdx = self._interp(jnp.ravel(xq), jnp.ravel(yq), dx=1, dy=0).reshape(shp)
        dVdy = self._interp(jnp.ravel(xq), jnp.ravel(yq), dx=0, dy=1).reshape(shp)
        J = jnp.stack([dVdx, dVdy], axis=-1)
        return J

    def hess_V_interp(self, xy):
        xq = xy[..., 0]
        yq = xy[..., 1]

        shp = xq.shape
        Vxx = self._interp(jnp.ravel(xq), jnp.ravel(yq), dx=2, dy=0).reshape(shp)
        Vxy = self._interp(jnp.ravel(xq), jnp.ravel(yq), dx=1, dy=1).reshape(shp)
        Vyy = self._interp(jnp.ravel(xq), jnp.ravel(yq), dx=0, dy=2).reshape(shp)
        H = jnp.stack([jnp.stack([Vxx, Vxy], -1), jnp.stack([Vxy, Vyy], -1)], -2)
        return H


@jdc.pytree_dataclass
class Biprism(Component):
    """
    Biprism with inverse sigmoid square aperture (blocked inside, open outside).

    Inside the (rotated / shifted) rectangle: amplitude ~ 0.
    Outside: amplitude ~ 1.
    Phase: linear in |u| (prism deflection).

    Parameters
    ----------
    strength : phase slope versus |u|
    width    : filament physical width (defines blocked square of side=width) along u
    length   : extent along v; if None => treated as infinite (no masking along v)
    theta    : rotation (radians)
    x0, y0   : center position
    sharpness: transition steepness
    eps      : fraction of half-width used to smooth |.| for differentiability
    """
    strength: float
    width: float
    length: float | None = None  # None => effectively infinite (no v aperture)

    theta: float = 0.0
    x0: float = 0.0
    y0: float = 0.0

    sharpness: float = 50.0
    eps: float = 1e-12

    def complex_action(self, xy):
        x, y = xy[..., 0], xy[..., 1]

        # shift & rotate into (u,v)
        xr, yr = x - self.x0, y - self.y0
        c, s = jnp.cos(self.theta), jnp.sin(self.theta)
        u = c * xr + s * yr
        v = -s * xr + c * yr

        # u half-width and smooth abs
        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)

        # Soft distance and inverse sigmoid aperture along u
        tx = self.sharpness * (au - hu)
        logA_u = -softplus(-tx)  # ≈ large negative inside (blocked), 0 outside

        # v dimension: only apply if finite length provided
        if self.length is None:
            logA_v = 0.0
        else:
            hv = 0.5 * self.length
            eps_v = self.eps * hu  # scale smoothing with filament width (could also use hv)
            av = jnp.sqrt(v * v + eps_v * eps_v)
            ty = self.sharpness * (av - hv)
            logA_v = -softplus(-ty)

        logA = logA_u + logA_v

        # Linear phase in |u|
        phi = -self.strength * au

        return phi - 1j * logA


@jdc.pytree_dataclass
class InverseSigmoidSquareAperture(Component):
    """
    Inverse of SigmoidSquareAperture.

    Inside the rectangle (|x| <= hx, |y| <= hy) amplitude -> ~0.
    Outside the rectangle amplitude ~1 (logA ≈ 0).
    Smoothly differentiable for gradient / Hessian use.

    Parameters
    ----------
    half_width_x : half-width along x
    half_width_y : half-width along y (defaults to half_width_x if None)
    sharpness    : controls steepness of the transition
    eps          : small number for smooth |x|
    """
    half_width_x: float
    half_width_y: float | None = None
    sharpness: float = 50.0
    eps: float = 1e-12

    def complex_action(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        hx = self.half_width_x
        hy = self.half_width_y if self.half_width_y is not None else hx

        # Smooth absolute values for differentiability
        ax = jnp.sqrt(x * x + self.eps)
        ay = jnp.sqrt(y * y + self.eps)

        # Signed (soft) distances from edges
        tx = self.sharpness * (ax - hx)
        ty = self.sharpness * (ay - hy)

        # Original square aperture used: logA = -softplus(tx) - softplus(ty)
        # which gives A≈1 inside, decays outside.
        # Invert: want A≈0 inside, A≈1 outside.
        # Use -softplus(-tx): for tx<<0 (inside) -> large negative; for tx>>0 (outside) -> ~0.
        logA = -softplus(-tx) - softplus(-ty)

        # Pure amplitude mask (φ=0)
        return -1j * logA


def jacobian_and_value(fn, argnums: int = 0, **jac_kwargs):
    def inner(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out, out
    return jax.jacobian(inner, argnums=argnums, has_aux=True, **jac_kwargs)


def passthrough_transform(component):
    def inner(ray):
        out = component(ray)
        return out, out
    return inner


def jacobian_transform(component):
    def inner(ray):
        jac, out = jacobian_and_value(component)(ray)
        return out, jac
    return inner


class BaseGaussianPropagator:
    """Base propagator for (Gaussian) rays in gaussian_taylor."""
    def propagate(self, ray: GaussianRayBeta, distance: float):
        raise NotImplementedError

    def with_distance(self, distance: float):
        return GaussianPropagator(distance, self)


class GaussianPropagator(NamedTuple):
    distance: float
    propagator: BaseGaussianPropagator

    def __call__(self, ray: GaussianRayBeta):
        return self.propagator.propagate(ray, self.distance)


class FreeSpaceParaxial(BaseGaussianPropagator):
    def propagate(self, ray: GaussianRayBeta, distance: float):
        L = distance

        x_new = ray.x + ray.dx * L
        y_new = ray.y + ray.dy * L
        z_new = ray.z + L

        dS_real = L * (1.0 + 0.5 * (ray.dx**2 + ray.dy**2))
        S0_new = ray.S.const + dS_real

        Q = ray.S.quad
        I2 = jnp.eye(2, dtype=jnp.complex128)
        M = I2 + L * Q
        M_inv = jnp.linalg.solve(M, I2)

        Q_new = Q @ M_inv
        eta_new = (jnp.swapaxes(M_inv, -1, -2) @ ray.S.lin[..., None])[..., 0]

        # geometric spreading (k-free)
        C_new = ray.C / jnp.sqrt(jnp.linalg.det(M))

        return ray.derive(
            x=x_new, y=y_new, z=z_new,
            S=jdc.replace(ray.S, const=S0_new, lin=eta_new, quad=Q_new),
            C=C_new,
        )


TransformT = Callable[[Any], Callable[[Any], tuple[Any, Any]]]


def run_iter(
    ray: Union[GaussianRayBeta, Any],
    components: Sequence[Any],
    transform: TransformT = passthrough_transform,
    propagator: BaseGaussianPropagator = FreeSpaceParaxial(),
) -> Generator[tuple[Any, Any], Any, None]:
    """
    Iterate a ray (GaussianRayBeta or standard Ray) through the model, yielding each step's output.
    Free-space (paraxial) propagation is inserted when component.z != ray.z.
    """
    for component in components:
        if isinstance(component, (Component, Detector)):  # Include Detector as a valid type
            ray_z = ray.z
            distance = component.z - ray_z
            propagator_d = propagator.with_distance(distance)
            ray, out = transform(propagator_d)(ray)
            yield propagator_d, out
        ray, out = transform(component)(ray)
        yield component, out


def run_to_end(
    ray: Union[GaussianRayBeta, Any],
    components: Sequence[Any],
    propagator: FreeSpaceParaxial = FreeSpaceParaxial(),
) -> Union[GaussianRayBeta, Any]:
    """
    Propagate a ray through all components and return the final state.
    """
    for _, ray in run_iter(ray, components, propagator=propagator):
        pass
    return ray
