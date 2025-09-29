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


def _grad_vjp(component, xy, k):
    """
    ∇_xy f(xy), where f: R^2 -> C. Returns shape (2,) complex.
    Uses VJP and seeds with ones_like(output) to match shape/dtype.
    """
    f = lambda x: component.complex_action(x, k)  # may return ()-shaped scalar or (1,) array
    y, pullback = jax.vjp(f, xy)
    seed = jnp.ones_like(y)  # ensures exact shape/dtype match ((),) or (1,)
    (g,) = pullback(seed)
    return g  # complex gradient (2,)


def grad_complex_action(component, xy, k):
    return _grad_vjp(component, xy, k)


def hess_complex_action(component, xy, k):
    """
    ∇^2_xy f(xy) via VJP-of-gradient. Returns (2,2) complex.
    """
    g = lambda x: _grad_vjp(component, x, k)  # R^2 -> C^2

    gxy, vjp_g = jax.vjp(g, xy)               # gxy: (2,) complex
    n = gxy.shape[-1]
    eye = jnp.eye(n, dtype=gxy.dtype)         # complex basis in C^n

    def apply(e):
        (ht_col,) = vjp_g(e)                  # H^T @ e
        return ht_col

    Ht = jax.vmap(apply)(eye)                 # (n,n) complex, this is H^T
    H = jnp.swapaxes(Ht, -1, -2)              # transpose to get H
    return 0.5 * (H + jnp.swapaxes(H, -1, -2))  # symmetrize (optional)


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
        L = jnp.log(t + 1e-16)  # attenuation length (meters), t in [0,1]
        return opd - 1j * (L / k)

    def __call__(self, ray: GaussianRayBeta):
        # Expect a single ray (no internal vmapping). Users should vmap this __call__ outside.
        r = jnp.asarray(ray.r_xy)  # shape (2,)
        k = ray.k

        # ΔS and its derivatives (meters, complex) for a single XY
        dS0 = self.complex_action(r, k)                     # complex scalar
        # # ensure a single 2-vector (drop leading 1-batch if present)
        # r_vec = jnp.asarray(r).reshape((2,))  # will convert (1,2) -> (2,) or leave (2,) unchanged

        dS1 = grad_complex_action(self, r, k)  # complex (2,)
        dS2 = hess_complex_action(self, r, k)  # complex (2,2)

        S_out = jdc.replace(
            ray.S,
            const=ray.S.const + dS0,
            lin=ray.S.lin + dS1,
            quad=ray.S.quad + dS2,
        )

        dxy_out = ray.d_xy + jnp.real(dS1)

        dx_new = jnp.atleast_1d(dxy_out[0])
        dy_new = jnp.atleast_1d(dxy_out[1])

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
        x, y = xy[0], xy[1]
        r = jnp.sqrt(x*x + y*y + self.eps)
        s = jax.nn.sigmoid(self.sharpness * (r - self.radius))  # 0 in → 1 out
        return (1.0 - s) + s * self.t_outside                   # 1 inside, t_out outside


@jdc.pytree_dataclass
class AberratedLens(Component):
    focal_length: float
    C_sph: float = 0.0
    C_coma_x: float = 0.0
    C_coma_y: float = 0.0

    def opl_shift(self, xy):
        x, y = xy[0], xy[1]
        rho2 = x*x + y*y

        opl = -(
            0.5 * rho2 / self.focal_length
            + self.C_sph * (rho2**2)
            + self.C_coma_x * (x**3 + x * y**2)
            + self.C_coma_y * (y**3 + y * x**2)
        )
        return opl

    def transmission(self, xy):
        return 0.0  # no amplitude change


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
    Biprism that follows the thin-element convention:
      - Real OPL shift encodes the prism's linear phase → deflection
      - Transmission encodes an inverse sigmoid rectangular aperture

    Inside the (rotated/shifted) rectangle: transmission ~ 0 (blocked).
    Outside: transmission ~ 1 (open).

    Parameters
    ----------
    strength : float
        Phase slope magnitude vs |u| (controls deflection).
        Units of radians per meter of |u| in OPL units.
    width : float
        Physical filament width along u; blocks |u| < width/2.
    length : float | None
        Extent along v; if None, no v-aperture (infinite length).
    theta : float
        Rotation angle (radians) for the (u,v) axes w.r.t lab (x,y).
    x0, y0 : float
        Center position of the filament.
    sharpness : float
        Transition steepness for the inverse sigmoid mask.
    eps : float
        Small smoothing factor for |.| to remain differentiable at u=0.
    """
    strength: float
    width: float
    length: float | None = None  # None => effectively infinite (no v aperture)

    theta: float = 0.0
    x0: float = 0.0
    y0: float = 0.0

    sharpness: float = 50.0
    eps: float = 1e-12

    # Helper: rotate/shift to (u,v)
    def _uv(self, xy):
        x, y = xy[0], xy[1]
        xr, yr = x - self.x0, y - self.y0
        c, s = jnp.cos(self.theta), jnp.sin(self.theta)
        u = c * xr + s * yr
        v = -s * xr + c * yr
        return u, v

    # Real OPL phase: linear in |u| → produces deflection via ∇_xy(opl)
    def opl_shift(self, xy):
        u, _ = self._uv(xy)
        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)
        return -self.strength * au

    # Amplitude transmission: inverse sigmoid rectangle (blocked inside)
    def transmission(self, xy):
        u, v = self._uv(xy)

        # u mask
        hu = 0.5 * self.width
        eps_u = self.eps * hu
        au = jnp.sqrt(u * u + eps_u * eps_u)
        tx = self.sharpness * (au - hu)
        logA_u = -softplus(-tx)  # ≈ large negative inside (blocked), 0 outside

        # v mask (optional)
        if self.length is None:
            logA_v = 0.0
        else:
            hv = 0.5 * self.length
            eps_v = self.eps * hu  # tie smoothing scale to u half-width
            av = jnp.sqrt(v * v + eps_v * eps_v)
            ty = self.sharpness * (av - hv)
            logA_v = -softplus(-ty)

        logA = logA_u + logA_v
        # Return transmission in [0,1]
        return jnp.exp(logA)


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
