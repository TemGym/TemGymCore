import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from .gaussian import GaussianRayBeta, map_reduce
from jax.nn import softplus
import interpax as interp


def _beam_field(r_m, C, eta, Q_inv, k, r2):
    delta = r2 - r_m
    lin = jnp.sum(delta * eta, axis=-1)
    quad = jnp.sum((delta @ Q_inv) * delta, axis=-1)

    k = jnp.asarray(k).reshape(())
    taylor = k * (lin + 0.5 * quad)

    taylor_phase = jnp.real(taylor)
    taylor_logA = jnp.imag(taylor)
    taylor_logA = jnp.clip(taylor_logA, a_min=-700.0, a_max=0.0)

    # Local linear and quadratic taylor expansions of amplitude and phase
    lin_and_quad_taylor_amp_phase = jnp.exp(taylor_logA) * jnp.exp(-1j * taylor_phase)
    final_field = C * lin_and_quad_taylor_amp_phase
    return final_field


def _beam_field_outer(xs, r2):
    r_m_i, C_i, eta_i, Q_i, k_i = xs
    return _beam_field(r_m_i, C_i, eta_i, Q_i, k_i, r2)


def evaluate_gaussian_packets_jax_scan(
    gaussian_ray: GaussianRayBeta,
    grid,
    *,
    batch_size: int | None = 128,
):
    r_m = gaussian_ray.r_xy
    C = gaussian_ray.C
    eta = gaussian_ray.eta
    Q_inv = gaussian_ray.Q_inv
    k = gaussian_ray.k

    r2 = grid.coords
    P = r2.shape[0]
    init = jnp.zeros((P,), dtype=jnp.complex128)

    def f_pack(xs):
        return _beam_field_outer(xs, r2)

    xs = (r_m, C, eta, Q_inv, k)
    out = map_reduce(f_pack, jnp.add, init, xs, batch_size=batch_size)
    return out.reshape(grid.shape)


# evaluate_gaussian_packets_jax_scan = jax.jit(
#     evaluate_gaussian_packets_jax_scan, static_argnames=["batch_size", "grid"]
# )


def evaluate_gaussian_packets_for_loop(
    gaussian_ray: GaussianRayBeta,
    grid,
):
    """
    A simple for-loop based evaluation for testing and debugging.
    This is not JIT-compatible and will be slow.
    """
    # Extract properties for all packets. These have a leading batch dimension.
    r_m = gaussian_ray.r_xy
    C = gaussian_ray.C
    eta = gaussian_ray.eta
    Q_inv = gaussian_ray.Q_inv
    k = gaussian_ray.k

    # Grid coordinates where the field is evaluated.
    r2 = grid.coords
    P = r2.shape[0]  # Total number of grid points
    num_packets = r_m.shape[0]

    # Initialize the total field accumulator.
    total_field = jnp.zeros((P,), dtype=jnp.complex128)

    # Iterate through each Gaussian packet one by one.
    for i in range(num_packets):
        # Calculate the field for the i-th packet on the entire grid.
        field_i = _beam_field(
            r_m=r_m[i],
            C=C[i],
            eta=eta[i],
            Q_inv=Q_inv[i],
            k=k[i],
            r2=r2,
        )
        # Add its contribution to the total field.
        total_field += field_i

    # Reshape the final flat array to match the grid's 2D shape.
    return total_field.reshape(grid.shape)


def _as_batch(r):
    r = r if r.ndim == 2 else r[None, ...]
    return r


def _split_real_imag(component, rr):
    z = component.complex_action(rr)
    return jnp.stack([jnp.real(z), jnp.imag(z)])


def grad_opl(component, xy_batch):
    # J shape: (B, 2_out=[Reψ, Imψ], 2_in=[x,y])
    J = jax.vmap(jax.jacfwd(_split_real_imag, argnums=1), in_axes=(None, 0))(component, xy_batch)
    g_phi = J[:, 0, :]  # ∇φ
    g_imag = J[:, 1, :]  # ∇(Im ψ) = ∇(-ℓ) = -∇ℓ
    return g_phi + 1j * g_imag  # == ∇φ - i∇ℓ


def hess_opl(component, xy_batch):
    # H shape: (B, 2_out=[Reψ, Imψ], 2, 2)
    H = jax.vmap(
        jax.jacfwd(jax.jacrev(_split_real_imag, argnums=1), argnums=1),
        in_axes=(None, 0)
    )(component, xy_batch)
    H_phi = H[:, 0, :, :]  # Hφ
    H_imag = H[:, 1, :, :]  # H(Im ψ) = H(-ℓ) = -Hℓ
    Hc = H_phi + 1j * H_imag  # == Hφ - iHℓ
    return 0.5 * (Hc + jnp.swapaxes(Hc, -1, -2))


def apply_thin_element_from_complex_opl(gp: GaussianRayBeta, opl, grad_opl, Hess_opl):
    """
    opl      : (B,)   complex = φ - iℓ
    grad_opl : (B,2)  complex = ∇φ - i∇ℓ
    Hess_opl : (B,2,2) complex = Hφ - iHℓ
    """
    k = gp.k
    phi_0 = jnp.real(opl)  # (B,)
    im_0 = jnp.imag(opl)  # (B,)  == -ℓ
    g_phi = jnp.real(grad_opl)  # (B,2)
    g_im = jnp.imag(grad_opl)  # (B,2) == -∇ℓ
    H_phi = jnp.real(Hess_opl)  # (B,2,2)
    H_im = jnp.imag(Hess_opl)  # (B,2,2) == -Hℓ

    # C' = C * exp(-ℓ) * exp(i φ) - Constant phase and amplitude factors
    C_out = gp.C * jnp.exp(-im_0) * jnp.exp(1j * k * phi_0)
    # η' = η - ∇φ - i ∇ℓ  (since g_im = -∇ℓ) - Linear phase and amplitude terms
    eta_out = gp.eta - g_phi - 1j * (g_im)
    # Qinv' = Qinv - Hφ - i Hℓ - Quadratic terms - Quadratic phase and amplitude terms
    Qinv_out = gp.Q_inv - H_phi - 1j * (H_im)

    dxy_in = gp.d_xy
    dxy_out = dxy_in - g_phi
    opl_out = gp.pathlength + opl

    return gp.derive(dx=dxy_out[..., 0], dy=dxy_out[..., 1], pathlength=opl_out,
                     Q_inv=Qinv_out, eta=eta_out, C=C_out)


@jdc.pytree_dataclass
class Component:
    z: float

    def complex_action(self, xy: jnp.ndarray) -> complex:
        """Return φ(x,y) - i ℓ(x,y). Default: no effect."""
        return jnp.zeros((), dtype=jnp.complex128)

    def __call__(self, ray):
        # (B,2) positions
        r_xy = ray.r_xy
        r_xy = _as_batch(r_xy)

        # φ - iℓ at points (for C update)
        opl = jax.vmap(self.complex_action)(r_xy)
        # Jac and Hessian
        g_opl = grad_opl(self, r_xy)    # (B,2) complex

        if isinstance(ray, GaussianRayBeta):
            H_opl = hess_opl(self, r_xy)    # (B,2,2) complex
            out = apply_thin_element_from_complex_opl(ray, opl, g_opl, H_opl)
            return out.derive(z=self.z)

        dxy = ray.d_xy - jnp.real(g_opl)
        return ray.derive(dx=dxy[..., 0], dy=dxy[..., 1], z=self.z)


@jdc.pytree_dataclass
class Lens(Component):
    focal_length: float

    def complex_action(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        rho2 = x*x + y*y
        phase = -0.5 * rho2 / self.focal_length
        log_amplitude = 0.0
        return phase - 1j * log_amplitude


@jdc.pytree_dataclass
class AberratedLens(Component):
    focal_length: float
    C_sph: float = 0.0
    C_coma_x: float = 0.0
    C_coma_y: float = 0.0

    def complex_action(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        rho2 = x*x + y*y

        # Aberration expansion
        phase = (
            -0.5 * rho2 / self.focal_length
            + self.C_sph * (rho2**2)
            + self.C_coma_x * (x**3 + x*y**2)
            + self.C_coma_y * (y**3 + y*x**2)
        )
        log_amplitude = 0.0
        return phase - 1j * log_amplitude


@jdc.pytree_dataclass
class SigmoidAperture(Component):
    radius: float
    sharpness: float = 50.0
    eps: float = 1e-12

    def complex_action(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        r = jnp.sqrt(x * x + y * y + self.eps)

        t = self.sharpness * (r - self.radius)
        logA = -softplus(t)

        # Pure amplitude mask: ψ = φ - iℓ, here φ=0, ℓ = logA (≤ 0)
        return -1j * logA


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
class FreeSpaceParaxial(Component):
    distance: float  # L

    def complex_action(self, xy):
        return jnp.array(0.0, dtype=jnp.complex128)

    def __call__(self, ray):
        L = self.distance

        x_new = ray.x + ray.dx * L
        y_new = ray.y + ray.dy * L
        z_new = ray.z + L

        path_inc = L * (1.0 + 0.5 * (ray.dx**2 + ray.dy**2))
        path_new = ray.pathlength + path_inc

        # Geometric ray path
        if not hasattr(ray, "Q_inv"):
            return ray.derive(x=x_new, y=y_new, z=z_new, pathlength=path_new)

        Qinv = ray.Q_inv
        I2 = jnp.eye(2, dtype=jnp.complex128)
        M = I2 + L * Qinv
        M_inv = jnp.linalg.solve(M, I2)
        Qinv_new = Qinv @ M_inv
        eta_new = (jnp.swapaxes(M_inv, -1, -2) @ ray.eta[..., None])[..., 0]
        C_new = ray.C / jnp.sqrt(jnp.linalg.det(M))

        return ray.derive(
            x=x_new, y=y_new, z=z_new, pathlength=path_new,
            Q_inv=Qinv_new, eta=eta_new, C=C_new
        )


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


def run_to_end(ray, components):
    r = ray
    for comp in components:
        r = comp(r)
    return r
