import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from .gaussian import GaussianRayBeta, map_reduce
from jax.nn import softplus
import numpy as np
from typing import List, Tuple, Dict, Any
import interpax as interp


def _beam_field(r_m, C, eta, Q_inv, opl, k, r2):
    delta = r2 - r_m
    lin = jnp.sum(delta * eta, axis=-1)
    quad = jnp.sum((delta @ Q_inv) * delta, axis=-1)

    k = jnp.asarray(k).reshape(())
    taylor = k * (lin + 0.5 * quad)

    taylor_phase = jnp.real(taylor)
    taylor_logA = jnp.imag(taylor)
    taylor_logA = jnp.clip(taylor_logA, a_min=-700.0, a_max=0.0)

    # New input constant amplitude and phase
    constant = C * jnp.exp(1j * k * opl)

    # Local linear and quadratic taylor expansions of amplitude and phase
    lin_and_quad_taylor_amp_phase = jnp.exp(taylor_logA) * jnp.exp(-1j * taylor_phase)
    final_field = constant * lin_and_quad_taylor_amp_phase
    return final_field


def _beam_field_outer(xs, r2):
    r_m_i, C_i, eta_i, Q_i, opl_i, k_i = xs
    return _beam_field(r_m_i, C_i, eta_i, Q_i, opl_i, k_i, r2)


def evaluate_gaussian_packets_jax_scan(gaussian_ray: GaussianRayBeta, grid, *, batch_size: int | None = 128):
    r_m = gaussian_ray.r_xy
    C = gaussian_ray.C
    eta = gaussian_ray.eta
    Q_inv = gaussian_ray.Q_inv
    k = gaussian_ray.k
    opl = gaussian_ray.pathlength

    r2 = grid.coords
    P = r2.shape[0]
    init = jnp.zeros((P,), dtype=jnp.complex128)

    def f_pack(xs):
        return _beam_field_outer(xs, r2)

    xs = (r_m, C, eta, Q_inv, opl, k)
    out = map_reduce(f_pack, jnp.add, init, xs, batch_size=batch_size)
    return out.reshape(grid.shape)


# evaluate_gaussian_packets_jax_scan = jax.jit(
#     evaluate_gaussian_packets_jax_scan, static_argnames=["batch_size", "grid"]
# )


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
    phi0 = jnp.real(opl)  # (B,)
    im0 = jnp.imag(opl)  # (B,)  == -ℓ
    C_out = gp.C * jnp.exp(-im0) * jnp.exp(1j * phi0)

    g_phi = jnp.real(grad_opl)  # (B,2)
    g_im = jnp.imag(grad_opl)  # (B,2) == -∇ℓ
    H_phi = jnp.real(Hess_opl)  # (B,2,2)
    H_im = jnp.imag(Hess_opl)  # (B,2,2) == -Hℓ

    # η' = η - ∇φ - i ∇ℓ  (since g_im = -∇ℓ) - Linear terms
    eta_out = gp.eta - g_phi - 1j * (g_im)
    # Qinv' = Qinv - Hφ - i Hℓ - Quadratic terms
    Qinv_out = gp.Q_inv - H_phi - 1j * (H_im)

    dxy_in = gp.d_xy
    dxy_out = dxy_in - g_phi

    return gp.derive(dx=dxy_out[..., 0], dy=dxy_out[..., 1],
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
        M_inv = jnp.linalg.inv(M)
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

        opl = ray.sigma * self.V_interp(r_xy) / ray.k

        g_real = ray.sigma * self.grad_V_interp(r_xy) / ray.k
        H_real = ray.sigma * self.hess_V_interp(r_xy) / ray.k

        # promote to complex (imaginary parts are zero)
        g_opl = g_real.astype(jnp.complex128)
        H_opl = H_real.astype(jnp.complex128)
        out = apply_thin_element_from_complex_opl(ray, opl, g_opl, H_opl)

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


def _poly2d_cubic_features(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Return cubic monomial features [1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3].

    Supports broadcasting. Last dimension is 10.
    """
    x2 = x * x
    y2 = y * y
    xy = x * y
    return jnp.stack([
        jnp.ones_like(x),  # 1
        x,
        y,
        x2,
        xy,
        y2,
        x2 * x,
        x2 * y,
        x * y2,
        y2 * y,
    ], axis=-1)


def _poly2d_cubic_eval(coeffs: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Evaluate 2D cubic polynomial with 10 coeffs at coordinates (x, y).

    coeffs order matches _poly2d_cubic_features.
    coeffs shape: (..., 10) or (10,)
    x,y: broadcastable arrays.
    """
    phi = _poly2d_cubic_features(x, y)
    return jnp.sum(phi * coeffs, axis=-1)


def fit_cubic_polynomial_to_slice(
    V2d: np.ndarray,
    sx: float,
    sy: float,
    x0: float,
    y0: float,
    *,
    stride: int = 1,
) -> np.ndarray:
    """Least-squares fit of a 2D cubic polynomial to a single slice.

    Parameters
    ----------
    V2d : (ny, nx) array of potential values (V·Å)
    sx, sy : pixel sizes in Å/px (x then y)
    x0, y0 : origin in Å so pixel (0,0) maps to (x0, y0)
    stride : sample every `stride` pixels along both axes to speed up LSQ

    Returns
    -------
    coeffs : (10,) array of cubic polynomial coefficients in order
        [1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3]
    """
    V2d = np.asarray(V2d)
    ny, nx = V2d.shape
    xs = x0 + sx * np.arange(nx)
    ys = y0 + sy * np.arange(ny)
    X, Y = np.meshgrid(xs, ys)

    Xs = X[::stride, ::stride].ravel()
    Ys = Y[::stride, ::stride].ravel()
    Zs = V2d[::stride, ::stride].ravel()

    # Build design matrix A with columns of monomials
    x = Xs
    y = Ys
    A = np.column_stack([
        np.ones_like(x),
        x,
        y,
        x * x,
        x * y,
        y * y,
        x * x * x,
        (x * x) * y,
        x * (y * y),
        y * y * y,
    ])

    # Solve least squares
    coeffs, *_ = np.linalg.lstsq(A, Zs, rcond=None)
    return coeffs.astype(np.float64)


@jdc.pytree_dataclass
class PolynomialPotential(Component):
    """Thin element whose phase is sigma * P_3(x, y), where P_3 is a cubic polynomial.

    coeffs order: [1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3]
    """
    coeffs: jnp.ndarray  # shape (10,)
    x0: float
    y0: float
    sigma: float

    def complex_action(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        phase = self.sigma * _poly2d_cubic_eval(self.coeffs, x, y)
        return phase - 1j * 0.0


def abtem_potential_3d(
    src,
    *,
    how: str = "slice",
    sampling_A: float = 0.05,
    parametrization: str = "lobato",
    projection: str = "infinite",
    periodic: bool = False,
    box_A: Tuple[float, float, float] | None = None,
    slice_thickness: float = 1.0,
    slice_index: int | None = None,
) -> Tuple[np.ndarray, float, float, float, float, float]:
    """Return a 3D potential volume from abTEM along with sampling and dz.

    If `src` is:
      - ase.Atoms: constructs abtem.Potential with given settings and builds slices.
      - abtem.Potential: builds slices.
      - abtem computed object with .array: if 3D, uses it directly.

    Returns
    -------
    V3d : np.ndarray with shape (nz, ny, nx)
    sx, sy : float pixel sizes in Å/px
    x0, y0 : float origins so that pixel (0,0) maps to (x0, y0)
    dz : float distance between slices in Å (taken as slice_thickness)
    """
    try:
        from ase import Atoms  # type: ignore
        import abtem  # type: ignore
    except Exception as e:
        raise ImportError("abtem and ase are required for abtem_potential_3d") from e

    decided = how
    if isinstance(src, Atoms):
        pot = abtem.Potential(
            src,
            sampling=sampling_A,
            parametrization=parametrization,
            slice_thickness=slice_thickness,
            projection=projection,
            periodic=periodic,
            box=None if box_A is None else tuple(map(float, box_A)),
        )
        decided = "slice" if how == "auto" else how
    elif hasattr(src, "project") and hasattr(src, "build"):
        pot = src
        if decided == "auto":
            decided = "slice"
    elif hasattr(src, "array"):
        A = np.asarray(src.array)
        sy, sx = getattr(src, "sampling", (None, None))
        if A.ndim != 3:
            raise ValueError("Expected a 3D array-like for abtem object with .array")
        nz, ny, nx = A.shape
        Lx, Ly = nx * sx, ny * sy
        x0, y0 = -Lx / 2.0, -Ly / 2.0
        dz = float(slice_thickness)
        return A, float(sx), float(sy), float(x0), float(y0), dz
    else:
        raise TypeError(
            "Unsupported src. Provide Atoms, abtem.Potential, or abtem object with .array."
        )

    if decided != "slice":
        raise ValueError("abtem_potential_3d requires how='slice' or 'auto' resolving to 'slice'.")

    slices = pot.build().compute()  # (nz, ny, nx)
    V3d = np.asarray(slices.array)
    if V3d.ndim != 3:
        raise ValueError("Built potential does not have 3D (nz, ny, nx) shape.")
    sy, sx = slices.sampling
    nz, ny, nx = V3d.shape
    Lx, Ly = nx * sx, ny * sy
    x0, y0 = -Lx / 2.0, -Ly / 2.0
    dz = float(slice_thickness)
    return V3d, float(sx), float(sy), float(x0), float(y0), dz


def build_polynomial_potential_components_from_abtem(
    src,
    *,
    accel_V: float = 200_000.0,
    how: str = "slice",
    sampling_A: float = 0.1,
    parametrization: str = "lobato",
    projection: str = "infinite",
    periodic: bool = False,
    box_A: Tuple[float, float, float] | None = None,
    slice_thickness: float = 1.0,
    z0: float = 0.0,
    fit_stride: int = 1,
) -> Tuple[List[Component], Dict[str, Any]]:
    """Create a list of thin polynomial potential components interleaved with free-space.

    For each z-slice of the abTEM potential volume, a 2D cubic polynomial is fitted
    and wrapped in a PolynomialPotential. Between successive slices, a FreeSpaceParaxial
    with distance = slice_thickness is inserted.

    Returns (components, meta).
    """
    from abtem.core.energy import energy2sigma  # lazy import to avoid hard dep at module import

    V3d, sx, sy, x0, y0, dz = abtem_potential_3d(
        src,
        how=how,
        sampling_A=sampling_A,
        parametrization=parametrization,
        projection=projection,
        periodic=periodic,
        box_A=box_A,
        slice_thickness=slice_thickness,
    )

    sigma = float(energy2sigma(accel_V))

    components: List[Component] = []
    nz = V3d.shape[0]
    for i in range(nz):
        coeffs = fit_cubic_polynomial_to_slice(V3d[i], sx, sy, x0, y0, stride=fit_stride)
        comp = PolynomialPotential(
            z=z0 + i * dz,
            coeffs=jnp.asarray(coeffs, dtype=jnp.float64),
            x0=float(x0),
            y0=float(y0),
            sigma=sigma,
        )
        components.append(comp)
        if i < nz - 1:
            components.append(FreeSpaceParaxial(distance=dz, z=z0 + i * dz))

    meta = dict(
        accel_V=accel_V,
        sampling=(sx, sy),
        origin=(x0, y0),
        dz=dz,
        nz=nz,
        sigma=sigma,
        src_type=type(src).__name__,
        how=how,
        parametrization=parametrization,
        projection=projection,
        periodic=periodic,
        box_A=box_A,
        fit_stride=fit_stride,
    )
    return components, meta


def run_gaussian_through_cubic_slices(
    ray: GaussianRayBeta,
    src,
    **kwargs,
):
    """Convenience: build polynomial components from abTEM volume and run to end.

    kwargs are forwarded to build_polynomial_potential_components_from_abtem.
    Returns (ray_out, components, meta).
    """
    components, meta = build_polynomial_potential_components_from_abtem(src, **kwargs)
    ray_out = run_to_end(ray, components)
    return ray_out, components, meta


def build_polynomial_potential_components_from_array(
    V3d: np.ndarray,
    *,
    sx: float,
    sy: float,
    x0: float,
    y0: float,
    dz: float,
    accel_V: float = 200_000.0,
    z0: float = 0.0,
    fit_stride: int = 1,
) -> Tuple[List[Component], Dict[str, Any]]:
    """Create polynomial thin elements + free space from a raw 3D volume.

    Parameters
    ----------
    V3d : np.ndarray (nz, ny, nx)
    sx, sy : Å/px
    x0, y0 : Å origin
    dz : Å between slices
    accel_V : accelerating voltage in volts
    z0 : starting z position
    fit_stride : subsampling factor for LSQ
    """
    from abtem.core.energy import energy2sigma  # lazy import

    V3d = np.asarray(V3d)
    assert V3d.ndim == 3, "V3d must have shape (nz, ny, nx)"

    sigma = float(energy2sigma(accel_V))
    components: List[Component] = []
    nz = V3d.shape[0]
    for i in range(nz):
        coeffs = fit_cubic_polynomial_to_slice(V3d[i], sx, sy, x0, y0, stride=fit_stride)
        comp = PolynomialPotential(
            z=z0 + i * dz,
            coeffs=jnp.asarray(coeffs, dtype=jnp.float64),
            x0=float(x0),
            y0=float(y0),
            sigma=sigma,
        )
        components.append(comp)
        if i < nz - 1:
            components.append(FreeSpaceParaxial(distance=dz, z=z0 + i * dz))

    meta = dict(
        accel_V=accel_V,
        sampling=(sx, sy),
        origin=(x0, y0),
        dz=dz,
        nz=nz,
        sigma=sigma,
        src_type="ndarray",
        fit_stride=fit_stride,
    )
    return components, meta


def run_gaussian_through_cubic_slices_from_array(
    ray: GaussianRayBeta,
    V3d: np.ndarray,
    *,
    sx: float,
    sy: float,
    x0: float,
    y0: float,
    dz: float,
    **kwargs,
):
    """Convenience: build polynomial components from array and run to end.

    kwargs are forwarded to build_polynomial_potential_components_from_array.
    Returns (ray_out, components, meta).
    """
    components, meta = build_polynomial_potential_components_from_array(
        V3d, sx=sx, sy=sy, x0=x0, y0=y0, dz=dz, **kwargs
    )
    ray_out = run_to_end(ray, components)
    return ray_out, components, meta
