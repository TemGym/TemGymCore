import dataclasses
import numpy as np
import jax.numpy as jnp
import jax
from .grid import Grid
from .run import run_to_end
from .utils import (
    custom_jacobian_matrix,
    energy2wavelength,
    fibonacci_spiral,
    uniform_disk,
)
from .ray import Ray
from .gaussian_action2D import GaussianBeamFactory, GaussianBeamBundle, GaussianBeam
import jax_dataclasses as jdc
from jax._src.lax.control_flow.loops import _batch_and_remainder
from jax import lax
from ase import units


def relativistic_mass_correction(energy: float) -> float:
    return 1 + units._e * energy / (units._me * units._c**2)


def w_z(w0, z, z_r):
    return w0 * jnp.sqrt(1 + (z / z_r) ** 2)


def zR(w0, wavelength):
    return (jnp.pi * w0**2) / wavelength


def R(z, z_r):
    cond = jnp.abs(z) < 1e-10
    z_r_over_z = jax.lax.cond(cond, lambda op: 0.0, lambda op: op[1] / op[0], (z, z_r))
    # z_r_over_z = jnp.where(cond, 0.0, z_r / z) - this gave me an error
    # and I don't know why it tried to evaluate z_r / z
    return jax.lax.cond(
        cond, lambda _: jnp.inf, lambda _: z * (1 + z_r_over_z**2), operand=None
    )


def gaussian_beam(x, y, q_inv, k, offset_x=0, offset_y=0):
    return jnp.exp(-1j * k * ((x + offset_x) ** 2 + (y + offset_y) ** 2) / 2 * q_inv)


def decompose_Q_inv(Q_inv, wavelength, eps=1e-12):
    """
    Decompose a 2x2 complex Q_inv matrix into beam parameters:
    returns (waist_x, waist_y, r_x, r_y, theta).

    Supports broadcasting over leading batch dimensions.
    """
    Q = jnp.asarray(Q_inv)

    # Use the imaginary part (real symmetric, negative-definite) to get rotation
    S = jnp.imag(Q)
    S = 0.5 * (S + jnp.swapaxes(S, -1, -2))  # enforce symmetry
    _, evecs = jnp.linalg.eigh(S)  # ascending order

    # Ensure a proper rotation (det = +1)
    det = jnp.linalg.det(evecs)
    sign = jnp.where(det < 0, -1.0, 1.0)
    evecs = evecs.at[..., :, 1].multiply(sign[..., None])

    # Rotate Q into principal axes and read diagonal
    Vt = jnp.swapaxes(evecs, -1, -2)
    Qd = Vt @ Q @ evecs
    qdiag = jnp.stack([Qd[..., 0, 0], Qd[..., 1, 1]], axis=-1)

    # Determine a consistent ordering: put larger waist first (major axis)
    imd_pre = jnp.imag(qdiag)  # = wavelength/(pi * w^2)
    waists_pre = jnp.sqrt(
        jnp.where(jnp.abs(imd_pre) > eps, jnp.abs(wavelength / (jnp.pi * imd_pre)), jnp.inf)
    )
    swap_mask = waists_pre[..., 0] < waists_pre[..., 1]

    # If needed, swap principal axes (qdiag and eigenvectors)
    qdiag = jnp.where(swap_mask[..., None], qdiag[..., ::-1], qdiag)
    evecs_swapped = evecs[..., :, ::-1]
    evecs = jnp.where(swap_mask[..., None, None], evecs_swapped, evecs)

    # Re-enforce right-handed rotation after possible swap
    det = jnp.linalg.det(evecs)
    sign = jnp.where(det < 0, -1.0, 1.0)
    evecs = evecs.at[..., :, 1].multiply(sign[..., None])

    imd = jnp.imag(qdiag)  # = wavelength/(pi * w^2)
    red = jnp.real(qdiag)  # = 1 / R

    # Waists
    waists = jnp.sqrt(jnp.where(jnp.abs(imd) > eps, jnp.abs(wavelength / (jnp.pi * imd)), jnp.inf))

    # Radii of curvature
    radii = jnp.where(jnp.abs(red) > eps, 1.0 / red, jnp.inf)

    # Rotation angle from first principal axis
    e1 = evecs[..., :, 0]
    theta = jnp.arctan2(e1[..., 1], e1[..., 0])

    return waists[..., 0], waists[..., 1], radii[..., 0], radii[..., 1], theta


def Qinv_ABCD(Qinv, A, B, C, D):
    # compute (C + D @ Qinv) @ inv(A + B @ Qinv) without explicit inv
    lhs = A + B @ Qinv
    rhs = C + D @ Qinv
    return jnp.linalg.solve(lhs, rhs)


def Qinv_ABCD_float(Qinv, A, B, C, D):
    return C + D * Qinv / (A + B * Qinv)


def q_inv(z, w0, wl):
    z_r = zR(w0, wl)
    cond = jnp.abs(z) < 1e-10
    wz_val = w_z(w0, z, z_r)
    R_val = R(z, z_r)

    q_inv = jnp.where(
        cond,
        -1j * wl / (jnp.pi * w0**2),
        1.0 / R_val - 1j * wl / (jnp.pi * wz_val**2),
    )
    return q_inv


@jdc.pytree_dataclass(kw_only=True)
class GaussianRay(Ray):
    amplitude: float
    waist_xy: jnp.ndarray
    radii_of_curv: jnp.ndarray
    wavelength: float
    theta: float

    def derive(self, **updates):
        # Like Ray.derive: allow passing values or callables that take self
        def resolve(v):
            return v(self) if callable(v) else v
        return jdc.replace(self, **{k: resolve(v) for k, v in updates.items()})

    def to_ray(self):
        return Ray(
            x=self.x,
            y=self.y,
            dx=self.dx,
            dy=self.dy,
            z=self.z,
            pathlength=self.pathlength,
            _one=self._one,
        )

    @property
    def q_inv(self):
        w_x, w_y = self.waist_xy.T
        R_x, R_y = self.radii_of_curv.T
        wavelength = self.wavelength
        # 1/q on each principal axis
        inv_qx = jnp.where(
            jnp.isinf(R_x),
            -1j * wavelength / ((jnp.pi * w_x**2)),
            1.0 / R_x - 1j * wavelength / (jnp.pi * w_x**2),
        )

        inv_qy = jnp.where(
            jnp.isinf(R_y),
            -1j * wavelength / ((jnp.pi * w_y**2)),
            1.0 / R_y - 1j * wavelength / (jnp.pi * w_y**2),
        )
        return inv_qx, inv_qy

    @property
    def Q_inv(self):
        from .gaussian import matrix_matrix_matrix_mul

        inv_qx, inv_qy = self.q_inv
        Q_inv_diag = jnp.stack(
            [
                jnp.stack([inv_qx, jnp.zeros_like(inv_qx)], axis=-1),
                jnp.stack([jnp.zeros_like(inv_qx), inv_qy], axis=-1),
            ],
            axis=-2,
        )
        c, s = jnp.cos(self.theta), jnp.sin(self.theta)
        R = jnp.stack(
            [
                jnp.stack([c, -s], axis=-1),
                jnp.stack([s, c], axis=-1),
            ],
            axis=-2,
        )

        Q_inv_diag = Q_inv_diag[None, ...] if Q_inv_diag.ndim == 2 else Q_inv_diag
        R = R[None, ...] if R.ndim == 2 else R
        return matrix_matrix_matrix_mul(R, Q_inv_diag, R)


@jdc.pytree_dataclass
class TaylorExpofAction:
    const: complex
    lin: jnp.ndarray
    quad: jnp.ndarray

    @classmethod
    def from_q_inv(
        cls,
        Q_inv,
        *,
        const=0.0 + 0.0j,
        lin=None,
    ) -> "TaylorExpofAction":
        """Build an action expansion with quadratic term ``Q_inv``.

        Parameters
        ----------
        Q_inv : array-like
            Quadratic coefficient(s) shaped (..., 2, 2).
        const : complex or array-like, optional
            Constant term to broadcast over the leading batch dimensions.
        lin : array-like, optional
            Linear term(s) shaped (..., 2). Defaults to zeros.
        """
        Q_inv_arr = jnp.asarray(Q_inv, dtype=jnp.complex128)
        batch_shape = Q_inv_arr.shape[:-2]
        const_arr = jnp.asarray(const, dtype=jnp.complex128)
        const_arr = jnp.broadcast_to(const_arr, batch_shape)
        if lin is None:
            lin_arr = jnp.zeros(batch_shape + (2,), dtype=jnp.complex128)
        else:
            lin_arr = jnp.asarray(lin, dtype=jnp.complex128)
            lin_arr = jnp.broadcast_to(lin_arr, batch_shape + (2,))
        return cls(const=const_arr, lin=lin_arr, quad=Q_inv_arr)


@jdc.pytree_dataclass(kw_only=True)
class GaussianRayBeta(Ray):
    """Gaussian ray carrying a quadratic expansion of the complex action.

    Notes
    -----
    The constant term ``S.const`` stores the accumulated complex action, while
    ``C`` holds only the geometric ABCD prefactor.
    """
    S: TaylorExpofAction  # Action
    C: complex = 1.0 + 0.0j  # geometric prefactor
    voltage: float

    def derive(self, **updates):
        def resolve(v): return v(self) if callable(v) else v
        return jdc.replace(self, **{k: resolve(v) for k, v in updates.items()})

    def to_ray(self):
        return Ray(x=self.x, y=self.y, dx=self.dx, dy=self.dy,
        z=self.z, pathlength=self.S.const, _one=self._one)

    def to_vector(self):
        params = {}
        for k, v in dataclasses.asdict(self).items():
            if k == "S" and isinstance(v, dict):
                # ensure each item in S is atleast_1d, then rebuild TaylorExpofAction
                s_params = {sk: jnp.atleast_1d(sv) for sk, sv in v.items()}
                params["S"] = TaylorExpofAction(**s_params)
            else:
                params[k] = jnp.atleast_1d(v)
        return type(self)(**params)

    @property
    def prefactor(self) -> complex:
        """Complex amplitude prefactor stored in ``C``."""
        return self.C

    @property
    def mass(self) -> float:
        """
        Relativistic electron mass [kg] for this ray's voltage (eV).
        """
        return relativistic_mass_correction(self.voltage) * units._me

    @property
    def wavelength(self) -> float:
        """
        Relativistic de Broglie wavelength [Å] for this ray's voltage (eV).
        """
        E = self.voltage
        return (
            units._hplanck
            * units._c
            / jnp.sqrt(E * (2 * units._me * units._c**2 / units._e + E))
            / units._e
        )

    @property
    def sigma(self) -> float:
        """
        Interaction parameter [1 / (Å * eV)] for this ray's voltage (eV).
        """
        return (
            2
            * jnp.pi
            * self.mass
            * units.kg
            * units._e
            * units.C
            * self.wavelength
            / (units._hplanck * units.s * units.J) ** 2
        )

    @property
    def k(self) -> float:
        """Wave number k = 2*pi / wavelength (1/Å)."""
        return 2 * jnp.pi / self.wavelength


# Gaussian beam sampling utilities are now provided by temgym_core.gaussian_action2D.
# The legacy GaussianRayBeta-based factory has been removed in favour of the
# GaussianBeam formulation. ``GaussianBeamFactory`` is imported from
# ``gaussian_action2D`` near the top of this module for backward-compatible
# access via ``temgym_core.gaussian``.


def make_gaussian_plane_wave_round_aperture(
    *,
    voltage: float = 200e3,
    aperture_radius: float = 50e-9,
    waist_radius: float | None = None,
    overlap_factor: float | None = None,
    num_rays: int = 1000,
    sampler=None,
    sampling: str | None = None,
    sampler_kwargs: dict | None = None,
    normalization: str = 'unit',
    initial_z: float = 0.0,
) -> GaussianRayBeta:
    if waist_radius is None and overlap_factor is None:
        waist_radius = 1e-9
    factory = GaussianBeamFactory(
        voltage=voltage,
        waist_radius=waist_radius,
        overlap_factor=overlap_factor,
        normalization=normalization,
        sampler=sampler,
        sampling=sampling,
        sampler_kwargs=sampler_kwargs,
        initial_z=initial_z,
    )
    return factory.round_aperture(
        aperture_radius=aperture_radius,
        num_rays=num_rays,
        initial_z=initial_z,
    )


def make_gaussian_plane_wave_elliptical_aperture(
    *,
    voltage: float = 200e3,
    semi_axis_x: float = 50e-9,
    semi_axis_y: float = 30e-9,
    rotation_radians: float = 0.0,
    waist_radius: float | None = None,
    overlap_factor: float | None = None,
    num_rays: int = 1000,
    sampler=None,
    sampling: str | None = None,
    sampler_kwargs: dict | None = None,
    normalization: str = 'unit',
    initial_z: float = 0.0,
) -> GaussianRayBeta:
    if waist_radius is None and overlap_factor is None:
        waist_radius = 1e-9
    factory = GaussianBeamFactory(
        voltage=voltage,
        waist_radius=waist_radius,
        overlap_factor=overlap_factor,
        normalization=normalization,
        sampler=sampler,
        sampling=sampling,
        sampler_kwargs=sampler_kwargs,
        initial_z=initial_z,
    )
    return factory.elliptical_aperture(
        semi_axis_x=semi_axis_x,
        semi_axis_y=semi_axis_y,
        rotation_radians=rotation_radians,
        num_rays=num_rays,
        initial_z=initial_z,
    )


def make_gaussian_plane_wave_square_aperture(
    *,
    voltage: float = 200e3,
    side_length: float = 100e-9,
    side_length_y: float | None = None,
    waist_radius: float | None = None,
    overlap_factor: float | None = None,
    samples_per_side: int | None = None,
    num_rays: int | None = None,
    sampling: str | None = None,
    normalization: str = 'unit',
    initial_z: float = 0.0,
) -> GaussianRayBeta:
    if waist_radius is None and overlap_factor is None:
        waist_radius = 1e-9
    factory = GaussianBeamFactory(
        voltage=voltage,
        waist_radius=waist_radius,
        overlap_factor=overlap_factor,
        normalization=normalization,
        sampling=sampling,
        initial_z=initial_z,
    )
    return factory.square_aperture(
        side_length=side_length,
        side_length_y=side_length_y,
        samples_per_side=samples_per_side,
        num_rays=num_rays,
        sampling=sampling,
        initial_z=initial_z,
    )


def matrix_vector_mul(M, v):
    """
    Batched matrix-vector multiplication.
    M: (nb,2,2)
    v: (nb,2)
    Returns (nb,2) result of M @ v for each batch.
    """
    return jnp.einsum("ij,j->i", M, v)


def matrix_matrix_mul(M1, M2):
    """
    Batched matrix-matrix multiplication.
    M1: (nb,2,2)
    M2: (nb,2,2)
    Returns (nb,2,2) result of M1 @ M2 for each batch.
    """
    return jnp.einsum("ij,jk->ik", M1, M2)


def matrix_quadratic_mul(v, M):
    """
    Batched quadratic multiplication -  v^T M v.
    v: (nb,2)
    M: (nb,2,2)
    Returns (nb,)
    """
    return jnp.einsum("i,ij,j->", v, M, v)


def matrix_linear_mul(v, M, w):
    """
    Batched linear multiplication - v^T M w
    v: (nb,2)
    M: (nb,2,2)
    w: (np,2)  -- observation coordinates (no batch)
    Returns (nb,np)
    """
    return jnp.einsum("i,ij,nj->n", v, M, w)


def matrix_matrix_matrix_mul(M1, M2, M3):
    return jnp.einsum("nij,njk,npk->nip", M1, M2, M3)


def make_gaussian_image(gaussian_rays, model, batch_size=128):

    rays = gaussian_rays
    assert isinstance(rays, GaussianRay)
    rays = rays.to_vector()

    grid = model[-1]
    assert isinstance(grid, Grid)

    vmap_fn = jax.vmap(jax.jacobian(run_to_end), in_axes=(0, None))
    central_rays = rays.to_ray()
    output_tm = vmap_fn(central_rays, model)
    output_rays = run_to_end(central_rays, model)

    model_ray_jacobians = custom_jacobian_matrix(output_tm)
    ABCDs = jnp.array(model_ray_jacobians)

    amplitudes = rays.amplitude

    Q1_invs = rays.Q_inv  # Should be of shape n x 2 x 2
    As = ABCDs[:, 0:2, 0:2]  # (nb,2,2)
    Bs = ABCDs[:, 0:2, 2:4]  # (nb,2,2)
    Cs = ABCDs[:, 2:4, 0:2]  # (nb,2,2)
    Ds = ABCDs[:, 2:4, 2:4]  # (nb,2,2)
    es = ABCDs[:, 0:2, 4]  # (nb,2)
    fs = ABCDs[:, 2:4, 4]  # (nb,2)
    r2 = grid.coords
    r1ms = jnp.stack([central_rays.x, central_rays.y], axis=-1)
    theta1ms = jnp.stack([central_rays.dx, central_rays.dy], axis=-1)
    wavelengths = rays.wavelength
    k = 2 * jnp.pi / wavelengths

    output_field = propagate_misaligned_gaussian_jax_scan(
        amplitudes,
        Q1_invs,
        As,
        Bs,
        Cs,
        Ds,
        es,
        fs,
        r1ms,
        theta1ms,
        k,
        r2=r2,
        batch_size=batch_size,
    ).reshape(grid.shape)
    return output_field


def _beam_field(amp, Q1_inv, Q2_inv, r1m, theta1m, A, B, e, f, k, r2):
    """Single-beam field at all observation points r2 -> (np,)
    r2 is at the end since it represents the grid, and is not batched.
    All other inputs are batched over the number of beams (nb, ...)"""
    I = jnp.eye(2, dtype=B.dtype)  # noqa

    # Safe inverses
    B_inv = jnp.linalg.solve(B, I)
    B_inv = jnp.nan_to_num(B_inv, nan=0.0, posinf=0.0, neginf=0.0)

    Q1 = jnp.linalg.solve(Q1_inv, I)
    Q1 = jnp.nan_to_num(Q1, nan=0.0, posinf=0.0, neginf=0.0)

    r2 = r2 - e
    # Central ray at output: r2m = A r1m + B theta1m
    r2m = matrix_vector_mul(A, r1m) + matrix_vector_mul(B, theta1m)  # (2,)

    # AB-q amplitude prefactor
    denom = A + matrix_matrix_mul(B, Q1_inv)  # (2,2)
    pref = amp / jnp.sqrt(jnp.linalg.det(denom))  # ()

    # Misalignment phase (input plane)
    ABinv = matrix_matrix_mul(A, B_inv)
    phi1 = matrix_quadratic_mul(r1m, ABinv) - 2 * matrix_linear_mul(
        r1m, B_inv, r2
    )  # (np,)

    # Misalignment phase (output plane)
    AQ1 = matrix_matrix_mul(A, Q1)
    B_over_AQ1B = jnp.linalg.solve(matrix_matrix_mul(B, AQ1 + B), I)  # (2,2)
    Q1B_over_AQ = matrix_matrix_mul(Q1, B_over_AQ1B)  # (2,2)
    phi2 = matrix_quadratic_mul(r2m, Q1B_over_AQ) - 2 * matrix_linear_mul(
        r2m, Q1B_over_AQ, r2
    )  # (np,)

    Q2t = jnp.einsum("ni,ij,nj->n", r2, Q2_inv, r2)  # (np,)

    # f is of shape 2, and r is (np,2), and we need f_offset * r2 to be (np,)
    f_offset = 2 * r2 @ f  # (np,)
    phase = (k / 2) * (Q2t + phi1 - phi2 + f_offset)  # (np,)
    return pref * jnp.exp(-1j * (phase))  # (np,)


def propagate_misaligned_gaussian_jax_scan(
    amp, Q1_inv, A, B, C, D, e, f, r1m, theta1m, k, r2, batch_size=128
):
    npix = r2.shape[0]
    Q2_inv = Qinv_ABCD(Q1_inv, A, B, C, D)  # (nb,2,2)

    def _beam_field_outer(xs):
        a_i, q1_i, q2_i, r1m_i, t1m_i, A_i, B_i, e_i, f_i, k_i = xs
        return _beam_field(a_i, q1_i, q2_i, r1m_i, t1m_i, A_i, B_i, e_i, f_i, k_i, r2)

    init = jnp.zeros((npix,), dtype=jnp.complex128)
    xs = (amp, Q1_inv, Q2_inv, r1m, theta1m, A, B, e, f, k)
    out = map_reduce(_beam_field_outer, jnp.add, init, xs, batch_size=batch_size)
    return out  # (npix,)


propagate_misaligned_gaussian_jax_scan = jax.jit(
    propagate_misaligned_gaussian_jax_scan, static_argnames=["batch_size"]
)


def map_reduce(f, reducer, init, xs, *, batch_size: int | None = None):
    def scan_fn(acc_inner, x):
        # combine f and reducer into function appropriate for normal lax.scan in reduce-only mode
        return reducer(acc_inner, f(x)), None

    if batch_size is not None:
        scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)

        def reduce_chunk(acc, x):
            # Reduce x into acc, assuming x have already been f'd
            return reducer(acc, x), None

        def map_reduce_chunk(acc, x):
            #  Vmap apply f to a chunk of x's, then reduce them sequentially into acc
            elements = jax.vmap(f)(x)
            return lax.scan(reduce_chunk, acc, elements)

        if scan_xs is not None:
            # Map f over each chunk of xs, and reduce each sequentially into init
            acc, _ = lax.scan(map_reduce_chunk, init, scan_xs)
        else:
            acc, _ = init, None

        if remainder_xs is not None:
            # normal scan-reduce the remainder chunk into acc (could also be vmapped?)
            acc, _ = lax.scan(scan_fn, acc, remainder_xs)
    else:
        # normal scan-reduce the remainder chunk into acc (could also be vmapped?)
        acc, _ = lax.scan(scan_fn, init, xs)
    return acc


def evaluate_gaussian_input_image(gaussian_rays, grid, batch_size=128):

    rays = gaussian_rays
    assert isinstance(rays, GaussianRay)
    rays = rays.to_vector()
    central_rays = rays.to_ray()
    amplitudes = rays.amplitude

    n_rays = amplitudes.shape[0]
    Q1_invs = rays.Q_inv  # Should be of shape n x 2 x 2
    r1 = grid.coords
    r1ms = jnp.stack([central_rays.x, central_rays.y], axis=-1)
    theta1ms = jnp.stack([central_rays.dx, central_rays.dy], axis=-1)
    wavelengths = jnp.full((n_rays,), rays.wavelength)
    k = 2 * jnp.pi / wavelengths
    phase_offset = k * rays.pathlength

    output_field = evaluate_misaligned_input_gaussian_jax_scan(
        amplitudes,
        phase_offset,
        Q1_invs,
        r1ms,
        theta1ms,
        k,
        r1,
        batch_size=batch_size,
    ).reshape(grid.shape)
    return output_field


def _input_beam_field(a, p, q1, r1m, t1m, k, r1):
    # r1: (np,2), r1m: (2,), q1: (2,2)
    r1_minus_r1m = r1 - r1m  # (np,2)
    # faster quadratic form: v = r1_minus_r1m @ q1.T -> (np,2)
    v = r1_minus_r1m @ q1.T
    r1_Q1_inv_r1 = jnp.sum(v * r1_minus_r1m, axis=1)  # (np,)
    misaligned_tilt_phase = 2 * (r1_minus_r1m @ t1m)  # (np,)
    phase = (k / 2) * (r1_Q1_inv_r1 + misaligned_tilt_phase)  # (np,)
    # combine exps into one to reduce work
    return a * jnp.exp(1j * (phase + p))  # (np,)


def evaluate_misaligned_input_gaussian_jax_scan(
    amp, phase_offset, Q1_inv, r1m, theta1m, k, r1, batch_size=128
):
    npix = r1.shape[0]

    def _input_beam_field_outer(xs):
        a_i, p_i, q1_i, r1m_i, t1m_i, k_i = xs
        return _input_beam_field(a_i, p_i, q1_i, r1m_i, t1m_i, k_i, r1)

    init = jnp.zeros((npix,), dtype=jnp.complex128)
    xs = (amp, phase_offset, Q1_inv, r1m, theta1m, k)
    out = map_reduce(_input_beam_field_outer, jnp.add, init, xs, batch_size=batch_size)
    return out  # (npix,)


# def evaluate_misaligned_input_gaussian_jax_scan(
#     amp, phase_offset, Q1_inv, r1m, theta1m, k, r1, batch_size=128
# ):
#     npix = r1.shape[0]
#     # Pick accumulator dtype that matches inputs to avoid upcasts:
#     complex_dtype = jnp.result_type(amp, 1j)
#     init = jnp.zeros((npix,), dtype=complex_dtype)

#     # Prefer a single vmap + sum when memory permits (faster than map_reduce):
#     # vmapped over the beam axis; each call returns shape (npix,)
#     vmapped = jax.vmap(lambda a, p, q, r0, t, kk: _input_beam_field(a, p, q, r0, t, kk, r1))
#     fields = vmapped(amp, phase_offset, Q1_inv, r1m, theta1m, k)  # (nbeams, npix)
#     out = jnp.sum(fields, axis=0)  # (npix,)
#     return out


evaluate_misaligned_input_gaussian_jax_scan = jax.jit(
    evaluate_misaligned_input_gaussian_jax_scan, static_argnames=["batch_size"]
)
