import jax.numpy as jnp
import jax
from .grid import Grid
from .run import run_to_end
from .utils import custom_jacobian_matrix
from dataclasses import fields
from jax._src.lax.control_flow.loops import _batch_and_remainder
from jax import lax


def w_z(w0, z, z_r):
    return w0 * jnp.sqrt(1 + (z / z_r) ** 2)


def zR(w0, wavelength):
    return (jnp.pi * w0**2) / wavelength


def R(z, z_r):
    cond = jnp.abs(z) < 1e-10
    z_r_over_z = jax.lax.cond(cond, lambda op: 0.0, lambda op: op[1] / op[0], (z, z_r))
    # z_r_over_z = jnp.where(cond, 0.0, z_r / z) - this gave me an error and I don't know why it tried to evaluate z_r / z
    return jax.lax.cond(cond, lambda _: jnp.inf, lambda _: z * (1 + z_r_over_z ** 2), operand=None)


def gaussian_beam(x, y, q_inv, k, offset_x=0, offset_y=0):
    return jnp.exp(1j * k * ((x+offset_x)**2 + (y+offset_y)**2) / 2 * q_inv)


def Qinv_ABCD(Qinv, A, B, C, D):
    # compute (C + D @ Qinv) @ inv(A + B @ Qinv) without explicit inv
    lhs = A + B @ Qinv
    rhs = C + D @ Qinv
    return jnp.linalg.solve(lhs, rhs)


def q_inv(z, w0, wl):
    z_r = zR(w0, wl)
    cond = jnp.abs(z) < 1e-10
    wz_val = w_z(w0, z, z_r)
    R_val = R(z, z_r)

    q_inv = jnp.where(
        cond,
        -1.0 / (1j * (jnp.pi * w0**2) / wl),
        1.0 / R_val - 1j * wl / (jnp.pi * wz_val**2),
    )
    return q_inv


def matrix_vector_mul(M, v):
    """
    Batched matrix-vector multiplication.
    M: (nb,2,2)
    v: (nb,2)
    Returns (nb,2) result of M @ v for each batch.
    """
    return jnp.einsum('ij,j->i', M, v)


def matrix_matrix_mul(M1, M2):
    """
    Batched matrix-matrix multiplication.
    M1: (nb,2,2)
    M2: (nb,2,2)
    Returns (nb,2,2) result of M1 @ M2 for each batch.
    """
    return jnp.einsum('ij,jk->ik', M1, M2)


def matrix_quadratic_mul(v, M):
    """
    Batched quadratic multiplication -  v^T M v.
    v: (nb,2)
    M: (nb,2,2)
    Returns (nb,)
    """
    return jnp.einsum('i,ij,j->', v, M, v)


def matrix_linear_mul(v, M, w):
    """
    Batched linear multiplication - v^T M w
    v: (nb,2)
    M: (nb,2,2)
    w: (np,2)  -- observation coordinates (no batch)
    Returns (nb,np)
    """
    return jnp.einsum('i,ij,nj->n', v, M, w)


def matrix_matrix_matrix_mul(M1, M2, M3):
    return jnp.einsum('nij,njk,npk->nip', M1, M2, M3)


def _beam_field(
    amp, Q1_inv, Q2_inv, r2, r1m, theta1m, k, A, B, e
):
    """Single-beam field at all observation points r2 -> (np,)"""
    I = jnp.eye(2, dtype=B.dtype)  # noqa

    # Safe inverses
    B_inv = jnp.linalg.solve(B, I)
    B_inv = jnp.nan_to_num(B_inv, nan=0., posinf=0., neginf=0.)

    Q1 = jnp.linalg.solve(Q1_inv, I)
    Q1 = jnp.nan_to_num(Q1, nan=0., posinf=0., neginf=0.)

    # Central ray at output: r2m = A r1m + B theta1m + e
    r2m = matrix_vector_mul(A, r1m) + matrix_vector_mul(B, theta1m) + e  # (2,)

    # AB-q amplitude prefactor
    denom = A + matrix_matrix_mul(B, Q1_inv)    # (2,2)
    pref = amp / jnp.sqrt(jnp.linalg.det(denom))  # ()

    # Misalignment phase (input plane)
    ABinv = matrix_matrix_mul(A, B_inv)
    phi1 = matrix_quadratic_mul(r1m, ABinv) - 2 * matrix_linear_mul(r1m, B_inv, r2)  # (np,)

    # Misalignment phase (output plane)
    AQ1 = matrix_matrix_mul(A, Q1)
    B_over_AQ1B = jnp.linalg.solve(matrix_matrix_mul(B, AQ1 + B), I)  # (2,2)
    Q1B_over_AQ = matrix_matrix_mul(Q1, B_over_AQ1B)                   # (2,2)
    phi2 = (
        matrix_quadratic_mul(r2m, Q1B_over_AQ)
        - 2 * matrix_linear_mul(r2m, Q1B_over_AQ, r2)
      )  # (np,)

    Q2t = jnp.einsum('ni,ij,nj->n', r2, Q2_inv, r2)  # (np,)

    phase = (k/2) * (Q2t + phi1 - phi2)  # (np,)
    return pref * jnp.exp(1j * phase)    # (np,)


def propagate_misaligned_gaussian_jax_scan(
    amp, Q1_inv, A, B, C, D, e, r2, r1m, theta1m, k, batch_size=128
):
    npix = r2.shape[0]
    Q2_inv = Qinv_ABCD(Q1_inv, A, B, C, D)  # (nb,2,2)

    def _beam_field_outer(xs):
        a_i, q1_i, q2_i, r1_i, t1_i, A_i, B_i, e_i, k_i = xs
        return _beam_field(a_i, q1_i, q2_i, r2, r1_i, t1_i, k_i, A_i, B_i, e_i)

    init = jnp.zeros((npix,), dtype=jnp.complex128)
    xs = (amp, Q1_inv, Q2_inv, r1m, theta1m, A, B, e, k)
    out = map_reduce(_beam_field_outer, jnp.add, init, xs, batch_size=batch_size)
    return out  # (npix,)


propagate_misaligned_gaussian_jax_scan = jax.jit(propagate_misaligned_gaussian_jax_scan, static_argnames=['batch_size'])


def get_image(gaussian_rays, model, batch_size=128):
    from .ray import GaussianRay

    rays = gaussian_rays
    assert isinstance(rays, GaussianRay)

    n_rays = rays.amplitude.shape[0]
    for param in fields(rays):
        if getattr(rays, param.name).shape[0] != n_rays:
            assert False, f"All ray parameters must have same leading dimension, but {param.name} has shape {getattr(rays, param.name).shape}"

    grid = model[-1]
    assert isinstance(grid, Grid)

    vmap_fn = jax.vmap(jax.jacobian(run_to_end), in_axes=(0, None))
    central_rays = rays.to_ray()
    output_tm = vmap_fn(central_rays, model)

    model_ray_jacobians = custom_jacobian_matrix(output_tm)
    ABCDs = jnp.array(model_ray_jacobians)

    amplitudes = rays.amplitude

    Q1_invs = rays.Q_inv  # Should be of shape n x 2 x 2
    As = ABCDs[:, 0:2, 0:2]   # (nb,2,2)
    Bs = ABCDs[:, 0:2, 2:4]   # (nb,2,2)
    Cs = ABCDs[:, 2:4, 0:2]   # (nb,2,2)
    Ds = ABCDs[:, 2:4, 2:4]   # (nb,2,2)
    es = ABCDs[:, 0:2, 4]  # (nb,2)
    r2 = grid.coords
    r1ms = jnp.stack([central_rays.x, central_rays.y], axis=-1)
    theta1ms = jnp.stack([central_rays.dx, central_rays.dy], axis=-1)
    wavelengths = rays.wavelength
    k = 2 * jnp.pi / wavelengths

    output_field = propagate_misaligned_gaussian_jax_scan(
        amplitudes, Q1_invs, As, Bs, Cs, Ds, es, r2, r1ms, theta1ms, k, batch_size=batch_size
    ).reshape(grid.shape)
    return output_field


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
