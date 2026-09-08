"""Small analytical matrices and ray transport through supplied matrices.

5×5 matrices act on [x, y, theta_x, theta_y, 1]; 3×3 matrices act on
[x, theta_x, 1]. Constructors default to SymPy for existing derivations;
pass xp=numpy or xp=jax.numpy for numerical arrays and JAX derivatives.

For a microscope built from components, use run.column_matrix instead.
"""

import jax.numpy as jnp
import numpy as np
import sympy as sp


def _matrix(rows, xp):
    """Construct each matrix once, then select its numerical or symbolic type."""
    if xp is sp:
        return sp.Matrix(rows)
    if xp is np or xp is jnp:
        return xp.array(rows, dtype=xp.float64)
    raise ValueError("xp must be sympy, numpy, or jax.numpy.")


def imaging_matrix(M, f, xp=sp):
    """Perfect imaging: A=M, B=0, C=-1/f, D=1/M."""
    return _matrix([[M, 0, 0], [-1 / f, 1 / M, 0], [0, 0, 1]], xp)


def imaging_matrix_5x5(M, f, xp=sp):
    """The same perfect imaging map in both transverse axes."""
    return _matrix([
        [M, 0, 0, 0, 0],
        [0, M, 0, 0, 0],
        [-1 / f, 0, 1 / M, 0, 0],
        [0, -1 / f, 0, 1 / M, 0],
        [0, 0, 0, 0, 1],
    ], xp)


def scale_matrix(M, xp=sp):
    """Scale position by M and slope by 1/M."""
    return _matrix([[M, 0, 0], [0, 1 / M, 0], [0, 0, 1]], xp)


def scale_matrix_5x5(M, xp=sp):
    """Scale both positions by M and both slopes by 1/M."""
    return _matrix([
        [M, 0, 0, 0, 0],
        [0, M, 0, 0, 0],
        [0, 0, 1 / M, 0, 0],
        [0, 0, 0, 1 / M, 0],
        [0, 0, 0, 0, 1],
    ], xp)


def propagation_matrix(z, xp=sp):
    """Free-space drift: x' = x + z theta_x."""
    return _matrix([[1, z, 0], [0, 1, 0], [0, 0, 1]], xp)


def propagation_matrix_5x5(z, xp=sp):
    """Free-space drift through distance z in both axes."""
    return _matrix([
        [1, 0, z, 0, 0],
        [0, 1, 0, z, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], xp)


def lens_matrix(f, xp=sp):
    """Thin lens: theta_x' = theta_x - x/f."""
    return _matrix([[1, 0, 0], [-1 / f, 1, 0], [0, 0, 1]], xp)


def lens_matrix_5x5(f, xp=sp):
    """Round thin lens with the same focal length in both axes."""
    return _matrix([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [-1 / f, 0, 1, 0, 0],
        [0, -1 / f, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], xp)


def biprism_matrix(biprism_deflection, xp=sp):
    """Constant angular kick; the caller supplies its sign for each wire side."""
    return _matrix([[1, 0, 0], [0, 1, biprism_deflection], [0, 0, 1]], xp)


def biprism_matrix_5x5(biprism_deflection, xp=sp):
    """Constant kick in theta_x only; y and theta_y are unchanged."""
    return _matrix([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, biprism_deflection],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], xp)


def double_deflector_matrix_5x5(
    spacing, drive_x, drive_y, balance_x=1.0, balance_y=1.0, xp=sp,
):
    """Two angular kicks separated by spacing: D2 @ P(spacing) @ D1.

    The first kick is (drive_x, drive_y); the second is its negative,
    scaled by the respective balance. Balanced kicks leave a pure shift.
    """
    return _matrix([
        [1, 0, spacing, 0, spacing * drive_x],
        [0, 1, 0, spacing, spacing * drive_y],
        [0, 0, 1, 0, drive_x - balance_x * drive_x],
        [0, 0, 0, 1, drive_y - balance_y * drive_y],
        [0, 0, 0, 0, 1],
    ], xp)


def propagation_refractive_index(z, n, xp=sp):
    """Drift with reduced distance z/n."""
    return propagation_matrix(z / n, xp=xp)


def refraction_flat(n_1, n_2, xp=sp):
    """Flat interface: scale slope by n_1/n_2."""
    return _matrix([[1, 0, 0], [0, n_1 / n_2, 0], [0, 0, 1]], xp)


def rotation_matrix(theta, xp=sp):
    """Legacy rotation of the two 3×3 state coordinates, not an x/y lens rotation."""
    c, s = xp.cos(theta), xp.sin(theta)
    return _matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]], xp)


def calculate_z1_and_z2_from_M_and_f(M, f):
    """Single-lens object/image coordinates relative to the lens at z=0."""
    return f * (1 / M - 1), f * (1 - M)


def calculate_z2_and_M_from_z1_and_f(z1, f):
    """Image coordinate and magnification for an object at signed coordinate z1."""
    M = 1 / (1 + z1 / f)
    return f * (1 - M), M


def calculate_M_and_f_from_z1_and_z2(z1, z2):
    """Magnification and focal length from signed object/image coordinates."""
    return z2 / z1, z1 * z2 / (z1 - z2)


def full_abcd_2lens(zs, f1, f2, symbolic=False):
    """Three drifts and two thin lenses, ordered from object to image."""
    z1, z2, z3 = zs
    xp = sp if symbolic else np
    return (
        propagation_matrix(z3, xp=xp) @ lens_matrix(f2, xp=xp)
        @ propagation_matrix(z2, xp=xp) @ lens_matrix(f1, xp=xp)
        @ propagation_matrix(z1, xp=xp)
    )


def accumulate_matrices(matrices):
    """Product M[-1] @ ... @ M[0]; supply a nonempty sequence in travel order."""
    total = matrices[-1]
    for matrix in reversed(matrices[:-1]):
        total = total @ matrix
    return total


def accumulate_matrices_cumulative(matrices):
    """JAX array of maps to each plane: M[0], M[1] @ M[0], ... ."""
    total = matrices[0]
    cumulative = [total]
    for matrix in matrices[1:]:
        total = matrix @ total
        cumulative.append(total)
    return jnp.stack(cumulative)


def transfer_rays(ray_coords, transfer_matrices):
    """Rays (N, 5) and M successive matrices → JAX coordinates (N, M, 5)."""
    cumulative = accumulate_matrices_cumulative(transfer_matrices)
    return jnp.einsum("mij,nj->nmi", cumulative, ray_coords)


def transfer_rays_pt_src(input_pos_xy, input_slopes_xy, transfer_matrix):
    """Point position, two N-element slope arrays and a 5×5 map → (4, N)."""
    dx, dy = map(jnp.asarray, input_slopes_xy)
    x, y = input_pos_xy
    rays = jnp.stack((jnp.full(dx.shape, x), jnp.full(dy.shape, y), dx, dy, jnp.ones_like(dx)))
    return (transfer_matrix @ rays)[:4]
