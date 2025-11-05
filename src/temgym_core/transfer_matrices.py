import sympy as sp
import numpy as np
import jax.numpy as jnp


def imaging_matrix(M, f, xp=sp):
    """
    Returns the 3x3 ABCD matrix for a perfect imaging system
    with magnification M and focal length f. Last column is [0,0,1].
    """
    if xp == sp:
        return xp.Matrix([[M, 0, 0], [-1/f, 1/M, 0], [0, 0, 1]])
    elif xp == np:
        return np.array([[M, 0, 0], [-1.0 / f, 1.0 / M, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64)
    elif xp == jnp:
        return jnp.array([[M, 0, 0], [-1.0 / f, 1.0 / M, 0.0], [0.0, 0.0, 1.0]],
                         dtype=jnp.float64)


def imaging_matrix_5x5(M, f, xp=sp):
    """
    5x5 imaging matrix acting on [x, y, theta_x, theta_y, 1]^T.
    """
    if xp == sp:
        return xp.Matrix([
            [M, 0, 0,   0,   0],
            [0, M, 0,   0,   0],
            [-1/f, 0, 1/M, 0, 0],
            [0, -1/f, 0, 1/M, 0],
            [0, 0, 0,   0,   1],
        ])
    elif xp == np:
        return np.array([
            [M, 0.0, 0.0, 0.0, 0.0],
            [0.0, M, 0.0, 0.0, 0.0],
            [-1.0/f, 0.0, 1.0/M, 0.0, 0.0],
            [0.0, -1.0/f, 0.0, 1.0/M, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
    elif xp == jnp:
        return jnp.array([
            [M, 0.0, 0.0, 0.0, 0.0],
            [0.0, M, 0.0, 0.0, 0.0],
            [-1.0/f, 0.0, 1.0/M, 0.0, 0.0],
            [0.0, -1.0/f, 0.0, 1.0/M, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=jnp.float64)


def scale_matrix(M, xp=sp):
    """
    3x3 scaling matrix with last column [0,0,1].
    """
    if xp == sp:
        return xp.Matrix([[M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    elif xp == np:
        return np.array([[M, 0, 0], [0, 1.0 / M, 0], [0, 0, 1.0]],
                        dtype=np.float64)
    elif xp == jnp:
        return jnp.array([[M, 0, 0], [0, 1.0 / M, 0], [0, 0, 1.0]],
                         dtype=jnp.float64)


def scale_matrix_5x5(M, xp=sp):
    """
    5x5 scaling matrix for [x,y,theta_x,theta_y,1]^T, scaling x,y by M and angles by 1/M.
    """
    if xp == sp:
        return xp.Matrix([
            [M, 0, 0,   0,   0],
            [0, M, 0,   0,   0],
            [0, 0, 1/M, 0,   0],
            [0, 0, 0, 1/M,   0],
            [0, 0, 0,   0,   1],
        ])
    elif xp == np:
        return np.array([
            [M, 0.0, 0.0, 0.0, 0.0],
            [0.0, M, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0/M, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0/M, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
    elif xp == jnp:
        return jnp.array([
            [M, 0.0, 0.0, 0.0, 0.0],
            [0.0, M, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0/M, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0/M, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=jnp.float64)


def propagation_matrix(z, xp=sp):
    """
    Returns the 3x3 ABCD matrix for free space propagation over distance z.
    """
    if xp == sp:
        return xp.Matrix([[1, z, 0], [0, 1, 0], [0, 0, 1]])
    elif xp == np:
        return np.array([[1.0, z, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64)
    elif xp == jnp:
        return jnp.array([[1.0, z, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                         dtype=jnp.float64)


def propagation_matrix_5x5(z, xp=sp):
    """
    5x5 free space propagation for [x,y,theta_x,theta_y,1]^T.
    x' = x + z*theta_x; y' = y + z*theta_y.
    """
    if xp == sp:
        return xp.Matrix([
            [1, 0, z, 0, 0],
            [0, 1, 0, z, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])
    elif xp == np:
        return np.array([
            [1.0, 0.0, z,   0.0, 0.0],
            [0.0, 1.0, 0.0, z,   0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
    elif xp == jnp:
        return jnp.array([
            [1.0, 0.0, z,   0.0, 0.0],
            [0.0, 1.0, 0.0, z,   0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=jnp.float64)


def lens_matrix(f, xp=sp):
    """
    Returns the 3x3 ABCD matrix for a thin lens with focal length f.
    """
    if xp == sp:
        return xp.Matrix([[1, 0, 0], [-1 / f, 1, 0], [0, 0, 1]])
    elif xp == np:
        return np.array([[1.0, 0.0, 0.0], [-1.0 / f, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64)
    elif xp == jnp:
        return jnp.array([[1.0, 0.0, 0.0], [-1.0 / f, 1.0, 0.0], [0.0, 0.0, 1.0]],
                         dtype=jnp.float64)


def lens_matrix_5x5(f, xp=sp):
    """
    5x5 thin lens (same focal length for x and y).
    theta_x' = theta_x - x/f; theta_y' = theta_y - y/f.
    """
    if xp == sp:
        return xp.Matrix([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [-1/f, 0, 1, 0, 0],
            [0, -1/f, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])
    elif xp == np:
        return np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0/f, 0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0/f, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
    elif xp == jnp:
        return jnp.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0/f, 0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0/f, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=jnp.float64)


def biprism_matrix(biprism_deflection, xp=sp):
    """
    Returns the 3x3 matrix adding constant deflection to angle term.
    """
    if xp == sp:
        return xp.Matrix([[1.0, 0.0, 0.0],
                          [0.0, 1.0, biprism_deflection],
                          [0.0, 0.0, 1.0]])
    elif xp == np:
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, biprism_deflection],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    elif xp == jnp:
        return jnp.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, biprism_deflection],
                          [0.0, 0.0, 1.0]], dtype=jnp.float64)


def biprism_matrix_5x5(biprism_deflection, xp=sp):
    """
    5x5 version adding same constant deflection to both theta_x and theta_y.
    """
    if xp == sp:
        return xp.Matrix([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, biprism_deflection],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])
    elif xp == np:
        return np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, biprism_deflection],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
    elif xp == jnp:
        return jnp.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, biprism_deflection],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=jnp.float64)


def propagation_refractive_index(z, n, xp=sp):
    """
    Returns the 3x3 ABCD matrix for propagation over distance z with refractive index n.
    """
    if xp == sp:
        return xp.Matrix([[1, z / n, 0], [0, 1, 0], [0, 0, 1]])
    elif xp == np:
        return np.array([[1.0, z / n, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64)
    elif xp == jnp:
        return jnp.array([[1.0, z / n, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                         dtype=jnp.float64)


def refraction_flat(n_1, n_2, xp=sp):
    """
    Returns the 3x3 ABCD matrix for refraction at a flat interface between
    two media with refractive indices n_1 and n_2.
    """
    if xp == sp:
        return xp.Matrix([[1, 0, 0], [0, n_1 / n_2, 0], [0, 0, 1]])
    elif xp == np:
        return np.array([[1.0, 0.0, 0.0], [0.0, (n_1 / n_2), 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64)
    elif xp == jnp:
        return jnp.array([[1.0, 0.0, 0.0], [0.0, (n_1 / n_2), 0.0], [0.0, 0.0, 1.0]],
                         dtype=jnp.float64)


def calculate_z1_and_z2_from_M_and_f(M, f):
    """
    Given the magnification M and focal length f,
    calculate the distances z1 and z2 for a two-lens system
    to satisfy the imaging condition.
    """
    z1 = f * (1/M - 1)
    z2 = f * (1 - M)
    return z1, z2


def calculate_z2_and_M_from_z1_and_f(z1, f):
    """
    Given the distance z1 and focal length f,
    calculate the distance z2 and magnification M
    for a two-lens system to satisfy the imaging condition.
    """
    M = 1 / (1 + z1/f)
    z2 = f * (1 - M)
    return z2, M


def full_abcd_2lens(zs, f1, f2, symbolic=False):
    """
    Build the full 3x3 ABCD matrix for a two-lens system (propagations + lenses).
    If symbolic is True, uses sympy and returns a sympy.Matrix, otherwise returns a numpy array.
    """
    z1, z2, z3 = zs
    xp = sp if symbolic else np

    P1 = propagation_matrix(z1, xp=xp)
    L1 = lens_matrix(f1, xp=xp)
    P2 = propagation_matrix(z2, xp=xp)
    L2 = lens_matrix(f2, xp=xp)
    P3 = propagation_matrix(z3, xp=xp)

    M = P3 @ L2 @ P2 @ L1 @ P1

    if symbolic:
        return M
    else:
        return np.array(M, dtype=np.float64)
