"""Shared magnetic-cube recipe for the holography examples."""

import jax.numpy as jnp
from temgym_core.constants import e_over_hbar, interaction_constant_CE

def rotation_matrix_x(theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c],
    ])


def make_tilted_uniform_cube(
    tilt_x,  # tilt angle [rad] about x-axis
    N=256,
    L=100e-9,  # simulation side length [m]
    cube_fraction=0.25,  # cube side length as fraction of L
    V0=10.0,  # mean inner potential [V]
    B0=0.6,  # internal B [Tesla]
    amplitude_dampening=1.0,  # target amplitude dampening inside the cube.
    x0=0.0, y0=0.0, z0=0.0,
):
    """
    Build a uniform MIP + uniform magnetisation cube, tilted by tilt_x
    around the lab x-axis.

    Returns:
        fields: (Nx, Ny, Nz, 3) with channels [V, A_z_lab, mu]
        x_coords, y_coords, z_coords: lab coordinates
    """
    # --- lab coordinates ---
    x_coords = jnp.linspace(-L / 2, L / 2, N)
    y_coords = jnp.linspace(-L / 2, L / 2, N)
    z_coords = jnp.linspace(-L / 2, L / 2, N)

    X_lab, Y_lab, Z_lab = jnp.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    # --- rotation: object -> lab ---
    R = rotation_matrix_x(tilt_x)
    R_T = R.T

    # Map lab coordinates to object coordinates: r_obj = R^T r_lab
    X_obj = R_T[0, 0] * X_lab + R_T[0, 1] * Y_lab + R_T[0, 2] * Z_lab
    Y_obj = R_T[1, 0] * X_lab + R_T[1, 1] * Y_lab + R_T[1, 2] * Z_lab
    Z_obj = R_T[2, 0] * X_lab + R_T[2, 1] * Y_lab + R_T[2, 2] * Z_lab

    # --- cube mask in object coordinates ---
    half_side = 0.5 * L * cube_fraction
    mask = (
        (jnp.abs(X_obj) <= half_side)
        & (jnp.abs(Y_obj) <= half_side)
        & (jnp.abs(Z_obj) <= half_side)
    )

    # --- MIP: uniform V0 inside cube ---
    V = V0 * mask / (L * cube_fraction)

    # --- Magnetic gauge in object frame ---
    # A_obj = (0, 0, B0 * x_obj)
    A_z_obj = B0 * X_obj * mask

    # Transform A to lab frame: A_lab = R @ A_obj
    # Since A_obj = (0,0,A_z_obj), we have A_z_lab = R[2,2] * A_z_obj
    A_z_lab = R[2, 2] * A_z_obj

    # --- Log-amplitude: uniform mu inside cube ---
    t_cube = 2 * half_side
    mu = jnp.log(amplitude_dampening) / t_cube
    mu_field = mu * mask

    fields = jnp.stack([V, A_z_lab, mu_field], axis=-1)  # (Nx, Ny, Nz, 3)

    # Shift the cube to (x0, y0, z0)
    x_coords = x_coords + x0
    y_coords = y_coords + y0
    z_coords = z_coords + z0

    return fields, x_coords, y_coords, z_coords


def project_phase_along_z(fields, z_coords, voltage):
    V = fields[..., 0]    # (Nx, Ny, Nz)
    A_z = fields[..., 1]  # (Nx, Ny, Nz)
    log_amplitude = fields[..., 2]  # (Nx, Ny, Nz)

    dz = z_coords[1] - z_coords[0]
    C_E = interaction_constant_CE(voltage)  # [rad / (V·m)]
    e_over_hbar_val = e_over_hbar()  # [rad / (T·m)]
    g = C_E * V + e_over_hbar_val * A_z   # generator per unit length

    phi = jnp.sum(g, axis=-1) * dz    # integrate over z → shape (Nx, Ny)

    amp = jnp.exp(jnp.sum(log_amplitude, axis=-1) * dz)  # integrate over z → shape (Nx, Ny)
    return amp, phi
