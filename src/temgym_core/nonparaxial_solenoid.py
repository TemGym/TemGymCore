"""Exact uniform-field ray benchmark, not a real pole-piece field model.

Coordinates are (x, pi_x, y, pi_y); canonical momenta are divided by the
fixed total relativistic momentum p0, NOT pz. Signed k=Q Bz/(2 p0), symmetric
gauge Q A/p0=(-k*y,k*x), so kinetic transverse momenta are
u=pi_x+k*y, v=pi_y-k*x. Mechanical slopes are (u,v)/sqrt(1-u*u-v*v).

At fixed energy and forward pz, H=-sqrt(1-u*u-v*v). The kinetic transverse
norm is constant in a uniform body; its exact map is the usual linear
solenoid map with distance L/sqrt(1-u*u-v*v). The quartic Hamiltonian
H2+H4=(u*u+v*v)/2+(u*u+v*v)**2/8 instead multiplies L by 1+(u*u+v*v)/2.

No finite-fringe geometry or higher radial field derivatives are supplied.
Joining these bodies by discontinuous k gives a particular idealized
symmetric-gauge field, not a Maxwell-consistent current-free bore fringe.
No spin, radiation, energy spread, quantum propagation or apertures are added.

Hamiltonian reference: H. Wiedemann, Particle Accelerator Physics (2015),
Eq. (5.40), https://doi.org/10.1007/978-3-319-18317-6_5.
The signed linear-map convention agrees with immersed_objective.py.
"""

import jax.numpy as jnp


def kinetic_transverse_momentum(state, k_m_inv):
    state = jnp.asarray(state)
    if state.shape[-1] != 4:
        raise ValueError('Last coordinate dimension must be (x,pi_x,y,pi_y).')
    return jnp.stack((state[..., 1]+k_m_inv*state[..., 2],
                      state[..., 3]-k_m_inv*state[..., 0]), axis=-1)


def mechanical_slopes(state, k_m_inv):
    uv = kinetic_transverse_momentum(state, k_m_inv)
    return uv/jnp.sqrt(1-jnp.sum(uv*uv, axis=-1))[..., None]


def canonical_from_mechanical_slopes(position_xy, slopes_xy, k_m_inv):
    xy, alpha = jnp.asarray(position_xy), jnp.asarray(slopes_xy)
    uv = alpha/jnp.sqrt(1+jnp.sum(alpha*alpha, axis=-1))[..., None]
    return jnp.stack((xy[..., 0], uv[..., 0]-k_m_inv*xy[..., 1],
                      xy[..., 1], uv[..., 1]+k_m_inv*xy[..., 0]), axis=-1)


def uniform_solenoid_hamiltonian(state, k_m_inv):
    uv = kinetic_transverse_momentum(state, k_m_inv)
    return -jnp.sqrt(1-jnp.sum(uv*uv, axis=-1))


def uniform_solenoid_ray(state, k_m_inv, distance_m, *, order='exact'):
    """JAX differentiable uniform body/drift; non-forward exact states -> NaN.

    ``quartic`` is the exact flow of H2+H4, not an exact Lorentz trajectory.
    ``paraxial`` is H2 and agrees with uniform_field_canonical_matrix.
    States and k/d broadcast. Zero field and zero distance are smooth.
    """
    state = jnp.asarray(state)
    uv = kinetic_transverse_momentum(state, k_m_inv)
    r2 = jnp.sum(uv*uv, axis=-1)
    if order == 'exact':
        distance = distance_m/jnp.sqrt(1-r2)
    elif order == 'quartic':
        distance = distance_m*(1+.5*r2)
    elif order == 'paraxial':
        distance = distance_m+jnp.zeros_like(r2)
    else:
        raise ValueError('order must be exact, quartic or paraxial.')
    phase = 2*k_m_inv*distance
    cosine, sine = jnp.cos(phase), jnp.sin(phase)
    a = jnp.sinc(phase/jnp.pi)
    b = .5*phase*jnp.sinc(.5*phase/jnp.pi)**2
    x = state[..., 0]+distance*(a*uv[..., 0]+b*uv[..., 1])
    y = state[..., 2]+distance*(a*uv[..., 1]-b*uv[..., 0])
    u = cosine*uv[..., 0]+sine*uv[..., 1]
    v = cosine*uv[..., 1]-sine*uv[..., 0]
    return jnp.stack((x, u-k_m_inv*y, y, v+k_m_inv*x), axis=-1)
