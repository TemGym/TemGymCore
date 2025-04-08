import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def rk5_step(f, t, x, h):
    k1 = f(t, x)
    k2 = f(t + (1./5)*h, x + h*(1./5*k1))
    k3 = f(t + (3./10)*h, x + h*(3./40*k1 + 9./40*k2))
    k4 = f(t + (4./5)*h, x + h*(44./45*k1 - 56./15*k2 + 32./9*k3))
    k5 = f(t + (8./9)*h, x + h*(19372./6561*k1 - 25360./2187*k2 + 64448./6561*k3 - 212./729*k4))
    k6 = f(t + h, x + h*(9017./3168*k1 - 355./33*k2 + 46732./5247*k3 + 49./176*k4 - 5103./18656*k5))
    v5 = 35./384*k1 + 500./1113*k3 + 125./192*k4 - 2187./6784*k5 + 11./84*k6
    
    x_next = x + h*v5
    return x_next


@partial(jax.jit, static_argnums=(0,))
def ode_solver_static(f, y0, t):
    y = y0
    t_prev = t[0]
    for t_curr in t[1:]:
        dt = t_curr - t_prev
        y = rk5_step(f, t_prev, y, dt)
        t_prev = t_curr
    return y


def complete_equation_of_motion(z, x, args):
    phi_lambda, E_lambda = args

    v = 1.0 + x[2]**2 + x[3]**2
    u = phi_lambda(x[0], x[1], z)
    Ex, Ey, Ez = E_lambda(x[0], x[1], z)
    return jnp.array([x[2], x[3], (-0.5 * u) * v * (Ex - x[2]*Ez), (-0.5 * u) * v * (Ey - x[3]*Ez)])