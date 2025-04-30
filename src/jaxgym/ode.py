import jax
import jax.numpy as jnp
from functools import partial
import diffrax
import numpy as np
from tqdm.autonotebook import trange, tqdm

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

#Custom odedopri solver for testing daceypy
def odedopri(f,  x0,  y0,  x1,  tol,  hmax,  hmin,  maxiter, args=()):
   # we trust that the compiler is smart enough to pre-evaluate the
   # value of the constants.
   a21 = (1.0/5.0)
   a31 = (3.0/40.0)
   a32 = (9.0/40.0)
   a41 = (44.0/45.0)
   a42 = (-56.0/15.0)
   a43 = (32.0/9.0)
   a51 = (19372.0/6561.0)
   a52 = (-25360.0/2187.0)
   a53 = (64448.0/6561.0)
   a54 = (-212.0/729.0)
   a61 = (9017.0/3168.0)
   a62 = (-355.0/33.0)
   a63 = (46732.0/5247.0)
   a64 = (49.0/176.0)
   a65 = (-5103.0/18656.0)
   a71 = (35.0/384.0)
   a72 = (0.0)
   a73 = (500.0/1113.0)
   a74 = (125.0/192.0)
   a75 = (-2187.0/6784.0)
   a76 = (11.0/84.0)

   c2 = (1.0 / 5.0)
   c3 = (3.0 / 10.0)
   c4 = (4.0 / 5.0)
   c5 = (8.0 / 9.0)
   c6 = (1.0)
   c7 = (1.0)

   b1 = (35.0/384.0)
   b2 = (0.0)
   b3 = (500.0/1113.0)
   b4 = (125.0/192.0)
   b5 = (-2187.0/6784.0)
   b6 = (11.0/84.0)
   b7 = (0.0)

   b1p = (5179.0/57600.0)
   b2p = (0.0)
   b3p = (7571.0/16695.0)
   b4p = (393.0/640.0)
   b5p = (-92097.0/339200.0)
   b6p = (187.0/2100.0)
   b7p = (1.0/40.0)

   x = x0
   y = y0
   h = hmax

   pbar = trange(maxiter)
   for i in pbar:
      # Update progress bar description with the current value of x using the walrus operator.
      pbar.set_description(f"x: {x:=}")

      # Compute the function values
      K1 = f(x,       y, args)
      K2 = f(x + c2*h, y+h*(a21*K1), args)
      K3 = f(x + c3*h, y+h*(a31*K1+a32*K2), args)
      K4 = f(x + c4*h, y+h*(a41*K1+a42*K2+a43*K3), args)
      K5 = f(x + c5*h, y+h*(a51*K1+a52*K2+a53*K3+a54*K4), args)
      K6 = f(x + h, y+h*(a61*K1+a62*K2+a63*K3+a64*K4+a65*K5), args)
      K7 = f(x + h, y+h*(a71*K1+a72*K2+a73*K3+a74*K4+a75*K5+a76*K6), args)

      error = abs((b1-b1p)*K1+(b3-b3p)*K3+(b4-b4p)*K4+(b5-b5p)*K5 +
               (b6-b6p)*K6+(b7-b7p)*K7)

      # Error in X controls tolerance
      error = max(error)

      # Error control
      if error != 0:
         delta = 0.84 * pow(tol / error, (1.0/5.0))
      else:
         delta = 4.0  # if error is 0, we can increase the step

      if error < tol:
        x = x + h
        y = y + h * (b1*K1+b3*K3+b4*K4+b5*K5+b6*K6)
        
      if delta <= 0.1:
        h = h * 0.1
      elif delta >= 4.0:
        h = h * 4.0
      else:
        h = delta * h
      
      if h > hmax:
        h = hmax

      if x >= x1:
        flag = 0
        print('z =', x)
        break

      elif x + h > x1:
        h = x1 - x

      elif h < hmin:
        # flag = 1
        print('z =', x)
        print('At hmin')
        break

   maxiter = maxiter - i
   if i <= 0:
      flag = 2

   return x, y

@partial(jax.jit, static_argnums=(3, 4))
def solve_ode(y0, z0, z1, phi_lambda, E_lambda, u0):

    # Set up the ODE solver.
    term = diffrax.ODETerm(electron_equation_of_motion)
    solver = diffrax.Dopri8()  # Tsit5 solver.
    stepsize_controller = diffrax.PIDController(rtol=1e-15, atol=1e-15, dtmax=10000, dtmin=1e-15)
    Adjoint = diffrax.ForwardMode()
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=z0,
        t1=z1,
        y0=y0,
        dt0=None,
        stepsize_controller=stepsize_controller,
        args=(phi_lambda, E_lambda, u0),
        adjoint=Adjoint,
    )

    return sol.ys[0], sol.ts[0]

def electron_equation_of_motion(z, x, args):
    #z
    #x = [x, y, px, py, opl]
    phi_lambda, E_lambda, u0 = args

    v = 1.0 + x[2]**2 + x[3]**2
    u = phi_lambda(x[0], x[1], z)
    Ex, Ey, Ez = E_lambda(x[0], x[1], z)

    dx = x[2]
    dy = x[3]
    ddx = (-0.5 / u) * v * (Ex - x[2]*Ez)
    ddy = (-0.5 / u) * v * (Ey - x[3]*Ez)
    dopl = (u/u0) ** (1/2) * (v) ** (1/2)

    return jnp.array([dx, dy, ddx, ddy, dopl])

 
def electron_equation_of_motion_DA(z, x, args):
    phi_lambda, E_lambda, u0 = args

    v = 1.0 + x[2]**2 + x[3]**2
    u = phi_lambda(x[0], x[1], z)
    Ex, Ey, Ez = E_lambda(x[0], x[1], z)

    dx = x[2]
    dy = x[3]
    ddx = (-0.5 / u) * v * (Ex - x[2]*Ez)
    ddy = (-0.5 / u) * v * (Ey - x[3]*Ez)
    dopl = (u/u0) ** (1/2) * (v) ** (1/2)

    return np.array([dx, dy, ddx, ddy, dopl])


