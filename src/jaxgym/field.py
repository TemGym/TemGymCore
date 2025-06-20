import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy import interpolate
import jax

from sympy.printing.numpy import NumPyPrinter, S


class CustomNumpyPrinter(NumPyPrinter):
    def _print_Piecewise(self, expr):
        "Piecewise function printer"
        from sympy.logic.boolalg import ITE, simplify_logic

        def print_cond(cond):
            """Problem having an ITE in the cond."""
            if cond.has(ITE):
                return self._print(simplify_logic(cond))
            else:
                return self._print(cond)

        ### Modifications
        exprs = [
            f"atleast_1d(asarray({self._print(arg.expr)})).astype(jnp.float64)"
            for arg in expr.args
        ]
        exprs = "({})".format(",".join(exprs))
        conds = [f"atleast_1d(asarray({print_cond(arg.cond)}))" for arg in expr.args]
        conds = "({})".format(",".join(conds))
        ###
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in expr.args.
        # If this is not the case, it may be triggered prematurely.
        return "{}({}, {}, default={})".format(
            self._module_format(self._module + ".select"),
            conds,
            exprs,
            self._print(S.NaN),
        )


def schiske_lens_expansion_xyz(X, Y, Z, phi_0, a, k):
    # Define Schiske's Electrostatic Field. See Principle of Electron Optics 2017 edition, (Ctrl-F Schiske) volume 2 Applied Geometric Optics for more details.
    phi = phi_0 - (phi_0 * ((k**2) / (1 + (Z / a) ** 2)))

    # Differentiate it so we can find the expansion around the centre of the lens.
    phi_prime = phi.diff(Z, 1)
    phi_double_prime = phi.diff(Z, 2)
    phi_quadruple_prime = phi.diff(Z, 4)
    phi_sextuple_prime = phi.diff(Z, 6)

    # Expand round lens field to include 5th order spherical aberrations
    phi_expansion_symbolic = (
        phi
        - ((X**2 + Y**2) / 4) * phi_double_prime
        + (((X**2 + Y**2) ** 2) / 64) * phi_quadruple_prime
        - (((X**2 + Y**2) ** 3) / 2304) * phi_sextuple_prime
    )

    # Define the Efield
    Ex = -1 * phi_expansion_symbolic.diff(X)
    Ey = -1 * phi_expansion_symbolic.diff(Y)
    Ez = -1 * phi_expansion_symbolic.diff(Z)

    # Lambdify the field and potential equations so we can call them for various experiments
    E_lambda = sp.lambdify([X, Y, Z], (Ex, Ey, Ez), "numpy")
    phi_lambda = sp.lambdify([X, Y, Z], phi_expansion_symbolic, "numpy")

    phi_lambda_axial = sp.lambdify([Z], phi, "numpy")
    phi_lambda_prime = sp.lambdify([Z], phi_prime)
    phi_lambda_double_prime = sp.lambdify([Z], phi_double_prime)
    phi_lambda_quadrupole_prime = sp.lambdify([Z], phi_quadruple_prime)
    phi_lambda_sextuple_prime = sp.lambdify([Z], phi_sextuple_prime)

    return (
        phi_expansion_symbolic,
        E_lambda,
        phi_lambda,
        phi_lambda_axial,
        phi_lambda_prime,
        phi_lambda_double_prime,
        phi_lambda_quadrupole_prime,
        phi_lambda_sextuple_prime,
    )


def obtain_first_order_electrostatic_lens_properties(
    z_init,
    phi_lambda_axial,
    phi_lambda_,
    phi_lambda__,
    z_sampling,
    z_final=10 * 1e7,
    a0=0,
    a_0=1.0,
    b0=1.0,
    b_0=0.0,
):
    def hit_optic_axis(z, x, phi_lambda_axial, phi_lambda_, phi_lambda__):
        return x[0]

    hit_optic_axis.terminal = True
    hit_optic_axis.direction = -1

    def hit_focal_plane(z, x, phi_lambda_axial, phi_lambda_, phi_lambda__):
        return x[0]

    params = {
        "max_step": np.inf,
        "rtol": 1e-13,
        "atol": 1e-15,
    }

    # Solve first order equation of motion for rays to obtain magnification, focal plane, and gaussian image plane.
    sol_h = solve_ivp(
        first_order_electrostatic_lens_equation_of_motion,
        (z_init, z_final),
        np.array([a0, a_0]),
        method="LSODA",
        **params,
        events=hit_optic_axis,
        args=(phi_lambda_axial, phi_lambda_, phi_lambda__),
        dense_output=True,
    )

    z_h_ray = sol_h.t
    h, h_ = sol_h.y[0], sol_h.y[1]

    z_image = z_h_ray[-1]

    # Solve first order equation of motion for rays to obtain magnification, focal plane, and gaussian image plane.
    sol_g = solve_ivp(
        first_order_electrostatic_lens_equation_of_motion,
        (z_init, z_image),
        np.array([b0, b_0]),
        method="LSODA",
        **params,
        events=(hit_focal_plane),
        args=(phi_lambda_axial, phi_lambda_, phi_lambda__),
        dense_output=True,
    )

    mag_real = sol_g.y[0][-1]

    z_g_ray = sol_g.t
    g, g_ = sol_g.y[0], sol_g.y[1]

    # Interpolate them so we can find values easier in their path
    fg = interpolate.CubicSpline(z_g_ray, g)
    fg_ = interpolate.CubicSpline(z_g_ray, g_)
    fh = interpolate.CubicSpline(z_h_ray, h)
    fh_ = interpolate.CubicSpline(z_h_ray, h_)

    z_pos = np.linspace(z_init, z_image, z_sampling, endpoint=True)

    g, g_, h, h_ = fg(z_pos), fg_(z_pos), fh(z_pos), fh_(z_pos)

    # Obtain the principle ray coordinates
    principal_gray_slope = g_[-1]
    principal_gray_intercept = g[-1] - g_[-1] * z_pos[-1]

    z_f = -principal_gray_intercept / principal_gray_slope

    z_p = (1 - principal_gray_intercept) / principal_gray_slope

    f = z_f - z_p

    return z_pos, g, g_, h, h_, mag_real, z_image, f, z_f, z_p


def first_order_electrostatic_lens_equation_of_motion(z, x, U, U_, U__):
    # Create first order linear lens equation:
    return np.array(
        [x[1], (-1 * (U_(z)) / (2 * U(z)) * x[1] - U__(z) / (4 * (U(z))) * x[0])]
    )


def make_potential_and_efield_non_rel(phi, x, y, z):
    phi_hat = phi

    # Get E field function. Notice there is no negative sign, as the way hawkes uses phi_hat, there is no minus before it (Equation 3.22).
    dphi_hat_wires_electron_dx = -phi_hat.diff(x)
    dphi_hat_wires_electron_dy = -phi_hat.diff(y)
    dphi_hat_wires_electron_dz = -phi_hat.diff(z)

    phi_hat_wires_electron_lambda = sp.lambdify([x, y, z], phi_hat, "jax")
    phi_hat_wires_electron_lambda_nb = jax.jit(phi_hat_wires_electron_lambda)

    dphi_hat_wires_electron_lambda = sp.lambdify(
        [x, y, z],
        [
            dphi_hat_wires_electron_dx,
            dphi_hat_wires_electron_dy,
            dphi_hat_wires_electron_dz,
        ],
        "jax",
    )
    dphi_hat_wires_electron_lambda_nb = jax.jit(dphi_hat_wires_electron_lambda)

    return phi_hat_wires_electron_lambda_nb, dphi_hat_wires_electron_lambda_nb
