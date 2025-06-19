import jax.numpy as jnp
import numpy as np
import sympy as sp
import os
import jax

from scipy.constants import h as h_planck, e, m_e
from daceypy import array, DA
from scipy.integrate import simpson

from jaxgym.field import schiske_lens_expansion_xyz, obtain_first_order_electrostatic_lens_properties
from jaxgym.ode import odedopri,  electron_equation_of_motion_DA
import jaxgym.components as comp
from jaxgym.taylor import poly_dict, order_indices, poly_dict_to_sympy_expr
from jaxgym.run import run_to_end, calculate_derivatives
from jaxgym.ray import Ray


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def test_ray_amplitude_pt_source_free_space():

    z_init = jnp.array(0.0)
    z_image = jnp.array(10.0)

    image_array = jnp.zeros((11, 11), jnp.complex128)
    image_array = image_array.at[5, 5].set(1.0+0.0j)

    wavelength = 1e-1
    wavenumber = 2 * jnp.pi / wavelength

    PointSourcePlane = comp.ImageGrid(z=z_init, image_array=image_array, image_pixel_size=(1e-8, 1e-8), image_shape=(10, 10), image_rotation=0.0)
    Detector = comp.Detector(z=z_image, det_pixel_size=(5e-1, 5e-1), det_shape = (257, 257))
    model = [PointSourcePlane, Detector]

    ray = Ray(0., 0., 0., 0., 0., z_init, 0.0)
    ray_out = run_to_end(ray, model)

    dray_out_dray_in = jax.jacobian(run_to_end, argnums=0)(ray, model)


def test_aberrations_schiske_electrostatic_lens():

    ### ELECTROSTATIC LENS SETUP ###
    X, Y, Z = sp.symbols('X Y Z')

    z_init = -0.02  # Convert m to um units
    a = 0.0004  # Convert m to um units
    phi_0 = 1000  # Volts

    k = 0.4**(1/2)  # Unitless

    (
        phi_expansion_symbolic,
        E_lambda, phi_lambda,
        phi_lambda_axial,
        phi_lambda_prime,
        phi_lambda_double_prime,
        phi_lambda_quadruple_prime,
        phi_lambda_sextuple_prime
    ) = schiske_lens_expansion_xyz(X, Y, Z, phi_0, a, k)

    wavelength = h_planck/(2*abs(e)*m_e*phi_lambda_axial(z_init))**(1/2)


    z_pos, g, g_, h, h_, mag_real, z_image, z_focal_real, z_focal_asymp, z_pi = obtain_first_order_electrostatic_lens_properties(
        z_init, phi_lambda_axial, phi_lambda_prime, phi_lambda_double_prime, z_sampling=1000)


    ### CS FROM ABERRATION INTEGRAL METHOD ###
    Uz0 = phi_lambda_axial(z_pos[0])

    U_val = phi_lambda_axial(z_pos)
    U_val__ = phi_lambda_double_prime(z_pos)
    U_val____ = phi_lambda_quadruple_prime(z_pos)

    # Weird Aberration integral formulas for simpsons rule
    def L_1():
        return (1/(32*jnp.sqrt(U_val)))*((U_val__**2)/(U_val)-U_val____)

    def L_2():
        return (1/(8*jnp.sqrt(U_val)))*(U_val__)

    def L_3():
        return 1/2*(jnp.sqrt(U_val))

    def F_020():
        return (L_1()/4)*h*h*h*h + (L_2()/2)*h*h*h_*h_ + (L_3()/4)*h_*h_*h_*h_

    # B aberration coefficient, needed for Cs
    B = simpson(F_020(), x=z_pos)
    Cs_aberint = 4 / jnp.sqrt(abs(Uz0)) * B * mag_real

    ### CS FROM DACEPY METHOD ###
    order = 4

    DA.init(order, 5)

    x0 = 0.
    y0 = 0.
    x0_slope = 0.0
    y0_slope = 0.0
    opl = 0.

    z_init_nondm = z_init
    z_image_nondm = z_image
    
    u0 = phi_lambda_axial(z_init)  # initial potential
    x = array([x0 + DA(1), y0 + DA(2), x0_slope + DA(3), y0_slope + DA(4), opl + DA(5)])

    # solve the equation of motion via the differential algebra method, which delivers the solution as a taylor expansion
    with DA.cache_manager():
        zf, x_f = odedopri(electron_equation_of_motion_DA, 
                        x0 = z_init_nondm,
                        y0 = x, 
                        x1 = z_image_nondm,  
                        tol = 1e-1, 
                        hmax = 10000, 
                        hmin = 1e-15,  
                        maxiter =int(1e5), 
                        args=(phi_lambda, E_lambda, u0))
        
    Cs_daceypy = x_f[0].getCoefficient([0, 0, 3, 0])
    Cs_daceypy_opl_polynomial = x_f[4].getCoefficient([0, 0, 4, 0, ]) * 4 / 3 * mag_real

    ### CS FROM JAX GRADIENT METHOD ###
    z_init = jnp.array(z_init)
    z_image = jnp.array(z_image)

    PointSource = comp.InputPlane(z=z_init)
    ElectrostaticLens = comp.ODE(z=z_init, z_end=z_image, phi_lambda=phi_lambda, E_lambda=E_lambda)
    Detector = comp.Detector(z=z_image, det_pixel_size=(5e-9, 5e-9), det_shape = (128, 128))
    model = [PointSource, ElectrostaticLens, Detector]

    ray = Ray(0., 0., 0., 0., 0., z_init, 0.0)

    derivatives = calculate_derivatives(ray, model, order)

    selected_vars = ['x', 'y', 'dx', 'dy', 'pathlength']
    multi_indices = order_indices(order, n_vars=len(selected_vars))
    poly_dicts = poly_dict(derivatives, selected_vars, multi_indices[1:])
    x_var, y_var, dx_var, dy_var, opl_var = sp.symbols("x y x' y' S", real=True)
    polynomials = poly_dict_to_sympy_expr(poly_dicts, selected_vars, sym_vars=[x_var, y_var, dx_var, dy_var, opl_var])

    Cs_jax = polynomials['x'].coeff(dx_var**3)
    Cs_jax_opl_polynomial = polynomials['pathlength'].coeff(dx_var**4)* 4 / 3 * mag_real

    np.testing.assert_allclose(float(Cs_aberint), float(Cs_daceypy), rtol=1e-5)
    np.testing.assert_allclose(float(Cs_aberint), float(Cs_jax), rtol=1e-5)

    ''' Print the coefficients of the polynomials in a readable format
    # var = 'pathlength'

    # if var == 'x':
    #     var_idx = 0
    # elif var == 'y':
    #     var_idx = 1
    # elif var == 'dx':
    #     var_idx = 2
    # elif var == 'dy':
    #     var_idx = 3
    # elif var == 'pathlength':
    #     var_idx = 4
    # else:
    #     raise ValueError(f"Unknown variable: {var}")

    # header = f"{'I':>6}  {'COEFFICIENT':>3}   {'ORDER':>16} {'EXPONENTS':>4}"
    # const = f"{1:6d}   {getattr(ray_out, var): .16e}   {0} {'  0  0  0  0  0':15s}"

    # print_jax = [header, const]
    # for idx, entry in enumerate(poly_dicts[var]):
    #     exponents = tuple(map(int, entry[:-1]))
    #     coeff = entry[-1]
    #     total_order = sum(exponents)
    #     exponents_str = " ".join(f"{e:2d}" for e in exponents)
    #     print_jax.append(f"{idx:6d}   {coeff: .16e}   {total_order}  {exponents_str}")
    # print_jax.append('------------------------------------------------')


    # header = f"{'I':>6}  {'COEFFICIENT':>3}   {'ORDER':>16} {'EXPONENTS':>4}"
    # const = f"{1:6d}   {x_f[var_idx].getCoefficient([0, 0]): .16e}   {0} {'  0  0  0  0  0':15s}"

    # print_daceypy = [header, const]

    # for idx, entry in enumerate(poly_dicts[var]):
    #     exponents = tuple(map(int, entry[:-1]))
    #     coeff = x_f[var_idx].getCoefficient(list(exponents))
    #     total_order = sum(exponents)
    #     exponents_str = " ".join(f"{e:2d}" for e in exponents)
    #     print_daceypy.append(f"{idx:6d}   {coeff: .16e}   {total_order}  {exponents_str}")
    # print_daceypy.append('------------------------------------------------')

    # # Print the two blocks side by side
    # for left, right in zip(print_jax, print_daceypy):
    #     print(f"{left:<60} {right}")
    '''
