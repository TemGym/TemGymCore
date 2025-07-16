from .ray import propagate
import jax
import jax.numpy as jnp
import jaxgym.components as comp
from .utils import custom_jacobian_matrix


def run_to_end(ray, components):
    for component in components:
        # if the component is an ODE component, then just run the step
        # function of the component, otherwise run the propagation function first
        if isinstance(component, comp.ODE):
            ray = component.step(ray)
        else:
            distance = (component.z - ray.z).squeeze()
            ray = propagate(distance, ray)
            ray = component.step(ray)

    return ray


def run_to_end_with_history(ray, components):
    rays = [ray]
    for component in components:
        if isinstance(component, comp.ODE):
            ray = component.step(ray)
        else:
            distance = (component.z - ray.z).squeeze()
            ray = propagate(distance, ray)
            ray = component.step(ray)
        rays.append(ray)
    return rays


def run_to_component(ray, component):
    distance = (component.z - ray.z).squeeze()
    ray = propagate(distance, ray)
    ray = component.step(ray)
    return ray


def calculate_derivatives(ray, model, order):
    derivs = []
    current_func = run_to_end
    for _ in range(order):
        current_func = jax.jacfwd(current_func, argnums=0)
        deriv_val = current_func(ray, model)
        derivs.append(deriv_val)
    return derivs


@jax.jit
def solve_model(ray, model):
    model_ray_jacobians = []

    # Run the step function of the first component at the starting plane
    component_jacobian = jax.jacobian(model[0].step)(ray)
    component_jacobian = custom_jacobian_matrix(component_jacobian)

    model_ray_jacobians.append(component_jacobian)

    for i in range(1, len(model)):
        distance = (model[i].z - ray.z).squeeze()

        """
        This block of code here is akin to calling value_and_grad on a function in jax
        to get it's gradients, except we are doing instead value_and_jacobian to get the
        ray transfer matrix.
        First we call the jacobian on the propagation step to get the transfer matrix. This
        does not actually propagate the ray, it only gets its derivatives, so to get the
        rays "value" from the
        propagation function, we then call the propagate step on the ray without calling
        the jacobian.
        Thus once the same function has been called with and without the jacobian, we have
        calculated the "value_and_jacobian" of the ray.
        """
        # Get the jacobian of the ray propagation
        # from the previous component to the current component
        propagate_jacobian = jax.jacobian(propagate, argnums=1)(distance, ray)
        propagate_jacobian = custom_jacobian_matrix(propagate_jacobian)
        model_ray_jacobians.append(propagate_jacobian)

        # Propagate the ray
        ray = propagate(distance, ray)

        # Get the jacobian of the step function of the current component
        component_jacobian = jax.jacobian(model[i].step)(ray)
        component_jacobian = custom_jacobian_matrix(component_jacobian)

        model_ray_jacobians.append(component_jacobian)

        # Step the ray
        ray = model[i].step(ray)

    ABCDs = jnp.array(model_ray_jacobians)  # ABCD matrices at each component

    return ABCDs


@jax.jit
def get_z_vals(ray, model):
    z_vals = [ray.z]    
    for i in range(1, len(model)):
        distance = (model[i].z - ray.z).squeeze()
        # Propagate the ray
        ray = propagate(distance, ray)
        # Step the ray
        z_vals.append(ray.z)
        ray = model[i].step(ray)
        z_vals.append(ray.z)
    return jnp.array(z_vals)
