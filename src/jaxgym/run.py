from itertools import chain
import dataclasses
from typing import TYPE_CHECKING, Sequence

import jax
import jax.numpy as jnp
import jaxgym.components as comp
from .utils import custom_jacobian_matrix
from .ray import propagate

if TYPE_CHECKING:
    from .tree_utils import PathBuilder
    from .ray import Ray


def run_to_end(ray: "Ray", components):
    for component in components:
        # # if the component is an ODE component, then just run the step
        # # function of the component, otherwise run the propagation function first
        # if isinstance(component, comp.ODE):
        #     ray = component.step(ray)
        # else:
        distance = component.z - ray.z
        ray = propagate(distance, ray)
        ray = component.step(ray)
    return ray


def run_to_end_with_history(ray: "Ray", components):
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


def run_to_component(ray: "Ray", component):
    distance = (component.z - ray.z).squeeze()
    ray = propagate(distance, ray)
    ray = component.step(ray)
    return ray


def calculate_derivatives(ray: "Ray", model, order):
    derivs = []
    current_func = run_to_end
    for _ in range(order):
        current_func = jax.jacfwd(current_func, argnums=0)
        deriv_val = current_func(ray, model)
        derivs.append(deriv_val)
    return derivs


@jax.jit
def solve_model(ray: "Ray", model):
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
def get_z_vals(ray: "Ray", model):
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


def run_with_grads(
    input_ray: "Ray", model: Sequence, grad_vars: Sequence["PathBuilder"]
):
    ray_params, ray_tree = jax.tree.flatten(input_ray)
    model_params, model_tree = jax.tree.flatten(model)

    grad_idxs = {}
    grad_ray_idxs = {}
    for var in grad_vars:
        if var is input_ray:
            for field in dataclasses.fields(var):
                builder = getattr(var.params, field.name)
                idx = builder._build()[-1].key
                path = builder._build(original=True)
                grad_ray_idxs[path] = idx
        elif var._resolve_root() is input_ray:
            idx = var._build()[-1].key
            path = var._build(original=True)
            grad_ray_idxs[path] = idx
        else:
            grad_idxs.update(var._find_in(model))

    def run_wrap(num_ray_params: int, *grad_params):
        # build the input ray from the params
        grad_iter = iter(grad_params[:num_ray_params])
        grad_ray_params = [
            p if ix not in grad_ray_idxs.values() else next(grad_iter)
            for ix, p in enumerate(ray_params)
        ]
        input_ray = jax.tree.unflatten(ray_tree, grad_ray_params)
        # build the model from the params
        grad_iter = iter(grad_params[num_ray_params:])
        grad_model_params = [
            p if ix not in grad_idxs.values() else next(grad_iter)
            for ix, p in enumerate(model_params)
        ]
        grad_model = jax.tree.unflatten(model_tree, grad_model_params)
        out = run_to_end(input_ray, grad_model)
        return out, out  # double return lets us do jacobian_and_value via has_aux=True

    jac_fn = jax.jacobian(
        run_wrap,
        argnums=tuple(range(1, len(grad_ray_idxs) + len(grad_idxs) + 1)),
        has_aux=True,  # return will be (jac, value)
    )
    grad_ray_params = list(ray_params[idx] for idx in grad_ray_idxs.values())
    grad_model_params = list(model_params[idx] for idx in grad_idxs.values())
    grads, value = jac_fn(len(grad_ray_params), *(grad_ray_params + grad_model_params))
    grads = {
        k: grads[i] for i, k in enumerate(chain(grad_ray_idxs.keys(), grad_idxs.keys()))
    }
    return value, grads
