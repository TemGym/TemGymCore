from itertools import chain
import dataclasses
from typing import TYPE_CHECKING, Sequence

import jax
import jax.numpy as jnp
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
        #     ray = component(ray)
        # else:
        distance = component.z - ray.z
        ray = propagate(distance, ray)
        ray = component(ray)
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
    component_jacobian = jax.jacobian(model[0])(ray)
    component_jacobian = custom_jacobian_matrix(component_jacobian)

    model_ray_jacobians.append(component_jacobian)

    def prop_aux(distance, ray):
        prop_ray = propagate(distance, ray)
        return prop_ray, prop_ray

    prop_jac_val_fn = jax.jacobian(prop_aux, argnums=1, has_aux=True)

    for i in range(1, len(model)):
        distance = (model[i].z - ray.z).squeeze()

        # Get the jacobian of the ray propagation
        # from the previous component to the current component
        propagate_jacobian, ray = prop_jac_val_fn(distance, ray)
        propagate_jacobian = custom_jacobian_matrix(propagate_jacobian)
        model_ray_jacobians.append(propagate_jacobian)

        # Get the jacobian of the step function of the current component
        def _step_val(ray):
            step_ray = model[i](ray)
            return step_ray, step_ray

        component_jacobian, ray = jax.jacobian(_step_val, has_aux=True)(ray)
        component_jacobian = custom_jacobian_matrix(component_jacobian)
        model_ray_jacobians.append(component_jacobian)

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
        ray = model[i](ray)
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
        num_vars = len(grad_idxs) + len(grad_ray_idxs)
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
        if (len(grad_idxs) + len(grad_ray_idxs)) == num_vars:
            raise RuntimeError(f"Cannot find {var._build(original=True)} in parameters")

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
