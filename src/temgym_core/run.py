from itertools import chain
import dataclasses
from typing import TYPE_CHECKING, Sequence, Union, Any, Callable, Generator

import jax
import jax.numpy as jnp
from .utils import custom_jacobian_matrix
from .propagator import FreeSpaceParaxial, BasePropagator, Propagator
from .components import Component
from .source import Source
from .ray import Ray

if TYPE_CHECKING:
    from .tree_utils import PathBuilder


def jacobian_and_value(fn, argnums: int = 0, **jac_kwargs):

    def inner(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out, out

    return jax.jacobian(inner, argnums=argnums, has_aux=True, **jac_kwargs)


def passthrough_transform(component: Union[Component, Source]):

    def inner(ray: Ray) -> tuple[Ray, Ray]:
        out = component(ray)
        return out, out

    return inner


def jacobian_transform(component: Union[Component, Source]):

    def inner(ray: Ray) -> tuple[Ray, Any]:
        jac, out = jacobian_and_value(component)(ray)
        return out, jac

    return inner


TransformT = Callable[[Union[Component, Source]], Callable[[Ray], tuple[Ray, Any]]]


def run_iter(
    ray: Ray,
    components: Sequence[Union[Component, Source]],
    transform: TransformT = passthrough_transform,
    propagator: BasePropagator = FreeSpaceParaxial(),
) -> Generator[tuple[Propagator | Source | Component, Ray], Any, None]:
    for component in components:
        if isinstance(component, (Source, Component)):
            distance = component.z - ray.z
            if distance != 0.:
                propagator_d = propagator.with_distance(distance)
                ray, out = transform(propagator_d)(ray)
                yield propagator_d, out
        ray, out = transform(component)(ray)
        yield component, out


def run_to_end(
    ray: Ray,
    components: Sequence[Union[Component, Source]],
    propagator: BasePropagator = FreeSpaceParaxial(),
) -> Ray:
    for _, ray in run_iter(ray, components, propagator=propagator):
        pass
    return ray


def calculate_derivatives(ray: Ray, model: Sequence[Union[Component, Source]], order: int):
    derivs = []
    current_func = run_to_end
    for _ in range(order):
        current_func = jax.jacfwd(current_func, argnums=0)
        deriv_val = current_func(ray, model)
        derivs.append(deriv_val)
    return derivs


def solve_model(
    ray: Ray,
    model: Sequence[Union[Component, Source]],
    propagator: BasePropagator = FreeSpaceParaxial(),
):
    model_ray_jacobians = []
    for _, jac in run_iter(ray, model, transform=jacobian_transform, propagator=propagator):
        jac = custom_jacobian_matrix(jac)
        model_ray_jacobians.append(jac)
    return jnp.array(model_ray_jacobians)  # ABCD matrices at each component


def run_with_grads(
    input_ray: Ray,
    model: Sequence[Union[Component, Source]],
    grad_vars: Sequence["PathBuilder"],
) -> tuple[Ray, dict[Sequence[Any], Ray]]:
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
