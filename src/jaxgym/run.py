
from .ray import propagate
import jax
import jax.numpy as jnp

def run_to_end(ray, components):
    for component in components:
        distance = (component.z - ray.z).squeeze()
        ray = propagate(distance, ray)
        ray = component.step(ray)

    return ray

def run_to_component(ray, component):
    distance = (component.z - ray.z).squeeze()
    ray = propagate(distance, ray)
    ray = component.step(ray)
    return ray

def solve_model(ray, model):

    model_ray_jacobians = []

    #Run the step function of the first component at the starting plane
    component_jacobian = jax.jacobian(model[0].step)(ray)
    model_ray_jacobians.append(component_jacobian)

    for i in range(1, len(model)):
        distance = (model[i].z - ray.z).squeeze()

        '''
        This block of code here is akin to calling value_and_grad on a function in jax
        to get it's gradients, except we are doing instead value_and_jacobian to get the ray transfer matrix. 
        First we call the jacobian on the propagation step to get the transfer matrix. This
        does not actually propagate the ray, it only gets its derivatives, so to get the rays "value" from the 
        propagation function, we then call the propagate step on the ray without calling the jacobian. 
        Thus once the same function has been called with and without the jacobian, we have calculated the "value_and_jacobian" of the ray.
        '''
        # Get the jacobian of the ray propagation
        # from the previous component to the current component
        propagate_jacobian = jax.jacobian(propagate, argnums=1)(distance, ray)
        model_ray_jacobians.append(propagate_jacobian)

        # Propagate the ray
        ray = propagate(distance, ray)

        # Get the jacobian of the step function of the current component
        component_jacobian = jax.jacobian(model[i].step)(ray)
        model_ray_jacobians.append(component_jacobian)

        #Step the ray
        ray = model[i].step(ray)

    # Edit the jacobian matrices to include shifts calculated 
    # from the optical path length derivative - not the best solution for now but it works.
    ABCDs = [] #ABCD matrices at each component

    for ray_jacobian in model_ray_jacobians:
        shift_vector = ray_jacobian.pathlength.matrix # This is the shift vector for each ray, dopl_out/dr_in
        ABCD = ray_jacobian.matrix.matrix # This is the ABCD matrix for each ray, dr_out/dr_in
        ABCD = ABCD.at[:, -1].set(shift_vector[0, :])
        ABCD = ABCD.at[-1, -1].set(1.0) # Add the final one to bottom right corner of the matrix.
        ABCDs.append(ABCD)

    return jnp.array(ABCDs)
