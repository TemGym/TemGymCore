# TemGymCore

A ray tracing package that uses the automatic differentiation tools of jax to solve optical systems
via a Taylor Expansion of a "ray" representing the optical axis. 
In TemGym, Linear optical systems are represented via the ABCD values of a ray 
transfer matrix determined using the Jacobian of the ray coordinates through the optical system, and non-linear optical systems are represented via 
coefficients of a Taylor Expansion polynomial of the ray coordinates through the optical system, calculated either via repeated Jacobian calls, 
or via the experimental library jax.jet.

The returned polynomial equations representing an optical system can then be solved to determine output ray positions, 
slopes, amplitudes and phases, enabling one to propagate input wavefronts through linear or non-linear optical systems. 

The specific use case implemented in this library is designed to solve a linear system representing the coordinate transformation of a defocused point source on 
a sample, creating a "shadow image" of the sample on the detector. Utilising the code in this repository, and a 4D STEM dataset of shadow images, we can solve the linear system of the shadow image projection, and by iteratively backprojecting each shadow image via ray tracing, can verify whether the coordinate system, and parameters such as scan step, camera length, scan rotation used in the 4D STEM experiment are correct. Such a verification step is neccessary in order to reliably apply iterative phase retrieval algorithms such as Ptychography to a 4D STEM experiment. 

The location of images on the detector in 4D STEM experiments can also suffer from a systematic error in the Scan/Descan system, which is used to raster the beam over the sample, and return it to the optical axis such that the beam is viewed on the centre of the detector. When shadow images are not returned to the centre of the detector, the STEM experiment suffers from Descan Error which must be corrected for before the coordinate system can be determined. We show how one can use a ray transfer matrix to represent Descan Error in the system, and how to fit it such that it can be corrected reliably for a range of camera lengths.

This work is a continuation of a non-differentiable ray tracing library designed for visualisation published in 2023
TemGym: Landers, D., Clancy, I., Weber, D., Dunin-Borkowski, R. & Stewart, A. (2023). J. Appl. Cryst. 56, https://doi.org/10.1107/S1600576723005174

**Usage**

## Quick start

```python
import jax
from temgym_core.ray import Ray
from temgym_core.components import Lens, Detector
from temgym_core.run import run_to_end

# Define an input ray
ray_in = Ray(x=0.1, y=0.2, dx=0.3, dy=0.4, z=0.0, pathlength=0.0, _one=1.0)

# Define a simple model: a lens at z=0.5, then a detector at z=1.0
lens = Lens(z=0.5, focal_length=1.0)
detector = Detector(z=1.0, pixel_size=(0.01, 0.01), shape=(128, 128))
model = [lens, detector]

# Run the ray through the model and query the output coordinates
ray_out = run_to_end(ray_in, model)
print(ray_out.x, ray_out.y, ray_out.dx, ray_out.dy, ray_out.z)
```

## Gradients through the model (w.r.t. parameters)

Use `run_with_grads` to get the output ray and the derivatives of the output w.r.t. selected parameters. Keys in the returned dict are tuples like `(component, 'parameter')` and map to a `Ray` of partial derivatives.

```python
from temgym_core.run import run_with_grads

# Choose which parameters to differentiate with respect to
grad_vars = (lens.params.z, lens.params.focal_length, )

ray_out, d_ray_out_d_params = run_with_grads(ray_in, model, grad_vars)

# Access derivatives using tuple keys
d_out__d_f = d_ray_out_d_params[(lens, 'focal_length')]  # a Ray of d(output)/d(focal_length)
d_out__d_z = d_ray_out_d_params[(lens, 'z')]

# Example: derivative of output x w.r.t lens focal_length
print(d_out__d_f.x)
```


## Gradients w.r.t. the input ray

You can also request derivatives w.r.t. the input ray fields.

```python
# Single input field
_, grads_in = run_with_grads(ray_in, model, grad_vars=(ray_in.params.x,))
print(grads_in[(ray_in, 'x')].x)  # d(output.x)/d(input.x)

# Or all input fields at once by passing the ray itself
_, grads_all_in = run_with_grads(ray_in, model, grad_vars=(ray_in,))
print(grads_all_in[(ray_in, 'dx')].y)  # d(output.y)/d(input.dx)
```


## Jacobian and ABCD matrix

The full Jacobian of `run_to_end` w.r.t. the input ray gives the 5×5 ray-transfer (ABCD) form when formatted. You can compute it directly with JAX:

```python
from temgym_core.utils import custom_jacobian_matrix

ray_jac = jax.jacobian(run_to_end, argnums=0)(ray_in, model)
ABCD = custom_jacobian_matrix(ray_jac)
print(ABCD)  # 5x5 matrix
```

Or get the ABCD matrices at each propagation/component step:

```python
from temgym_core.run import solve_model

per_step_ABCD = solve_model(ray_in, model)  # shape: (num_steps, 5, 5)
```


## Multiple rays from a source

`Source` classes generate many rays at once;

```python
from temgym_core.source import PointSource

src = PointSource(z=0.0, semi_conv=0.01)
rays = src.make_rays(num=256, random=False)  # returns a Ray with vector fields

rays_out = run_to_end(rays, model)
print(rays_out.x.shape)  # (256,) for this example
```

## Iterative ray tracing with `run_iter`

The `run_iter` function allows you to trace rays step-by-step through the model, returning intermediate results at each component. This is useful for analyzing the behavior of rays at each stage of the optical system.

```python
from temgym_core.run import run_iter

# Run the ray iteratively through the model
for step, (component, ray) in enumerate(run_iter(ray_in, model)):
    print(f"Step {step}: x={ray.x}, y={ray.y}, dx={ray.dx}, dy={ray.dy}, z={ray.z}")

```
