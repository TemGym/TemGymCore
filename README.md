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
```

Define an input ray

```python
>>> ray_in = Ray(x=0.1, y=0.2, dx=0.3, dy=0.4, z=0.0, pathlength=0.0)
Ray(x=0.1, y=0.2, dx=0.3, dy=0.4, z=0.0, pathlength=0.0)
```

Define a simple model: a lens at `z=0.5`, then a detector at `z=1.0`

```python
lens = Lens(z=0.5, focal_length=1.0)
detector = Detector(z=1.0, pixel_size=(0.01, 0.01), shape=(128, 128))
model = (lens, detector)
```

Run the ray through the model and query the output coordinates:

```python
>>> ray_out = run_to_end(ray_in, model)
>>> print(ray_out)
Ray(x=0.275, y=0.4, dx=0.05, dy=0.0, z=1.0, pathlength=0.89)
```

## What is a ray, a model, and what are components?

- A ray is an set of coordinates and slopes in an optical system
- A model is a sequence of components
- Components are a function which take a ray as input, apply operation, and return a new ray

To create a standard Component we need two parts:

A [Jax dataclass](https://github.com/brentyi/jax_dataclasses) that holds parameters:

```python
@jdc.pytree_dataclass
class Lens(Component):
    z: float
    focal_length: float
```

and a `__call__` method implemented on it to operate on a ray, so that we can write `ray_out = component(ray_in)`:

```python
    def __call__(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)
        one = ray._one * 1.0

        return Ray(
            x=x, y=y, dx=new_dx, dy=new_dy, _one=one, pathlength=pathlength, z=ray.z
        )
```

Functions can apply any operation available to Jax on a ray.

## Closer look at sending a ray through the model

The primary function which propagates rays through a `model`:

``` python 
run_to_end(ray, model)
```

is a convenience function that repeatedly propagates a ray from it's location to the next component in the model
until end. The basic functionality of `run_to_end` is the following:

``` python
    ray = ...
    propagator = FreeSpaceParaxial()
    # step through all components in the model
    for component in model:
        # compute the distance between the current ray and the component
        distance = component.z - ray.z
        if distance != 0.:
            # propagate the ray over the distance
            propagator_d = propagator.with_distance(distance)
            ray = propagator_d(ray)
        # apply the component to the propagated ray
        ray = component(ray)
    return ray
```

## Gradients through the model (w.r.t. parameters)

Write a wrapper to get gradients of a ray through the model with respect to a specific parameter. 

```python
## Gradients of rays through the model with a wrapper + jax.jacobian
def run_with_params(f, z):
    # Rebuild the lens with differentiable parameters and run the model
    lens_local = Lens(z=z, focal_length=f)
    return run_to_end(ray_in, [lens_local, detector])

# Jacobians of the output ray w.r.t. lens parameters f and z
>>> deriv_func = jax.jacobian(run_with_params, argnums=(0, 1))
>>> grads = deriv_func(lens.focal_length, lens.z)

>>> print(grads.x)  # derivative of output x-coordinate wrt. (f, z)
(0.125, 0.09999999)
```

## Gradients w.r.t. the input ray

You can also take derivatives w.r.t. the input ray fields.

```python
# Gradients of a specific coordinate of the output ray w.r.t. a single input ray parameter using jax.grad directly
def run_with_params(x):
    ray_in = Ray(x=x, y=0.2, dx=0.3, dy=0.4, z=0.0, pathlength=0.6)
    return run_to_end(ray_in, model).dx  # return only the output slope in x

>>> d_dx_d_x = jax.grad(run_with_params)(0.01)
>>> print(d_dx_d_x)  # d(output.dx)/d(input.x)
-1.0

# Gradients w.r.t. all of the input ray parameters using jax.jacobian directly
# We can query a specific gradient then from the dataclass
>>> d_out_d_in = jax.jacobian(run_to_end)(ray_in, model)

# Query a specific value in the ray dataclass
>>> print(d_out_d_in.dy.x)  # d(output.dy)/d(input.x)
0.0
```

## Jacobian and ABCD matrices

The structure of the ray object becomes clearer when we introduce the concept of $ABCD$ transfer matrices from optics: 

$ABCD$ transfer matrices are used to describe how an optical component affects the position and slope 
of a light or electron ray which is made up of the following coordinates:

$$
r = 
\begin{bmatrix}
x_{1}\\
\theta_{1}\\
1
\end{bmatrix}
$$

 * $x_{1}$ -  input position
 * $\theta_{1}$ - input slope
 * 1 - an extra variable to carry offsets through the optical system

$ABCD$ matrices act on a ray in the following way: 

$$
\begin{bmatrix}
x_2\\
\theta_{x,2}\\
1
\end{bmatrix}=
\left[
\begin{array}{ccc}
A & B & t\\
C & D & \kappa\\
0 & 0 & 1
\end{array}
\right]
\begin{bmatrix}
x_1\\
\theta_{x,1}\\
1
\end{bmatrix}.
$$

with $t$ and $\kappa$ representing offsets in position and slope in the system. 

A set of chosen $ABCD$ values can be combined to describe how an optical component modifies a ray position or slope.

For instance, to describe how a ray propagating in free space moves a certain distance $z$
we can use the $ABCD$ matrix, with $A=0,\  B=z,\  C=0,\  D=1$

$$
Free \ Space = 
\begin{bmatrix}
1 & z & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

See [here](https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis) for more examples of $ABCD$ matrices.

Using TemGym, one can create a ray and a model, and call ```jax.jacobian``` on the ```run_to_end``` function to calculate the partial derivatives of the output ray with respect to the input ray, which will directly enable us to complete $ABCD$ transfer matrix of the system, and in $2D$ with an offset column, is of shape $5 \times 5$. In particular, $ABCD$ matrices are a helpful tool to propagate rays through any number of components in a model in one step, without the burden of propgating from one component to the next each time. 

```python
from temgym_core.utils import custom_jacobian_matrix

ray_jac = jax.jacobian(run_to_end, argnums=0)(ray_in, model)
ABCD = custom_jacobian_matrix(ray_jac)
print(ABCD)  # 5x5 matrix
>>> [[ 0.5   0.    0.75  0.    0.  ]
 [ 0.    0.5   0.    0.75  0.  ]
 [-1.    0.    0.5   0.    0.  ]
 [ 0.   -1.    0.    0.5   0.  ]
 [ 0.    0.    0.    0.    1.  ]]
```

If needed one can also obtain the ABCD matrices at each propagation/component step:

```python
from temgym_core.run import solve_model

>>> per_step_ABCD = solve_model(ray_in, model)  # shape: (num_steps, 5, 5)
print(per_step_ABCD)

[[[ 1.   0.   0.5  0.   0. ]
  [ 0.   1.   0.   0.5  0. ]
  [ 0.   0.   1.   0.   0. ]
  [ 0.   0.   0.   1.   0. ]
  [ 0.   0.   0.   0.   1. ]]

 [[ 1.   0.   0.   0.   0. ]
  [ 0.   1.   0.   0.   0. ]
  [-1.   0.   1.   0.   0. ]
  [ 0.  -1.   0.   1.   0. ]
  [ 0.   0.   0.   0.   1. ]]

 [[ 1.   0.   0.5  0.   0. ]
  [ 0.   1.   0.   0.5  0. ]
  [ 0.   0.   1.   0.   0. ]
  [ 0.   0.   0.   1.   0. ]
  [ 0.   0.   0.   0.   1. ]]

 [[ 1.   0.   0.   0.   0. ]
  [ 0.   1.   0.   0.   0. ]
  [ 0.   0.   1.   0.   0. ]
  [ 0.   0.   0.   1.   0. ]
  [ 0.   0.   0.   0.   1. ]]]
```


## Multiple rays from a source

`Source` classes generate many rays at once;

```python
from temgym_core.source import PointSource

>>> src = PointSource(z=0.0, semi_conv=0.01)
>>> rays = src.make_rays(num=256, random=False)  # returns a Ray with vector fields

>>> rays_out = run_to_end(rays, model)
>>> print(rays_out.size)
256
```

## Iterative ray tracing with `run_iter`

The `run_iter` function allows you to trace rays step-by-step through the model, returning intermediate results at each component. This is useful for analyzing individually the behavior of rays at each stage of the optical system.

```python
from temgym_core.run import run_iter

# Run the ray iteratively through the model
for step_idx, (component, ray) in enumerate(run_iter(ray_in, model)):
    print(f"Step {step_idx}: {ray}")
```
