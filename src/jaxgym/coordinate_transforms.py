import jax.numpy as jnp
from . import Degrees, Radians
RadiansJNP = jnp.float64


def _rotate_with_deg_to_rad(degrees: 'Degrees'):
    # From libertem.corrections.coordinates v0.11.1
    return _rotate(jnp.pi / 180 * degrees)


def _identity():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.eye(3)


def _rotate(radians: 'Radians'):
    # From libertem.corrections.coordinates v0.11.1
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    # This function is written with the y-axis flipped - i.e a pi/2 rotation sends
    # a (1, 0) vector to (0, -1). This is the opposition direction to what one 
    # would expect in a conventional cartesian coordinate system.
    return jnp.array([
        (jnp.cos(radians), jnp.sin(radians), 0.),
        (-jnp.sin(radians), jnp.cos(radians), 0.),
        (0., 0., 1.)
    ])


def _scale(pixel_size_yx):
    return jnp.array([
        (1 / pixel_size_yx[0], 0., 0.),
        (0., 1 / pixel_size_yx[1], 0.),
        (0., 0., 1.)
    ])


def _shift(centre_yx):
    return jnp.array([
        (1., 0., centre_yx[0]),
        (0., 1., centre_yx[1]),
        (0., 0., 1.)
    ])


def _flip_y():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.array([
        (-1., 0., 0.),
        (0., 1., 0.),
        (0., 0., 1.)
    ])


def metres_to_pixels_transform(centre_yx, 
                               pixel_size_yx, 
                               shape_yx, 
                               flip_y=False, 
                               rotation = 0. # Degrees
):
    if flip_y:
        flip_transform = _flip_y()
    else:
        flip_transform = _identity()

    centre_shift_transform = _shift(centre_yx)
    rotation_transform = _rotate_with_deg_to_rad(rotation)
    scale_transform = _scale(pixel_size_yx)
    shape_yx = jnp.array(shape_yx)
    pixel_shift_transform = _shift(shape_yx / 2)

    transform = pixel_shift_transform @ scale_transform @ centre_shift_transform @ rotation_transform @ flip_transform

    return transform


def rotation_shift_transform(centre_yx, 
                             flip_y=False, 
                             rotation = 0. # Degrees
):
    if flip_y:
        flip_transform = _flip_y()
    else:
        flip_transform = _identity()

    rotation_transform = _rotate_with_deg_to_rad(rotation)
    centre_shift_transform = _shift(centre_yx)

    transform = centre_shift_transform @ rotation_transform @ flip_transform

    return transform


def apply_transformation(y, x, transformation):
    # All of our coordinate transforms are 3x3 transformations, 
    # so we need to add an array of 1s to the end of our coordinates array
    r = jnp.stack([y, x, jnp.ones_like(y)], axis=-1)
    r_transformed = transformation @ r.T 
    y_transformed, x_transformed, _ = r_transformed
    return y_transformed, x_transformed