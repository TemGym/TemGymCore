import jax.numpy as jnp
from . import Degrees, Radians, Shape_YX, Coords_XY, Coords_YX, Scale_YX, Pixels_YX
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
        (pixel_size_yx[0], 0., 0.),
        (0., pixel_size_yx[1], 0.),
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


def pixels_to_metres_transform(centre: Coords_XY, 
                               pixel_size: Scale_YX,
                               shape: Shape_YX, 
                               flip_y=False, 
                               rotation: Degrees = 0.0):
    """
    Transforms pixel coordinates into metre coordinates using a series of matrix operations.

    Parameters:
        centre (Coords_XY): The translation vector representing the center coordinate.
        pixel_size (Scale_YX): The scaling factors that convert pixel dimensions to metres.
        shape (Shape_YX): The shape of the pixel grid (e.g., image size) in (height, width) format.
        flip_y (bool, optional): If True, applies a flip along the y-axis to the transformation.
                                    Defaults to False.
        rotation (Degrees, optional): The rotation angle (in degrees) applied to the transformation.
                                        Defaults to 0.0.

    Returns:
        jax.numpy.ndarray: A transformation matrix that converts pixel coordinates to metre coordinates.
        
    The transformation is composed by applying:
        - A conditional flip along the y-axis if flip_y is True.
        - A fixed flip transformation for the y-axis to go from xy of ray space to yx of pixel space.
        - A rotation transformation given by the specified angle.
        - A translation to the provided centre in metres.
        - A scaling transformation with the given pixel size.
        - A shift to center the pixel coordinates based on the shape.
    """

    if flip_y:
        flip_transform = _flip_y()
    else:
        flip_transform = _identity()

    metres_to_px_flip_y = _flip_y()
    shape = jnp.array(shape)
    pixel_shift_transform = _shift(-shape / 2)
    scale_transform = _scale(pixel_size)
    centre_shift_transform = _shift(centre)
    rotation_transform = _rotate_with_deg_to_rad(rotation)

    transform = flip_transform @ metres_to_px_flip_y @ rotation_transform @ centre_shift_transform @ scale_transform @ pixel_shift_transform

    return transform


def _metres_to_pixels(ray_coords: Coords_XY, 
                     centre: Coords_XY, 
                     step: Scale_YX, 
                     shape: Shape_YX,
                     rotation: Degrees) -> Pixels_YX: 

    """
    Convert a coordinate point specified in metres to its corresponding
    pixel indices. It does so by first obtaining a transformation matrix (from pixels to metres)
    using the provided centre, scale (step), shape, and rotation parameters, then computing its
    inverse to map metre coordinates back to pixel space. The resulting pixel values are rounded 
    to the nearest integers.
    Parameters:
        ray_coords (Coords_XY - tuple of float): A tuple (x, y) representing the coordinate in metres
                                        that needs to be converted.
        centre (Coords_XY - tuple of float): A tuple (x, y) representing the centre of the transformation,
                                    typically the midpoint of the pixel space.
        step (Scale_YX - tuple of float): A tuple (y_step, x_step) representing the scale factor along the y and x axes.
        shape (Shape_YX - tuple of int): A tuple (height, width) specifying the overall dimensions of the 
                                pixel space.
        rotation (Degrees - float): The rotation angle in degrees to be applied during the transformation.
    Returns:
        (Pixels_YX - tuple of int): A tuple (pixel_y, pixel_x) representing the converted pixel coordinates.
    """

    ray_coords_x, ray_coords_y = ray_coords

    pixels_to_metres_transformation = pixels_to_metres_transform(centre, 
                                                                    step, 
                                                                    shape, 
                                                                    False, 
                                                                    rotation)
    
    metres_to_pixels_transformation = jnp.linalg.inv(pixels_to_metres_transformation)
    
    ray_pixels_y, ray_pixels_x = apply_transformation(ray_coords_y, ray_coords_x, metres_to_pixels_transformation) 
    
    ray_pixels_y = jnp.round(ray_pixels_y).astype(jnp.int32)       
    ray_pixels_x = jnp.round(ray_pixels_x).astype(jnp.int32)   

    return ray_pixels_y, ray_pixels_x


def _pixels_to_metres(ray_pixels: Pixels_YX, 
                     centre: Coords_XY, 
                     step: Scale_YX, 
                     shape: Shape_YX,
                     rotation: Degrees) -> Coords_XY:
    
    """
    Convert a coordinate point specified in pixels to its corresponding coordinate in metres.
    This function obtains a transformation matrix that maps pixels to metres using the
    provided centre, scale (step), shape, and rotation parameters. It then applies this transformation
    on the input pixel coordinates, converting them into metre coordinates.
    Parameters:
        ray_pixels (Pixels_YX - tuple of float): A tuple (y, x) representing the coordinate in the pixel space
                                        that needs to be converted.
        centre (Coords_XY - tuple of float): A tuple (x, y) representing the centre of the transformation,
                                    typically the midpoint of the pixel space.
        step (Scale_YX - tuple of float): A tuple (y_step, x_step) representing the scale along the
                                y and x axes in metre units.
        shape (Shape_YX - tuple of int): A tuple (height, width) specifying the overall dimensions of the pixel space.
        rotation (Degrees - float): The rotation angle in degrees to be applied during the transformation.
    Returns:
        (Coords_XY - tuple of float): A tuple (metre_x, metre_y) representing the converted coordinates in metres.
    """

    ray_pixels_y, ray_pixels_x = ray_pixels

    pixels_to_metres_transformation = pixels_to_metres_transform(centre, 
                                                                 step, 
                                                                 shape, 
                                                                 0, 
                                                                 rotation)
    

    ray_coords_y, ray_coords_x = apply_transformation(ray_pixels_y, ray_pixels_x, pixels_to_metres_transformation) 

    return ray_coords_x, ray_coords_y


def apply_transformation(y, x, transformation):
    # All of our coordinate transforms are 3x3 transformations, 
    # so we need to add an array of 1s to the end of our coordinates array
    r = jnp.stack([y, x, jnp.ones_like(y)], axis=-1)
    r_transformed = transformation @ r.T 
    y_transformed, x_transformed, _ = r_transformed
    return y_transformed, x_transformed