import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import jax.numpy as jnp
import jax.lax as lax
from . import Degrees, Radians, ShapeYX, CoordsXY, ScaleYX


def _rotate_with_deg_to_rad(degrees: "Degrees"):
    # From libertem.corrections.coordinates v0.11.1
    return _rotate(jnp.pi / 180 * degrees)


def _identity():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.eye(3)


def _rotate(radians: "Radians"):
    # From libertem.corrections.coordinates v0.11.1
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    # This function is written with the y-axis flipped - i.e a pi/2 rotation sends
    # a (1, 0) vector to (0, -1). This is the opposition direction to what one
    # would expect in a conventional cartesian coordinate system.
    return jnp.array(
        [
            (jnp.cos(radians), jnp.sin(radians), 0.0),
            (-jnp.sin(radians), jnp.cos(radians), 0.0),
            (0.0, 0.0, 1.0),
        ]
    )


def _scale(pixel_size_yx):
    return jnp.array(
        [(pixel_size_yx[0], 0.0, 0.0), (0.0, pixel_size_yx[1], 0.0), (0.0, 0.0, 1.0)]
    )


def _shift(centre_yx):
    return jnp.array(
        [(1.0, 0.0, centre_yx[0]), (0.0, 1.0, centre_yx[1]), (0.0, 0.0, 1.0)]
    )


def _flip_y():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.array([(-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])


def pixels_to_metres_transform(
    centre: CoordsXY,
    pixel_size: ScaleYX,
    shape: ShapeYX,
    flip_y=False,
    rotation: Degrees = 0.0,
):
    """
    Transforms pixel coordinates into metre coordinates using a series of matrix operations.

    Parameters:
        centre (CoordsXY): The translation vector representing the center coordinate.
        pixel_size (ScaleYX): The scaling factors that convert pixel dimensions to metres.
        shape (ShapeYX): The shape of the pixel grid (e.g., image size) in (height, width) format.
        flip_y (bool, optional): If True, applies a flip along the y-axis to the transformation.
                                    Defaults to False.
        rotation (Degrees, optional): The rotation angle (in degrees) applied to the transformation.
                                        Defaults to 0.0.

    Returns:
        jax.numpy.ndarray: A transformation matrix that converts
        pixel coordinates to metre coordinates.

    The transformation is composed by applying:
        - A conditional flip along the y-axis if flip_y is True.
        - A fixed flip transformation for the y-axis to go from xy
        of ray space to yx of pixel space.
        - A rotation transformation given by the specified angle.
        - A translation to the provided centre in metres.
        - A scaling transformation with the given pixel size.
        - A shift to center the pixel coordinates based on the shape.
    """

    flip_transform = lax.cond(flip_y, _flip_y, _identity)

    metres_to_px_flip_y = _flip_y()
    shape = jnp.array(shape)
    pixel_shift_transform = _shift(-(shape - 1) / 2.0)
    scale_transform = _scale(pixel_size)
    centre_shift_transform = _shift(centre)
    rotation_transform = _rotate_with_deg_to_rad(rotation)

    transform = (
        flip_transform
        @ metres_to_px_flip_y
        @ rotation_transform
        @ centre_shift_transform
        @ scale_transform
        @ pixel_shift_transform
    )

    return transform


def apply_transformation(y, x, transformation):
    """Apply a 3×3 homogeneous transform to pixel or metre coordinates.

    Parameters
    ----------
    y : jnp.ndarray
        First coordinate (row if pixels, y if metres). Any shape.
    x : jnp.ndarray
        Second coordinate (col if pixels, x if metres). Broadcastable to `y`.
    transformation : jnp.ndarray, shape (3, 3)
        Homogeneous transform mapping [y, x, 1] to [y', x', 1].

    Returns
    -------
    y_out : jnp.ndarray
        Transformed first coordinate with the same shape as `y`.
    x_out : jnp.ndarray
        Transformed second coordinate with the same shape as `x`.

    Notes
    -----
    Pure and JIT-friendly. Differentiable w.r.t. inputs and transform.

    Examples
    --------
    >>> T = jnp.eye(3)
    >>> y, x = jnp.array([0., 1.]), jnp.array([2., 3.])
    >>> y2, x2 = apply_transformation(y, x, T)
    >>> (y2 == y).all(), (x2 == x).all()
    (Array(True, dtype=bool), Array(True, dtype=bool))
    """
    # All of our coordinate transforms are 3x3 transformations,
    # so we need to add an array of 1s to the end of our coordinates array
    r = jnp.stack([y, x, jnp.ones_like(y)], axis=-1)
    r_transformed = transformation @ r.T
    y_transformed, x_transformed, _ = r_transformed
    return y_transformed, x_transformed
