from typing import Union
import numpy as np
from jax.numpy import ndarray as NDArray
import jax.numpy as jnp
import jax.lax as lax
from . import Degrees, Radians, ShapeYX, CoordsXY, ScaleYX, PixelsYX
from .ray import Ray
from .utils import inplace_sum


class Grid:
    z: float
    centre: CoordsXY
    shape: ShapeYX
    pixel_size: ScaleYX
    rotation: Degrees
    flip_y: bool

    @property
    def pixels_to_metres_mat(self) -> NDArray:
        return pixels_to_metres_transform(
            self.centre, self.pixel_size, self.shape, self.flip_y, self.rotation
        )

    @property
    def metres_to_pixels_mat(self) -> NDArray:
        return jnp.linalg.inv(
            pixels_to_metres_transform(
                self.centre, self.pixel_size, self.shape, self.flip_y, self.rotation
            )
        )

    @property
    def coords(self) -> NDArray:
        shape = self.shape
        y_px = jnp.arange(shape[0])
        x_px = jnp.arange(shape[1])
        yy_px, xx_px = jnp.meshgrid(y_px, x_px, indexing="ij")
        yy_px = yy_px.ravel()
        xx_px = xx_px.ravel()
        coords_x, coords_y = self.pixels_to_metres((yy_px, xx_px))
        coords_xy = jnp.stack((coords_x, coords_y), axis=-1).reshape(-1, 2)
        return coords_xy

    def metres_to_pixels(self, coords: CoordsXY, cast: bool = True) -> PixelsYX:
        coords_x, coords_y = coords
        metres_to_pixels_mat = self.metres_to_pixels_mat
        pixels_y, pixels_x = apply_transformation(
            try_ravel(coords_y), try_ravel(coords_x), metres_to_pixels_mat
        )
        if cast:
            pixels_y = jnp.round(pixels_y).astype(jnp.int32)
            pixels_x = jnp.round(pixels_x).astype(jnp.int32)
        return try_reshape(pixels_y, coords_y), try_reshape(pixels_x, coords_x)

    def pixels_to_metres(self, pixels: PixelsYX) -> CoordsXY:
        pixels_y, pixels_x = pixels
        pixels_to_metres_mat = self.pixels_to_metres_mat
        metres_y, metres_x = apply_transformation(
            try_ravel(pixels_y), try_ravel(pixels_x), pixels_to_metres_mat
        )
        return try_reshape(metres_x, pixels_x), try_reshape(metres_y, pixels_y)

    def ray_at_grid(self, px_y: float, px_x: float, dx: float = 0., dy: float = 0.):
        x, y = self.pixels_to_metres((px_y, px_x))
        return Ray(
            x=x, y=y, dx=dx, dy=dy, z=self.z, pathlength=0.,
        )

    def ray_to_grid(self, ray: "Ray", cast: bool = False) -> PixelsYX:
        return self.metres_to_pixels((ray.x, ray.y), cast=cast)

    def into_image(self, ray: Union[Ray, PixelsYX], acc: np.ndarray | None = None):
        if isinstance(ray, Ray):
            yy, xx = self.ray_to_grid(ray, cast=True)
        else:
            yy, xx = ray
        if acc is None:
            acc = np.zeros(self.shape, dtype=int)
        inplace_sum(
            np.asarray(yy),
            np.asarray(xx),
            np.ones(yy.shape, dtype=bool),
            np.ones(xx.shape, dtype=np.float32),
            acc,
        )
        return acc


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
    # All of our coordinate transforms are 3x3 transformations,
    # so we need to add an array of 1s to the end of our coordinates array
    r = jnp.stack([y, x, jnp.ones_like(y)], axis=-1)
    r_transformed = transformation @ r.T
    y_transformed, x_transformed, _ = r_transformed
    return y_transformed, x_transformed


def try_ravel(val):
    try:
        return val.ravel()
    except AttributeError:
        return val


def try_reshape(val, maybe_has_shape):
    try:
        return val.reshape(maybe_has_shape.shape)
    except AttributeError:
        return val
