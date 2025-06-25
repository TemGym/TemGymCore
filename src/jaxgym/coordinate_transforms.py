import abc
from jax.numpy import ndarray as NDArray
import jax.numpy as jnp
import jax.lax as lax
from . import Degrees, Radians, Shape_YX, Coords_XY, Scale_YX, Pixels_YX

RadiansJNP = jnp.float64


class GridBase(abc.ABC):
    metres_to_pixels_mat: jnp.ndarray
    pixels_to_metres_mat: jnp.ndarray

    def __post_init__(self):
        object.__setattr__(
            self, "metres_to_pixels_mat", self.get_metres_to_pixels_transform()
        )
        object.__setattr__(
            self, "pixels_to_metres_mat", self.get_pixels_to_metres_transform()
        )

    @property
    @abc.abstractmethod
    def pixel_size(self) -> Scale_YX: ...

    @property
    @abc.abstractmethod
    def shape(self) -> Shape_YX: ...

    @property
    @abc.abstractmethod
    def rotation(self) -> Degrees: ...

    @property
    @abc.abstractmethod
    def centre(self) -> Coords_XY: ...

    @property
    @abc.abstractmethod
    def flip(self) -> bool: ...

    def get_coords(self) -> NDArray:
        shape = self.shape
        y_px = jnp.arange(shape[0])
        x_px = jnp.arange(shape[1])
        yy_px, xx_px = jnp.meshgrid(y_px, x_px, indexing="ij")
        yy_px = yy_px.ravel()
        xx_px = xx_px.ravel()
        coords_x, coords_y = self.pixels_to_metres((yy_px, xx_px))
        coords_xy = jnp.stack((coords_x, coords_y), axis=-1).reshape(-1, 2)
        return coords_xy

    def step(self, ray):
        return ray

    def get_metres_to_pixels_transform(self) -> NDArray:
        # Use the common transform using centre, pixel_size, shape and rotation.
        pixels_to_metres_mat = pixels_to_metres_transform(
            self.centre, self.pixel_size, self.shape, self.flip, self.rotation
        )
        return jnp.linalg.inv(pixels_to_metres_mat)

    def get_pixels_to_metres_transform(self) -> NDArray:
        return pixels_to_metres_transform(
            self.centre, self.pixel_size, self.shape, self.flip, self.rotation
        )

    def metres_to_pixels(self, coords: Coords_XY) -> Pixels_YX:
        coords_x, coords_y = coords
        pixels_y, pixels_x = apply_transformation(
            coords_y, coords_x, self.metres_to_pixels_mat
        )
        pixels_y = jnp.round(pixels_y).astype(jnp.int32)
        pixels_x = jnp.round(pixels_x).astype(jnp.int32)
        return pixels_y, pixels_x

    def pixels_to_metres(self, pixels: Pixels_YX) -> Coords_XY:
        pixels_y, pixels_x = pixels
        metres_y, metres_x = apply_transformation(
            pixels_y, pixels_x, self.pixels_to_metres_mat
        )
        return metres_x, metres_y

    @property
    def coords(self) -> NDArray:
        return self.get_coords()


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
    centre: Coords_XY,
    pixel_size: Scale_YX,
    shape: Shape_YX,
    flip_y=False,
    rotation: Degrees = 0.0,
):
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
