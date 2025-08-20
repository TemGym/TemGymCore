from typing import Union
import numpy as np
import jax.numpy as jnp

from . import Degrees, ShapeYX, CoordsXY, ScaleYX, PixelsYX
from .ray import Ray
from .utils import inplace_sum, try_ravel, try_reshape
from .coordinate_transforms import pixels_to_metres_transform, apply_transformation


class Grid:
    z: float
    centre: CoordsXY
    shape: ShapeYX
    pixel_size: ScaleYX
    rotation: Degrees
    flip_y: bool

    @property
    def pixels_to_metres_mat(self) -> jnp.ndarray:
        return pixels_to_metres_transform(
            self.centre, self.pixel_size, self.shape, self.flip_y, self.rotation
        )

    @property
    def metres_to_pixels_mat(self) -> jnp.ndarray:
        return jnp.linalg.inv(
            pixels_to_metres_transform(
                self.centre, self.pixel_size, self.shape, self.flip_y, self.rotation
            )
        )

    @property
    def coords_px(self) -> PixelsYX:
        shape = self.shape
        y_px = jnp.arange(shape[0])
        x_px = jnp.arange(shape[1])
        yy_px, xx_px = jnp.meshgrid(y_px, x_px, indexing="ij")
        return yy_px, xx_px

    @property
    def coords(self) -> jnp.ndarray:
        yy_px, xx_px = self.coords_px
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

    def ray_at_grid(
        self, px_y: float, px_x: float, dx: float = 0., dy: float = 0., z: float | None = None
    ):
        if z is None:
            z = self.z
        x, y = self.pixels_to_metres((px_y, px_x))
        return Ray(
            x=x, y=y, dx=dx, dy=dy, z=z, pathlength=0.,
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
