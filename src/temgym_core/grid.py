from typing import Union
import numpy as np
import jax.numpy as jnp

from . import Degrees, ShapeYX, CoordXY, CoordsXY, ScaleYX, PixelsYX
from .ray import Ray
from .utils import inplace_sum, try_ravel, try_reshape
from .coordinate_transforms import pixels_to_metres_transform, apply_transformation


class Grid:
    """Mixin with convenience methods for coordinate transformations.

    Attributes
    ----------
    z : float
        Axial position in metres.
    centre : CoordXY
        Grid centre in metres (x, y).
    shape : ShapeYX
        Grid shape (y, x) in pixels.
    pixel_size : ScaleYX
        Pixel size as (y, x) in metres/pixel.
    rotation : Degrees
        Rotation of the grid in degrees.
    flip_y : bool
        If True, apply an additional vertical flip.
    """
    z: float
    centre: CoordXY
    shape: ShapeYX
    pixel_size: ScaleYX
    rotation: Degrees
    flip_y: bool

    @property
    def pixels_to_metres_mat(self) -> jnp.ndarray:
        """Return the 3×3 transform from pixels to metres.

        Returns
        -------
        T : jnp.ndarray, shape (3, 3)
            Homogeneous transform mapping [y_px, x_px, 1] to [y_m, x_m, 1].
        """
        return pixels_to_metres_transform(
            self.centre, self.pixel_size, self.shape, self.flip_y, self.rotation
        )

    @property
    def metres_to_pixels_mat(self) -> jnp.ndarray:
        """Return the 3×3 transform from metres to pixels.

        Returns
        -------
        Tinv : jnp.ndarray, shape (3, 3)
            Inverse homogeneous transform mapping metres to pixels.
        """
        return jnp.linalg.inv(
            pixels_to_metres_transform(
                self.centre, self.pixel_size, self.shape, self.flip_y, self.rotation
            )
        )

    @property
    def coords_px(self) -> PixelsYX:
        """Return meshgrid of pixel coordinates.

        Returns
        -------
        yy_px : jnp.ndarray, shape (H, W), int32
            Row indices [0..H-1].
        xx_px : jnp.ndarray, shape (H, W), int32
            Column indices [0..W-1].
        """
        shape = self.shape
        y_px = jnp.arange(shape[0])
        x_px = jnp.arange(shape[1])
        yy_px, xx_px = jnp.meshgrid(y_px, x_px, indexing="ij")
        return yy_px, xx_px

    @property
    def coords(self) -> jnp.ndarray:
        """Return pixel centres in metres as (x, y) pairs.

        Returns
        -------
        coords_xy : jnp.ndarray, shape (H*W, 2), float32
            For each pixel, returns (x_m, y_m) coordinates in metres.

        Notes
        -----
        The order is flattened row-major.
        """
        yy_px, xx_px = self.coords_px
        yy_px = yy_px.ravel()
        xx_px = xx_px.ravel()
        coords_x, coords_y = self.pixels_to_metres((yy_px, xx_px))
        coords_xy = jnp.stack((coords_x, coords_y), axis=-1).reshape(-1, 2)
        return coords_xy

    @property
    def coords_1d(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return 1D coordinates for x and y dimensions in metres.

        Returns
        -------
        x_coords : jnp.ndarray, shape (W,), float32
            X coordinates of pixel centres along the first row.
        y_coords : jnp.ndarray, shape (H,), float32
            Y coordinates of pixel centres along the first column.
        """
        yy_px, xx_px = self.coords_px
        # Get x_coords from first row
        x_coords = self.pixels_to_metres((yy_px[0, :], xx_px[0, :]))[0]
        # Get y_coords from first column
        y_coords = self.pixels_to_metres((yy_px[:, 0], xx_px[:, 0]))[1]
        return x_coords, y_coords

    def metres_to_pixels(self, coords: CoordsXY, cast: bool = True) -> PixelsYX:
        """Convert metric coordinates to pixel indices.

        Parameters
        ----------
        coords : CoordsXY
            Coordinates in metres (x, y). Any broadcastable shape.
        cast : bool, default True
            If True, round and cast to int32 pixel indices.

        Returns
        -------
        y_px : jnp.ndarray, int32 or float
            Row index/indices. Shape matches input shape.
        x_px : jnp.ndarray, int32 or float
            Column index/indices. Shape matches input shape.

        Notes
        -----
        If `cast=False`, fractional pixel coordinates are returned. Pure and
        JIT-friendly.
        """
        coords_x, coords_y = coords
        metres_to_pixels_mat = self.metres_to_pixels_mat
        pixels_y, pixels_x = apply_transformation(
            try_ravel(coords_y), try_ravel(coords_x), metres_to_pixels_mat
        )
        if cast:
            pixels_y = jnp.round(pixels_y).astype(jnp.int32)
            pixels_x = jnp.round(pixels_x).astype(jnp.int32)
        return PixelsYX(
            y=try_reshape(pixels_y, coords_y),
            x=try_reshape(pixels_x, coords_x)
        )

    def pixels_to_metres(self, pixels: PixelsYX) -> CoordsXY:
        """Convert pixel indices to metric coordinates.

        Parameters
        ----------
        pixels : PixelsYX
            Pixel coordinates as (y_px, x_px). Integer or float arrays.

        Returns
        -------
        x_m : jnp.ndarray
            X coordinate(s) in metres. Shape matches input.
        y_m : jnp.ndarray
            Y coordinate(s) in metres. Shape matches input.

        Notes
        -----
        Pure and JIT-friendly.
        """
        pixels_y, pixels_x = pixels
        pixels_to_metres_mat = self.pixels_to_metres_mat
        metres_y, metres_x = apply_transformation(
            try_ravel(pixels_y), try_ravel(pixels_x), pixels_to_metres_mat
        )
        return CoordsXY(
            x=try_reshape(metres_x, pixels_x),
            y=try_reshape(metres_y, pixels_y)
        )

    def ray_at_grid(
        self, px_y: float, px_x: float, dx: float = 0., dy: float = 0., z: float | None = None
    ):
        """Build a ray positioned at a pixel location on this grid.

        Parameters
        ----------
        px_y : float
            Row index (can be fractional).
        px_x : float
            Column index (can be fractional).
        dx : float, default 0.0
            Slope in x, radians.
        dy : float, default 0.0
            Slope in y, radians.
        z : float or None, default None
            If None, use the grid's `z` position; otherwise override, metres.

        Returns
        -------
        ray : temgym_core.ray.Ray
            Ray positioned at the given pixel with specified slopes.
        """
        if z is None:
            z = self.z
        x, y = self.pixels_to_metres((px_y, px_x))
        return Ray(
            x=x, y=y, dx=dx, dy=dy, z=z, pathlength=0.,
        )

    def ray_to_grid(self, ray: "Ray", cast: bool = False) -> PixelsYX:
        """Map a ray's (x, y) to this grid's pixel coordinates.

        Parameters
        ----------
        ray : temgym_core.ray.Ray
            Input ray.
        cast : bool, default False
            If True, round and cast to int32 pixel indices.

        Returns
        -------
        pixels : PixelsYX
            Pixel coordinates (y_px, x_px).
        """
        return self.metres_to_pixels((ray.x, ray.y), cast=cast)

    def into_image(self, ray: Union[Ray, PixelsYX], acc: np.ndarray | None = None):
        """Rasterize one or more rays into an integer accumulator image.

        Parameters
        ----------
        ray : Ray or PixelsYX
            Ray(s) to rasterize, or pixel coordinates.
        acc : numpy.ndarray or None, default None
            Accumulator image of shape `self.shape`. If None, a new array
            is created and returned.

        Returns
        -------
        acc : numpy.ndarray, dtype=int
            Updated accumulator image.

        Notes
        -----
        Uses an in-place summation kernel for performance. Not JIT-compiled;
        uses numpy arrays internally.

        Raises
        ------
        ValueError
            If `acc` has a different shape than `self.shape`.

        Examples
        --------
        >>> import numpy as np
        >>> from temgym_core.components import Detector
        >>> det = Detector(z=0., pixel_size=(1e-3, 1e-3), shape=(4, 4))
        >>> r = det.ray_at_grid(1.0, 2.0)
        >>> img = det.into_image(r)
        >>> img.sum()
        1
        """
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
