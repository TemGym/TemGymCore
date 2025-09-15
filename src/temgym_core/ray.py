import dataclasses
import jax_dataclasses as jdc
import jax.numpy as jnp
from .tree_utils import HasParamsMixin


@jdc.pytree_dataclass
class Ray(HasParamsMixin):
    """Parametric ray with positions, slopes, z, and pathlength.

    Parameters
    ----------
    x : float or jnp.ndarray
        X position(s), metres.
    y : float or jnp.ndarray
        Y position(s), metres.
    dx : float or jnp.ndarray
        X slope(s), radians (paraxial small-angle).
    dy : float or jnp.ndarray
        Y slope(s), radians (paraxial small-angle).
    z : float or jnp.ndarray
        Axial position(s), metres.
    pathlength : float or jnp.ndarray
        Accumulated distance, metres.
    _one : float, default 1.0
        Homogeneous coordinate carrier; do not modify.

    Notes
    -----
    Instances are immutable; use `derive()` to create modified copies.
    Vectorized rays are supported by using array fields with matching size.

    Examples
    --------
    >>> Ray.origin()
    Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=0.0, pathlength=0.0)
    """
    x: float
    y: float
    dx: float
    dy: float
    z: float
    pathlength: float
    _one: float = 1.0

    @classmethod
    def origin(cls):
        """Create a ray at the origin with zero slopes and zero pathlength.

        Returns
        -------
        ray : Ray
            Ray(x=0, y=0, dx=0, dy=0, z=0, pathlength=0).
        """
        return cls(*((0.0,) * 6))

    @property
    def size(self):
        """Return the number of elements represented by this ray.

        Returns
        -------
        n : int
            1 for scalar rays, otherwise the vector length.

        Raises
        ------
        AssertionError
            If fields have mismatched sizes.
        """
        sizes = set(
            1 if jnp.isscalar(v) else jnp.asarray(v).size
            for v in dataclasses.asdict(self).values()
        )
        assert len(sizes) == 1
        return tuple(sizes)[0]

    def __getitem__(self, arg):
        """Index a vectorized ray to get a single-element Ray.

        Parameters
        ----------
        arg : int or slice
            Index or slice over the vector dimension.

        Returns
        -------
        ray : Ray
            Ray with fields indexed as requested.
        """
        params = {k: v[arg] for k, v in dataclasses.asdict(self).items()}
        return type(self)(**params)

    def to_ray(self):
        return self

    def item(self):
        """Convert a single-element ray to scalars.

        Returns
        -------
        ray : Ray
            Ray with scalar Python floats instead of arrays.

        Notes
        -----
        Useful to extract a single result from vectorized computations.
        """
        params = {
            k: v.item()
            if hasattr(v, "size")
            else v
            for k, v
            in dataclasses.asdict(self).items()
        }
        return type(self)(**params)

    def to_vector(self):
        params = {
            k: jnp.atleast_1d(v)
            for k, v
            in dataclasses.asdict(self).items()
        }
        return type(self)(**params)

    def derive(
        self,
        x: float | None = None,
        y: float | None = None,
        dx: float | None = None,
        dy: float | None = None,
        z: float | None = None,
        pathlength: float | None = None
    ) -> 'Ray':
        """Return a modified copy of the ray with selected fields changed.

        Parameters
        ----------
        x, y : float or None, default None
            New positions in metres.
        dx, dy : float or None, default None
            New slopes in radians.
        z : float or None, default None
            New axial position in metres.
        pathlength : float or None, default None
            New pathlength in metres.

        Returns
        -------
        ray : Ray
            Modified copy of the ray.

        Notes
        -----
        `_one` is preserved unchanged.
        """
        return Ray(
            x=x if x is not None else self.x,
            y=y if y is not None else self.y,
            dx=dx if dx is not None else self.dx,
            dy=dy if dy is not None else self.dy,
            z=z if z is not None else self.z,
            pathlength=pathlength if pathlength is not None else self.pathlength,
            _one=self._one,
        )
