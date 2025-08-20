import dataclasses
import jax_dataclasses as jdc
import jax.numpy as jnp

from .tree_utils import HasParamsMixin


@jdc.pytree_dataclass
class Ray(HasParamsMixin):
    x: float
    y: float
    dx: float
    dy: float
    z: float
    pathlength: float
    _one: float = 1.0

    @classmethod
    def origin(cls):
        return cls(*((0.0,) * 6))

    @property
    def size(self):
        sizes = set(
            1 if jnp.isscalar(v) else jnp.asarray(v).size
            for v in dataclasses.asdict(self).values()
        )
        assert len(sizes) == 1
        return tuple(sizes)[0]

    def __getitem__(self, arg):
        params = {k: v[arg] for k, v in dataclasses.asdict(self).items()}
        return type(self)(**params)

    def item(self):
        params = {
            k: v.item()
            if hasattr(v, "size")
            else v
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
        """
        Return a modified copy.
        Use this to modify some parameters while keeping others as-is
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
