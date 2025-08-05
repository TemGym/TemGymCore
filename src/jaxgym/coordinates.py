from math import sin, cos
import jax_dataclasses as jdc
import jax.numpy as jnp


@jdc.pytree_dataclass
class XYVector:
    x: float
    y: float
    _one: float = 1.

    def derive(
        self,
        x: float | None = None,
        y: float | None = None
    ) -> "XYVector":
        return XYVector(
            x=x if x is not None else self.x,
            y=y if y is not None else self.y,
            _one=self._one,
        )

    def __neg__(self) -> 'XYVector':
        return self.derive(x=-self.x, y=-self.y)

    def __add__(self, other: 'XYVector') -> 'XYVector':
        return self.derive(
            x=self.x + other.x,
            y=self.y + other.y,
        )

    def __sub__(self, other: 'XYVector') -> 'XYVector':
        return self.derive(
            x=self.x - other.x,
            y=self.y - other.y,
        )

    def __mul__(self, other: float) -> 'XYVector':
        return self.derive(
            x=self.x * other,
            y=self.y * other,
        )

    __rmul__ = __mul__


@jdc.pytree_dataclass
class XYCoordinateSystem:
    x_vector: XYVector
    y_vector: XYVector
    origin: XYVector

    def __call__(self, coords: XYVector) -> XYVector:
        return coords.derive(
            x=self.origin.x * coords._one + coords.x * self.x_vector.x + coords.y * self.y_vector.x,
            y=self.origin.y * coords._one + coords.x * self.x_vector.y + coords.y * self.y_vector.y,
        )

    def matrix(self) -> jnp.ndarray:
        return jnp.array((
            (self.x_vector.x, self.y_vector.x, self.origin.x),
            (self.x_vector.y, self.y_vector.y, self.origin.y),
            (0., 0., 1.),
        ))

    @classmethod
    def from_matrix(cls, mat: jnp.ndarray) -> 'XYCoordinateSystem':
        return cls(
            origin=XYVector(x=float(mat[0, 2]), y=float(mat[1, 2])),
            x_vector=XYVector(x=float(mat[0, 0]), y=float(mat[1, 0])),
            y_vector=XYVector(x=float(mat[0, 1]), y=float(mat[1, 1])),
        )

    def invert(self) -> 'XYCoordinateSystem':
        mat = self.matrix()
        inv = jnp.linalg.inv(mat)
        return self.from_matrix(inv)

    @classmethod
    def identity(cls, _one: float = 1.) -> 'XYCoordinateSystem':
        return cls(
            x_vector=XYVector(x=1., y=0., _one=_one),
            y_vector=XYVector(x=0, y=1., _one=_one),
            origin=XYVector(x=0., y=0., _one=_one),
        )

    def shift(self, shift_vector: XYVector) -> 'XYCoordinateSystem':
        return XYCoordinateSystem(
            x_vector=self.x_vector,
            y_vector=self.y_vector,
            origin=self.origin + shift_vector,
        )

    def rotate(self, radians, center: XYVector | None = None) -> 'XYCoordinateSystem':
        if center is None:
            center = XYVector(x=0, y=0)

        shifted = self.shift(-center)

        x_x = cos(radians)
        x_y = sin(radians)
        y_x = -sin(radians)
        y_y = cos(radians)

        rotated = self.__class__(
            x_vector=shifted.x_vector.derive(
                x=shifted.x_vector.x * x_x + shifted.x_vector.y * x_y,
                y=shifted.x_vector.x * y_x + shifted.x_vector.y * y_y,
            ),
            y_vector=shifted.y_vector.derive(
                x=shifted.y_vector.x * x_x + shifted.y_vector.y * x_y,
                y=shifted.y_vector.x * y_x + shifted.y_vector.y * y_y,
            ),
            origin=shifted.y_vector.derive(
                x=shifted.origin.x * x_x + shifted.origin.y * x_y,
                y=shifted.origin.x * y_x + shifted.origin.y * y_y,
            ),
        )
        return rotated.shift(center)

    def scale(self, factor: float, center: XYVector | None = None) -> 'XYCoordinateSystem':
        if center is None:
            center = XYVector(x=0, y=0)
        shifted = self.shift(-center)
        scaled = self.__class__(
            x_vector=shifted.x_vector * factor,
            y_vector=shifted.y_vector * factor,
            origin=shifted.origin * factor,
        )
        return scaled.shift(center)

    def flip_y(self, flip_y: bool = True) -> 'XYCoordinateSystem':
        factor = -1. if flip_y is True else 1
        return self.__class__(
            x_vector=self.x_vector,
            y_vector=self.y_vector * factor,
            origin=self.origin,
        )
