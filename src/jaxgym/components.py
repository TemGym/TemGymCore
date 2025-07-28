from .ray import Ray, propagate
from .utils import random_coords, concentric_rings
from .ode import solve_ode
from . import Degrees

from typing import Union, List, Literal
import dataclasses
import jax_dataclasses as jdc
from jax_dataclasses._dataclasses import FieldInfo, JDC_STATIC_MARKER, get_type_hints_partial
import jax.numpy as jnp
from jax.tree_util import FlattenedIndexKey, SequenceKey, DictKey
import numpy as np

from typing_extensions import TypeAlias, get_type_hints

Radians: TypeAlias = jnp.float64  # type: ignore


@jdc.pytree_dataclass
class PointSource:
    z: float
    semi_conv: float
    offset_xy: tuple[float, float] = (0.0, 0.0)

    def step(self, ray: Ray):
        return ray

    def generate(self, num_rays: int, random: bool = False):
        semi_conv = self.semi_conv
        offset_xy = self.offset_xy

        if random:
            y, x = random_coords(num_rays) * semi_conv
        else:
            y, x = concentric_rings(num_rays, semi_conv)

        r = np.zeros((num_rays, 5), dtype=np.float64)  # x, y, theta_x, theta_y, 1

        r[:, 0] += offset_xy[0]
        r[:, 1] += offset_xy[1]
        r[:, 2] = x
        r[:, 3] = y
        r[:, 4] = 1.

        return r


def get_key(k):
    try:
        return k.idx
    except AttributeError:
        pass
    try:
        return k.key
    except AttributeError:
        pass
    raise TypeError(f"Unrecognized key type {k}")


class PathBuilder:
    def __init__(
        self,
        parent: Union['PathBuilder', 'Component'],
        key: int | str,
        getter: Literal["attr", "item"],
    ):
        self._parent = parent
        self._original_key = key
        self._getter = getter
        true_parent = self._resolve_parent()
        if dataclasses.is_dataclass(type(true_parent)):
            field_info = get_field_info(type(true_parent))
            if len(field_info.static_field_names):
                raise NotImplementedError("Static fields not supported")
            key = FlattenedIndexKey(field_info.child_node_field_names.index(key))
        elif isinstance(true_parent, (list, tuple)):
            key = SequenceKey(key)
        elif isinstance(true_parent, dict):
            key = DictKey(key)
        else:
            raise NotImplementedError(f"Unknown key type for cls {type(true_parent)}")
        self._key = key

    def __getattr__(self, name: str):
        # if name in ("_parent", "_key", "_build", "_original_key"):
        #     return super().__getattribute__(name)
        return type(self)(self, name, "attr")

    def __getitem__(self, idx: int):
        return type(self)(self, idx, "item")

    def _resolve_parent(self):
        return (
            self._parent._resolve()
            if isinstance(self._parent, PathBuilder)
            else self._parent
        )

    def _resolve(self):
        get_from = self._resolve_parent()
        if self._getter == "attr":
            return getattr(get_from, self._original_key)
        elif self._getter == "item":
            return get_from[self._original_key]
        else:
            raise ValueError(f'Unknown get with {self._getter}')

    def _build(self, children: tuple[int | str] | None = None, original: bool = False):
        if children is None:
            children = tuple()
        if original:
            key = self._original_key
        else:
            key = self._key
        this = (key,) + children
        if not isinstance(self._parent, PathBuilder):
            return (self._parent,) + this
        return self._parent._build(this, original=original)

    def _find_in(self, tree: Sequence[Any]) -> dict[Sequence[str | int], int]:
        original_path = self._build(original=True)
        this_path = self._build()
        root = this_path[0]
        for idx, el in enumerate(tree):
            if el is root:
                idx = SequenceKey(idx)
                break
        assert el is root, f"First item {root} not found in model"
        node_path = (idx,) + this_path[1:]

        param_idxs = {}
        paths_vals, _ = jax.tree.flatten_with_path(tree)
        all_paths = list(p[0] for p in paths_vals)
        for idx, param_path in enumerate(all_paths):
            if param_path[:len(node_path)] == node_path:
                param_idxs[
                    original_path
                    + tuple(get_key(k) for k in param_path[len(node_path):])
                ] = idx
        return param_idxs


def get_field_info(cls) -> FieldInfo:
    # Determine which fields are static and part of the treedef, and which should be
    # registered as child nodes.
    child_node_field_names: List[str] = []
    static_field_names: List[str] = []

    # We don't directly use field.type for postponed evaluation; we want to make sure
    # that our types are interpreted as proper types and not as (string) forward
    # references.
    #
    # Note that there are ocassionally situations where the @jdc.pytree_dataclass
    # decorator is called before a referenced type is defined; to suppress this error,
    # we resolve missing names to our subscriptible placeholder object.

    try:
        type_from_name = get_type_hints(cls, include_extras=True)  # type: ignore
    except Exception:
        # Try again, but suppress errors from unresolvable forward
        # references. This should be rare.
        type_from_name = get_type_hints_partial(cls, include_extras=True)  # type: ignore

    for field in dataclasses.fields(cls):
        if not field.init:
            continue

        field_type = type_from_name[field.name]

        # Two ways to mark a field as static: either via the Static[] type or
        # jdc.static_field().
        if (
            hasattr(field_type, "__metadata__")
            and JDC_STATIC_MARKER in field_type.__metadata__
        ):
            static_field_names.append(field.name)
            continue
        if field.metadata.get(JDC_STATIC_MARKER, False):
            static_field_names.append(field.name)
            continue

        child_node_field_names.append(field.name)
    return FieldInfo(child_node_field_names, static_field_names)


class Component:
    @property
    def params(self):
        # could be @classproperty if this existed or could write own descriptor
        params = {
            k.name: PathBuilder(self, k.name, "attr")
            for k
            in dataclasses.fields(self)
        }
        # return within instance of self purely to get type hints while building path
        return type(self)(**params)


@jdc.pytree_dataclass
class Lens(Component):
    z: float
    focal_length: float

    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)
        one = ray._one * 1.0

        return Ray(
            x=x, y=y, dx=new_dx, dy=new_dy, _one=one, pathlength=pathlength, z=ray.z
        )


@jdc.pytree_dataclass
class ThickLens:
    z_po: float
    z_pi: float
    focal_length: float

    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)

        new_z = ray.z - (self.z_po - self.z_pi)

        one = ray._one * 1.0

        return Ray(
            x=x, y=y, dx=new_dx, dy=new_dy, _one=one, pathlength=pathlength, z=new_z
        )

    @property
    def z(self):
        return self.z_po


@jdc.pytree_dataclass
class ODE:
    z: float
    z_end: float
    phi_lambda: callable
    E_lambda: callable

    def step(self, ray: Ray) -> Ray:
        in_state = jnp.array([ray.x, ray.y, ray.dx, ray.dy, ray.pathlength])

        z_start = self.z
        z_end = self.z_end

        u0 = self.phi_lambda(0.0, 0.0, z_start).astype(jnp.float64)

        out_state, out_z = solve_ode(
            in_state, z_start, z_end, self.phi_lambda, self.E_lambda, u0
        )

        x, y, dx, dy, opl = out_state

        return Ray(x=x, y=y, dx=dx, dy=dy, _one=ray._one, pathlength=opl, z=out_z)


@jdc.pytree_dataclass
class Deflector:
    z: float
    def_x: float
    def_y: float

    def step(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        new_dx = dx + self.def_x
        new_dy = dy + self.def_y

        pathlength = ray.pathlength + dx * x + dy * y

        return Ray(
            x=x,
            y=y,
            dx=new_dx,
            dy=new_dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )


@jdc.pytree_dataclass
class Rotator:
    z: float
    angle: Degrees

    def step(self, ray: Ray):
        angle = jnp.deg2rad(self.angle)

        # Rotate the ray's position
        new_x = ray.x * jnp.cos(angle) - ray.y * jnp.sin(angle)
        new_y = ray.x * jnp.sin(angle) + ray.y * jnp.cos(angle)
        # Rotate the ray's slopes
        new_dx = ray.dx * jnp.cos(angle) - ray.dy * jnp.sin(angle)
        new_dy = ray.dx * jnp.sin(angle) + ray.dy * jnp.cos(angle)

        pathlength = ray.pathlength

        return Ray(
            x=new_x,
            y=new_y,
            dx=new_dx,
            dy=new_dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )


@jdc.pytree_dataclass
class DoubleDeflector:
    z: float
    first: Deflector
    second: Deflector

    def step(self, ray: Ray):
        ray = self.first.step(ray)
        z_step = self.second.z - self.first.z
        ray = propagate(z_step, ray)
        ray = self.second.step(ray)

        return ray


@jdc.pytree_dataclass
class InputPlane:
    z: float

    def step(self, ray: Ray):
        return ray


@jdc.pytree_dataclass
class Biprism:
    z: float
    offset: float = 0.0
    rotation: Degrees = 0.0
    deflection: float = 0.0

    def step(
        self,
        ray: Ray,
    ) -> Ray:
        pos_x, pos_y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        deflection = self.deflection
        offset = self.offset
        rot = jnp.deg2rad(self.rotation)

        rays_v = jnp.array([pos_x, pos_y]).T

        biprism_loc_v = jnp.array([offset * jnp.cos(rot), offset * jnp.sin(rot)])

        biprism_v = jnp.array([-jnp.sin(rot), jnp.cos(rot)])
        biprism_v /= jnp.linalg.norm(biprism_v)

        rays_v_centred = rays_v - biprism_loc_v

        dot_product = jnp.dot(rays_v_centred, biprism_v) / jnp.dot(biprism_v, biprism_v)
        projection = jnp.outer(dot_product, biprism_v)

        rejection = rays_v_centred - projection
        rejection = rejection / jnp.linalg.norm(rejection, axis=1, keepdims=True)

        # If the ray position is located at [zero, zero], rejection_norm returns a nan,
        # so we convert it to a zero, zero.
        rejection = jnp.nan_to_num(rejection)

        xdeflection_mag = rejection[:, 0]
        ydeflection_mag = rejection[:, 1]

        new_dx = (dx + xdeflection_mag * deflection).squeeze()
        new_dy = (dy + ydeflection_mag * deflection).squeeze()

        pathlength = ray.pathlength + (
            xdeflection_mag * deflection * pos_x + ydeflection_mag * deflection * pos_y
        )

        return Ray(
            x=pos_x.squeeze(),
            y=pos_y.squeeze(),
            dx=new_dx,
            dy=new_dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )


# Base class for grid transforms


# @jdc.pytree_dataclass
# class ImageGrid(GridBase):
#     z: float
#     image_pixel_size: Scale_YX
#     image_shape: Shape_YX
#     image_rotation: Degrees
#     image_centre: Coords_XY = (0., 0.)
#     image_array: jnp.ndarray = None  # Added image array variable specific to ImageGrid
#     metres_to_pixels_mat: jnp.ndarray = jdc.field(init=False)
#     pixels_to_metres_mat: jnp.ndarray = jdc.field(init=False)

#     @property
#     def pixel_size(self) -> Scale_YX:
#         return self.image_pixel_size

#     @property
#     def shape(self) -> Shape_YX:
#         return self.image_shape

#     @property
#     def rotation(self) -> Degrees:
#         return self.image_rotation

#     @property
#     def centre(self) -> Coords_XY:
#         return self.image_centre
