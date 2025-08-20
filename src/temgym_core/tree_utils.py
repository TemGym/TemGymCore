from typing import List, Literal, Any, Sequence
from typing_extensions import get_type_hints
import dataclasses

import jax
from jax_dataclasses._dataclasses import (
    FieldInfo,
    JDC_STATIC_MARKER,
    get_type_hints_partial,
)
from jax.tree_util import FlattenedIndexKey, SequenceKey, DictKey, GetAttrKey


def isnamedtuple(obj) -> bool:
    return (
        isinstance(obj, tuple)
        and hasattr(obj, '_asdict')
        and hasattr(obj, '_fields')
    )


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
        parent: Any,
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
        return type(self)(self, name, "attr")

    def __getitem__(self, idx: int):
        return type(self)(self, idx, "item")

    def _resolve_root(self):
        return (
            self._parent._resolve_root()
            if isinstance(self._parent, PathBuilder)
            else self._parent
        )

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
            raise ValueError(f"Unknown get with {self._getter}")

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
                if isnamedtuple(tree):
                    idx = GetAttrKey(tree._fields[idx])
                else:
                    idx = SequenceKey(idx)
                break
        assert el is root, f"First item {root} not found in model"
        node_path = (idx,) + this_path[1:]

        param_idxs = {}
        paths_vals, _ = jax.tree.flatten_with_path(tree)
        all_paths = list(p[0] for p in paths_vals)
        for path_idx, param_path in enumerate(all_paths):
            if param_path[: len(node_path)] == node_path:
                param_idxs[
                    original_path
                    + tuple(get_key(k) for k in param_path[len(node_path):])
                ] = path_idx
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


class HasParamsMixin:
    def new_with(self, **kwargs):
        fields = dataclasses.asdict(self)
        fields.update(kwargs)
        return type(self)(**fields)

    @property
    def params(self):
        # could be @classproperty if this existed or could write own descriptor
        params = {
            k.name: PathBuilder(self, k.name, "attr") for k in dataclasses.fields(self)
        }
        # return within instance of self purely to get type hints while building path
        return type(self)(**params)
