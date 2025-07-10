import os
import tomli
from dataclasses import dataclass
from typing import Any

# from . import components as comp


component_map = {
    # "ParallelBeam": comp.ParallelBeam,
    # "Lens": comp.Lens,
    # "Detector": comp.Detector,
}


@dataclass
class ParamTuple:
    val: Any
    meta: dict[str, Any]


def to_model(config: dict[str, list]):
    components = []
    for comp_def in config.get("components", tuple()):
        comp_type = comp_def.pop("type", None)
        if comp_type is None:
            raise
        cls = component_map.get(comp_type, None)
        if cls is None:
            raise
        comp_def = {k: v.val if isinstance(v, ParamTuple) else v for k, v in comp_def.items()}
        components.append(cls(**comp_def))
    return Model(components)


def from_str(model_str: str):
    config = tomli.loads(model_str)
    return to_model(config)


def from_file(filename: os.PathLike):
    with open(filename, 'rb') as fp:
        config = tomli.load(fp)
    return to_model(config)
