from dataclasses import dataclass
from typing import Any

from jaxgym import components as comp

component_map = {
    "PointSource": comp.PointSource,
    "Lens": comp.Lens,
    "ThickLens": comp.ThickLens,
    "Deflector": comp.Deflector,
    "Rotator": comp.Rotator,
    "DoubleDeflector": comp.DoubleDeflector,
    "Plane": comp.Plane,
    "Biprism": comp.Biprism,
    "Detector": comp.Detector,
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
            raise ValueError("Component type not specified")
        cls = component_map.get(comp_type, None)
        if cls is None:
            raise ValueError(f"Unknown component type: {comp_type}")
        comp_def = {
            k: v.val if isinstance(v, ParamTuple) else v for k, v in comp_def.items()
        }
        components.append(cls(**comp_def))
    return components


# def from_str(model_str: str):
#     config = tomli.loads(model_str)
#     return to_model(config)


# def from_file(filename: os.PathLike):
#     with open(filename, 'rb') as fp:
#         config = tomli.load(fp)
#     return to_model(config)
