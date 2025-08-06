from typing import TypedDict, NamedTuple, TYPE_CHECKING

from jaxgym import CoordsXY
from jaxgym.components import DescanError

if TYPE_CHECKING:
    from jaxgym.components import PointSource, Descanner, ScanGrid, Detector


class ModelParameters(TypedDict):
    semi_conv: float
    defocus: float
    camera_length: float
    scan_shape: tuple[int, int]
    det_shape: tuple[int, int]
    scan_step: tuple[float, float]
    det_px_size: tuple[float, float]
    scan_rotation: float
    descan_error: DescanError
    flip_y: bool


class Model(NamedTuple):
    source: "PointSource"
    scan_grid: "ScanGrid"
    descanner: "Descanner"
    detector: "Detector"


def create_stem_model(
    params_dict: ModelParameters, scan_pos_xy: CoordsXY = (0.0, 0.0)
) -> Model:
    # delay import to avoid circular dependency
    from jaxgym import components as comp

    PointSource = comp.PointSource(z=0., semi_conv=params_dict["semi_conv"])

    ScanGrid = comp.ScanGrid(
        z=params_dict["defocus"],
        scan_step=tuple(params_dict["scan_step"]),
        scan_shape=tuple(params_dict["scan_shape"]),
        scan_rotation=params_dict["scan_rotation"],
    )

    Descanner = comp.Descanner(
        z=params_dict["defocus"],
        descan_error=params_dict["descan_error"],
        scan_pos_x=scan_pos_xy[0],
        scan_pos_y=scan_pos_xy[1],
    )

    Detector = comp.Detector(
        z=params_dict["camera_length"] + params_dict["defocus"],
        det_shape=tuple(params_dict["det_shape"]),
        det_pixel_size=tuple(params_dict["det_px_size"]),
        flip_y=params_dict["flip_y"],
    )

    return Model(PointSource, ScanGrid, Descanner, Detector)
