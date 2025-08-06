from typing import NamedTuple, TYPE_CHECKING

from jaxgym import CoordsXY
from jaxgym.components import DescanError

if TYPE_CHECKING:
    from jaxgym.components import PointSource, Descanner, ScanGrid, Detector


class Parameters4DSTEM(NamedTuple):
    overfocus: float  # m
    scan_pixel_pitch: float  # m
    scan_cy: float  # px
    scan_cx: float  # px
    scan_shape: tuple[int, int]
    scan_rotation: float  # rad
    camera_length: float  # m
    detector_pixel_pitch: float  # m
    detector_cy: float  # px
    detector_cx: float  # px
    detector_shape: tuple[int, int]
    semiconv: float  # rad
    flip_y: bool
    descan_error: DescanError


class Model4DSTEM(NamedTuple):
    source: "PointSource"
    scan_grid: "ScanGrid"
    descanner: "Descanner"
    detector: "Detector"


def create_stem_model(
    params: Parameters4DSTEM, scan_pos_xy: CoordsXY = (0.0, 0.0)
) -> Model4DSTEM:
    # delay import to avoid circular dependency
    from jaxgym import components as comp

    PointSource = comp.PointSource(z=0., semi_conv=params.semiconv)

    ScanGrid = comp.ScanGrid(
        z=params.overfocus,
        scan_step=(params.scan_pixel_pitch, params.scan_pixel_pitch),
        scan_shape=tuple(params.scan_shape),
        scan_rotation=params.scan_rotation,
    )

    Descanner = comp.Descanner(
        z=params.overfocus,
        descan_error=params.descan_error,
        scan_pos_x=scan_pos_xy[0],
        scan_pos_y=scan_pos_xy[1],
    )

    Detector = comp.Detector(
        z=params.camera_length + params.overfocus,
        det_shape=tuple(params.detector_shape),
        det_pixel_size=(params.detector_pixel_pitch, params.detector_pixel_pitch),
        flip_y=params.flip_y,
    )

    return Model4DSTEM(PointSource, ScanGrid, Descanner, Detector)
