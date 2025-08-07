from typing import NamedTuple, TYPE_CHECKING

from jaxgym import CoordsXY
from jaxgym.components import DescanError
from jaxgym import components as comp
from jaxgym.ray import Ray
from jaxgym.run import run_to_end

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

    def at_scan_pos(
        self,
        scan_px_y: int,
        scan_px_x: int,
    ):
        scan_pos = self.scan_grid.pixels_to_metres(
            (scan_px_y, scan_px_x)
        )
        return type(self)(
            self.source.new_with(offset_xy=scan_pos),
            self.scan_grid,
            self.descanner.new_with(
                scan_pos_x=scan_pos[0],
                scan_pos_y=scan_pos[1],
            ),
            self.detector,
        )

    def make_rays(self, num: int = 1, random: bool = False):
        r = self.source.generate(num, random=random)
        return Ray(
            x=r[:, 0],
            y=r[:, 1],
            dx=r[:, 2],
            dy=r[:, 3],
            z=self.source.z,
            pathlength=0.,
        )

    def trace(self, ray: Ray):
        return run_to_end(ray, self)


def create_stem_model(
    params: Parameters4DSTEM, scan_pos_xy: CoordsXY = (0.0, 0.0)
) -> Model4DSTEM:

    PointSource = comp.PointSource(
        z=0.,
        semi_conv=params.semiconv,
        offset_xy=scan_pos_xy,
    )

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
