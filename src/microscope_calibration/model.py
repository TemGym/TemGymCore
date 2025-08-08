from typing import NamedTuple, Generator, Any, overload, Literal, TYPE_CHECKING, Sequence

from jaxgym import CoordsXY, PixelsYX
from jaxgym.components import DescanError, Component
from jaxgym import components as comp
from jaxgym.source import PointSource
from jaxgym.ray import Ray
from jaxgym.run import run_to_end, run_iter, run_with_grads
from jaxgym.propagator import Propagator

if TYPE_CHECKING:
    from jaxgym.tree_utils import PathBuilder


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

    def new_with(self, **kwargs):
        fields = self._asdict()
        fields.update(kwargs)
        return type(self)(**fields)


class Model4DSTEM(NamedTuple):
    source: PointSource
    scan_grid: comp.ScanGrid
    descanner: comp.Descanner
    detector: comp.Detector

    def at_scan_pos(
        self, scan_px_y: int, scan_px_x: int,
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

    def make_rays_at_source(self, num: int = 1, random: bool = False):
        return self.source.make_rays(num, random=random)

    @overload
    def trace(self, ray: Ray, output_as_pixels: Literal[True]) -> PixelsYX: ...

    @overload
    def trace(self, ray: Ray, output_as_pixels: Literal[False]) -> Ray: ...

    def trace(self, ray: Ray, output_as_pixels: bool = False):
        to_run = self
        if output_as_pixels:
            to_run = to_run + (self.detector.ray_to_grid,)
        return run_to_end(ray, to_run)

    def trace_with_grads(
        self,
        ray: Ray,
        grad_vars: Sequence["PathBuilder"],
        output_as_pixels: bool = False,
    ) -> tuple[Ray, dict[Sequence[Any], Ray]]:
        to_run = self
        if output_as_pixels:
            to_run = to_run + (self.detector.ray_to_grid,)
        return run_with_grads(ray, to_run, grad_vars)

    def trace_iter(
        self, ray: Ray
    ) -> Generator[tuple[Propagator | PointSource | Component, Ray], Any, None]:
        yield from run_iter(ray, self)


def create_stem_model(
    params: Parameters4DSTEM, scan_pos_xy: CoordsXY = (0.0, 0.0)
) -> Model4DSTEM:

    pointsource = PointSource(
        z=0.,
        semi_conv=params.semiconv,
        offset_xy=scan_pos_xy,
    )

    scangrid = comp.ScanGrid(
        z=params.overfocus,
        pixel_size=(params.scan_pixel_pitch, params.scan_pixel_pitch),
        shape=tuple(params.scan_shape),
        rotation=params.scan_rotation,
        centre=(params.scan_cx, params.scan_cy),
        flip_y=False,
    )

    descanner = comp.Descanner(
        z=params.overfocus,
        descan_error=params.descan_error,
        scan_pos_x=scan_pos_xy[0],
        scan_pos_y=scan_pos_xy[1],
    )

    detector = comp.Detector(
        z=params.camera_length + params.overfocus,
        pixel_size=(params.detector_pixel_pitch, params.detector_pixel_pitch),
        shape=tuple(params.detector_shape),
        rotation=0.,
        centre=(params.detector_cx, params.detector_cy),
        flip_y=params.flip_y,
    )

    return Model4DSTEM(pointsource, scangrid, descanner, detector)
