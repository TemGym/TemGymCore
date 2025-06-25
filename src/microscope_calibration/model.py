from typing_extensions import TypedDict, NamedTuple
from numpy.typing import NDArray
import jax.numpy as jnp
from microscope_calibration import components as comp
from jaxgym import Coords_XY


class ModelParameters(TypedDict):
    semi_conv: float
    defocus: float
    camera_length: float
    scan_shape: tuple[int, int]
    det_shape: tuple[int, int]
    scan_step: tuple[float, float]
    det_px_size: tuple[float, float]
    scan_rotation: float
    descan_error: NDArray
    flip_y: bool


class DescannerErrorParameters(NamedTuple):
    pxo_pxi: float = 0.0  # How position x output scales with respect to position x input
    pxo_pyi: float = 0.0  # How position x output scales with respect to position y input
    pyo_pxi: float = 0.0  # How position y output scales with respect to position x input
    pyo_pyi: float = 0.0  # How position y output scales with respect to position y input
    sxo_pxi: float = 0.0  # How slope x output slope scales with respect to position x input
    sxo_pyi: float = 0.0  # How slope x output slope scales with respect to position y input
    syo_pxi: float = 0.0  # How slope y output slope scales with respect to position x input
    syo_pyi: float = 0.0  # How slope y output slope scales with respect to position x input
    offpxi: float = 0.0  # Offset position in x input
    offpyi: float = 0.0  # Offset position in y input
    offsxi: float = 0.0  # Offset slope in x input
    offsyi: float = 0.0  # Offset slope in y input

    def as_array(self) -> NDArray:
        return jnp.array(self)


class Model(NamedTuple):
    source: comp.PointSource
    scan_grid: comp.ScanGrid
    descanner: comp.Descanner
    detector: comp.Detector


def create_stem_model(
    params_dict: ModelParameters, scan_pos_xy: Coords_XY = (0.0, 0.0)
) -> Model:
    PointSource = comp.PointSource(z=jnp.zeros((1)), semi_conv=params_dict["semi_conv"])

    ScanGrid = comp.ScanGrid(
        z=jnp.array([params_dict["defocus"]]),
        scan_step=tuple(params_dict["scan_step"]),
        scan_shape=tuple(params_dict["scan_shape"]),
        scan_rotation=params_dict["scan_rotation"],
    )

    Descanner = comp.Descanner(
        z=jnp.array([params_dict["defocus"]]),
        descan_error=params_dict["descan_error"],
        scan_pos_x=scan_pos_xy[0],
        scan_pos_y=scan_pos_xy[1],
    )

    Detector = comp.Detector(
        z=jnp.array([params_dict["camera_length"] + params_dict["defocus"]]),
        det_shape=tuple(params_dict["det_shape"]),
        det_pixel_size=tuple(params_dict["det_px_size"]),
    )

    return Model(PointSource, ScanGrid, Descanner, Detector)
