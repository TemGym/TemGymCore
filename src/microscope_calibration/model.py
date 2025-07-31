from typing import TypedDict, NamedTuple, TYPE_CHECKING
import jax.numpy as jnp

from jaxgym import CoordsXY

if TYPE_CHECKING:
    from jaxgym.components import PointSource, Descanner, ScanGrid, Detector


class DescanErrorParameters(NamedTuple):
    pxo_pxi: float = 0.0  # How position x output scales with respect to scan x position
    pxo_pyi: float = 0.0  # How position x output scales with respect to scan y position
    pyo_pxi: float = 0.0  # How position y output scales with respect to scan x position
    pyo_pyi: float = 0.0  # How position y output scales with respect to scan y position
    sxo_pxi: float = 0.0  # How slope x output scales with respect to scan x position
    sxo_pyi: float = 0.0  # How slope x output scales with respect to scan y position
    syo_pxi: float = 0.0  # How slope y output scales with respect to scan x position
    syo_pyi: float = 0.0  # How slope y output scales with respect to scan y position
    offpxi: float = 0.0  # Constant additive error in x position
    offpyi: float = 0.0  # Constant additive error in y position
    offsxi: float = 0.0  # Constant additive error in x slope
    offsyi: float = 0.0  # Constant additive error in y slope

    def as_array(self) -> jnp.ndarray:
        return jnp.array(self)

    def as_matrix(self) -> jnp.ndarray:
        # Not used but represents the equations in descanner.step()
        return jnp.array(
            [
                [self.pxo_pxi, self.pxo_pyi, 0.0, 0.0, self.offpxi],
                [self.pyo_pxi, self.pyo_pyi, 0.0, 0.0, self.offpyi],
                [self.sxo_pxi, self.sxo_pyi, 0.0, 0.0, self.offsyi],
                [self.syo_pxi, self.syo_pyi, 0.0, 0.0, self.offsyi],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )


class ModelParameters(TypedDict):
    semi_conv: float
    defocus: float
    camera_length: float
    scan_shape: tuple[int, int]
    det_shape: tuple[int, int]
    scan_step: tuple[float, float]
    det_px_size: tuple[float, float]
    scan_rotation: float
    descan_error: DescanErrorParameters
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
        flip_y=params_dict["flip_y"],
    )

    return Model(PointSource, ScanGrid, Descanner, Detector)
