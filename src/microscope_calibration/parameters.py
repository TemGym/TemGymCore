from typing_extensions import TypedDict
from numpy.typing import NDArray


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


def stem_model(test_params_dict):
    params_dict = test_params_dict

    #Create ray input z plane
    crossover_z = jnp.zeros((1))

    PointSource = comp.PointSource(z=crossover_z, semi_conv=params_dict['semi_conv'])

    ScanGrid = comp.ScanGrid(z=jnp.array([params_dict['defocus']]), 
                            scan_step=params_dict['scan_step'], 
                            scan_shape=params_dict['scan_shape'], 
                            scan_rotation=params_dict['scan_rotation'])

    Descanner = comp.Descanner(z=jnp.array([params_dict['defocus']]), 
                                            descan_error=params_dict['descan_error'], 
                                            scan_pos_x=0., 
                                            scan_pos_y=0.)

    Detector = comp.Detector(z=jnp.array([params_dict['camera_length'] + params_dict['defocus']]), 
                            det_shape=params_dict['det_shape'], 
                            det_pixel_size=params_dict['det_px_size'])

    model = [PointSource, ScanGrid, Descanner, Detector]

    return model
