from jaxgym.run import run_with_grads
from jaxgym.ray import Ray
from jaxgym import components
from microscope_calibration.model import DescanErrorParameters
import microscope_calibration.components as comp


if __name__ == "__main__":
    model = (
        comp.PointSource(0., 0.01),
        (lens := components.Lens(0.5, 0.1)),
        (desc := comp.Descanner(0.75, 0.2, 0.1, DescanErrorParameters())),
        (det := comp.Detector(1., (0.001,) * 2, (128., 128.))),
    )

    ray = Ray(0.2, 0., 0., 0., 0., 0.)
    val, grads = run_with_grads(
        ray,
        model,
        (ray, lens.params.focal_length, (det.params.det_shape[1])),
    )
    for k, v in grads.items():
        print(k)
        print(f"\t{v.item() if v.size == 1 else v}")
