from jaxgym.run import run_with_grads
from jaxgym.ray import PixelsRay
import jaxgym.components as comp
from microscope_calibration.model import DescanError


if __name__ == "__main__":
    model = (
        (scan_grid := comp.ScanGrid(0., (0.001, 0.001), (64, 64), 0.)),
        (lens := comp.Lens(0.33, 0.1)),
        comp.Descanner(0.75, 0.2, 0.1, DescanError()),
        (det := comp.Detector(1., (0.001,) * 2, (128., 128.)))
    )

    ray = PixelsRay(10, 30, 0., 0., 0., 0.)
    out_ray, grads = run_with_grads(
        ray,
        model,
        (lens.params.focal_length, (det.params.z)),
    )
    print("Output")
    print(f"\t{out_ray.item()}")
    for k, v in grads.items():
        print(k)
        print(f"\t{v.item() if v.size == 1 else v}")
