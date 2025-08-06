from jaxgym.run import run_with_grads
from jaxgym.ray import PixelsRay
from microscope_calibration.model import DescanError, create_stem_model, Parameters4DSTEM


if __name__ == "__main__":
    params = Parameters4DSTEM(
        overfocus=0.001,
        scan_pixel_pitch=0.001,
        scan_cy=0.0,
        scan_cx=0.0,
        scan_shape=(11, 11),
        scan_rotation=0.0,
        camera_length=0.1,
        detector_pixel_pitch=0.001,
        detector_cy=0.0,
        detector_cx=0.0,
        detector_shape=(11, 11),
        semiconv=1e-12,
        flip_y=False,
        descan_error=DescanError(),
    )

    model = create_stem_model(params, (0., 0.))

    ray = PixelsRay(10, 30, 0., 0., 0., 0.)
    out_ray, grads = run_with_grads(
        ray,
        model,
        (model.detector.params.z,),
    )
    print("Output")
    print(f"\t{out_ray.item()}")
    for k, v in grads.items():
        print(k)
        print(f"\t{v.item() if v.size == 1 else v}")
