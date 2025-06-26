import jax
import matplotlib.pyplot as plt
import libertem.api as lt
from microscope_calibration.udf import ShiftedSumUDF

from microscope_calibration.model import (
    ModelParameters,
    DescannerErrorParameters,
)
from microscope_calibration.generate import generate_dataset_from_image


jax.config.update("jax_platform_name", "cpu")

ctx = lt.Context.make_with("inline")
sample_image = plt.imread(
    r"SilverFast_Resolution_Target_USAF_1951.png"
)[:, :, 0]
sample_image = sample_image[1:-1, 1:-1]

params = ModelParameters(
    semi_conv=0.1,
    defocus=0.01,  # Distance from the crossover to the sample
    camera_length=1.0,  # Distance from the point source to the detector
    scan_shape=(32, 32),  # YX!
    det_shape=(128, 128),  # YX!
    scan_step=(0.0001, 0.0001),  # YX!
    det_px_size=(0.004, 0.004),  # YX!
    scan_rotation=33.,
    descan_error=DescannerErrorParameters(),
    flip_y=False,
)


data = generate_dataset_from_image(
    params,
    sample_image,
    method="linear",
    progress=True,
)
ds = ctx.load("memory", data=data.copy())

print("Start")
try:
    udf = ShiftedSumUDF(
        model_parameters=params,
    )
    results = ctx.run_udf(ds, udf, progress=True)
finally:
    pass
print("Done")
