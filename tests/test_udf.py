import numpy as np
import libertem.api as lt

from microscope_calibration.model import ModelParameters
from microscope_calibration.udf import ShiftedSumUDF


def test_functional():
    data = np.zeros((3, 4, 16, 16), dtype=np.float32)
    cy, cx = np.asarray(data.shape[-2:]) // 2
    data[0, 0, cy, cx] = 1
    data[0, -1, cy, cx] = 2
    data[-1, -1, cy, cx] = 3
    data[-1, 0, cy, cx] = 4

    ctx = lt.Context.make_with("inline")
    ds = ctx.load("memory", data=data, num_partitions=1)

    parameters = ModelParameters(
        semi_conv=0.000001,
        defocus=0.001,
        camera_length=1.,
        scan_step=(0.001, 0.001),
        det_px_size=(0.001, 0.001),
        scan_rotation=0.,
        descan_error=np.zeros((12,)),
        flip_y=False,
        scan_shape=ds.shape.nav,
        det_shape=ds.shape.sig,
    )
    udf = ShiftedSumUDF(parameters)
    res = ctx.run_udf(ds, udf)
    s_sum = res["shifted_sum"].data
    np.testing.assert_allclose(s_sum, data.sum(axis=(-2, -1)))
