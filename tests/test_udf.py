import numpy as np
import pytest
try:
    import libertem.api as lt
except ImportError:
    pytest.skip("libertem not installed, skipping tests", allow_module_level=True)

from microscope_calibration.model import Parameters4DSTEM, DescanError
from microscope_calibration.udf import ShiftedSumUDF


def test_functional():
    data = np.zeros((11, 11, 16, 16), dtype=np.float32)
    cy, cx = np.asarray(data.shape[-2:]) // 2
    data[0, 0, cy, cx] = 1
    data[0, -1, cy, cx] = 2
    data[-1, -1, cy, cx] = 3
    data[-1, 0, cy, cx] = 4

    ctx = lt.Context.make_with("inline")
    ds = ctx.load("memory", data=data, num_partitions=1)

    descanner_error_params = DescanError()
    parameters = Parameters4DSTEM(
        overfocus=0.0,
        scan_pixel_pitch=0.01,
        scan_cy=0.0,
        scan_cx=0.0,
        scan_shape=(11, 11),
        scan_rotation=0.0,
        camera_length=0.5,
        detector_pixel_pitch=0.01,
        detector_cy=0.0,
        detector_cx=0.0,
        detector_shape=(16, 16),
        semiconv=1,
        flip_y=False,
        descan_error=descanner_error_params,
    )

    udf = ShiftedSumUDF(parameters)
    res = ctx.run_udf(ds, udf)
    s_sum = res["shifted_sum"].data
    np.testing.assert_allclose(s_sum, data.sum(axis=(-2, -1)))
