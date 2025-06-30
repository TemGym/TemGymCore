import numpy as np
import copy

from concurrent.futures import ProcessPoolExecutor

from microscope_calibration.fitting import fit_descan_error_matrix
from microscope_calibration.model import DescannerErrorParameters, ModelParameters
from microscope_calibration.generate import generate_dataset_from_image

import libertem.api as lt
from libertem.udf.sum import SumUDF
from libertem.udf.com import CoMUDF


def descan_error_params_random():
    # Randomize descan error parameters
    return DescannerErrorParameters(
        pxo_pxi=np.random.uniform(-2.0, 2.0),
        pxo_pyi=np.random.uniform(-2.0, 2.0),
        pyo_pxi=np.random.uniform(-2.0, -2.0),
        pyo_pyi=np.random.uniform(-2.0, 2.0),
        sxo_pxi=np.random.uniform(-2.0, 2.0),
        sxo_pyi=np.random.uniform(-2.0, 2.0),
        syo_pxi=np.random.uniform(-2.0, 2.0),
        syo_pyi=np.random.uniform(-2.0, 2.0),
        offpxi=np.random.uniform(-0.04, 0.04),
        offsxi=np.random.uniform(-0.04, 0.04),
        offpyi=np.random.uniform(-0.04, 0.04),
        offsyi=np.random.uniform(-0.04, 0.04)
    )


def test_fit_descan_error_matrix_random():

    ctx = lt.Context.make_with("inline")

    scan_shape = (41, 41)
    det_shape = (129, 129)
    scan_step = (0.0005, 0.0005)
    det_px_size = (0.01, 0.01)

    test_image = np.ones(scan_shape, dtype=np.uint8)

    descan_error = descan_error_params_random()

    params = ModelParameters(
        semi_conv=1e-4,
        defocus=0.01,
        camera_length=0.5,
        scan_shape=scan_shape,
        det_shape=det_shape,
        scan_step=scan_step,
        det_px_size=det_px_size,
        scan_rotation=0.0,
        descan_error=descan_error,
        flip_y=False,
    )

    datasets = {}
    clengths = (0.5, 1.0, 1.5)
    with ProcessPoolExecutor(max_workers=3) as p:
        futures = []
        for cl in clengths:
            _params = copy.deepcopy(params)
            _params["camera_length"] = cl
            f = p.submit(generate_dataset_from_image,
                        _params,
                        test_image,
                        method="linear",
                        sample_scale=1.0,
                        progress=False)
            futures.append(f)

        for f, cl in zip(futures, clengths):
            data = f.result()
            datasets[cl] = ctx.load("memory", data=data, num_partitions=1)

    com_r = {}
    com_udf = CoMUDF.with_params()
    for i, (cl, ds) in enumerate(datasets.items()):
        sum_res, com_res = ctx.run_udf(ds, [SumUDF(), com_udf])
        com_r[cl] = com_res

    err = fit_descan_error_matrix(params, com_r)

    for key in err._fields:
        fitted_val = getattr(err, key)
        known_val = getattr(descan_error, key)
        np.testing.assert_allclose(
            fitted_val, known_val, atol=3e-1,
            err_msg=f"Field {key} does not match: {fitted_val} vs {known_val}"
        )
