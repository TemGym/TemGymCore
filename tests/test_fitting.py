import numpy as np
import copy

from microscope_calibration.fitting import fit_descan_error_matrix
from microscope_calibration.model import DescanError, ModelParameters
from microscope_calibration.generate import generate_dataset_from_image

import pytest

try:
    import libertem.api as lt
    from libertem.udf.sum import SumUDF
    from libertem.udf.com import CoMUDF
except ImportError:
    pytest.skip("libertem not installed, skipping tests", allow_module_level=True)


def descan_error_params_random():
    # Randomize descan error parameters
    return DescanError(
        pxo_pxi=np.random.uniform(-0.2, 0.2),
        pxo_pyi=np.random.uniform(-0.2, 0.2),
        pyo_pxi=np.random.uniform(-0.2, -0.2),
        pyo_pyi=np.random.uniform(-0.2, 0.2),
        sxo_pxi=np.random.uniform(-0.2, 0.2),
        sxo_pyi=np.random.uniform(-0.2, 0.2),
        syo_pxi=np.random.uniform(-0.2, 0.2),
        syo_pyi=np.random.uniform(-0.2, 0.2),
        offpxi=np.random.uniform(-0.001, 0.001),
        offsxi=np.random.uniform(-0.001, 0.001),
        offpyi=np.random.uniform(-0.001, 0.001),
        offsyi=np.random.uniform(-0.001, 0.001)
    )


def test_fit_descan_error_matrix():

    ctx = lt.Context.make_with("inline")

    scan_shape = (19, 19)
    det_shape = (29, 29)
    scan_step = (0.0005, 0.0005)
    det_px_size = (0.01, 0.01)

    test_image = np.ones(scan_shape, dtype=np.uint8)

    descan_error = DescanError(
        pxo_pxi=3,
        pxo_pyi=-3.,
        pyo_pxi=1.,
        pyo_pyi=0.,
        sxo_pxi=2.,
        sxo_pyi=-2.,
        syo_pxi=3.,
        syo_pyi=-1.,
        offpxi=0.,
        offsxi=0.,
        offpyi=0.,
        offsyi=0.,
    )

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
    for cl in clengths:
        _params = copy.deepcopy(params)
        _params["camera_length"] = cl
        data = generate_dataset_from_image(
            _params,
            test_image,
            method="linear",
            sample_scale=1.0,
            progress=False
        )
        datasets[cl] = ctx.load("memory", data=data, num_partitions=1)

    com_r = {}
    com_udf = CoMUDF.with_params()
    for i, (cl, ds) in enumerate(datasets.items()):
        sum_res, com_res = ctx.run_udf(ds, [SumUDF(), com_udf])
        com_r[cl] = com_res

    # Uncomment the following lines to visualize the results
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, len(datasets), figsize=(12, 6))
    # for i, (cl, ds) in enumerate(datasets.items()):
    #     sum_res, com_res = ctx.run_udf(ds, [SumUDF(), com_udf])
    #     com_r[cl] = com_res
    #     axs[0, i].imshow(sum_res["intensity"].data, cmap="gray")
    #     axs[0, i].set_title(f"Sum cl={cl}")

    # for i, (cl, ds) in enumerate(datasets.items()):
    #     axs[1, i].imshow(com_r[cl]["magnitude"].data)
    #     axs[1, i].set_title(f"CoM shift cl={cl}")

    # fig.savefig('com_results.png')

    err = fit_descan_error_matrix(params, com_r)

    for known_val, fitted_val in zip(descan_error, err):
        np.testing.assert_allclose(
            fitted_val, known_val, atol=3e-1,
            err_msg=f"Field does not match: {fitted_val} vs {known_val}"
        )
