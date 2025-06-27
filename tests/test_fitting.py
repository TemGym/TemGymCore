import numpy as np
import copy
import jax.numpy as jnp

import microscope_calibration.components as comp
from concurrent.futures import ProcessPoolExecutor

from microscope_calibration.fitting import fit_descan_error_matrix
from microscope_calibration.model import DescannerErrorParameters
from microscope_calibration.generate import generate_dataset_from_image
from microscope_calibration.model import ModelParameters

import libertem.api as lt
from libertem.udf.sum import SumUDF
from libertem.udf.com import CoMUDF


def descan_error_scale():
    # Example descan error parameters
    return DescannerErrorParameters(
        pxo_pxi=1.0, pxo_pyi=0.0,
        sxo_pxi=0.0, sxo_pyi=0.0,
        offpxi=0.0, offsxi=0.0,
        pyo_pxi=0.0, pyo_pyi=1.0,
        syo_pxi=0.0, syo_pyi=0.0,
        offpyi=0.0, offsyi=0.0,
    )


def test_fit_descan_error_matrix_perfect_data():
    N = 200
    scan_coords = np.random.uniform(-1, 1, (N, 2))
    camera_lengths = np.random.uniform(0.5, 2.0, N)

    # true model parameters for x and y in one NamedTuple
    true_params = DescannerErrorParameters(
        pxo_pxi=1.2, pxo_pyi=-0.7,
        sxo_pxi=0.1, sxo_pyi=-0.2,
        offpxi=0.05, offsxi=-0.03,
        pyo_pxi=-1.5, pyo_pyi=2.3,
        syo_pxi=0.15, syo_pyi=-0.25,
        offpyi=-0.02, offsyi=0.04,
    )

    # build perfect detector coordinates
    spx = scan_coords[:, 0]
    spy = scan_coords[:, 1]
    B = camera_lengths

    det_x = (
        true_params.pxo_pxi * spx + true_params.pxo_pyi * spy + true_params.offpxi
        + B * (spx * true_params.sxo_pxi + spy * true_params.sxo_pyi + true_params.offsxi)
    )
    det_y = (
        true_params.pyo_pxi * spx + true_params.pyo_pyi * spy + true_params.offpyi
        + B * (spx * true_params.syo_pxi + spy * true_params.syo_pyi + true_params.offsyi)
    )
    det_coords = np.vstack([det_x, det_y]).T

    # fit error matrix
    err = fit_descan_error_matrix(scan_coords, det_coords, camera_lengths,
                                  num_samples=N)

    # verify recovered parameters close to true values
    atol = 1e-2
    # check x-related fields
    for field in ('pxo_pxi', 'pxo_pyi', 'sxo_pxi', 'sxo_pyi', 'offpxi', 'offsxi'):
        rec = getattr(err, field)
        exp = getattr(true_params, field)
        assert np.isclose(rec, exp, atol=atol), f"{field}: {rec} vs {exp}"
    # check y-related fields
    for field in ('pyo_pxi', 'pyo_pyi', 'syo_pxi', 'syo_pyi', 'offpyi', 'offsyi'):
        rec = getattr(err, field)
        exp = getattr(true_params, field)
        assert np.isclose(rec, exp, atol=atol), f"{field}: {rec} vs {exp}"


# def test_fit_descan_error_generated_data(camera_lengths=(0.5, 1.0, 1.5)):
#     # That that we can fit the descan error parameters
#     # generated from propagating a single pixel with
#     # descan error.

#     ctx = lt.Context.make_with("inline")

#     grid_shape = (12, 12)
#     scan_step = (0.01, 0.01)
#     det_px_size = (0.01, 0.01)

#     test_image = np.zeros(grid_shape, dtype=np.uint8)
#     test_image[0, 0] = 1

#     descan_error = descan_error_scale()

#     params = ModelParameters(
#         semi_conv=1e-4,
#         defocus=0.0,
#         camera_length=0.5,
#         scan_shape=grid_shape,
#         det_shape=grid_shape,
#         scan_step=scan_step,
#         det_px_size=det_px_size,
#         scan_rotation=0.0,
#         descan_error=descan_error,
#     )

#     datasets = {}
#     clengths = camera_lengths
#     with ProcessPoolExecutor(max_workers=3) as p:
#         futures = []
#         for cl in clengths:
#             _params = copy.deepcopy(params)
#             _params["camera_length"] = cl
#             f = p.submit(generate_dataset_from_image, _params, test_image, method="linear",
#                          sample_scale=1.0, progress=False)

#             futures.append(f)

#         for f, cl in zip(futures, clengths):
#             data = f.result()
#             datasets[cl] = data

#     com_r = {}
#     com_udf = CoMUDF.with_params()
#     for i, (cl, ds) in enumerate(datasets.items()):
#         sum_res, com_res = ctx.run_udf(ds, [SumUDF(), com_udf])
#         com_r[cl] = com_res

#     scan_coords = []
#     det_coords = []
#     b_vals = []
#     for camera_length, ds in datasets.items():
#         ScanGrid = comp.ScanGrid(
#             z=jnp.array(params['defocus']),
#             scan_step=params['scan_step'],
#             scan_shape=ds.shape.nav.to_tuple(),
#             scan_rotation=params['scan_rotation'],
#         )
#         scan_coords.append(ScanGrid.coords)
#         Detector = comp.Detector(
#             z=jnp.array(camera_length),
#             det_shape=ds.shape.sig.to_tuple(),
#             det_pixel_size=params['det_px_size'],
#             flip_y=params['flip_y'],
#         )
#         yx_px_det = com_r[camera_length]["raw_com"].data.reshape(-1, 2)
#         det_coords.append(np.stack(Detector.pixels_to_metres(yx_px_det.T), axis=1))
#         b_vals.append(camera_length - params['defocus'])

#     bvals = np.concatenate(
#         tuple(np.full((c.shape[0],), b) for b, c in zip(b_vals, scan_coords))
#     )
#     scan_coords = np.concatenate(scan_coords, axis=0)
#     det_coords = np.concatenate(det_coords, axis=0)
#     bvals.shape

#     print(det_coords)
