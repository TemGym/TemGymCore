import numpy as np
from microscope_calibration.fitting import fit_descan_error_matrix
from microscope_calibration.model import DescannerErrorParameters
from microscope_calibration.generate import generate_dataset_from_image
from microscope_calibration.model import ModelParameters


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


def test_fit_descan_erro_generated_data(descan_error, camera_lengths=(0.5, 1.0, 1.5)):
    # Test that we can predict where a single pixel will end up after the descanner with scale error
    grid_shape = (12, 12)
    scan_step = (0.01, 0.01)
    det_px_size = (0.01, 0.01)

    test_image = np.zeros(grid_shape, dtype=np.uint8)
    test_image[0, 0] = 1

    descan_error = DescannerErrorParameters(*descan_error)

    params = ModelParameters(
        semi_conv=1e-4,
        defocus=0.0,
        camera_length=0.5,
        scan_shape=grid_shape,
        det_shape=grid_shape,
        scan_step=scan_step,
        det_px_size=det_px_size,
        scan_rotation=0.0,
        descan_error=descan_error,
    )

    for camera_length in camera_lengths:
        params.camera_length = camera_length
        fourdstem_array = generate_dataset_from_image(params, test_image)
