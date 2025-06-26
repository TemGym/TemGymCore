import numpy as np
from scipy.optimize import curve_fit

from microscope_calibration.model import DescannerErrorParameters


def descan_model_x(vars, pxo_pxi, pxo_pyi, sxo_pxi, sxo_pyi, offpxi, offsxi):
    spx, spy, B = vars
    return (spx * pxo_pxi + spy * pxo_pyi + offpxi +
            B * (spx * sxo_pxi + spy * sxo_pyi + offsxi))


def descan_model_y(vars, pyo_pxi, pyo_pyi, syo_pxi, syo_pyi, offpyi, offsyi):
    spx, spy, B = vars
    return (spx * pyo_pxi + spy * pyo_pyi + offpyi +
            B * (spx * syo_pxi + spy * syo_pyi + offsyi))


def fit_descan_error_matrix(scan_coords, det_coords, camera_lengths, num_samples=100):

    indices = np.random.choice(camera_lengths.size, num_samples, replace=False)

    popt_x, pcov_x = curve_fit(
        descan_model_x,
        (scan_coords[:, 0][indices], scan_coords[:, 1][indices], camera_lengths[indices]),
        det_coords[:, 0][indices],
        p0=np.zeros(6),
    )
    pxo_pxi, pxo_pyi, sxo_pxi, sxo_pyi, offpxi, offsxi = popt_x

    popt_y, pcov_y = curve_fit(
        descan_model_y,
        (scan_coords[:, 0][indices], scan_coords[:, 1][indices], camera_lengths[indices]),
        det_coords[:, 1][indices],
        p0=np.zeros(6),
    )
    pyo_pxi, pyo_pyi, syo_pxi, syo_pyi, offpyi, offsyi = popt_y

    return DescannerErrorParameters(pxo_pxi=pxo_pxi,
                                    pxo_pyi=pxo_pyi,
                                    pyo_pxi=pyo_pxi,
                                    pyo_pyi=pyo_pyi,
                                    sxo_pxi=sxo_pxi,
                                    sxo_pyi=sxo_pyi,
                                    syo_pxi=syo_pxi,
                                    syo_pyi=syo_pyi,
                                    offpxi=offpxi,
                                    offpyi=offpyi,
                                    offsxi=offsxi,
                                    offsyi=offsyi)
