import numpy as np
from scipy.optimize import curve_fit

from microscope_calibration.model import DescannerErrorParameters
from microscope_calibration.model import create_stem_model


def descan_model_x(vars, pxo_pxi, pxo_pyi, sxo_pxi, sxo_pyi, offpxi, offsxi):
    spx, spy, B = vars
    return (spx * pxo_pxi + spy * pxo_pyi + offpxi) + \
        B * (spx * sxo_pxi + spy * sxo_pyi + offsxi)


def descan_model_y(vars, pyo_pxi, pyo_pyi, syo_pxi, syo_pyi, offpyi, offsyi):
    spx, spy, B = vars
    return (spx * pyo_pxi + spy * pyo_pyi + offpyi) + \
        B * (spx * syo_pxi + spy * syo_pyi + offsyi)


def fit_descan_error_matrix(model_params, com_dict, num_samples=100):

    scan_coords = []
    det_coords = []
    camera_lengths = []

    for camera_length in com_dict:

        PointSource, ScanGrid, Descanner, Detector = create_stem_model(model_params)

        scan_coords.append(ScanGrid.coords)
        yx_px_det = com_dict[camera_length]["raw_com"].data.reshape(-1, 2)
        det_coords.append(np.stack(Detector.pixels_to_metres(yx_px_det.T), axis=1))
        camera_lengths.append(camera_length)

    # Make an array of camera lengths
    # that matches the number of scan coordinates
    # and detector coordinates

    camera_lengths = np.concatenate(
        tuple(np.full((c.shape[0],), b) for b, c in zip(camera_lengths, scan_coords))
    )
    scan_coords = np.concatenate(scan_coords, axis=0)
    det_coords = np.concatenate(det_coords, axis=0)

    mask = ~(np.all(det_coords == 0.0, axis=1))
    camera_lengths = camera_lengths[mask]
    scan_coords = scan_coords[mask]
    det_coords = det_coords[mask]

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
