"""A shared image calibration must preserve correlations between specimen beams."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/microscope_models"))
from condenser_voltage_calibration import combine_shared_calibration_covariance


def test_eliminated_calibration_covariance_matches_full_joint_fit():
    rng = np.random.default_rng(809)
    full = np.zeros((24, 10))
    covariances, responses = [], []
    information = 0.0
    for index in range(3):
        design, common = rng.normal(size=(8, 3)), rng.normal(size=8)
        full[index * 8 : (index + 1) * 8, index * 3 : (index + 1) * 3] = design
        full[index * 8 : (index + 1) * 8, -1] = common
        covariance = np.linalg.inv(design.T @ design)
        response = -covariance @ design.T @ common
        projected = common + design @ response
        information += projected @ projected
        covariances.append(covariance)
        responses.append(response)
    # Two beam values per setting; the third local value represents exposure.
    select = [0, 1, 3, 4, 6, 7, 9]
    expected = np.linalg.inv(full.T @ full)[np.ix_(select, select)]
    result = combine_shared_calibration_covariance(
        covariances, responses, information, beam_parameters=2
    )
    np.testing.assert_allclose(result, expected, atol=1e-13, rtol=1e-12)
    assert abs(result[0, 2]) > 1e-4  # Different beams are correlated.
