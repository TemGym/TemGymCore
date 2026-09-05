from pathlib import Path
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/lens_inversion'))
from neoarm_measured_specimen import detector_footprint, require_view_coverage
from neoarm_mixed_gaussian_validation import imaging_map
from temgym_core.multiscale_sample import SampleLevel2D, SampleView
from temgym_core.neoarm_calibration import DetectorGeometry

jax.config.update('jax_enable_x64', True)


def test_measured_specimen_footprint_uses_edges_rotation_and_detector_affine():
    matrix = imaging_map(40000., 1.44, 0.)
    detector = DetectorGeometry(shape_yx=(32, 48), pixel_size_yx_m=(55e-6, 45e-6),
        centre_xy_m=(1e-3, -2e-3), affine_xy=np.array([[1.1, .12], [-.08, .9]]))
    specimen = detector_footprint(matrix, detector)
    optical = specimen@matrix[np.ix_([0, 2], [0, 2])].T
    physical = (optical-detector.centre_xy_m)@detector.affine_xy.T
    expected = np.array([[-48*45e-6/2, -32*55e-6/2], [-48*45e-6/2, 32*55e-6/2],
                         [48*45e-6/2, -32*55e-6/2], [48*45e-6/2, 32*55e-6/2]])
    np.testing.assert_allclose(physical, expected, rtol=1e-13, atol=1e-17)


def test_crop_coverage_does_not_confuse_missing_tile_with_vacuum():
    x, y = np.linspace(-8e-9, 8e-9, 17), np.linspace(-5e-9, 5e-9, 11)
    level = SampleLevel2D(np.zeros((11, 17)), np.zeros((11, 17)), x, y)
    view = SampleView.from_level(level, centre_xy_m=(2e-9, -1e-9), rotation_rad=.4)
    assert require_view_coverage(view, [[2e-9, -1e-9]], label='test') > 0
    with pytest.raises(ValueError, match='does not cover'):
        require_view_coverage(view, [[20e-9, -1e-9]], label='test')


def test_footprint_rejects_nonfinite_matrix():
    with pytest.raises(ValueError, match='finite'):
        detector_footprint(np.full((4, 4), np.nan), DetectorGeometry())
