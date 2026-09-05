"""Measurement selection preserves physical image pairs and reported image counts."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples/microscope_models'))
from projector_adaptive_experiments import CalibrationMeasurements
from projector_dataset_reduction import subset_measurements, image_count


def test_subset_keeps_both_axes_and_their_covariance_without_duplicating_frames():
    controls = np.array([[1., .5, .6, .7, .8], [1.001, .8, .9, 1., 1.1]])
    pairs = dict(lens_controls=np.repeat(controls, 2, axis=0),
                 angle_changes_rad=np.tile(np.eye(2) * .006, (2, 1)),
                 displacements_pixels=np.arange(8).reshape(4, 2),
                 covariance_pixels=np.tile(np.array([[2., .3], [.3, 1.]]), (4, 1, 1)),
                 pair_labels=['first x', 'first y', 'second x', 'second y'])
    shifts = dict(pairs, position_changes_m=np.tile(np.eye(2)*20e-9, (2, 1)),
                  angle_changes_rad=np.zeros((4, 2)))
    measured = CalibrationMeasurements(controls, np.arange(6).reshape(2, 3),
                                      np.tile(np.eye(3), (2, 1, 1)), pairs, shifts)
    selected = subset_measurements(measurements=measured, lens_controls=controls[1:])
    assert image_count(measured) == 18
    assert image_count(selected) == 9
    assert selected.tilts['pair_labels'] == ['second x', 'second y']
    np.testing.assert_array_equal(selected.values, measured.values[1:])
    np.testing.assert_array_equal(selected.shifts['covariance_pixels'], shifts['covariance_pixels'][2:])
    np.testing.assert_array_equal(selected.shifts['position_changes_m'], shifts['position_changes_m'][2:])
    assert len(measured.controls) == 2
    with pytest.raises(ValueError, match='unique'):
        subset_measurements(measurements=measured, lens_controls=np.repeat(controls[:1], 2, axis=0))
    with pytest.raises(ValueError, match='no recorded'):
        subset_measurements(measurements=measured, lens_controls=controls[:1] + .1)
