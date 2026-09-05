"""Contracts for measured-geometry condenser moment reconnaissance."""

import importlib.util
from pathlib import Path
import sys

import jax
import numpy as np


EXAMPLES = Path(__file__).parents[1] / "examples/lens_inversion"
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location("neoarm_measured_information_audit",
                                             EXAMPLES / "neoarm_measured_information_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
jax.config.update("jax_enable_x64", True)


def test_source_is_physical_without_using_archived_pullback_covariance():
    from temgym_core.gaussian_image_likelihood import validate_phase_space_covariance
    covariance = AUDIT.source_covariance()
    wavelength = float(AUDIT.energy2wavelength(200000.))
    validate_phase_space_covariance(covariance, wavelength)
    np.testing.assert_allclose(AUDIT.symplectic_emittances(covariance), [2.5e-13, 2.7e-13])


def test_column_uses_actual_diagram_aperture_distance_and_cl3_name():
    column = AUDIT.measured_column()
    anchors = AUDIT.geometry_anchors_from_notebook(AUDIT.DIAGRAM)
    assert column.sample_z_m == 0
    assert "CL3" in column.channel_names
    np.testing.assert_allclose(anchors["C-aperture"], -.42083333333333334)
    cl3 = next(lens for lens in column.lenses if lens.name == "CL3")
    np.testing.assert_allclose(anchors["C-aperture"] - cl3.exit_z, .0008333333333333)


def test_source_plane_shift_is_covariance_transport_gauge():
    column = AUDIT.measured_column()
    parameters = column.nominal_parameters
    covariance = AUDIT.source_covariance()
    controls = AUDIT.control_for_q(column, np.array((.285, .22522727272727272, .34984826301857863)))
    first = np.asarray(column.matrix(parameters, controls, segment="source_to_sample"))
    shift = .001
    drift = np.eye(4); drift[0, 1] = shift; drift[2, 3] = shift
    changed = parameters.copy(); changed[0] += shift
    second = np.asarray(column.matrix(changed, controls, segment="source_to_sample"))
    np.testing.assert_allclose(second @ drift, first, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(second @ (drift @ covariance @ drift.T) @ second.T,
                               first @ covariance @ first.T, rtol=1e-8, atol=1e-22)


def test_capture_probability_is_not_silently_normalized_to_one():
    covariances = np.array([np.eye(2) * (30e-9)**2, np.eye(2) * (200e-9)**2])
    capture = AUDIT.capture_lower_bound(covariances)
    assert capture[0] > .999
    assert capture[1] < .5


def test_numerical_rank_is_not_confused_with_relative_reporting_threshold():
    matrix = np.pad(np.diag(np.r_[1e8, np.ones(10), .01]), ((0, 2), (0, 1)))
    matrix[-1, -1] = 1.
    result = AUDIT.profile_measured_targets(matrix)
    assert result["numerical_target_rank"] == 12
    assert result["relative_reporting_rank"] < 12
    np.testing.assert_allclose(result["numerically_full_rank_maximum_scaled_sigma"], 100.)


def test_search_finds_capture_compatible_measured_geometry_states():
    candidates = AUDIT.search_illumination_candidates(grid_points=21)
    assert len(candidates["q"]) >= 16
    assert np.all(candidates["minimum_capture_probability"] >= .995)
    assert np.all(candidates["aperture_eigen_rms_m"] >= 3e-6)
    assert np.all(candidates["aperture_eigen_rms_m"] <= 30e-6)
    selected = AUDIT.select_spread_candidates(candidates, 16)
    assert len(np.unique(selected)) == 16
