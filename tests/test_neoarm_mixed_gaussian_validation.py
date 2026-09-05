from dataclasses import replace
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/lens_inversion"))
from neoarm_mixed_gaussian_validation import (
    default_source, source_modes, physical_emittances, imaging_map,
    render_mixed_counts, detector_quadrature,
)
from temgym_core.gaussian_image_likelihood import (
    make_gaussian_detector_operator, validate_phase_space_covariance,
)
from temgym_core.microscope_inversion import hard_edge_solenoid_matrix
from temgym_core.neoarm_calibration import DetectorGeometry

jax.config.update("jax_enable_x64", True)


def test_pure_mode_has_minimum_emittance_and_mixture_is_physically_valid():
    source = default_source()
    epsilon = source.wavelength_m/(4*np.pi)
    np.testing.assert_allclose(physical_emittances(source.pure_covariance, source.wavelength_m), epsilon, rtol=1e-12)
    assert np.all(physical_emittances(source.total_covariance, source.wavelength_m) > epsilon)
    validate_phase_space_covariance(source.total_covariance, source.wavelength_m)
    assert np.count_nonzero(np.tril(source.centre_covariance)) == 10


def test_positive_covariance_is_not_automatically_a_physical_electron_beam():
    source = default_source()
    illegal = np.diag(np.array([60e-9, .5e-6, 60e-9, .5e-6])**2)
    with pytest.raises(ValueError, match="uncertainty"):
        physical_emittances(illegal, source.wavelength_m)
    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(source, centre_covariance=-source.centre_covariance)


def test_positive_four_dimensional_quadrature_preserves_source_mean_and_covariance():
    source = default_source()
    modes, weights = source_modes(source, 3)
    centres = np.stack((modes.x, modes.dx, modes.y, modes.dy), axis=-1)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-14)
    assert np.all(weights > 0)
    mean = weights@centres
    np.testing.assert_allclose(mean, source.mean_state, rtol=1e-12, atol=1e-21)
    difference = centres-mean
    covariance = (difference.T*weights)@difference
    scales = np.sqrt(np.diag(source.centre_covariance))
    np.testing.assert_allclose(covariance/scales[:, None]/scales[None, :],
                               source.centre_covariance/scales[:, None]/scales[None, :], atol=1e-13)


@pytest.mark.parametrize("defocus", [-100e-6, 0., 100e-6])
def test_incoherent_gaussian_beam_pixel_images_match_covariance_transport(defocus):
    source = default_source()
    detector = DetectorGeometry(shape_yx=(19, 27), pixel_size_yx_m=(55e-6, 45e-6),
                                affine_xy=np.array([[1.01, .08], [-.03, .98]]),
                                centre_xy_m=np.array([.15e-3, -.09e-3]), mtf_sigma_px=.8, z_m=1.)
    matrix = imaging_map(12000., .43, defocus)
    position = matrix[[0, 2]]
    expected = np.asarray(make_gaussian_detector_operator(detector)(
        position@source.mean_state, position@source.total_covariance@position.T, 2e7))
    actual = np.asarray(render_mixed_counts(source, matrix, detector, hermite_order=7,
                                             require_gpu=False, pixel_order=4, pixel_tile=128))
    assert np.sqrt(np.mean(((actual-expected)/np.sqrt(expected+.36))**2)) < .01


def test_arbitrary_solenoid_map_and_identical_incoherent_modes():
    source = replace(default_source(), centre_covariance=np.zeros((4, 4)))
    detector = DetectorGeometry(shape_yx=(17, 23), pixel_size_yx_m=(55e-6, 55e-6), z_m=1.)
    matrix = imaging_map(10000., .31, 0.)@np.asarray(hard_edge_solenoid_matrix(.43, .01))
    position = matrix[[0, 2]]
    expected = np.asarray(make_gaussian_detector_operator(detector)(
        position@source.mean_state, position@source.total_covariance@position.T, 2e7))
    single = np.asarray(render_mixed_counts(source, matrix, detector, hermite_order=1, require_gpu=False))
    many = np.asarray(render_mixed_counts(source, matrix, detector, hermite_order=3, require_gpu=False))
    np.testing.assert_allclose(single, many, rtol=2e-13, atol=1e-9)
    np.testing.assert_allclose(single, expected, rtol=1e-9, atol=1e-5)


def test_general_mixed_source_through_finite_solenoid_map():
    source = default_source()
    detector = DetectorGeometry(shape_yx=(17, 23), pixel_size_yx_m=(55e-6, 55e-6),
                                mtf_sigma_px=.65, z_m=1.)
    matrix = imaging_map(10000., -.27, 100e-6)@np.asarray(hard_edge_solenoid_matrix(.43, .01))
    response = matrix[[0, 2]]
    expected = np.asarray(make_gaussian_detector_operator(detector)(
        response@source.mean_state, response@source.total_covariance@response.T, 2e7))
    actual = np.asarray(render_mixed_counts(source, matrix, detector, hermite_order=9,
                                             require_gpu=False, pixel_tile=128))
    assert np.sqrt(np.mean(((actual-expected)/np.sqrt(expected+.36))**2)) < .02


def test_pixel_quadrature_integrates_optical_area_and_mtf_halo():
    detector = DetectorGeometry(shape_yx=(17, 23), pixel_size_yx_m=(3., 2.),
                                affine_xy=np.array([[1.1, .2], [-.3, .9]]),
                                mtf_sigma_px=.8, z_m=1.)
    coordinates, weights, shape, radius = detector_quadrature(detector, 4)
    assert radius == 4 and shape == (25, 31)
    assert coordinates.shape == (25*31, 16, 2)
    assert weights.sum() == pytest.approx(6/abs(np.linalg.det(detector.affine_xy)))


def test_non_symplectic_map_rejected():
    with pytest.raises(ValueError, match="symplectic"):
        render_mixed_counts(default_source(), np.eye(4)*2,
                            DetectorGeometry(shape_yx=(17, 17), z_m=1.), hermite_order=3, require_gpu=False)
