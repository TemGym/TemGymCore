"""Acquisition roles and data-only initialization, without CPU optical renders."""
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]/"examples/lens_inversion"
if str(EXAMPLES) not in sys.path: sys.path.insert(0, str(EXAMPLES))

from neoarm_condenser_extension_images import validate_roles, data_only_image_initializer, fit_data_only_image
from temgym_core.neoarm_calibration import DetectorGeometry


def test_roles_remain_disjoint_and_complete():
    controls = np.zeros((3, 7))
    validate_roles(controls, np.array([True, True, False]), np.array([False, False, True]))
    with pytest.raises(ValueError, match="exactly one"):
        validate_roles(controls, np.array([True, True, False]), np.array([True, False, True]))
    with pytest.raises(ValueError, match="exactly one"):
        validate_roles(controls, np.array([True, False, False]), np.array([False, False, True]))
    with pytest.raises(ValueError, match="seven-channel"):
        validate_roles(np.zeros((3, 6)), np.ones(3, bool), np.zeros(3, bool))


def arbitrary_test_image():
    # A literal pixel patch, not a Gaussian/optical forward simulation.
    image = np.zeros((64, 64))
    image[28:33, 29:36] = 120.
    image[29:32, 30:35] += 80.
    return image


def test_initializer_tracks_data_translation_and_dose_not_external_truth():
    detector = DetectorGeometry(shape_yx=(64, 64), mtf_sigma_px=.65)
    first = arbitrary_test_image(); second = np.roll(np.roll(first, -4, axis=0), 6, axis=1)*3
    p = data_only_image_initializer(first, detector)
    q = data_only_image_initializer(second, detector)
    np.testing.assert_allclose(q[:2]-p[:2], [6., -4.], atol=1e-14)
    np.testing.assert_allclose(q[2:5], p[2:5], atol=1e-14)
    np.testing.assert_allclose(q[5]-p[5], np.log(3), atol=1e-14)


def test_unilluminated_signed_pixel_noise_not_used_as_second_moment_signal():
    detector = DetectorGeometry(shape_yx=(64, 64), mtf_sigma_px=.65)
    image = arbitrary_test_image(); noisy = image.copy()
    noisy[0, 0] = 2.; noisy[-1, -1] = -2.
    np.testing.assert_allclose(data_only_image_initializer(noisy, detector), data_only_image_initializer(image, detector), atol=1e-14)


def test_initializer_rejects_invalid_or_missing_signal():
    detector = DetectorGeometry(shape_yx=(64, 64))
    with pytest.raises(ValueError): data_only_image_initializer(np.zeros((64, 64)), detector)
    with pytest.raises(ValueError): data_only_image_initializer(np.full((64, 64), np.nan), detector)
    with pytest.raises(ValueError): data_only_image_initializer(np.ones((32, 32)), detector)


def test_data_only_fit_requires_gpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="GPU"):
        fit_data_only_image(arbitrary_test_image(), DetectorGeometry(shape_yx=(64, 64)))
