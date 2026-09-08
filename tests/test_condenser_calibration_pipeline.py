"""Check information content, physical transport, and the truth-free fitting boundary."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

EXAMPLES = Path(__file__).parents[1] / "src/temgym_core/calibration/presets"
jax.config.update("jax_enable_x64", True)
from temgym_core.calibration.condenser_images import make_image_operator
from temgym_core.calibration.projector_model import matrix5 as projector_matrix5


def test_calibrated_views_are_real_projector_maps_with_fixed_objective():
    record = json.loads((EXAMPLES / "condenser_projector_views.json").read_text())
    for view in record["views"]:
        assert view["controls"][0] == 1.0
        expected = projector_matrix5(
            np.asarray(record["calibrated_projector_parameters"]),
            np.asarray(view["controls"]),
        )
        np.testing.assert_allclose(view["matrix"], expected, rtol=2e-12, atol=1e-08)


def test_image_operator_conserves_dose_on_a_large_detector_and_rejects_astigmatism():
    template = {
        "axis_m": np.array([-5e-09, 5e-09]),
        "intensity": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    detector = {
        "shape_yx": [64, 64],
        "sample_kernel_sigma_m": 1e-09,
        "pixel_pitch_m": 1e-06,
        "psf_sigma_pixels": 0.5,
    }
    matrices = np.eye(5)[None].copy()
    matrices[0, :2, :2] *= 1000
    operator = make_image_operator(
        projector_matrices=matrices, template=template, detector=detector
    )
    beam = np.array(
        [0, 0, 0, 0, np.log(1e-08), 0, np.log(1e-05), np.log(100000.0), np.log(2)]
    )
    frame = np.asarray(operator(beam))[0]
    assert (frame - 2).sum() == pytest.approx(100000.0, rel=1e-08)
    changed = matrices.copy()
    changed[0, :2, :2] *= 1.01
    rebuilt = make_image_operator(
        projector_matrices=changed, template=template, detector=detector
    )
    np.testing.assert_allclose(operator(beam, changed), rebuilt(beam), rtol=1e-12)
    matrices[0, 0, 0] *= 2
    with pytest.raises(ValueError, match="round"):
        make_image_operator(
            projector_matrices=matrices, template=template, detector=detector
        )
