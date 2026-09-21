"""Projector matrix units and image displacement measurements."""

import json

import numpy as np
import pytest

from temgym_core.calibration import projector_model as pipeline
from temgym_core.calibration.workflow import PRESETS


def test_zero_excitation_is_full_axial_drift_in_grouped_coordinates():
    p = pipeline.initial_guess()
    p[11] = 0.0
    actual = np.asarray(pipeline.matrix5(p, np.zeros(4)))
    distance = float(pipeline.decode_geometry(p)[-1])
    expected = np.eye(5)
    expected[0, 2] = expected[1, 3] = distance
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_objective_control_is_independent_and_legacy_controls_are_unchanged():
    p = pipeline.initial_guess()
    projector_controls = np.array([0.9, 0.8, 0.7, 0.9])
    legacy = np.asarray(pipeline.matrix5(p, projector_controls))
    np.testing.assert_allclose(
        pipeline.matrix5(p, np.r_[1.0, projector_controls]), legacy, atol=0, rtol=0
    )
    changed = p.copy()
    changed[11] *= 1.002
    np.testing.assert_allclose(
        pipeline.matrix5(p, np.r_[1.002, projector_controls]),
        pipeline.matrix5(changed, projector_controls),
        rtol=1e-13,
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="four projector controls or five"):
        pipeline.matrix5(p, np.ones(3))


def test_bounded_position_coordinates_preserve_the_column_and_prevent_overlap():
    priors = json.loads((PRESETS / "projector_position_priors.json").read_text())
    encode, decode, lower, upper = pipeline.position_prior_coordinates(priors)
    p = pipeline.initial_guess()
    np.testing.assert_allclose(decode(encode(p)), p, atol=1e-13, rtol=0)
    bounds = np.asarray(priors["position_bounds_mm"])
    length_bounds = np.asarray(priors["length_bounds_mm"])
    rng = np.random.default_rng(61)
    for _ in range(30):
        u = np.asarray(encode(p)).copy()
        # Corners, including maximum lens thicknesses, are more stringent than
        # random interior points for gap positivity.
        u[:6] = np.where(rng.integers(0, 2, 6), lower, upper)
        u[6:11] = np.log(length_bounds[:, 1] / 1000)
        gaps, lengths, centres, detector = map(
            np.asarray, pipeline.decode_geometry(decode(u))
        )
        assert np.all(gaps > 0)
        positions = np.r_[centres, detector] * 1000
        assert np.all(positions >= bounds[:, 0] - 1e-10)
        assert np.all(positions <= bounds[:, 1] + 1e-10)
        np.testing.assert_allclose(lengths * 1000, length_bounds[:, 1], atol=1e-12)


def test_inconsistent_position_priors_are_rejected():
    priors = json.loads((PRESETS / "projector_position_priors.json").read_text())
    priors["position_bounds_mm"][2][0] = 240.0
    with pytest.raises(ValueError, match="non-overlapping"):
        pipeline.position_prior_coordinates(priors)


def test_readable_model_parameters_round_trip(monkeypatch):
    from temgym_core.calibration.projector_model import (
        model_to_parameters,
        parameters_to_model,
    )

    parameters = pipeline.initial_guess()
    model = parameters_to_model(parameters)
    assert set(model["lens_positions_mm"]) == set(pipeline.LENSES)
    assert len(parameters) == 16
    np.testing.assert_allclose(
        model_to_parameters(model), parameters, rtol=0, atol=1e-12
    )
