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
    from temgym_core.calibration.projector_images import (
        model_to_parameters,
        parameters_to_model,
    )

    parameters = pipeline.initial_guess()
    model = parameters_to_model(parameters)
    assert set(model["lens_positions_mm"]) == set(pipeline.LENSES)
    assert model["angular_spread_mrad"] == pytest.approx(1.2)
    np.testing.assert_allclose(
        model_to_parameters(model), parameters, rtol=0, atol=1e-12
    )


def test_beam_centres_are_recovered_from_visible_texture_and_boundary(monkeypatch):
    from temgym_core.calibration.projector_images import (
        analyse_beam_shift_images,
        beam_centre_pixels,
        make_beam_shift_image_operator,
    )

    spec = pipeline.public_specification()
    spec.update(
        shape_yx=[48, 48],
        read_noise_std_e=2.0,
        specimen=dict(illumination_radius_m=40e-9, illumination_edge_m=2.5e-9),
    )
    axis = np.linspace(-65e-9, 65e-9, 27)
    x, y = np.meshgrid(axis, axis)
    intensity = 0.2 + 0.5 * (np.sin(x / 5e-9) > 0.3) + 0.2 * (y > 4e-9)
    template = dict(axis_m=axis, intensity=intensity)
    operator = make_beam_shift_image_operator(spec, template)
    p = np.r_[
        np.log(40000.0), 0.23, np.log(0.85), 0.37, -0.18, np.log([2e6, 3.0]), -7.0, 3.0
    ]
    other = p.copy()
    other[7:9] = [7.0, -3.0]
    images = np.array([operator(p), operator(other)])
    seed = np.r_[np.log(40000.0), 0.18, np.log(1.0), 0.2, -0.3, np.log([2e6, 3.0])]
    result = analyse_beam_shift_images(
        images=images,
        nominal_magnifications=[40000.0, 40000.0],
        specimen_template=template,
        specimen_and_detector=spec,
        initial_image_parameters=np.tile(seed, (2, 1)),
        verbose=False,
    )
    expected = np.array(
        [beam_centre_pixels(v, spec["pixel_pitch_m"]) for v in (p, other)]
    )
    np.testing.assert_allclose(result["beam_centres_pixels"], expected, atol=1e-6)
    np.testing.assert_allclose(
        result["image_parameters"][:, 7:9], [p[7:9], other[7:9]], atol=1e-6
    )
    # The texture translation is unchanged even though illumination has moved.
    np.testing.assert_allclose(
        result["image_parameters"][:, 3:5], [p[3:5], p[3:5]], atol=1e-6
    )
    assert np.all(np.linalg.eigvalsh(result["beam_centre_covariances"]) > 0)
    assert np.max(result["image_residual_rms"]) < 1e-6


def test_illumination_centre_matches_ray_transport_with_curvature_and_specimen_rotation(
    monkeypatch,
):
    from temgym_core.calibration.projector_images import beam_centre_pixels

    p = pipeline.initial_guess()
    p[16], p[18] = 0.4, 0.31
    commands = np.array([0.9, 0.8, 0.7, 0.9])
    spec = pipeline.public_specification()
    matrix = np.asarray(pipeline.matrix5(p, commands))
    position, angle = np.array([10e-9, -4e-9]), np.array([0.001, -0.002])
    rotation = np.array(
        [[np.cos(p[18]), -np.sin(p[18])], [np.sin(p[18]), np.cos(p[18])]]
    )
    texture_shift = (
        matrix[:2, 2:4] @ (angle - p[16] * 1e4 * position) / spec["pixel_pitch_m"]
    )
    image_parameters = np.r_[
        pipeline.predict_observables(p, commands, spec),
        texture_shift,
        np.log([2e6, 3.0]),
        rotation.T @ position * 1e9,
    ]
    expected = (matrix @ np.r_[position, angle, 1.0])[:2] / spec["pixel_pitch_m"]
    np.testing.assert_allclose(
        beam_centre_pixels(image_parameters, spec["pixel_pitch_m"]),
        expected,
        rtol=1e-12,
        atol=1e-9,
    )


def test_tilt_registration_recovers_shifts_without_lens_truth(monkeypatch):
    from temgym_core.calibration.projector_images import (
        analyse_magical_arrays,
        make_magical_image_operator,
        pair_tilt_measurements,
    )

    spec = pipeline.public_specification()
    spec["read_noise_std_e"] = 2.0
    template = dict(
        centres_m=np.asarray(spec["feature_centres_m"]),
        weights=np.array([0.4, 0.25, 0.35]),
    )
    operator = make_magical_image_operator(spec, template)
    p = np.r_[np.log(40000.0), 0.23, np.log(1.4), 1.7, -1.1, np.log(2e6), np.log(3.0)]
    delta = np.array([0.17, -0.08])
    plus = p.copy()
    plus[3:5] += delta
    results = analyse_magical_arrays(
        images=np.array([operator(p), operator(plus)]),
        nominal_magnifications=[40000.0, 40000.0],
        specimen_template=template,
        specimen_and_detector=spec,
        initial_image_parameters=np.array([p, p]),
    )
    covariance = results["parameter_covariance"][:, 3:5, 3:5]
    measurements = pair_tilt_measurements(
        lens_controls=np.ones((2, 4)),
        input_tilts_rad=np.array([[-0.001, 0.0], [0.001, 0.0]]),
        positions_pixels=results["image_parameters"][:, 3:5],
        position_covariances=covariance,
        drift_std_pixels=0.003,
        pair_labels=["x tilt"],
    )
    np.testing.assert_allclose(
        measurements["displacements_pixels"][0], delta, atol=1e-8
    )
    np.testing.assert_allclose(
        measurements["angle_changes_rad"][0], [0.002, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(
        measurements["covariance_pixels"][0],
        covariance.sum(axis=0) + 2 * 0.003**2 * np.eye(2),
    )
    assert np.all(np.linalg.eigvalsh(measurements["covariance_pixels"]) > 0)


def test_tilt_pairs_reject_a_changed_projector_setting(monkeypatch):
    from temgym_core.calibration.projector_images import pair_tilt_measurements

    with pytest.raises(ValueError, match="fixed within each tilt pair"):
        pair_tilt_measurements(
            lens_controls=np.array([[1.0, 1.0, 1.0, 1.0], [1.01, 1.0, 1.0, 1.0]]),
            input_tilts_rad=np.array([[-0.001, 0.0], [0.001, 0.0]]),
            positions_pixels=np.zeros((2, 2)),
            position_covariances=np.tile(np.eye(2), (2, 1, 1)),
            drift_std_pixels=0.003,
            pair_labels=["x tilt"],
        )
