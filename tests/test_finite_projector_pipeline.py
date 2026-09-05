"""Check observable information and coordinate conventions in the tutorial."""

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
_path = Path(__file__).parents[1] / "examples/microscope_models/finite_projector_pipeline.py"
_spec = importlib.util.spec_from_file_location("finite_projector_pipeline", _path)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def test_zero_excitation_is_full_axial_drift_in_grouped_coordinates():
    p = pipeline.initial_guess()
    p[11] = 0.
    actual = np.asarray(pipeline.matrix5(p, np.zeros(4)))
    distance = float(pipeline.decode_geometry(p)[-1])
    expected = np.eye(5)
    expected[0, 2] = expected[1, 3] = distance
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_axial_scale_preserves_images_across_lens_wobbles():
    p, _ = pipeline._generator_parameters()
    controls = np.array([.96, .78, .87, .91])
    scaled = pipeline.axial_scale_family(p, 1.3)
    spec = pipeline.public_specification()
    similarity = np.diag([1., 1., 1 / 1.3, 1 / 1.3, 1.])
    for perturbation in [np.zeros(4), *(.025 * np.eye(4)), *(-.025 * np.eye(4))]:
        row = controls * (1 + perturbation)
        matrix = np.asarray(pipeline.matrix5(p, row))
        other = np.asarray(pipeline.matrix5(scaled, row))
        assert np.linalg.norm(matrix[:2, 2:4]) > 1e-3  # genuinely away from focus
        np.testing.assert_allclose(other, similarity @ matrix @ np.linalg.inv(similarity), rtol=1e-10, atol=1e-8)
        original = pipeline.predict_observables(p, row, spec)
        transformed = pipeline.predict_observables(scaled, row, spec)
        np.testing.assert_allclose(original, transformed, rtol=0, atol=1e-10)


def test_image_analysis_recovers_measurements_without_an_optical_model(tmp_path):
    spec = pipeline.public_specification()
    spec.update(nominal_magnifications=[40000.], read_noise_std_e=2.)
    # Scale, angle, and width are independently chosen; there is no lens truth
    # file, control table, or parameter initialisation in this directory.
    p = np.r_[np.log(45000.), .23, np.log(1.4), .3, -.4,
              np.log([8e5, 6e5, 1e6]), np.log(3.)]
    image = np.asarray(pipeline.make_image_operator(spec)(p))
    (tmp_path / "acquisition.json").write_text(json.dumps(spec))
    np.savez(tmp_path / "images.npz", images=image[None])
    observed, covariance, residual = pipeline.analyse_images(tmp_path)
    np.testing.assert_allclose(observed[0], p[:3], rtol=0, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(covariance[0]) > 0)
    assert residual[0] < 1e-7


def test_bounded_position_coordinates_preserve_the_column_and_prevent_overlap():
    priors = json.loads((_path.parent / "projector_position_priors.json").read_text())
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
        gaps, lengths, centres, detector = map(np.asarray, pipeline.decode_geometry(decode(u)))
        assert np.all(gaps > 0)
        positions = np.r_[centres, detector] * 1000
        assert np.all(positions >= bounds[:, 0] - 1e-10)
        assert np.all(positions <= bounds[:, 1] + 1e-10)
        np.testing.assert_allclose(lengths * 1000, length_bounds[:, 1], atol=1e-12)


def test_inconsistent_position_priors_are_rejected():
    priors = json.loads((_path.parent / "projector_position_priors.json").read_text())
    priors["position_bounds_mm"][2][0] = 240.0
    with pytest.raises(ValueError, match="non-overlapping"):
        pipeline.position_prior_coordinates(priors)


def test_texture_renderer_reproduces_the_three_mark_special_case(monkeypatch):
    monkeypatch.syspath_prepend(str(_path.parent))
    from magical_projector_pipeline import make_magical_image_operator

    spec = pipeline.public_specification()
    counts = np.array([8e5, 6e5, 1e6])
    template = {"centres_m": np.asarray(spec["feature_centres_m"]), "weights": counts / counts.sum()}
    p = np.r_[np.log(120000.), .71, np.log(2.1), .31, -.22, np.log(counts), np.log(3.)]
    texture_p = np.r_[p[:5], np.log(counts.sum()), p[8]]
    actual = np.asarray(make_magical_image_operator(spec, template)(texture_p))
    expected = np.asarray(pipeline.make_image_operator(spec)(p))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)


def test_readable_model_parameters_round_trip(monkeypatch):
    monkeypatch.syspath_prepend(str(_path.parent))
    from projector_notebook_helpers import model_to_parameters, parameters_to_model

    parameters = pipeline.initial_guess()
    model = parameters_to_model(parameters)
    assert set(model["lens_positions_mm"]) == set(pipeline.LENSES)
    assert model["angular_spread_mrad"] == pytest.approx(1.2)
    np.testing.assert_allclose(model_to_parameters(model), parameters, rtol=0, atol=1e-12)


def test_two_magnifications_have_eighteen_settings_and_no_blob_images(tmp_path):
    spec = pipeline.generate_dataset(
        directory=tmp_path, magnifications=[40000, 150000], wobble_fraction=.025,
        settings_only=True,
    )
    assert len(spec["controls"]) == 18
    assert all(spec["training_mask"])
    assert not (tmp_path / "images.npz").exists()
    assert all("withheld" not in label for label in spec["labels"])
    for start, magnification in [(0, 40000), (9, 150000)]:
        controls = np.array(spec["controls"])[start:start+9]
        np.testing.assert_allclose((controls[1:] / controls[0] - 1),
                                   np.repeat(np.eye(4), 2, axis=0) * np.tile([-.025,.025],4)[:,None],
                                   atol=1e-14)
        assert spec["nominal_magnifications"][start:start+9] == [magnification]*9


def test_in_memory_fitter_checks_matching_inputs():
    with pytest.raises(ValueError, match="matching frame counts"):
        pipeline.fit_column_from_measurements(
            lens_controls=np.zeros((2,4)), measured_values=np.zeros((3,3)),
            measurement_covariances=np.tile(np.eye(3),(2,1,1)),
            specimen_and_detector=pipeline.public_specification(),
            starting_parameters=pipeline.initial_guess(),
        )
