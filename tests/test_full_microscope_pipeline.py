"""The split fit must agree with a full linear likelihood, without hidden maps."""

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/microscope_models"))
import full_microscope_inference as inference
from finite_projector_pipeline import initial_guess, matrix5
from full_microscope_projector import at_voltage


def test_shared_projector_covariance_matches_full_pixel_fisher():
    rng = np.random.default_rng(184)
    full = np.zeros((31, 11))  # Three local beams and a two-parameter calibration.
    covariances, responses, projected = [], [], []
    for index in range(3):
        local, common = rng.normal(size=(9, 3)), rng.normal(size=(9, 2))
        full[9 * index : 9 * (index + 1), 3 * index : 3 * (index + 1)] = local
        full[9 * index : 9 * (index + 1), -2:] = common
        inverse = np.linalg.pinv(local)
        covariances.append(np.linalg.inv(local.T @ local)[:2, :2])
        responses.append((-inverse @ common)[:2])
        projected.append(common - local @ inverse @ common)
    calibration = rng.normal(size=(4, 2))
    full[-4:, -2:] = calibration
    projected.append(calibration)
    design = np.concatenate(projected)
    common_covariance = np.linalg.inv(design.T @ design)
    beam_covariance, cross = inference.joint_measurement_covariance(
        covariances, responses, common_covariance
    )
    actual = np.block([[beam_covariance, cross], [cross.T, common_covariance]])
    keep = [0, 1, 3, 4, 6, 7, 9, 10]
    expected = np.linalg.inv(full.T @ full)[np.ix_(keep, keep)]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
    assert abs(actual[0, 2]) > 1e-5


def test_conditioned_split_fit_matches_dense_joint_solution():
    rng = np.random.default_rng(581)
    raw = rng.normal(size=(8, 8))
    covariance = raw @ raw.T + np.eye(8)
    # Measurements consist of five beam values and three projector parameters.
    beam_map = rng.normal(size=(5, 2))
    design = np.zeros((8, 5))
    design[:5, 3:] = beam_map
    design[5:, :3] = np.eye(3)
    observed = rng.normal(size=8)
    precision = np.linalg.inv(covariance)
    expected_covariance = np.linalg.inv(design.T @ precision @ design)
    expected_mean = expected_covariance @ design.T @ precision @ observed
    beam_precision = np.linalg.inv(covariance[:5, :5])
    condenser_covariance = np.linalg.inv(beam_map.T @ beam_precision @ beam_map)
    condenser_mean = condenser_covariance @ beam_map.T @ beam_precision @ observed[:5]
    projector, actual_covariance = inference.condition_projector(
        projector_mean=observed[5:],
        beam_mean=observed[:5],
        beam_covariance=covariance[:5, :5],
        beam_projector_covariance=covariance[:5, 5:],
        projector_covariance=covariance[5:, 5:],
        fitted_beams=beam_map @ condenser_mean,
        beam_derivative=beam_map,
        condenser_covariance=condenser_covariance,
    )
    np.testing.assert_allclose(
        np.r_[projector, condenser_mean], expected_mean, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        actual_covariance, expected_covariance, rtol=1e-12, atol=1e-12
    )


def test_voltage_changes_strength_without_changing_geometry():
    p = np.asarray(initial_guess())
    voltage, reference = 195000.0, 200000.0
    # Electron rest energy in eV, matching the physical constants in the model.
    from scipy.constants import electron_mass, elementary_charge, speed_of_light

    rest = electron_mass * speed_of_light**2 / elementary_charge
    ratio = np.sqrt(
        reference * (reference + 2 * rest) / (voltage * (voltage + 2 * rest))
    )
    commands = np.array([1.0, 0.9, 1.1, 0.8, 1.2])
    adjusted = np.asarray(at_voltage(p, voltage, reference))
    np.testing.assert_array_equal(adjusted[:11], p[:11])
    np.testing.assert_allclose(
        matrix5(adjusted, commands), matrix5(p, commands * ratio), rtol=1e-11, atol=1e-8
    )
    assert commands[0] == 1.0


def test_analysis_replaces_hidden_projector_maps(tmp_path, monkeypatch):
    source = tmp_path / "condenser"
    source.mkdir()
    (source / "experiment.json").write_text(
        json.dumps({"voltages_v": [200000.0, 195000.0]})
    )
    for index in [0, 1]:
        folder = source / f"voltage_{index}"
        folder.mkdir()
        np.savez(
            folder / "images.npz",
            images=np.ones((1, 1, 2, 2)),
            controls=np.ones((1, 3)),
            projector_controls=np.ones((1, 5)),
            projector_matrices=np.full((1, 5, 5), np.nan),
        )
        np.savez(folder / "template.npz", axis_m=[0.0])
        np.savez(
            folder / "counts.npz",
            transmitted_electrons=[[50.0]],
            reference_electrons=[100.0],
            controls=np.ones((1, 3)),
        )
        for filename in ["detector.json", "aperture.json"]:
            (folder / filename).write_text("{}")
    # Neither a scoring-truth file nor a generating-projector file exists.
    monkeypatch.setattr(inference, "measure_images", lambda **kwargs: None)
    monkeypatch.setattr(
        inference,
        "projector_at_voltage",
        lambda p, command, voltage, **kwargs: jnp.eye(5) * p[0],
    )
    output = inference.prepare_condenser_analysis(
        images_directory=source,
        projector_fit={"best_fit": {"parameters": [2.0]}},
        reference_voltage_v=200000.0,
    )
    for index in [0, 1]:
        data = np.load(output / f"voltage_{index}/images.npz")
        np.testing.assert_array_equal(data["projector_matrices"], [np.eye(5) * 2.0])
        assert not (output / f"voltage_{index}/scoring_truth.json").exists()


def test_condenser_starts_use_literal_engineering_model():
    root = Path(__file__).parents[1] / "examples/microscope_models"
    path = root / "microscope_condenser_starting_guess.json"
    guesses, lower, upper = inference.condenser_starting_guesses(
        starting_guess_path=path
    )
    assert len(guesses) == 8
    for guess in guesses:
        assert len(guess) == 29
        assert np.all((guess > lower) & (guess < upper))
    settings = json.loads(path.read_text())
    np.testing.assert_allclose(
        np.exp(guesses[0][:5]) * 1000, settings["model"]["gaps_mm"]
    )
    assert "best_fit" not in settings
