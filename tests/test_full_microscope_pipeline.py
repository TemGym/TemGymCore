"""The split fit must agree with a full linear likelihood, without hidden maps."""

import json
import sys
from pathlib import Path

import jax
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


def test_gaussian_analysis_copies_known_sample_and_replaces_generating_maps(
    tmp_path, monkeypatch
):
    source = tmp_path / "condenser"
    source.mkdir()
    (source / "experiment.json").write_text(
        json.dumps(
            {"voltages_v": [200000.0, 195000.0], "image_model": "coherent_gaussian"}
        )
    )
    (source / "scoring_truth.json").write_text(
        "This generating file must never be opened by analysis."
    )
    for index in (0, 1):
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
        np.savez(folder / "moment_specimen.npz", known_transmission=[0.8, 0.9])
        np.savez(folder / "moment_specimen_local.npz", known_transmission=[0.85])
        np.savez(folder / "gaussian_specimen.npz", not_needed_for_fitting=[123.0])
        np.savez(
            folder / "counts.npz",
            transmitted_electrons=[[50.0]],
            reference_electrons=[100.0],
            controls=np.ones((1, 3)),
        )
        for name in ("detector.json", "aperture.json"):
            (folder / name).write_text("{}")
    seen = []
    monkeypatch.setattr(
        inference, "measure_condenser_moments", lambda *, folder: seen.append(folder)
    )
    monkeypatch.setattr(
        inference,
        "measure_images",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Intensity fitting must not run.")
        ),
    )
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
    assert len(seen) == 2
    for folder in seen:
        np.testing.assert_array_equal(
            np.load(folder / "images.npz")["projector_matrices"], [np.eye(5) * 2]
        )
        np.testing.assert_array_equal(
            np.load(folder / "moment_specimen.npz")["known_transmission"], [0.8, 0.9]
        )
        assert not (folder / "gaussian_specimen.npz").exists()
        np.testing.assert_array_equal(
            np.load(folder / "moment_specimen_local.npz")["known_transmission"], [0.85]
        )
    assert not (output / "scoring_truth.json").exists()


def test_beam_surrogate_refinement_matches_original_nonlinear_likelihood():
    def beam(x):
        return jnp.array([x[0] * x[1], x[1] ** 2, x[0], x[1]])

    def image(b):
        return jnp.array([jnp.exp(0.3 * b[0]), jnp.sin(b[1]), b[2] ** 2, b[3] ** 3])

    measured = image(beam(jnp.array([1.2, 0.8]))) + jnp.array(
        [0.003, -0.002, 0.004, -0.001]
    )

    def residual(x):
        return (image(beam(x)) - measured) / 0.01

    def make_state(x):
        b = beam(x)
        return (b, residual(x), jax.jacfwd(image)(b) / 0.01)

    def surrogate(x, anchor, error, jac):
        return error + jac @ (beam(x) - anchor)

    start = jnp.array([0.6, 0.6])
    bounds = (0.1, 2.0)
    fit = inference._refine_with_surrogate(
        residual=jax.jit(residual),
        derivative=jax.jit(jax.jacfwd(residual)),
        surrogate=jax.jit(surrogate),
        make_state=make_state,
        start=start,
        bounds=bounds,
    )
    exact = inference.least_squares(residual, start, bounds=bounds, x_scale="jac")
    assert fit.success, fit.message
    np.testing.assert_allclose(fit.x, exact.x, atol=1e-6)
    np.testing.assert_allclose(fit.jacobian, jax.jacfwd(residual)(fit.x), rtol=1e-12)
    assert all(r["actual_reduction"] >= 0 for r in fit.history if r["accepted"])


def test_surrogate_iteration_budget_does_not_imply_convergence():
    fit = inference._refine_with_surrogate(
        residual=lambda x: x - 1.0,
        derivative=lambda x: jnp.eye(1),
        surrogate=lambda x: x - 1.0,
        make_state=lambda x: (),
        start=[0.0],
        bounds=(-2.0, 2.0),
        max_iterations=0,
    )
    assert not fit.success


def test_aperture_centre_search_retains_distinct_current_solutions():
    p = (
        jnp.zeros(31)
        .at[20:22]
        .set(jnp.array([0.2, -0.3]))
        .at[29:31]
        .set(jnp.array([-0.15, -0.1]))
    )

    def currents(p):
        # A symmetric current measurement admits two separated x centres.
        return jnp.array(
            [p[20] - 0.2, p[21] + 0.3, 30 * (p[29] ** 2 - 0.15**2), p[30] + 0.1]
        )

    branches = inference._aperture_centre_branches(
        parameters=p,
        residual=jax.jit(currents),
        bounds=(np.full(31, -3.0), np.full(31, 3.0)),
        centering_sigma_um=0.2,
    )
    second = [r for r in branches if r["indices"] == [29, 30]]
    assert len(second) == 2
    np.testing.assert_allclose(
        sorted(r["centre_um"][0] for r in second), [-0.15, 0.15], atol=1e-6
    )
    assert all(abs(r["conditional_delta_chi_squared"]) < 1e-10 for r in second)
