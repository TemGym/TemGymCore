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


def test_objective_control_is_independent_and_legacy_controls_are_unchanged():
    p = pipeline.initial_guess()
    projector_controls = np.array([.9,.8,.7,.9])
    legacy = np.asarray(pipeline.matrix5(p, projector_controls))
    np.testing.assert_allclose(pipeline.matrix5(p,np.r_[1.,projector_controls]),legacy,atol=0,rtol=0)
    changed = p.copy(); changed[11] *= 1.002
    np.testing.assert_allclose(pipeline.matrix5(p,np.r_[1.002,projector_controls]),
                               pipeline.matrix5(changed,projector_controls),rtol=1e-13,atol=1e-9)
    with pytest.raises(ValueError,match='four projector controls or five'):
        pipeline.matrix5(p,np.ones(3))


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
        directory=tmp_path, magnifications=[40000, 60000], wobble_fraction=.025,
        settings_only=True,
    )
    assert len(spec["controls"]) == 18
    assert all(spec["training_mask"])
    assert not (tmp_path / "images.npz").exists()
    assert all("withheld" not in label for label in spec["labels"])
    for start, magnification in [(0, 40000), (9, 60000)]:
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


def test_calibrated_tilt_has_physical_units_and_breaks_axial_scale_ambiguity():
    p = pipeline.initial_guess()
    p[11] = 0.
    controls = np.zeros((2, 4))
    angles = np.array([[.002, 0.], [0., -.002]])
    pixel_pitch = 110e-6
    distance = float(pipeline.decode_geometry(p)[-1])
    shifts = pipeline.predict_tilt_displacements(p, controls, angles, pixel_pitch)
    np.testing.assert_allclose(shifts, distance*angles/pixel_pitch, atol=1e-12)
    np.testing.assert_allclose(pipeline.predict_tilt_displacements(p, controls, 3*angles, pixel_pitch),
                               3*shifts, atol=1e-12)
    scaled = pipeline.axial_scale_family(p, 1.1)
    np.testing.assert_allclose(
        pipeline.predict_tilt_displacements(scaled, controls, angles, pixel_pitch),
        1.1*shifts, atol=1e-12,
    )
    # Without the known tilt, the very same models give identical observables.
    np.testing.assert_allclose(pipeline.predict_observables(p, controls[0], pipeline.public_specification()),
                               pipeline.predict_observables(scaled, controls[0], pipeline.public_specification()),
                               atol=1e-12)


def test_calibrated_beam_shift_uses_a_and_known_angle_coupling():
    p = pipeline.initial_guess()
    controls = np.array([[.9, .8, .7, .9], [1., .8, .7, .9]])
    positions = np.array([[20e-9, 0.], [0., -20e-9]])
    angles = np.array([[.0001, -.0002], [.0003, 0.]])
    pitch = 110e-6
    expected = np.array([(np.asarray(pipeline.matrix5(p, c)) @ np.r_[s, a, 1.])[:2]/pitch
                         for c, s, a in zip(controls, positions, angles)])
    actual = pipeline.predict_beam_shift_displacements(p, controls, positions, angles, pitch)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    other = p.copy()
    other[16] = 1.2  # unknown illumination curvature cannot replace A in this measurement
    np.testing.assert_allclose(pipeline.predict_beam_shift_displacements(other, controls, positions, angles, pitch), actual)
    p[11] = 0.
    distance = float(pipeline.decode_geometry(p)[-1])
    np.testing.assert_allclose(pipeline.predict_beam_shift_displacements(p, np.zeros((2,4)), positions, angles, pitch),
                               (positions + distance*angles)/pitch, atol=1e-12)


def test_beam_centres_are_recovered_from_visible_texture_and_boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(_path.parent))
    from projector_beam_shift_pipeline import (
        make_beam_shift_image_operator, analyse_beam_shift_images, beam_centre_pixels,
    )

    spec = pipeline.public_specification()
    spec.update(shape_yx=[48,48], read_noise_std_e=2.,
                specimen=dict(illumination_radius_m=40e-9, illumination_edge_m=2.5e-9))
    axis = np.linspace(-65e-9,65e-9,27)
    x, y = np.meshgrid(axis,axis)
    intensity = .2 + .5*(np.sin(x/5e-9) > .3) + .2*(y > 4e-9)
    template = dict(axis_m=axis, intensity=intensity)
    operator = make_beam_shift_image_operator(spec, template)
    p = np.r_[np.log(40000.), .23, np.log(.85), .37, -.18, np.log([2e6,3.]), -7., 3.]
    other = p.copy()
    other[7:9] = [7., -3.]
    images = np.array([operator(p), operator(other)])
    seed = np.r_[np.log(40000.), .18, np.log(1.), .2, -.3, np.log([2e6,3.])]
    result = analyse_beam_shift_images(
        images=images, nominal_magnifications=[40000.,40000.], specimen_template=template,
        specimen_and_detector=spec, initial_image_parameters=np.tile(seed,(2,1)), verbose=False,
    )
    expected = np.array([beam_centre_pixels(v,spec['pixel_pitch_m']) for v in (p,other)])
    np.testing.assert_allclose(result['beam_centres_pixels'], expected, atol=1e-6)
    np.testing.assert_allclose(result['image_parameters'][:,7:9], [p[7:9],other[7:9]], atol=1e-6)
    # The texture translation is unchanged even though illumination has moved.
    np.testing.assert_allclose(result['image_parameters'][:,3:5], [p[3:5],p[3:5]], atol=1e-6)
    assert np.all(np.linalg.eigvalsh(result['beam_centre_covariances']) > 0)
    assert np.max(result['image_residual_rms']) < 1e-6



def test_illumination_centre_matches_ray_transport_with_curvature_and_specimen_rotation(monkeypatch):
    monkeypatch.syspath_prepend(str(_path.parent))
    from projector_beam_shift_pipeline import beam_centre_pixels

    p = pipeline.initial_guess()
    p[16], p[18] = .4, .31
    commands = np.array([.9,.8,.7,.9])
    spec = pipeline.public_specification()
    matrix = np.asarray(pipeline.matrix5(p, commands))
    position, angle = np.array([10e-9,-4e-9]), np.array([.001,-.002])
    rotation = np.array([[np.cos(p[18]),-np.sin(p[18])], [np.sin(p[18]),np.cos(p[18])]])
    texture_shift = matrix[:2,2:4] @ (angle-p[16]*1e4*position) / spec['pixel_pitch_m']
    image_parameters = np.r_[pipeline.predict_observables(p,commands,spec), texture_shift,
                             np.log([2e6,3.]), rotation.T @ position*1e9]
    expected = (matrix @ np.r_[position,angle,1.])[:2] / spec['pixel_pitch_m']
    np.testing.assert_allclose(beam_centre_pixels(image_parameters,spec['pixel_pitch_m']),
                               expected, rtol=1e-12, atol=1e-9)


def test_beam_displacements_enter_the_fit_with_their_covariance():
    p = pipeline.initial_guess()
    spec = pipeline.public_specification()
    controls = np.array([[.9, .8, .7, .9], [1., .8, .7, .9]])
    positions = np.eye(2)*20e-9
    angles = np.zeros((2,2))
    predicted = np.asarray(pipeline.predict_beam_shift_displacements(p, controls, positions, angles, spec['pixel_pitch_m']))
    offset = np.array([[.1, -.2], [.3, -.4]])
    beam = dict(lens_controls=controls, position_changes_m=positions, angle_changes_rad=angles,
                displacements_pixels=predicted+offset, covariance_pixels=np.tile(np.eye(2)*.25, (2,1,1)))
    problem = pipeline.prepare_column_fit(
        lens_controls=controls, measured_values=np.array([pipeline.predict_observables(p,c,spec) for c in controls]),
        measurement_covariances=np.tile(np.eye(3), (2,1,1)), specimen_and_detector=spec,
        prior_parameters=p, beam_shift_measurements=beam,
    )
    result = problem.solve(p, max_evaluations=1)
    np.testing.assert_allclose(result.data_residual[-4:], -2*offset.ravel(), atol=1e-9)
    assert len(result.data_residual) == 10
    beam['position_changes_m'] = np.zeros((1,2))
    with pytest.raises(ValueError, match='matching pair counts'):
        pipeline.prepare_column_fit(
            lens_controls=controls, measured_values=np.zeros((2,3)),
            measurement_covariances=np.tile(np.eye(3), (2,1,1)), specimen_and_detector=spec,
            prior_parameters=p, beam_shift_measurements=beam,
        )


def test_tilt_registration_recovers_shifts_without_lens_truth(monkeypatch):
    monkeypatch.syspath_prepend(str(_path.parent))
    from magical_projector_pipeline import make_magical_image_operator, analyse_magical_arrays
    from projector_tilt_pipeline import pair_tilt_measurements

    spec = pipeline.public_specification()
    spec['read_noise_std_e'] = 2.
    template = dict(centres_m=np.asarray(spec['feature_centres_m']), weights=np.array([.4,.25,.35]))
    operator = make_magical_image_operator(spec, template)
    p = np.r_[np.log(40000.), .23, np.log(1.4), 1.7, -1.1, np.log(2e6), np.log(3.)]
    delta = np.array([.17, -.08])
    plus = p.copy()
    plus[3:5] += delta
    results = analyse_magical_arrays(
        images=np.array([operator(p), operator(plus)]), nominal_magnifications=[40000., 40000.],
        specimen_template=template, specimen_and_detector=spec,
        initial_image_parameters=np.array([p, p]),
    )
    covariance = results['parameter_covariance'][:, 3:5, 3:5]
    measurements = pair_tilt_measurements(
        lens_controls=np.ones((2,4)), input_tilts_rad=np.array([[-.001,0.],[.001,0.]]),
        positions_pixels=results['image_parameters'][:,3:5], position_covariances=covariance,
        drift_std_pixels=.003, pair_labels=['x tilt'],
    )
    np.testing.assert_allclose(measurements['displacements_pixels'][0], delta, atol=1e-8)
    np.testing.assert_allclose(measurements['angle_changes_rad'][0], [.002,0.], atol=1e-15)
    np.testing.assert_allclose(measurements['covariance_pixels'][0], covariance.sum(axis=0)+2*.003**2*np.eye(2))
    assert np.all(np.linalg.eigvalsh(measurements['covariance_pixels']) > 0)


def test_tilt_pairs_reject_a_changed_projector_setting(monkeypatch):
    monkeypatch.syspath_prepend(str(_path.parent))
    from projector_tilt_pipeline import pair_tilt_measurements

    with pytest.raises(ValueError, match='fixed within each tilt pair'):
        pair_tilt_measurements(
            lens_controls=np.array([[1.,1.,1.,1.], [1.01,1.,1.,1.]]),
            input_tilts_rad=np.array([[-.001,0.],[.001,0.]]),
            positions_pixels=np.zeros((2,2)), position_covariances=np.tile(np.eye(2),(2,1,1)),
            drift_std_pixels=.003, pair_labels=['x tilt'],
        )


def test_optical_fit_rejects_mismatched_tilt_measurements():
    with pytest.raises(ValueError, match='matching pair counts'):
        pipeline.fit_column_from_measurements(
            lens_controls=np.ones((2,4)), measured_values=np.zeros((2,3)),
            measurement_covariances=np.tile(np.eye(3),(2,1,1)),
            specimen_and_detector=pipeline.public_specification(),
            starting_parameters=pipeline.initial_guess(),
            tilt_measurements=dict(lens_controls=np.ones((2,4)), angle_changes_rad=np.ones((1,2)),
                                   displacements_pixels=np.zeros((2,2)), covariance_pixels=np.tile(np.eye(2),(2,1,1))),
        )


def test_multiple_starts_keep_the_same_prior_and_detector_profiles_pin_one_coordinate():
    p = pipeline.initial_guess()
    spec = pipeline.public_specification()
    controls = np.array([[.9,.8,.7,.9], [1.,.8,.7,.9]])
    values = np.array([pipeline.predict_observables(p, row, spec) for row in controls])
    priors = json.loads((_path.parent / 'projector_position_priors.json').read_text())
    problem = pipeline.prepare_column_fit(
        lens_controls=controls, measured_values=values,
        measurement_covariances=np.tile(np.eye(3), (2,1,1)),
        specimen_and_detector=spec, prior_parameters=p, position_priors=priors,
    )
    other = p.copy()
    other[12] += .1
    result = problem.solve(other, max_evaluations=1)
    np.testing.assert_allclose(result.prior_parameters, p, atol=0)
    # With only one evaluation, the optimizer remains at the other start;
    # its gain prior must still penalize the displacement from the fixed prior.
    assert result.fun[-13:][6] == pytest.approx(.1)
    np.testing.assert_allclose(result.fun[-13:], result.x[6:]-p[6:], atol=1e-12)
    fixed = problem.solve(other, max_evaluations=2, fixed_detector_position_mm=700.)
    assert float(pipeline.decode_geometry(fixed.x)[-1])*1000 == pytest.approx(700., abs=1e-9)
    assert fixed.free_coordinate_mask.sum() == 18
    assert not fixed.free_coordinate_mask[5]
    assert np.all(fixed.solver_x >= problem.lower-1e-12)
    assert np.all(fixed.solver_x <= problem.upper+1e-12)
    with pytest.raises(ValueError, match='outside'):
        problem.solve(p, max_evaluations=1, fixed_detector_position_mm=900.)


def test_extra_wobbles_preserve_reference_controls_and_add_sixteen_images(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_path.parent))
    from projector_notebook_helpers import Experiment
    from projector_landscape import simulate_extra_wobbles
    from projector_tilt_pipeline import simulate_tilt_images

    spec = pipeline.generate_dataset(tmp_path, magnifications=[40000, 60000], settings_only=True)
    template = dict(centres_m=np.array(spec['feature_centres_m']), weights=np.array([.4,.25,.35]))
    experiment = Experiment(np.zeros((18,128,128)), np.array(spec['controls']),
                            np.array(spec['nominal_magnifications']), spec['labels'], template, spec, tmp_path)
    original = experiment.lens_controls.copy()
    extra = simulate_extra_wobbles(experiment=experiment, reference_image_index=0,
                                   wobble_fractions=[.05,.10], random_seed=1)
    assert extra.images.shape == (16,128,128)
    assert np.isfinite(extra.images).all()
    np.testing.assert_array_equal(experiment.lens_controls, original)
    for start, fraction in [(0,.05), (8,.10)]:
        expected = np.repeat(np.eye(4), 2, axis=0)*np.tile([-fraction, fraction],4)[:,None]
        np.testing.assert_allclose(extra.lens_controls[start:start+8]/original[0]-1, expected, atol=1e-14)
    assert np.all(extra.nominal_magnifications == 40000)
    # Adding the larger tilt amplitude must preserve the existing exposures.
    one = simulate_tilt_images(experiment=extra, lens_setting_indices=[0], tilt_mrad=1., random_seed=1)
    saved_one = (extra.output_directory/'tilt_images.npz').read_bytes()
    three = simulate_tilt_images(experiment=extra, lens_setting_indices=[0], tilt_mrad=3., random_seed=2)
    assert (extra.output_directory/'tilt_images.npz').read_bytes() == saved_one
    assert (extra.output_directory/'tilt_images_3mrad.npz').exists()
    np.testing.assert_array_equal(three.lens_controls, one.lens_controls)
    np.testing.assert_allclose(three.input_tilts_rad, 3*one.input_tilts_rad)
