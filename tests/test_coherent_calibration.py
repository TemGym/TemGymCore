from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_coherent_microscope import model

from temgym_core.coherent_calibration import (
    CalibrationState,
    FitParameter,
    Parameterization,
    RecordedSettings,
    fit_microscope,
    make_predictor,
    microscope_parameters,
)
from temgym_core.coherent_microscope import WaveNumerics
from temgym_core.components import Plane


def test_named_parameters_and_complete_geometry_roundtrip():
    state = CalibrationState(model(), jnp.ones(2), jnp.zeros(2), jnp.ones(2))
    params = Parameterization(state, microscope_parameters(state, fit_exposures=True))
    assert len(params.parameters) == 49
    restored = params.decode(params.initial)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-20)
    for p in params.parameters:
        if p.logarithmic and p.sign < 0:
            assert p.lower < p.upper < 0
    with pytest.raises(ValueError, match="distinct"):
        Parameterization(state, [params.parameters[0], params.parameters[0]])


def test_two_voltage_coherent_pixels_currents_jacobian_and_blind_fit():
    microscope = model()
    settings = RecordedSettings(
        tuple(
            {
                "voltage": v,
                "source_index": i,
                "condenser": [1.0, 1.0, 1.0, 1.0],
                "projector": [1.0, 1.0, 1.0, 1.0, 1.0],
                "tilt_xy": [0.0, 0.0],
                "shift_xy": [0.0, 0.0],
            }
            for i, v in enumerate((200000.0, 195000.0))
        ),
        tuple(
            {
                "voltage": v,
                "source_index": i,
                "condenser": [1.0, 1.0, 1.0, 1.0],
                "hole": h,
            }
            for i, v in enumerate((200000.0, 195000.0))
            for h in (-1, 0, 1)
        ),
    )
    numerical = WaveNumerics(
        specimen_half_width=80e-9,
        beamlet_width=8e-9,
        follow_source=True,
        detector_shape=(6, 6),
        detector_pitch=0.01,
        pixel_samples=2,
        radial_order=48,
        azimuthal_order=16,
    )
    predictor = make_predictor(settings, Plane(z=0.0), numerical)
    truth = CalibrationState(
        microscope, jnp.full(2, 1e6), jnp.ones(2), jnp.full(2, 1e7)
    )
    pixels, currents = predictor(truth)
    observations = {
        "images": pixels,
        "currents": currents,
        "image_sigma": jnp.ones_like(pixels),
        "current_sigma": jnp.ones_like(currents),
    }
    specs = [
        FitParameter(
            "CL1 gain", ("microscope", "condenser", "strengths", 0), 0.16, 0.24
        ),
        FitParameter(
            "PL1 gain", ("microscope", "projector", "strengths", 4), -0.9, -0.6
        ),
        FitParameter(
            "source size", ("microscope", "sources", 0, "rms_size"), 6e-9, 10e-9, 1e-9
        ),
    ]
    # Only the independent starting state is available to the fit, not truth.
    starting = replace(
        truth,
        microscope=replace(
            microscope,
            condenser=replace(
                microscope.condenser,
                strengths=microscope.condenser.strengths.at[0].set(0.202),
            ),
            projector=replace(
                microscope.projector,
                strengths=microscope.projector.strengths.at[4].set(-0.753),
            ),
            sources=(
                replace(microscope.sources[0], rms_size=8.1e-9),
                microscope.sources[1],
            ),
        ),
    )
    params = Parameterization(starting, specs)

    def flattened(x):
        images, counts = predictor(params.decode(x))
        return jnp.concatenate((images.ravel(), counts))

    derivative = jax.jacfwd(flattened)(params.initial)
    assert np.isfinite(derivative).all()
    direction = jnp.array([0.3, -0.2, 0.1])
    finite = (
        flattened(params.initial + direction * 1e-5)
        - flattened(params.initial - direction * 1e-5)
    ) / 2e-5
    np.testing.assert_allclose(derivative @ direction, finite, rtol=2e-5, atol=1e-2)
    fitted, report = fit_microscope(
        predictor=predictor,
        observations=observations,
        parameterization=params,
        max_evaluations=60,
        relative_cost_tolerance=1e-12,
    )
    assert report["starts"][0]["success"]
    assert report["rank"] == 3
    np.testing.assert_allclose(params.encode(fitted), params.encode(truth), rtol=1e-5)


def test_nonfinite_solver_attempt_is_reported_without_covariance():
    state = CalibrationState(model(), jnp.ones(1), jnp.zeros(1), jnp.ones(2))
    params = Parameterization(
        state, [FitParameter("dose", ("image_dose", 0), 0.1, 3.0)]
    )

    def invalid_predictor(state):
        return jnp.array([jnp.nan]) * state.image_dose, jnp.zeros(0)

    _, report = fit_microscope(
        predictor=invalid_predictor,
        observations={
            "images": jnp.zeros(1),
            "currents": jnp.zeros(0),
            "image_sigma": jnp.ones(1),
            "current_sigma": jnp.zeros(0),
        },
        parameterization=params,
        max_evaluations=2,
    )
    assert not report["starts"][0]["success"]
    assert report["rank"] is None
    assert report["covariance"] is None
    assert not report["data_fit_accepted"]


def test_image_centroid_translation_and_noise_covariance():
    from temgym_core.coherent_calibration import measure_image_centroids

    y, x = np.mgrid[-32:33, -32:33]
    images = np.stack([
        5000 * np.exp(-((x - dx)**2 + (y - dy)**2) / 18) + 3
        for dx, dy in ((0, 0), (2, -3))
    ])
    result = measure_image_centroids(
        images=images, background_counts=3, read_noise_counts=2,
    )
    np.testing.assert_allclose(result['centres_xy_pixels'], [[0, 0], [2, -3]], atol=1e-10)
    assert np.linalg.eigvalsh(result['covariances_pixels2']).min() > 0
    # In the shot-noise limit centroid covariance is spatial covariance / flux.
    shot = measure_image_centroids(
        images=images[:1] - 3, background_counts=0, read_noise_counts=0,
    )
    np.testing.assert_allclose(
        shot['covariances_pixels2'][0], np.eye(2) * 9 / shot['signal_counts'][0],
        rtol=1e-12, atol=1e-16,
    )
    with pytest.raises(ValueError, match='positive'):
        measure_image_centroids(images=np.ones((1, 3, 3)), background_counts=3, read_noise_counts=2)


def test_matrix_tilt_fit_recovers_physical_parameters_without_rendering(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / 'examples/microscope_models'))
    from coherent_microscope_workflow import fit_gaussian_tilt_measurements
    from finite_projector_pipeline import (
        _generator_parameters,
        predict_tilt_displacements,
    )

    from temgym_core import coherent_image

    def forbidden_renderer(*args, **kwargs):
        raise AssertionError('A measurement-space fit must not render Gaussian images.')

    monkeypatch.setattr(coherent_image, 'blurred_gaussian_image', forbidden_renderer)
    truth, _ = _generator_parameters()
    controls = np.repeat(np.array([
        [1., .61568697, .56752706, 1.75623022, .83875841],
        [1., .64291112, .80838409, 1.92561045, 1.5376724],
        [1., .47997970, .12337942, 1.49849115, 1.9175293],
    ]), 2, axis=0)
    changes = np.tile(np.eye(2) * .002, (3, 1))
    indices = np.array([5, 12, 15])
    known = truth.tolist()
    for index in indices:
        known[index] = None
    settings = {
        'fixed_projector_parameters': known,
        'lens_controls': np.repeat(controls, 2, axis=0),
        'tilts_xy_rad': np.stack((-changes / 2, changes / 2), axis=1).reshape(-1, 2),
        'pixel_pitch_m': 55e-6,
    }
    def prediction(p):
        full = jnp.asarray(truth).at[indices].set(p)
        return predict_tilt_displacements(full, controls, changes, 55e-6)

    p = jnp.asarray(truth[indices])
    direction = jnp.array([.2, -.3, .1])
    derivative = jax.jacfwd(prediction)(p) @ direction
    finite = (prediction(p + 1e-5 * direction) - prediction(p - 1e-5 * direction)) / 2e-5
    np.testing.assert_allclose(derivative, finite, rtol=1e-7, atol=1e-6)
    fit = fit_gaussian_tilt_measurements(
        measurements={'displacements_xy_pixels': prediction(p),
                      'covariances_pixels2': np.tile(np.eye(2) * 1e-4, (6, 1, 1))},
        known_settings=settings,
        starting_parameters={'detector_gap_m': .34, 'IL1_gain': .50, 'PL1_gain': .70},
        parameter_bounds={'detector_gap_m': (.2, .45), 'IL1_gain': (.4, .8), 'PL1_gain': (.5, .95)},
    )
    assert fit['local_rank'] == 3
    assert fit['starts'][0]['success']
    assert fit['gaussian_images_rendered_during_fit'] == 0
    np.testing.assert_allclose(list(fit['fitted_parameters'].values()), np.exp(truth[indices]), rtol=1e-7)
