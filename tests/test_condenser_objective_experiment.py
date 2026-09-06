"""Objective-sweep physics and independent image measurement representation."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLES = Path(__file__).parents[1] / "examples/microscope_models"
sys.path.insert(0, str(EXAMPLES))
jax.config.update("jax_enable_x64", True)

from condenser_aperture_experiment import equivalent_aperture_model
from condenser_calibration_model import generating_parameters
from condenser_image_pipeline import make_image_operator, predict_beam_coordinates
from condenser_objective_experiment import (
    make_pattern_image_operator,
    post_field_change,
    predict_pattern,
)
from finite_projector_pipeline import matrix5


def setup_models():
    views = json.loads((EXAMPLES / "condenser_projector_views.json").read_text())
    p = jnp.asarray(views["calibrated_projector_parameters"])
    controls = jnp.asarray(views["views"][0]["controls"])
    return p, controls, np.asarray([v["matrix"] for v in views["views"]])


def test_post_field_change_matches_physical_projector_and_keeps_reference():
    p, controls, _ = setup_models()
    np.testing.assert_allclose(
        post_field_change(p, 1.0, jnp.array([1.2, 0.4, 0.1])), np.eye(4), atol=1e-12
    )
    for u in [0.97, 1.03]:
        e = np.eye(5)
        e[:4, :4] = post_field_change(p, u, jnp.array([1.0, 0.0, 0.0]))
        expected = np.asarray(matrix5(p, controls.at[0].set(u)))
        np.testing.assert_allclose(
            np.asarray(matrix5(p, controls)) @ e, expected, rtol=1e-9, atol=1e-7
        )
        length0 = np.exp(float(p[6]))
        length = length0 * np.exp(0.3 * (u - 1))
        changed = (
            p.at[6]
            .set(np.log(length))
            .at[0]
            .set(np.log(np.exp(float(p[0])) + (length0 - length) / 2))
            .at[1]
            .set(np.log(np.exp(float(p[1])) + (length0 - length) / 2))
        )
        e[:4, :4] = post_field_change(p, u, jnp.array([1.0, 0.3, 0.0]))
        np.testing.assert_allclose(
            np.asarray(matrix5(p, controls)) @ e,
            matrix5(changed, controls.at[0].set(u)),
            rtol=1e-9,
            atol=1e-7,
        )


def test_independent_pattern_renderer_matches_direct_specimen_transport():
    projector, _, maps = setup_models()
    template = {
        "axis_m": np.linspace(-30e-9, 30e-9, 7),
        "intensity": np.random.default_rng(7).uniform(0.1, 1.0, (7, 7)),
    }
    detector = {
        "shape_yx": [24, 24],
        "sample_kernel_sigma_m": 2.5e-9,
        "pixel_pitch_m": 220e-6,
        "psf_sigma_pixels": 0.5,
    }
    p = jnp.asarray(generating_parameters())
    c = jnp.asarray(
        json.loads((EXAMPLES / "condenser_measurement_protocol.json").read_text())[
            "calibration_controls"
        ][0]
    )
    response = jnp.array([1.08, 0.2, 0.03])
    pattern_render = make_pattern_image_operator(
        projector_matrices=maps, template=template, detector=detector
    )
    exposure = jnp.tile(jnp.log(jnp.array([4e7, 3.0])), 5)
    for u in [0.98, 1.0, 1.02]:
        measured = predict_pattern(p, c, u, projector, 2.5e-9, response)
        e = np.eye(5)
        e[:4, :4] = post_field_change(projector, u, response)
        nominal = np.eye(5)
        nominal[:4, :4] = post_field_change(projector, u, jnp.array([1.0, 0.0, 0.0]))
        direct = make_image_operator(
            projector_matrices=maps @ e, template=template, detector=detector
        )
        beam = predict_beam_coordinates(p.at[12].add(jnp.log(u)), c)
        np.testing.assert_allclose(
            pattern_render(jnp.r_[measured, exposure], jnp.asarray(maps @ nominal)),
            direct(jnp.r_[beam, exposure]),
            rtol=1e-10,
            atol=1e-7,
        )


def test_objective_changes_distinguish_the_old_exact_family():
    projector, _, _ = setup_models()
    p = np.r_[generating_parameters(), -3.0, 0.2, -0.15]
    alternative = equivalent_aperture_model(p, objective_strength=-0.6)
    c = jnp.asarray(
        json.loads((EXAMPLES / "condenser_measurement_protocol.json").read_text())[
            "calibration_controls"
        ][0]
    )
    response = jnp.array([1.0, 0.0, 0.0])
    f = lambda p, u: predict_pattern(p, c, u, projector, 2.5e-9, response)
    np.testing.assert_allclose(f(p, 1.0), f(alternative, 1.0), rtol=1e-9, atol=1e-10)
    assert np.linalg.norm(f(p, 1.02) - f(alternative, 1.02)) > 0.01
    # A post-field slope derivative must be present away from reference and
    # vanish at the calibrated reference setting.
    derivative = jax.jacfwd(lambda r: predict_pattern(p, c, 1.0, projector, 2.5e-9, r))(
        response
    )
    np.testing.assert_allclose(derivative, 0.0, atol=1e-12)
    derivative = jax.jacfwd(
        lambda r: predict_pattern(p, c, 1.02, projector, 2.5e-9, r)
    )(response)
    step = 1e-5
    numerical = (
        predict_pattern(p, c, 1.02, projector, 2.5e-9, response.at[1].add(step))
        - predict_pattern(p, c, 1.02, projector, 2.5e-9, response.at[1].add(-step))
    ) / (2 * step)
    np.testing.assert_allclose(derivative[:, 1], numerical, rtol=1e-5, atol=1e-7)


def test_local_information_survives_three_unknown_post_response_terms():
    from condenser_aperture_experiment import (
        aperture_specification,
        make_transmission_operator,
    )

    projector, _, _ = setup_models()
    p = jnp.r_[
        generating_parameters(),
        np.log(0.0025 / (0.056 - 0.0025)),
        0.2,
        -0.15,
        1.08,
        0.2,
        0.03,
    ]
    controls = jnp.asarray(
        json.loads((EXAMPLES / "condenser_measurement_protocol.json").read_text())[
            "calibration_controls"
        ]
    )
    a, b = np.meshgrid([0.75, 1.05, 1.35], [0.72, 1.02, 1.32])
    aperture_controls = jnp.asarray(np.c_[a.ravel(), b.ravel(), np.ones(a.size)])
    transmission = make_transmission_operator(
        specification=aperture_specification(), side="after"
    )

    def observations(p, include_sweep):
        baseline = jax.vmap(
            lambda c: predict_pattern(p, c, 1.0, projector, 2.5e-9, p[22:])
        )(controls)
        currents = jax.vmap(transmission, in_axes=(None, 0))(p[:22], aperture_controls)
        result = jnp.r_[baseline.ravel(), currents.ravel()]
        if include_sweep:
            sweep = jax.vmap(
                lambda u: jax.vmap(
                    lambda c: predict_pattern(p, c, u, projector, 2.5e-9, p[22:])
                )(controls[::4])
            )(jnp.array([0.98, 1.02]))
            result = jnp.r_[result, sweep.ravel()]
        return result

    before = np.asarray(jax.jacfwd(lambda p: observations(p, False))(p))
    after = np.asarray(jax.jacfwd(lambda p: observations(p, True))(p))

    def rank(matrix):
        scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-30)
        s = np.linalg.svd(matrix / scale, compute_uv=False)
        return int(np.sum(s > s[0] * 1e-9))

    assert rank(before) == 21
    assert rank(after) == 25
