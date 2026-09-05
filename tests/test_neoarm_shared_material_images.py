from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
from neoarm_shared_material_images import (
    TranslatedActionView, fitting_box, box_coverage, measured_initializer, fit_image,
)
from neoarm_condenser_endpoint_calibration import EndpointRows
from temgym_core.multiscale_sample import SampleLevel2D, SampleView
from temgym_core.neoarm_calibration import DetectorGeometry

jax.config.update("jax_enable_x64", True)


def sample(extent=450e-9):
    x, y = np.linspace(-extent, extent, 17), np.linspace(-extent, extent, 19)
    xx, yy = np.meshgrid(x, y)
    level = SampleLevel2D(1e-6*xx+2e-6*yy, -.2-(xx/extent)**2-.3*(yy/extent)**2, x, y)
    return SampleView.from_level(level, rotation_rad=.31, tilt_xy_rad=(.1, -.07))


def test_rigid_translation_preserves_action_field_and_query_derivatives():
    view = sample()
    shift = jnp.array([3e-9, -4e-9])
    moved = TranslatedActionView(view, shift)
    points = jnp.array([[7e-9, -2e-9], [-11e-9, 13e-9]])
    np.testing.assert_allclose(moved.evaluate_actions(points), view.evaluate_actions(points-shift))
    np.testing.assert_allclose(jax.jacfwd(moved.evaluate_actions)(points),
        jax.jacfwd(view.evaluate_actions)(points-shift), rtol=1e-12, atol=1e-10)


def test_complete_fitting_box_coverage_encloses_random_rotations_and_translations():
    view, detector = sample(), DetectorGeometry(mtf_sigma_px=.65)
    bounds = fitting_box(40000., 2.7)
    coverage = box_coverage(view, detector, bounds, grid_halfwidth_m=330e-9, source_centre=[2e-9, -1e-9])
    assert coverage["full_image_fitting_box_covered"]
    rng = np.random.default_rng(618)
    p = rng.uniform(*bounds, (200, 7))
    corners = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])*131*55e-6
    local_points = []
    for row in p:
        c, s = np.cos(row[1]), np.sin(row[1])
        rotation = np.array([[c, -s], [s, c]])
        query = corners@rotation*np.exp(-row[0])+view.centre_xy_m-row[2:4]*1e-9
        local_points.extend(np.asarray(view.world_to_local(query)))
    local = np.asarray(local_points)
    xmin, xmax, ymin, ymax = view.domain_bounds_xyxy_m
    assert np.all(local >= [xmin, ymin]) and np.all(local <= [xmax, ymax])
    with pytest.raises(ValueError, match="complete image-fitting box"):
        box_coverage(sample(240e-9), detector, bounds, grid_halfwidth_m=100e-9, source_centre=[0., 0.])


def test_initializer_uses_only_measured_image_and_signed_calibration():
    detector = DetectorGeometry()
    endpoints = EndpointRows(-872.)
    encoded = np.array([3.63, -.17, 2.68, 3.65, .15, 2.66])
    response = np.asarray(endpoints.response(encoded))
    yy, xx = np.indices(detector.shape_yx)
    image = 1000*np.exp(-((xx-128)**2+(yy-127)**2)/(2*25**2))
    bounds = fitting_box(40000., 2.67)
    initial = measured_initializer(image, detector, response, -872., bounds)
    brighter = measured_initializer(2*image, detector, response, -872., bounds)
    assert np.isfinite(initial).all()
    np.testing.assert_allclose(initial[1], 2.67, atol=1e-14)
    np.testing.assert_array_equal(initial[2:4], [0., 0.])
    np.testing.assert_allclose(brighter[4]-initial[4], np.log(2))
    np.testing.assert_allclose(brighter[5], initial[5])
    with pytest.raises(ValueError, match="positive measured signal"):
        measured_initializer(np.zeros_like(image), detector, response, -872., bounds)


def test_feasible_gls_updates_from_fitted_counts_without_generating_weights():
    design = jnp.asarray(np.random.default_rng(612).normal(size=(128, 7)))
    predict = jax.jit(lambda p: (500.+20*design@p).reshape(8, 16))
    target = np.array([.2, -.4, .3, -.1, .1, .4, -.2])
    result, variance = fit_image(np.asarray(predict(target)), predict, np.zeros(7),
        (-np.ones(7), np.ones(7)))
    assert result["accepted"] and result["rank"] == 7
    assert len(result["history"]) >= 2
    np.testing.assert_allclose(result["parameters"], target, atol=2e-7)
    np.testing.assert_allclose(variance, predict(jnp.asarray(result["parameters"]))+.6**2)
