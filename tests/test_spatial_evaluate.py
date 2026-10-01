from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import Detector
from temgym_core.evaluate import describe_plan, evaluate_gaussians, gpu_kernel_available
from temgym_core.gaussian import FreeSpacePropagator, make_gaussian
from temgym_core.render import RenderConfig

jax.config.update("jax_enable_x64", True)

SHAPE = (17, 19)


def bundle():
    axis = np.linspace(-55e-9, 55e-9, 23)
    x, y = np.meshgrid(axis, axis)
    i = np.arange(x.size)
    return make_gaussian(
        x=x.ravel(), y=y.ravel(), dx=2e-5 * np.sin(i), dy=1e-5 * np.cos(i),
        amp=0.2 + 0.05 * np.cos(i), phase=1.7 * i,
        waist_x=(0.7 + 0.15 * np.sin(i)) * 1e-9,
        waist_y=(0.9 + 0.1 * np.cos(i)) * 1e-9,
        voltage=200e3 + 100 * np.sin(i),
    ).derive(pathlength=jnp.full(x.size, 0.71875))


def grid():
    """A regular detector: it has a pixel-to-metre map, so the sparse renderer plans on it."""
    return Detector(z=0.0, pixel_size=(130e-9 / 16, 136e-9 / 18), shape=SHAPE)


def irregular_grid():
    """Only pixel positions: the renderers take their dense kernels."""
    x, y = np.meshgrid(np.linspace(-68e-9, 68e-9, 19), np.linspace(-65e-9, 65e-9, 17))
    coords = jnp.asarray(np.column_stack((x.ravel(), y.ravel())))
    return SimpleNamespace(shape=x.shape, coords=coords)


def numpy_reference(beam, coords):
    """Independent coherent sum with path phase separated to retain precision."""
    b = beam.to_vector()
    delta = np.asarray(coords)[:, None, :] - np.asarray(b.r_xy)[None]
    q = np.asarray(b.Q_inv)
    k = np.asarray(b.k)
    linear = np.einsum("pbi,bi->pb", delta, np.asarray(b.d_xy))
    quadratic = np.einsum("pbi,bij,pbj->pb", delta, q, delta)
    amplitude = np.asarray(b.amplitude) * np.exp(1j * k * np.asarray(b.pathlength))
    return np.sum(amplitude * np.exp(1j * k * (linear + 0.5 * quadratic)), axis=1)


@pytest.fixture
def gpu():
    if not gpu_kernel_available():
        pytest.skip("requires verified Pallas GPU backend")


@pytest.mark.parametrize("tile", [16, 32, 64])
def test_spatial_complex_interference_matches_independent_sum_and_tail_bound(gpu, tile):
    beam, points = bundle(), grid()
    cutoff = 1e-10
    plan = describe_plan(beam, points, RenderConfig(cutoff=cutoff, bins_per_axis=32))
    assert plan.dense == 0 and sum(plan.class_counts) > 0, "test must exercise the sparse kernels"
    actual = np.asarray(evaluate_gaussians(beam, points, method="spatial",
                                          tile_beams=tile, spatial_bins=32,
                                          spatial_cutoff=cutoff))
    expected = numpy_reference(beam, points.coords).reshape(points.shape)
    bound = cutoff * np.abs(np.asarray(beam.amplitude)).sum()
    assert actual.dtype == np.complex128
    assert np.isfinite(actual).all()
    assert np.max(abs(actual - expected)) < bound + 2e-12
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-9)


@pytest.mark.parametrize("make_grid", [grid, irregular_grid], ids=["raster", "irregular"])
def test_jitted_calls_rebuild_the_plan_after_translation_rotation_and_broadening(gpu, make_grid):
    beam, points = bundle(), make_grid()
    render = jax.jit(lambda b: evaluate_gaussians(b, points, method="spatial", tile_beams=32))
    reference = jax.jit(lambda b: evaluate_gaussians(b, points, method="gpu_kernel", tile_beams=32))
    angle = 0.71
    rotation = jnp.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = beam.r_xy @ rotation.T + jnp.array([17e-9, -9e-9])
    slopes = beam.d_xy @ rotation.T
    changed = beam.derive(x=xy[:, 0], y=xy[:, 1], dx=slopes[:, 0], dy=slopes[:, 1],
                          Q_inv=rotation @ beam.Q_inv @ rotation.T)
    moderately_broadened = FreeSpacePropagator()(changed, 1e-6)
    if make_grid is grid:
        assert describe_plan(moderately_broadened, points).dense == 0
    broadened = FreeSpacePropagator()(changed, 5e-6)
    results = []
    for value in (beam, changed, moderately_broadened, broadened, beam):
        actual, expected = np.asarray(render(value)), np.asarray(reference(value))
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-10)
        results.append(actual)
    assert np.linalg.norm(results[0] - results[1]) > 0.1
    assert np.linalg.norm(results[1] - results[2]) > 0.1
    np.testing.assert_array_equal(results[0], results[-1])


def test_crowded_and_broad_beamlets_are_summed_without_losing_any(gpu):
    points = grid()
    crowded = bundle().derive(x=jnp.zeros(529), y=jnp.zeros(529)).to_vector()
    broad = make_gaussian(x=jnp.linspace(-1e-9, 1e-9, 71),
                          waist_x=1e-6, waist_y=3e-6, voltage=200e3).to_vector()
    assert describe_plan(crowded, points).dense == 0     # one crowded bin, still sparse
    assert describe_plan(broad, points).dense == 71      # footprints beyond the last class edge
    for beam in (crowded, broad):
        actual = evaluate_gaussians(beam, points, method="spatial", tile_beams=32)
        expected = evaluate_gaussians(beam, points, method="gpu_kernel", tile_beams=32)
        bound = 2e-12 * np.abs(np.asarray(beam.amplitude)).sum()
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=bound + 1e-13)


def test_occupied_bin_spans_many_tiles_without_capacity_truncation(gpu):
    axis = jnp.linspace(-200e-9, 200e-9, 41)
    x, y = jnp.meshgrid(axis, axis)
    x, y = x.ravel().at[:128].set(0.), y.ravel().at[:128].set(0.)
    beam = make_gaussian(x=x, y=y, phase=jnp.arange(x.size) * 0.3,
                         waist_x=0.5e-9, waist_y=0.5e-9, voltage=200e3)
    points = grid()
    plan = describe_plan(beam, points)
    assert plan.largest_bin >= 128
    assert plan.dense == 0
    actual = evaluate_gaussians(beam, points, method="spatial", tile_beams=32)
    expected = evaluate_gaussians(beam, points, method="gpu_kernel", tile_beams=32)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_bin_boundaries_and_far_outside_pixels(gpu):
    beam = bundle()
    xy = np.concatenate([np.linspace(-70e-9, 70e-9, 33)[:, None] * np.ones((1, 2)),
                         [[1e5, -1e5]]])
    points = SimpleNamespace(shape=(1, len(xy)), coords=jnp.asarray(xy))
    field = np.asarray(evaluate_gaussians(beam, points, method="spatial", spatial_bins=32,
                                         spatial_cutoff=1e-9))
    expected = numpy_reference(beam, points.coords).reshape(points.shape)
    bound = 1e-9 * np.abs(np.asarray(beam.amplitude)).sum()
    assert np.max(abs(field - expected)) < bound + 1e-12
    assert field[0, -1] == 0


def test_invalid_and_nearly_singular_curvature_are_summed_densely():
    beam, points = bundle(), grid()
    for q in (beam.Q_inv.at[:, 0, 0].set(-1j),
              jnp.broadcast_to(jnp.array([[1j, 0.], [0., 1e-18j]]), (529, 2, 2))):
        plan = describe_plan(beam.derive(Q_inv=q), points, RenderConfig(bins_per_axis=32))
        assert plan.dense == 529 and sum(plan.class_counts) == 0


@pytest.mark.parametrize("make_grid", [grid, irregular_grid], ids=["raster", "irregular"])
def test_nonfinite_beamlets_contribute_zero_and_do_not_poison_the_others(gpu, make_grid):
    beam, points = bundle(), make_grid()
    poisoned = beam.derive(amplitude=beam.amplitude.at[0].set(jnp.nan))
    healthy = beam.derive(amplitude=beam.amplitude.at[0].set(0.0))
    for method in ("spatial", "gpu_kernel"):
        actual = evaluate_gaussians(poisoned, points, method=method, tile_beams=32)
        expected = evaluate_gaussians(healthy, points, method=method, tile_beams=32)
        assert np.isfinite(np.asarray(actual)).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_nonfinite_pixel_positions_stay_local_to_their_pixel(gpu):
    beam, points = bundle(), irregular_grid()
    points.coords = points.coords.at[0, 0].set(jnp.nan)
    actual = np.asarray(evaluate_gaussians(beam, points, method="spatial", tile_beams=32))
    expected = np.asarray(evaluate_gaussians(beam, points, method="gpu_kernel", tile_beams=32))
    assert np.isnan(actual.ravel()[0]) and np.isnan(expected.ravel()[0])
    assert np.isfinite(actual.ravel()[1:]).all()
    np.testing.assert_allclose(actual.ravel()[1:], expected.ravel()[1:], rtol=1e-11, atol=1e-11)


def test_empty_observation_grid(gpu):
    points = SimpleNamespace(shape=(0, 1), coords=jnp.zeros((0, 2)))
    result = evaluate_gaussians(bundle(), points, method="spatial")
    assert result.shape == (0, 1)
    assert result.dtype == jnp.complex128


def test_portable_fallback_and_shared_scalar_beam_fields(monkeypatch):
    import importlib
    evaluate_module = importlib.import_module("temgym_core.evaluate")
    monkeypatch.setattr(evaluate_module, "gpu_kernel_available", lambda: False)
    beam = make_gaussian(waist_x=2e-9, waist_y=3e-9, voltage=200e3)
    points = irregular_grid()
    actual = evaluate_gaussians(beam, points, method="spatial")
    expected = evaluate_gaussians(beam, points, method="scan")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("settings", [
    {"spatial_cutoff": 0.}, {"spatial_cutoff": 1.}, {"spatial_cutoff": np.nan},
    {"spatial_bins": 1}, {"spatial_bins": 2048}, {"tile_beams": 17},
])
def test_invalid_spatial_configuration(settings):
    with pytest.raises(ValueError):
        evaluate_gaussians(bundle(), grid(), method="spatial", **settings)
