"""Regression on the real workload: the hologram notebook's 160,801 beamlets on a 1024^2 camera.

The bundle has phases of up to ~1e10 rad and footprint widths from 169 px to 85,000 px, and only
about 12,500 beamlets reach the sensor.  The sparse renderer (radius classes, no host-side
culling) must reproduce the dense sums: the forward field and the gradient of an intensity loss,
in both coordinate modes.  Run with ``JAX_PLATFORMS=cuda``; each dense whole-bundle pass takes
seconds.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legacy_dense_kernel import legacy_dense_kernel

from temgym_core.evaluate import _prepare_gaussian_params, describe_plan, evaluate_gaussians
from temgym_core.render import RenderConfig, evaluate_packed
from temgym_core.render.packing import (
    affine_from_grid,
    index_coords,
    pack_gaussian_params,
    to_index_space,
)

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def peak_error(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


@pytest.fixture(scope="module")
def gpu():
    from temgym_core.evaluate import gpu_kernel_available

    if not gpu_kernel_available():
        pytest.skip("requires a verified Pallas GPU backend (JAX_PLATFORMS=cuda)")


@pytest.fixture(scope="module")
def hologram(gpu):
    from hologram_bundle import hologram_bundle

    beam, detector = hologram_bundle()
    assert beam.x.shape[0] == 160801 and tuple(detector.shape) == (1024, 1024)
    return beam, detector


def test_the_bundle_is_classified_by_radius_and_mostly_culled(hologram):
    beam, detector = hologram
    plan = describe_plan(beam, detector)
    assert 12000 < sum(plan.class_counts) + plan.dense < 13000
    assert plan.inactive > 148000
    assert 300 <= plan.dense <= 500                     # the extremely broad lattice line
    assert max(plan.class_counts) > 10000               # the bulk shares one radius class


def test_index_mode_reproduces_the_dense_sum_and_its_gradient(hologram):
    beam, detector = hologram
    j, o = affine_from_grid(detector)
    packed = to_index_space(pack_gaussian_params(*_prepare_gaussian_params(beam)), j, o)
    shape = tuple(detector.shape)
    config = RenderConfig()
    pixels = index_coords(shape)                        # dense kernels, same predicate
    sparse = jax.jit(lambda p: evaluate_packed(p, shape=shape, config=config))(packed)
    same_pairs = jax.jit(lambda p: evaluate_packed(p, pixels, config=config))(packed)
    exact = jax.jit(lambda p: evaluate_packed(p, pixels, config=config, exact=True))(packed)
    assert peak_error(sparse, same_pairs) <= 1e-11
    assert peak_error(sparse, exact) <= 1e-10           # truncation at cutoff 1e-12

    weights = jnp.asarray(np.random.default_rng(0).normal(size=shape[0] * shape[1]))

    def loss(render):
        return lambda p: jnp.sum(weights * jnp.abs(render(p)) ** 2)

    g_sparse = jax.jit(jax.grad(loss(lambda p: evaluate_packed(p, shape=shape, config=config))))(packed)
    g_pairs = jax.jit(jax.grad(loss(lambda p: evaluate_packed(p, pixels, config=config))))(packed)
    scale = float(jnp.max(jnp.abs(g_pairs)))
    assert float(jnp.max(jnp.abs(g_sparse - g_pairs))) <= 1e-9 * scale
    g_exact = jax.jit(jax.grad(loss(lambda p: evaluate_packed(p, pixels, config=config, exact=True))))(packed)
    assert float(jnp.max(jnp.abs(g_sparse - g_exact))) <= 1e-6 * scale


def test_metre_mode_agrees_with_the_historical_dense_kernel(hologram):
    beam, detector = hologram
    legacy = legacy_dense_kernel(*_prepare_gaussian_params(beam), detector.coords,
                                 tile_pixels=16, tile_beams=16).reshape(detector.shape)
    metre = RenderConfig(coordinates="metre")
    dense = evaluate_gaussians(beam, detector, method="dense", config=metre)
    assert peak_error(dense, legacy) <= 1e-12
    tight = RenderConfig(coordinates="metre", cutoff=1e-14)     # truncation below the tolerance
    sparse = evaluate_gaussians(beam, detector, method="sparse", config=tight)
    assert peak_error(sparse, legacy) <= 1e-12
    default = evaluate_gaussians(beam, detector, method="sparse", config=metre)
    assert peak_error(default, legacy) <= 3e-12         # cutoff 1e-12 times the total amplitude
    index = evaluate_gaussians(beam, detector, method="dense")
    assert peak_error(index, legacy) <= 1e-6            # canonicalisation re-rounds ~1e10 rad
