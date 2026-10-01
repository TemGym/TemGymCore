"""The pixel-index frame is an exact reparametrisation and gives detector-pose gradients for free."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from render_bundles import field_scan, random_packed, rel

from temgym_core.components import Detector
from temgym_core.render import RenderConfig, evaluate_packed
from temgym_core.render.packing import affine_from_grid, index_coords, to_index_space

jax.config.update("jax_enable_x64", True)

CONFIG = RenderConfig(impl="reference", pixel_chunk=64)
SHAPE = (13, 17)


def detector(rotation=27.0, centre=(0.2, -0.1), pixel_size=(0.11, 0.09), flip_y=False):
    return Detector(z=0.0, pixel_size=pixel_size, shape=SHAPE, rotation=rotation,
                    centre=centre, flip_y=flip_y)


@pytest.mark.parametrize("flip", [False, True])
def test_affine_map_reproduces_the_grid_coordinates(flip):
    grid = detector(flip_y=flip)
    j, o = affine_from_grid(grid)
    xy = index_coords(SHAPE) @ j.T + o
    assert float(jnp.max(jnp.abs(xy - grid.coords))) <= 1e-14


def test_grids_without_a_pixel_map_have_no_affine():
    class Plain:
        shape = SHAPE
        coords = jnp.zeros((SHAPE[0] * SHAPE[1], 2))

    assert affine_from_grid(Plain()) is None


@pytest.mark.parametrize("flip", [False, True])
def test_index_space_field_equals_metre_space_field(flip):
    rng = np.random.default_rng(0)
    packed = jnp.asarray(random_packed(rng, 30, width=0.3, box=0.6, tilt=15.0, curv=20.0))
    grid = detector(flip_y=flip)
    j, o = affine_from_grid(grid)
    metre = evaluate_packed(packed, grid.coords, config=CONFIG, exact=True)
    index = evaluate_packed(to_index_space(packed, j, o), shape=SHAPE, config=CONFIG, exact=True)
    assert rel(index, metre) <= 1e-12


def test_metre_frame_sparse_planning_uses_the_affine_map():
    """Metre-space packed bundles plan in the index frame through ``affine``; results agree."""
    rng = np.random.default_rng(3)
    packed = jnp.asarray(random_packed(rng, 40, width=0.15, box=0.5, tilt=8.0, curv=10.0))
    grid = detector(flip_y=True)
    j, o = affine_from_grid(grid)
    affine = jnp.concatenate([j.ravel(), o])
    config = RenderConfig(impl="reference", cutoff=1e-9, pixel_chunk=64)
    metre = evaluate_packed(packed, grid.coords, shape=SHAPE, affine=affine, config=config)
    index = evaluate_packed(to_index_space(packed, j, o), shape=SHAPE, config=config)
    assert rel(metre, index) <= 1e-11
    assert rel(metre, field_scan(packed, grid.coords, config.decay_limit)) <= 1e-11


def test_pose_gradients_are_ordinary_autodiff_through_the_canonical_map():
    """d loss / d (rotation, centre, pixel size): kernels see indices only, JAX does the rest."""
    rng = np.random.default_rng(1)
    packed = jnp.asarray(random_packed(rng, 25, width=0.3, box=0.6, tilt=10.0, curv=15.0))
    weights = jnp.asarray(rng.normal(size=SHAPE[0] * SHAPE[1]))
    pose = jnp.array([27.0, 0.2, -0.1, 0.11, 0.09])

    def loss_of(field_fn):
        def loss(p):
            grid = detector(rotation=p[0], centre=(p[1], p[2]), pixel_size=(p[3], p[4]))
            return jnp.sum(weights * jnp.abs(field_fn(grid)) ** 2)
        return loss

    def canonical(grid):
        j, o = affine_from_grid(grid)
        return evaluate_packed(to_index_space(packed, j, o), shape=SHAPE, config=CONFIG, exact=True)

    def metre_scan(grid):  # coordinates enter autodiff directly
        return field_scan(packed, grid.coords)

    mine = jax.grad(loss_of(canonical))(pose)
    expected = jax.grad(loss_of(metre_scan))(pose)
    assert float(jnp.max(jnp.abs(mine - expected)) / jnp.max(jnp.abs(expected))) <= 1e-10
