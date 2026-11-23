import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.components import ABCDTransfer, Plane
from temgym_core.plotting import plot_model
from temgym_core.ray import Ray
from temgym_core.run import run_to_end

jax.config.update("jax_enable_x64", True)


def ray():
    return Ray(x=1e-6, y=-2e-6, dx=3e-3, dy=1e-3, z=0.0, pathlength=0.0)


def test_free_space_region_matches_a_drift():
    """A free-space map over a region gives the same ray as no component at all."""
    length = 0.05
    drift = ABCDTransfer(A=jnp.eye(2), B=length * jnp.eye(2), z=0.1, length=length)
    with_region = run_to_end(ray(), [drift, Plane(z=0.3)])
    without = run_to_end(ray(), [Plane(z=0.3)])
    for field in ("x", "y", "dx", "dy", "z"):
        np.testing.assert_allclose(getattr(with_region, field), getattr(without, field), rtol=1e-12)


def test_minus_one_transfer_inverts_the_ray_at_the_exit():
    """A transfer of magnification -1 gives, at its exit, minus the ray at its entrance."""
    length = 0.05
    transfer = ABCDTransfer(
        A=-jnp.eye(2), B=jnp.zeros((2, 2)), D=-jnp.eye(2), z=0.1, length=length
    )
    at_entrance = run_to_end(ray(), [Plane(z=0.1)])
    at_exit = run_to_end(ray(), [transfer, Plane(z=0.1 + length)])
    for field in ("x", "y", "dx", "dy"):
        np.testing.assert_allclose(getattr(at_exit, field), -getattr(at_entrance, field), rtol=1e-12)
    np.testing.assert_allclose(at_exit.z, 0.1 + length)


def test_legacy_plot_draws_the_region_as_a_box():
    transfer = ABCDTransfer(A=jnp.eye(2), B=0.05 * jnp.eye(2), z=0.1, length=0.05)
    _, ax = plot_model([transfer, Plane(z=0.3)], rays=ray())
    boxes = [patch for patch in ax.patches if isinstance(patch, matplotlib.patches.Rectangle)]
    assert any(np.isclose(box.get_height(), 0.05) and np.isclose(box.get_y(), 0.1) for box in boxes)
