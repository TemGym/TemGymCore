import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from temgym_core.constants import energy2wavelength
from temgym_core.components import ElectromagneticLens, Plane
from temgym_core.plotting import plot_model
from temgym_core.source import make_waist_divergence_rays
from temgym_core.ray import Ray


def _single_center_ray(*, z: float = 0.0) -> Ray:
    return Ray(
        x=np.asarray([0.0], dtype=float),
        y=np.asarray([0.0], dtype=float),
        dx=np.asarray([0.0], dtype=float),
        dy=np.asarray([0.0], dtype=float),
        z=np.asarray([z], dtype=float),
        pathlength=np.asarray([0.0], dtype=float),
    )


def test_make_waist_divergence_rays_from_voltage():
    waist = 10e-9
    voltage = 200e3
    z0 = 0.25
    x0 = 1e-6
    y0 = -2e-6
    p0 = 0.5

    rays = make_waist_divergence_rays(
        waist,
        voltage=voltage,
        z=z0,
        x0=x0,
        y0=y0,
        pathlength=p0,
    )

    wavelength = float(np.asarray(energy2wavelength(voltage)))
    theta = wavelength / (np.pi * waist)

    np.testing.assert_allclose(np.asarray(rays.x), np.asarray([x0 + waist, x0]))
    np.testing.assert_allclose(np.asarray(rays.y), np.asarray([y0, y0]))
    np.testing.assert_allclose(np.asarray(rays.dx), np.asarray([0.0, theta]))
    np.testing.assert_allclose(np.asarray(rays.dy), np.asarray([0.0, 0.0]))
    np.testing.assert_allclose(np.asarray(rays.z), np.asarray([z0, z0]))
    np.testing.assert_allclose(np.asarray(rays.pathlength), np.asarray([p0, p0]))


def test_make_waist_divergence_rays_requires_single_wave_specifier():
    with pytest.raises(ValueError, match="exactly one"):
        make_waist_divergence_rays(10e-9)

    with pytest.raises(ValueError, match="exactly one"):
        make_waist_divergence_rays(10e-9, voltage=200e3, wavelength=2.5e-12)


def test_make_waist_divergence_rays_rejects_nonpositive_waist():
    with pytest.raises(ValueError, match="must be > 0"):
        make_waist_divergence_rays(0.0, voltage=200e3)


def test_plot_model_overlays_solution_rays_and_expands_extent():
    components = (Plane(z=1.0),)
    rays = _single_center_ray()
    solution_rays = make_waist_divergence_rays(5e-4, voltage=200e3)

    fig, ax = plot_model(
        components,
        rays=rays,
        solution_rays=solution_rays,
        include_input_rays=True,
    )

    try:
        # main bundle line + waist line + divergence line
        assert len(ax.lines) == 3
        xlim = ax.get_xlim()
        assert max(abs(xlim[0]), abs(xlim[1])) > 4e-4
    finally:
        plt.close(fig)


def test_plot_model_rejects_solution_bundle_that_is_not_two_rays():
    components = (Plane(z=1.0),)
    rays = _single_center_ray()
    bad_solution = _single_center_ray()

    with pytest.raises(ValueError, match="exactly 2 rays"):
        plot_model(components, rays=rays, solution_rays=bad_solution)


def test_plot_model_breaks_same_z_position_jumps():
    components = (
        ElectromagneticLens(z=1.0, I0=1.0, Gc=1.0, Rc=float(np.pi / 2.0)),
        Plane(z=2.0),
    )
    rays = Ray(
        x=np.asarray([1e-6], dtype=float),
        y=np.asarray([0.0], dtype=float),
        dx=np.asarray([0.0], dtype=float),
        dy=np.asarray([0.0], dtype=float),
        z=np.asarray([0.0], dtype=float),
        pathlength=np.asarray([0.0], dtype=float),
    )

    fig, ax = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
    )

    try:
        ray_lines = [
            line for line in ax.lines
            if np.any(np.isnan(np.asarray(line.get_xdata(), dtype=float)))
        ]
        assert ray_lines
        x = np.asarray(ray_lines[0].get_xdata(), dtype=float)
        z = np.asarray(ray_lines[0].get_ydata(), dtype=float)

        valid = ~(np.isnan(x) | np.isnan(z))
        x_valid = x[valid]
        z_valid = z[valid]
        same_z = np.isclose(np.diff(z_valid), 0.0, rtol=0.0, atol=1e-15)
        jump_x = ~np.isclose(np.diff(x_valid), 0.0, rtol=0.0, atol=1e-18)
        assert not np.any(same_z & jump_x)
    finally:
        plt.close(fig)
