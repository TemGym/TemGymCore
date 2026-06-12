import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from temgym_core.constants import energy2wavelength, compute_Rc_from_voltage
from temgym_core.components import ElectromagneticLens, Plane
from temgym_core.plotting import plot_model, PlotParams, _stack_ray_positions, _compute_cumulative_rotation
from temgym_core.source import make_waist_divergence_rays
from temgym_core.ray import Ray
from temgym_core.gaussian import make_gaussian


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
        # main bundle line + waist line + divergence line (exclude side guides)
        ray_like_lines = [
            line for line in ax.lines
            if np.asarray(line.get_xdata(), dtype=float).size >= 3
        ]
        assert len(ray_like_lines) == 3
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
        ElectromagneticLens(z=1.0, turns=1.0, current=1.0, Gc=1.0),
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
        break_same_z_jumps=True,
    )

    try:
        ray_lines = [
            line for line in ax.lines
            if np.any(np.isnan(np.asarray(line.get_xdata(), dtype=float)))
        ]
        assert ray_lines
        x = np.asarray(ray_lines[0].get_xdata(), dtype=float)
        z = np.asarray(ray_lines[0].get_ydata(), dtype=float)
        nan_idx = np.flatnonzero(np.isnan(x) | np.isnan(z))
        assert nan_idx.size > 0

        has_expected_break = False
        for i in nan_idx:
            if i == 0 or i == x.size - 1:
                continue
            same_z = np.isclose(z[i - 1], z[i + 1], rtol=0.0, atol=1e-15)
            jump_x = ~np.isclose(x[i - 1], x[i + 1], rtol=0.0, atol=1e-18)
            if same_z and jump_x:
                has_expected_break = True
                break
        assert has_expected_break
    finally:
        plt.close(fig)


def test_plot_model_can_keep_same_z_position_jumps_connected():
    components = (
        ElectromagneticLens(z=1.0, turns=1.0, current=1.0, Gc=1.0),
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
        break_same_z_jumps=False,
    )

    try:
        # No split-with-NaN should be present when disabled.
        nan_lines = [
            line for line in ax.lines
            if np.any(np.isnan(np.asarray(line.get_xdata(), dtype=float)))
        ]
        assert not nan_lines

        # A same-z x jump should still exist in the connected polyline.
        found_same_z_jump = False
        for line in ax.lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            z = np.asarray(line.get_ydata(), dtype=float)
            if x.size < 4:
                continue
            same_z = np.isclose(np.diff(z), 0.0, rtol=0.0, atol=1e-15)
            jump_x = ~np.isclose(np.diff(x), 0.0, rtol=0.0, atol=1e-18)
            if np.any(same_z & jump_x):
                found_same_z_jump = True
                break
        assert found_same_z_jump
    finally:
        plt.close(fig)


def test_plot_model_r_coordinate_is_rotation_invariant():
    components = (
        ElectromagneticLens(z=1.0, turns=1.0, current=1.0, Gc=1.0),
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

    fig_x, ax_x = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="x",
        break_same_z_jumps=True,
    )
    fig_r, ax_r = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="r",
        break_same_z_jumps=True,
    )

    try:
        nan_lines_x = [
            line for line in ax_x.lines
            if np.any(np.isnan(np.asarray(line.get_xdata(), dtype=float)))
        ]
        nan_lines_r = [
            line for line in ax_r.lines
            if np.any(np.isnan(np.asarray(line.get_xdata(), dtype=float)))
        ]
        # In x-space the in-plane rotation introduces a same-z jump, in r-space it does not.
        assert nan_lines_x
        assert not nan_lines_r

        xlim_r = ax_r.get_xlim()
        assert xlim_r[0] >= 0.0
    finally:
        plt.close(fig_x)
        plt.close(fig_r)


def test_plot_model_x_rot_removes_rotation_jump():
    """x_rot mode should eliminate same-z jumps caused by EM lens rotation."""
    components = (
        ElectromagneticLens(z=1.0, turns=1.0, current=1.0, Gc=1.0),
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

    fig_x, ax_x = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="x",
        break_same_z_jumps=True,
    )
    fig_rot, ax_rot = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="x_rot",
        break_same_z_jumps=True,
    )

    try:
        nan_lines_x = [
            line for line in ax_x.lines
            if np.any(np.isnan(np.asarray(line.get_xdata(), dtype=float)))
        ]
        nan_lines_rot = [
            line for line in ax_rot.lines
            if np.any(np.isnan(np.asarray(line.get_xdata(), dtype=float)))
        ]
        # In lab-frame x the rotation introduces a same-z jump;
        # in x_rot that jump vanishes.
        assert nan_lines_x
        assert not nan_lines_rot
    finally:
        plt.close(fig_x)
        plt.close(fig_rot)


def test_plot_model_y_mode_extracts_y_coordinate():
    """y mode should plot lab-frame y position."""
    components = (Plane(z=1.0),)
    rays = Ray(
        x=np.asarray([0.0], dtype=float),
        y=np.asarray([5e-4], dtype=float),
        dx=np.asarray([0.0], dtype=float),
        dy=np.asarray([0.0], dtype=float),
        z=np.asarray([0.0], dtype=float),
        pathlength=np.asarray([0.0], dtype=float),
    )
    params = PlotParams(solid_beam=False, show_side_guides=False)

    fig, ax = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="y",
        break_same_z_jumps=False,
        plot_params=params,
    )

    try:
        # Single ray -> single polyline; check it carries the y value.
        assert len(ax.lines) >= 1
        x_data = np.asarray(ax.lines[0].get_xdata(), dtype=float)
        np.testing.assert_allclose(x_data, 5e-4, atol=1e-12)
    finally:
        plt.close(fig)


def test_compute_cumulative_rotation_matches_em_lenses():
    em1 = ElectromagneticLens(z=1.0, turns=1.0, current=1.0, Gc=1.0)
    em2 = ElectromagneticLens(z=2.0, turns=1.0, current=2.0, Gc=1.0)
    components = (em1, Plane(z=1.5), em2)

    angles = _compute_cumulative_rotation(components, voltage=200e3, include_input_rays=True)

    Rc_val = float(compute_Rc_from_voltage(200e3))  # default voltage

    # include_input_rays adds 1 entry; then 2 per component = 7 total
    assert angles.shape == (7,)
    # Input ray: 0
    assert angles[0] == 0.0
    # After propagation to em1: 0
    assert angles[1] == 0.0
    # After em1 applied: Rc(V) * excitation = Rc_val * 1
    np.testing.assert_allclose(angles[2], Rc_val)
    # After propagation to Plane: still Rc_val
    np.testing.assert_allclose(angles[3], Rc_val)
    # After Plane applied: still Rc_val
    np.testing.assert_allclose(angles[4], Rc_val)
    # After propagation to em2: still Rc_val
    np.testing.assert_allclose(angles[5], Rc_val)
    # After em2 applied: Rc_val + Rc_val * 2 = 3 * Rc_val
    np.testing.assert_allclose(angles[6], 3 * Rc_val)


def test_compute_cumulative_rotation_accepts_per_ray_voltage():
    em = ElectromagneticLens(z=1.0, turns=1.0, current=2.0, Gc=1.0)
    components = (em,)
    voltages = np.asarray([100e3, 200e3], dtype=float)

    angles = _compute_cumulative_rotation(
        components,
        voltage=voltages,
        include_input_rays=True,
    )

    expected = np.asarray(compute_Rc_from_voltage(voltages), dtype=float) * 2.0
    assert angles.shape == (3, 2)
    np.testing.assert_allclose(angles[0], np.zeros_like(voltages))
    np.testing.assert_allclose(angles[1], np.zeros_like(voltages))
    np.testing.assert_allclose(angles[2], expected)


def test_plot_model_accepts_gaussian_bundle_voltage_array():
    rays = make_gaussian(
        x=np.asarray([-1e-6, 1e-6], dtype=float),
        y=0.0,
        waist_x=10e-9,
        waist_y=10e-9,
        voltage=200e3,
    )
    components = (Plane(z=1.0),)

    fig, ax = plot_model(components, rays=rays, include_input_rays=True)

    try:
        assert ax.lines
    finally:
        plt.close(fig)


def test_plot_model_x_corot_alias_still_works():
    """Legacy alias x_corot should produce identical results to x_rot."""
    components = (
        ElectromagneticLens(z=1.0, turns=1.0, current=1.0, Gc=1.0),
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

    fig_rot, ax_rot = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="x_rot",
        break_same_z_jumps=False,
    )
    fig_alias, ax_alias = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="x_corot",
        break_same_z_jumps=False,
    )

    try:
        assert len(ax_rot.lines) == len(ax_alias.lines)
        for l_rot, l_alias in zip(ax_rot.lines, ax_alias.lines):
            np.testing.assert_allclose(
                np.asarray(l_rot.get_xdata(), dtype=float),
                np.asarray(l_alias.get_xdata(), dtype=float),
            )
    finally:
        plt.close(fig_rot)
        plt.close(fig_alias)


def test_plot_model_rejects_invalid_ray_coordinate():
    components = (Plane(z=1.0),)
    rays = _single_center_ray()

    with pytest.raises(ValueError, match="ray_coordinate"):
        plot_model(components, rays=rays, ray_coordinate="bad-mode")


def test_stack_ray_positions_rejects_radial_sign_with_non_r_mode():
    rays = _single_center_ray()

    with pytest.raises(ValueError, match="radial_sign"):
        _stack_ray_positions((rays,), ray_coordinate="x_rot", radial_sign=np.asarray([1.0]))


def test_plot_model_ray_half_sign_selects_positive_or_negative_half():
    components = (Plane(z=1.0),)
    rays = Ray(
        x=np.asarray([0.0, 0.0], dtype=float),
        y=np.asarray([0.0, 0.0], dtype=float),
        dx=np.asarray([-1e-3, 1e-3], dtype=float),
        dy=np.asarray([0.0, 0.0], dtype=float),
        z=np.asarray([0.0, 0.0], dtype=float),
        pathlength=np.asarray([0.0, 0.0], dtype=float),
    )
    params = PlotParams(solid_beam=False, show_side_guides=False)

    fig_pos, ax_pos = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="x",
        ray_half_sign=1,
        break_same_z_jumps=False,
        plot_params=params,
    )
    fig_neg, ax_neg = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="x",
        ray_half_sign=-1,
        break_same_z_jumps=False,
        plot_params=params,
    )

    try:
        # Each half selection should leave a single ray polyline.
        assert len(ax_pos.lines) == 1
        assert len(ax_neg.lines) == 1

        x_pos = np.asarray(ax_pos.lines[0].get_xdata(), dtype=float)
        x_neg = np.asarray(ax_neg.lines[0].get_xdata(), dtype=float)
        assert x_pos[-1] > 0.0
        assert x_neg[-1] < 0.0
    finally:
        plt.close(fig_pos)
        plt.close(fig_neg)


def test_plot_model_rejects_invalid_ray_half_sign():
    components = (Plane(z=1.0),)
    rays = _single_center_ray()

    with pytest.raises(ValueError, match="ray_half_sign"):
        plot_model(components, rays=rays, ray_half_sign=0)


def test_plot_model_r_side_by_sign_places_rays_on_both_sides_without_mirroring():
    components = (Plane(z=1.0),)
    rays = Ray(
        x=np.asarray([0.0, 0.0], dtype=float),
        y=np.asarray([0.0, 0.0], dtype=float),
        dx=np.asarray([-1e-3, 1e-3], dtype=float),
        dy=np.asarray([0.0, 0.0], dtype=float),
        z=np.asarray([0.0, 0.0], dtype=float),
        pathlength=np.asarray([0.0, 0.0], dtype=float),
    )
    params = PlotParams(solid_beam=False, show_side_guides=False)

    fig, ax = plot_model(
        components,
        rays=rays,
        include_input_rays=True,
        ray_coordinate="r",
        r_side_by_sign=True,
        break_same_z_jumps=False,
        plot_params=params,
    )

    try:
        xlim = ax.get_xlim()
        assert xlim[0] < 0.0 < xlim[1]
        assert len(ax.lines) >= 2

        has_pos = False
        has_neg = False
        for line in ax.lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            if np.max(x) > 0.0:
                has_pos = True
            if np.min(x) < 0.0:
                has_neg = True
        assert has_pos and has_neg
    finally:
        plt.close(fig)
