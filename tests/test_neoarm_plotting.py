from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from temgym_core.components import ElectromagneticLens, Lens, Plane
from temgym_core.plotting import (
    NEOARM_RAY_COLORS,
    AxialRayEvent,
    ColumnElement,
    ColumnSection,
    PositionAngleTrace,
    _format_engineering_length,
    _spread_label_boxes,
    basis_trace_events,
    column_elements_from_components,
    cumulative_larmor_profile,
    export_ray_figure,
    find_gaussian_waists,
    find_ray_crossings,
    gaussian_basis_envelope,
    infer_neoarm_sections,
    make_position_angle_basis_rays,
    plot_neoarm_ray_diagram,
    plot_neoarm_ray_small_multiples,
    trace_gaussian_basis,
    trace_position_angle_basis,
)


def _synthetic_trace(*, label: str = "truth", scale: float = 1.0) -> PositionAngleTrace:
    z = np.linspace(0.0, 1.0, 21)
    position = scale * 2.0e-6 * (1.0 - 2.0 * z)
    angle = scale * 0.7e-6 * np.sin(np.pi * z)
    centroid = 0.08e-6 * z
    return PositionAngleTrace(
        z=z,
        centroid=centroid,
        position=position,
        angle=angle,
        radius=gaussian_basis_envelope(position, angle),
        coordinate="x",
        position_scale=2.0e-6,
        angle_scale=0.7e-6,
        label=label,
    )


def test_make_position_angle_basis_rays_is_calibrated_on_each_axis():
    rays_x = make_position_angle_basis_rays(
        2.5e-6,
        3.0e-3,
        z=-0.2,
        voltage=100e3,
        x0=4.0e-6,
        y0=-7.0e-6,
        axis="x",
    )
    np.testing.assert_allclose(rays_x.x, [6.5e-6, 4.0e-6])
    np.testing.assert_allclose(rays_x.y, [-7.0e-6, -7.0e-6])
    np.testing.assert_allclose(rays_x.dx, [0.0, 3.0e-3])
    np.testing.assert_allclose(rays_x.dy, [0.0, 0.0])
    np.testing.assert_allclose(rays_x.z, [-0.2, -0.2])
    np.testing.assert_allclose(rays_x.voltage, [100e3, 100e3])

    rays_y = make_position_angle_basis_rays(1.0e-6, -2.0e-3, axis="y")
    np.testing.assert_allclose(rays_y.x, [0.0, 0.0])
    np.testing.assert_allclose(rays_y.y, [1.0e-6, 0.0])
    np.testing.assert_allclose(rays_y.dy, [0.0, -2.0e-3])

    with pytest.raises(ValueError, match="axis"):
        make_position_angle_basis_rays(1.0, 1.0, axis="r")


def test_trace_position_angle_basis_matches_thin_lens_geometry():
    components = (Lens(z=1.0, focal_length=0.5), Plane(z=2.0))
    trace = trace_position_angle_basis(
        components,
        position_scale=2.0e-6,
        angle_scale=3.0e-6,
        z=0.0,
    )

    # The run records input, pre/post component states, so select the final
    # sample rather than relying on an implementation-specific step index.
    assert trace.z[-1] == pytest.approx(2.0)
    assert trace.position[-1] == pytest.approx(-2.0e-6)
    assert trace.angle[-1] == pytest.approx(0.0, abs=1e-15)
    assert trace.radius[-1] == pytest.approx(2.0e-6)
    np.testing.assert_allclose(trace.centroid, 0.0, atol=1e-15)


def test_trace_gaussian_basis_uses_both_orthogonal_ray_responses():
    propagation_distance = 0.15
    waist = 2.0e-6
    components = (Plane(z=propagation_distance),)
    trace_x = trace_gaussian_basis(
        components,
        waist,
        voltage=200e3,
        ray_coordinate="x",
    )
    trace_y = trace_gaussian_basis(
        components,
        waist,
        voltage=200e3,
        ray_coordinate="y",
    )

    expected_angle_response = trace_x.angle_scale * propagation_distance
    expected_radius = np.hypot(waist, expected_angle_response)
    assert trace_x.position[-1] == pytest.approx(waist)
    assert trace_x.angle[-1] == pytest.approx(expected_angle_response)
    assert trace_x.radius[-1] == pytest.approx(expected_radius)
    np.testing.assert_allclose(trace_y.position, trace_x.position)
    np.testing.assert_allclose(trace_y.angle, trace_x.angle)
    np.testing.assert_allclose(trace_y.radius, trace_x.radius)


def test_gaussian_envelope_and_event_finders_are_scale_safe():
    assert _format_engineering_length(1.2e-12) == "1.2 pm"
    assert _format_engineering_length(4.15e-8) == "41.5 nm"
    np.testing.assert_allclose(
        gaussian_basis_envelope(np.array([3.0e-12]), np.array([4.0e-12])),
        [5.0e-12],
    )

    crossings = find_ray_crossings(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0e-12, 0.5e-12, -0.5e-12]),
    )
    assert len(crossings) == 1
    assert crossings[0].z == pytest.approx(1.5)

    pair_crossings = find_ray_crossings(
        np.array([0.0, 2.0]),
        np.array([-2.0e-9, 2.0e-9]),
        np.array([2.0e-9, -2.0e-9]),
    )
    assert pair_crossings == (
        AxialRayEvent("crossover", pytest.approx(1.0), pytest.approx(0.0), None, 0),
    )

    waists = find_gaussian_waists(
        np.array([0.0, 2.0]),
        np.array([-2.0e-12, 2.0e-12]),
        np.array([1.0e-12, 1.0e-12]),
    )
    assert len(waists) == 1
    assert waists[0].z == pytest.approx(1.0)
    assert waists[0].radius == pytest.approx(1.0e-12)


def test_variable_height_label_layout_has_non_overlapping_boxes():
    values = [0.300, 0.301, 0.302, 0.303, 0.304]
    line_counts = [1, 1, 2, 1, 2]
    positions = _spread_label_boxes(
        values,
        line_counts,
        0.25,
        0.40,
        line_height_fraction=0.05,
        gap_fraction=0.01,
    )
    span = 0.15
    half_heights = 0.5 * np.asarray(line_counts) * 0.05 * span
    order = np.argsort(positions)
    for left, right in pairwise(order):
        assert positions[right] - half_heights[right] > positions[left] + half_heights[left]
    assert positions.min() - half_heights[np.argmin(positions)] >= 0.25
    assert positions.max() + half_heights[np.argmax(positions)] <= 0.40


def test_basis_trace_events_distinguishes_abcd_conditions_and_waists():
    trace = PositionAngleTrace(
        z=np.array([0.0, 1.0, 2.0]),
        centroid=np.array([0.0, 0.2e-6, 0.4e-6]),
        position=np.array([1.0e-6, 0.0, -1.0e-6]),
        angle=np.array([0.0, 1.0e-6, 0.0]),
        radius=np.array([1.0e-6, 1.0e-6, 1.0e-6]),
        coordinate="x",
        position_scale=1.0e-6,
        angle_scale=1.0e-6,
    )
    events = basis_trace_events(trace)

    assert [event.z for event in events["crossovers"]] == pytest.approx([1.0])
    assert [event.z for event in events["conjugates"]] == pytest.approx([2.0])
    assert events["crossovers"][0].x == pytest.approx(0.2e-6)
    assert events["conjugates"][0].x == pytest.approx(0.4e-6)


def test_component_semantics_and_neoarm_sections_are_explicit():
    components = (
        Lens(z=0.10, focal_length=0.08),
        Plane(z=0.25),
        Lens(z=0.42, focal_length=0.03),
        Plane(z=0.50),
        Lens(z=0.60, focal_length=0.08),
        Plane(z=1.00),
    )
    labels = ("CL1", "C aperture", "Obj post", "Sample", "IL1", "Detector")
    elements = column_elements_from_components(
        components,
        labels=labels,
        extra_elements=(ColumnElement(0.32, "Corrector", "corrector_off", 0.38),),
    )
    roles = {element.label: element.role for element in elements}
    assert roles == {
        "CL1": "condenser_lens",
        "C aperture": "aperture",
        "Corrector": "corrector_off",
        "Obj post": "objective_lens",
        "Sample": "sample",
        "IL1": "intermediate_lens",
        "Detector": "detector",
    }

    sections = infer_neoarm_sections(elements, 0.0, 1.0)
    assert len(sections) == 3
    assert sections[0].z_max == pytest.approx(0.25)
    assert sections[1].z_min == pytest.approx(0.25)
    assert sections[2].z_min == pytest.approx(0.50)
    assert all(section.z_max > section.z_min for section in sections)


def test_larmor_profile_and_finite_thickness_use_calibrated_lens_properties():
    thick_em_lens = ElectromagneticLens(
        z=0.10,
        turns=4.0,
        current=0.5,
        Gc=1.0,
        Tc=0.025,
    )
    thin_em_lens = ElectromagneticLens(
        z=0.55,
        turns=3.0,
        current=-0.25,
        Gc=1.0,
        Tc=0.0,
    )
    elements = column_elements_from_components(
        (thick_em_lens, thin_em_lens),
        labels=("CL1", "IL1"),
    )
    assert elements[0].z_end == pytest.approx(
        thick_em_lens.z + float(thick_em_lens.thickness)
    )
    assert elements[1].z_end is None

    z, theta = cumulative_larmor_profile(
        (thick_em_lens, thin_em_lens),
        voltage=200e3,
        z_min=0.0,
        z_max=1.0,
    )
    expected = float(thick_em_lens.rotation_angle(200e3)) + float(
        thin_em_lens.rotation_angle(200e3)
    )
    assert z[0] == pytest.approx(0.0)
    assert z[-1] == pytest.approx(1.0)
    assert theta[-1] == pytest.approx(expected)
    assert np.count_nonzero(np.isclose(z, thin_em_lens.z)) == 2


def test_neoarm_diagram_has_linear_overview_details_glyphs_and_comparison():
    trace = _synthetic_trace()
    fit = replace(
        _synthetic_trace(label="fit", scale=0.96),
        centroid=trace.centroid + 0.03e-6,
    )
    cl1 = ElectromagneticLens(
        z=0.12,
        turns=4.0,
        current=0.5,
        Gc=1.0,
        Tc=0.01,
    )
    components = (
        cl1,
        Plane(z=0.30),
        Plane(z=0.48),
        Lens(z=0.70, focal_length=0.10),
        Plane(z=1.00),
    )
    sections = (
        ColumnSection("Illumination", 0.0, 0.34),
        ColumnSection("Objective", 0.28, 0.66),
        ColumnSection("Imaging", 0.46, 1.0),
    )
    fig, axes = plot_neoarm_ray_diagram(
        components,
        trace,
        component_labels=("CL1", "C aperture", "Sample", "IL1", "Detector"),
        extra_elements=(ColumnElement(0.37, "Corrector", "corrector_off", 0.44),),
        sections=sections,
        comparison_traces={"fit": fit},
        title="Synthetic NEOARM",
    )
    fig.canvas.draw()

    assert set(axes) == {
        "overview",
        "detail_1",
        "detail_2",
        "detail_3",
        "larmor",
    }
    ray_axes = [axes[key] for key in ("overview", "detail_1", "detail_2", "detail_3")]
    assert all(ax.get_yscale() == "linear" for ax in axes.values())
    assert all(ax.get_xscale() == "linear" for ax in axes.values())
    assert all(ax.get_ylim()[0] > ax.get_ylim()[1] for ax in ray_axes)
    assert all("[" in ax.get_xlabel() for ax in axes.values())
    texts = [text.get_text() for ax in axes.values() for text in ax.texts]
    assert "CL1" in texts
    assert any("Corrector" in text and "off" in text for text in texts)
    assert any("Gaussian waist" in text and "nm" in text for text in texts)
    assert len(axes["overview"].patches) > 0
    assert NEOARM_RAY_COLORS["position"] != NEOARM_RAY_COLORS["angle"]
    assert any(text.get_text() == "fit" for text in fig.legends[0].get_texts())
    assert axes["larmor"].get_ylabel() == "θ [deg]"
    assert axes["larmor"].child_axes[0].get_ylabel() == "θ [rad]"
    expected_rotation_deg = np.rad2deg(float(cl1.rotation_angle(trace.voltage)))
    assert axes["larmor"].lines[0].get_ydata()[-1] == pytest.approx(
        expected_rotation_deg
    )
    plt.close(fig)


def test_neoarm_diagram_uses_union_extent_for_longer_comparison_trace():
    trace = _synthetic_trace(label="truth")
    comparison = replace(
        _synthetic_trace(label="fit", scale=0.97),
        z=np.linspace(-0.20, 1.20, trace.z.size),
    )

    fig, axes = plot_neoarm_ray_diagram(
        (),
        trace,
        comparison_traces={"extended fit": comparison},
        annotate_events=False,
    )
    fig.canvas.draw()

    overview = axes["overview"]
    z_limits = overview.get_ylim()
    assert z_limits[0] > 1.20
    assert z_limits[1] < -0.20
    comparison_height = next(
        line
        for line in overview.lines
        if line.get_label() == "extended fit: height basis"
    )
    assert np.min(comparison_height.get_ydata()) == pytest.approx(-0.20)
    assert np.max(comparison_height.get_ydata()) == pytest.approx(1.20)
    source_markers = [line for line in overview.lines if line.get_marker() == "*"]
    assert len(source_markers) == 1
    assert source_markers[0].get_ydata()[0] == pytest.approx(0.0)
    assert axes["larmor"].get_xlim() == pytest.approx((-0.20, 1.20))
    plt.close(fig)


def test_calibrated_response_diagram_does_not_claim_gaussian_envelope_or_waist():
    trace = _synthetic_trace(label="calibrated reference")
    fit = _synthetic_trace(label="fit", scale=0.98)

    fig, axes = plot_neoarm_ray_diagram(
        (),
        trace,
        comparison_traces={"fit": fit},
        trace_semantics="calibrated_response",
    )
    fig.canvas.draw()

    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    annotation_labels = [
        text.get_text() for axis in axes.values() for text in axis.texts
    ]
    assert "calibrated sample-position response" in legend_labels
    assert "calibrated sample-tilt response" in legend_labels
    assert not any("Gaussian" in label for label in legend_labels)
    assert not any("Gaussian waist" in label for label in annotation_labels)
    assert not axes["overview"].collections
    plt.close(fig)


def test_small_multiples_share_each_rows_physical_extents():
    traces = {
        "truth": _synthetic_trace(label="truth"),
        "initial": _synthetic_trace(label="initial", scale=1.18),
        "fitted": _synthetic_trace(label="fitted", scale=1.01),
    }
    components = (
        Lens(z=0.12, focal_length=0.08),
        Plane(z=0.30),
        Plane(z=0.48),
        Lens(z=0.70, focal_length=0.10),
        Plane(z=1.00),
    )
    sections = (
        ColumnSection("Illumination", 0.0, 0.34),
        ColumnSection("Objective", 0.28, 0.66),
        ColumnSection("Imaging", 0.46, 1.0),
    )
    fig, axes = plot_neoarm_ray_small_multiples(
        components,
        traces,
        component_labels=("CL1", "C aperture", "Sample", "IL1", "Detector"),
        sections=sections,
        title="Truth / initial / fitted",
    )
    fig.canvas.draw()

    assert tuple(axes) == ("truth", "initial", "fitted")
    for row_key in ("overview", "detail_1", "detail_2", "detail_3"):
        row_axes = [axes[scenario][row_key] for scenario in traces]
        x_limits = [ax.get_xlim() for ax in row_axes]
        y_limits = [ax.get_ylim() for ax in row_axes]
        assert all(limit == pytest.approx(x_limits[0]) for limit in x_limits[1:])
        assert all(limit == pytest.approx(y_limits[0]) for limit in y_limits[1:])
        assert all(ax.get_xscale() == "linear" for ax in row_axes)
        assert all(ax.get_yscale() == "linear" for ax in row_axes)
    plt.close(fig)


def test_small_multiples_union_extent_and_scenario_source_positions():
    starts_and_ends = {
        "truth": (0.00, 1.00),
        "initial": (-0.10, 0.92),
        "fitted": (0.04, 1.15),
    }
    traces = {
        scenario: replace(
            _synthetic_trace(label=scenario),
            z=np.linspace(z_start, z_end, 21),
        )
        for scenario, (z_start, z_end) in starts_and_ends.items()
    }

    fig, axes = plot_neoarm_ray_small_multiples((), traces)
    fig.canvas.draw()

    overview_limits = [
        axes[scenario]["overview"].get_ylim() for scenario in starts_and_ends
    ]
    assert all(
        limits == pytest.approx(overview_limits[0])
        for limits in overview_limits[1:]
    )
    assert overview_limits[0][0] > 1.15
    assert overview_limits[0][1] < -0.10
    for scenario, (z_start, z_end) in starts_and_ends.items():
        overview = axes[scenario]["overview"]
        height_line = next(
            line
            for line in overview.lines
            if line.get_label() == f"{scenario}: height basis"
        )
        assert np.min(height_line.get_ydata()) == pytest.approx(z_start)
        assert np.max(height_line.get_ydata()) == pytest.approx(z_end)
        source_markers = [line for line in overview.lines if line.get_marker() == "*"]
        assert len(source_markers) == 1
        assert source_markers[0].get_ydata()[0] == pytest.approx(z_start)
    plt.close(fig)


def test_small_multiple_semantic_labels_stay_in_axes_and_do_not_overlap():
    traces = {
        "CL3 − wobble": _synthetic_trace(label="minus"),
        "Nominal": _synthetic_trace(label="nominal"),
        "CL3 + wobble": _synthetic_trace(label="plus"),
    }
    objective_labels = (
        "C-mini",
        "Corrector (off)",
        "Obj-prefield",
        "Sample: ML4 · P=10.00 nm · grid Δ=0.072 nm",
        "Obj-post",
    )
    condenser_labels = ("CL1", "CL3")
    extra_elements = (
        ColumnElement(0.200, condenser_labels[0], "condenser_lens", 0.205),
        ColumnElement(0.245, condenser_labels[1], "condenser_lens", 0.250),
        ColumnElement(0.300, objective_labels[0], "condenser_lens", 0.303),
        ColumnElement(0.306, objective_labels[1], "corrector_off", 0.316),
        ColumnElement(0.319, objective_labels[2], "objective_lens", 0.322),
        ColumnElement(0.325, objective_labels[3], "sample"),
        ColumnElement(0.329, objective_labels[4], "objective_lens", 0.332),
        ColumnElement(1.000, "Medipix 256×256", "detector"),
    )
    sections = (
        ColumnSection("Condenser", 0.0, 0.324),
        ColumnSection("Objective and sample", 0.293, 0.365),
        ColumnSection("Intermediate/projector", 0.365, 1.0),
    )
    fig, axes = plot_neoarm_ray_small_multiples(
        (),
        traces,
        extra_elements=extra_elements,
        sections=sections,
        title="Clipping-safe semantic labels",
        figsize=(12.45, 10.0),
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def normalized(text: str) -> str:
        return " ".join(text.split())

    for scenario in traces:
        for row_key, expected_labels in (
            ("detail_1", (*condenser_labels, *objective_labels[:3])),
            ("detail_2", objective_labels),
        ):
            detail_axis = axes[scenario][row_key]
            detail_texts = [
                text
                for text in detail_axis.texts
                if normalized(text.get_text()) in expected_labels
            ]
            assert len(detail_texts) == len(expected_labels)
            detail_boxes = [text.get_window_extent(renderer) for text in detail_texts]
            for left_index, left_box in enumerate(detail_boxes):
                for right_box in detail_boxes[left_index + 1 :]:
                    overlap_x = min(left_box.x1, right_box.x1) - max(
                        left_box.x0, right_box.x0
                    )
                    overlap_y = min(left_box.y1, right_box.y1) - max(
                        left_box.y0, right_box.y0
                    )
                    assert overlap_x <= 0.0 or overlap_y <= 0.0

        for row_key in ("overview", "detail_1", "detail_2", "detail_3"):
            axis = axes[scenario][row_key]
            axis_box = axis.get_window_extent(renderer)
            semantic_texts = [
                text
                for text in axis.texts
                if normalized(text.get_text())
                in (*condenser_labels, *objective_labels, "Medipix 256×256")
            ]
            for text in semantic_texts:
                text_box = text.get_window_extent(renderer)
                assert text_box.x0 >= axis_box.x0 - 0.5
                assert text_box.x1 <= axis_box.x1 + 0.5
                assert text_box.y0 >= axis_box.y0 - 0.5
                assert text_box.y1 <= axis_box.y1 + 0.5

    plt.close(fig)


def test_export_ray_figure_writes_all_formats_and_provenance(tmp_path):
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    outputs = export_ray_figure(
        fig,
        tmp_path / "neoarm-rays.png",
        metadata={
            "title": "NEOARM ray solution",
            "author": "TemGymCore test",
            "fit_id": 17,
            "lens_values": np.array([1.0, 2.0]),
        },
        dpi=72,
    )
    assert set(outputs) == {"png", "svg", "pdf", "metadata"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())

    payload = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert payload["renderer"] == "temgym_core.plotting.export_ray_figure"
    assert payload["metadata"]["fit_id"] == 17
    assert payload["metadata"]["lens_values"] == [1.0, 2.0]
    assert "NEOARM ray solution" in outputs["svg"].read_text(encoding="utf-8")
    plt.close(fig)
