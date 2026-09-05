from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from temgym_core.condenser_astigmatic_design import (
    load_astigmatic_condenser_design,
    recompute_astigmatic_design_audit,
)
from temgym_core.condenser_astigmatic_visualization import (
    PLANNING_PREVIEW_DISCLAIMER,
    PLANNING_PREVIEW_SCHEMA,
    SI_111_BASIS_SEPARATION_RAD,
    AstigmaticPlanningPreviewError,
    build_astigmatic_planning_preview,
    export_astigmatic_planning_preview,
    nominal_astigmatic_covariance_summaries,
    plot_astigmatic_planning_preview,
)


@pytest.fixture(scope="module")
def design_audit_summaries():
    design = load_astigmatic_condenser_design()
    audit = recompute_astigmatic_design_audit(design)
    summaries = nominal_astigmatic_covariance_summaries(design)
    return design, audit, summaries


@pytest.fixture(scope="module")
def preview(design_audit_summaries):
    design, audit, summaries = design_audit_summaries
    return build_astigmatic_planning_preview(
        design,
        audit=audit,
        summaries=summaries,
    )


def test_preview_selects_the_most_demanding_strict_state_deterministically(
    preview,
    design_audit_summaries,
):
    design, audit, summaries = design_audit_summaries

    assert preview.selected_state_index == 5
    assert preview.selected_repeat_count == 8
    np.testing.assert_array_equal(preview.strict_state_mask, True)
    np.testing.assert_allclose(
        preview.strict_sample_equivalent_rms_filter_m,
        (3.0e-8, 2.0e-7),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        preview.selected_control_CL1_CL3_Cmini,
        design.controls[5, :3],
        rtol=0.0,
        atol=0.0,
    )
    assert np.argmax(preview.angular_axis_rms_px[:, 1]) == 5
    assert preview.audit_target_rank == 12
    assert preview.audit_maximum_scaled_standard_uncertainty == pytest.approx(
        audit.maximum_scaled_standard_uncertainty
    )
    assert preview.nominal_detector_geometry_pass
    assert not preview.hardware_ready

    duplicate = build_astigmatic_planning_preview(
        design,
        audit=audit,
        summaries=summaries,
    )
    np.testing.assert_array_equal(
        duplicate.realspace_relative_intensity_yx,
        preview.realspace_relative_intensity_yx,
    )
    np.testing.assert_array_equal(
        duplicate.diffraction_relative_density_yx,
        preview.diffraction_relative_density_yx,
    )
    np.testing.assert_array_equal(
        duplicate.relative_spatial_orientation_rad,
        preview.relative_spatial_orientation_rad,
    )


def test_realspace_panel_is_a_resolved_covariance_prediction(preview):
    assert preview.realspace_relative_intensity_yx.shape == (401, 401)
    assert preview.realspace_x_m[200] == pytest.approx(0.0, abs=1.0e-24)
    assert preview.realspace_y_m[200] == pytest.approx(0.0, abs=1.0e-24)
    assert preview.realspace_relative_intensity_yx[200, 200] == pytest.approx(1.0)
    np.testing.assert_allclose(
        preview.realspace_relative_intensity_yx,
        preview.realspace_relative_intensity_yx[::-1, ::-1],
        rtol=2.0e-13,
        atol=2.0e-14,
    )
    minor_nm, major_nm = preview.selected_spatial_axis_rms_m * 1.0e9
    assert minor_nm == pytest.approx(152.68394, rel=2.0e-7)
    assert major_nm == pytest.approx(218.46019, rel=2.0e-7)
    assert major_nm / minor_nm > 1.4
    assert not preview.realspace_relative_intensity_yx.flags.writeable


def test_native_diffraction_geometry_has_two_indexed_noncollinear_vectors(
    preview,
):
    assert preview.diffraction_relative_density_yx.shape == (256, 256)
    np.testing.assert_allclose(
        np.diff(preview.detector_x_m),
        55.0e-6,
        rtol=0.0,
        atol=2.0e-18,
    )
    np.testing.assert_allclose(
        np.diff(preview.detector_y_m),
        55.0e-6,
        rtol=0.0,
        atol=2.0e-18,
    )
    assert preview.camera_length_m == pytest.approx(0.4)
    assert preview.detector_pixel_size_m == pytest.approx(55.0e-6)
    assert preview.si_111_radius_px == pytest.approx(58.16915496486905)
    np.testing.assert_allclose(
        np.linalg.norm(preview.bragg_vectors_px_xy, axis=1),
        preview.si_111_radius_px,
        rtol=2.0e-15,
    )
    cosine = np.dot(*preview.bragg_vectors_px_xy) / preview.si_111_radius_px**2
    assert cosine == pytest.approx(1.0 / 3.0, abs=2.0e-15)
    assert abs(np.linalg.det(preview.bragg_vectors_px_xy)) > 0.0
    assert preview.bragg_labels == (r"Si $(111)$", r"Si $(1\bar{1}1)$")
    assert np.rad2deg(SI_111_BASIS_SEPARATION_RAD) == pytest.approx(
        70.52877936550931
    )

    np.testing.assert_allclose(
        preview.selected_angular_axis_rms_rad,
        (0.0003031442865, 0.000454110042),
        rtol=3.0e-8,
    )
    np.testing.assert_allclose(
        preview.selected_angular_axis_rms_px,
        (2.20468572, 3.30261849),
        rtol=3.0e-8,
    )
    assert np.min(preview.diffraction_relative_density_yx) >= 0.0
    assert np.max(preview.diffraction_relative_density_yx) == pytest.approx(1.0)
    assert not preview.diffraction_relative_density_yx.flags.writeable


def test_relative_orientation_uses_unoriented_principal_axes(preview):
    assert preview.relative_spatial_orientation_rad[0] == pytest.approx(0.0)
    assert preview.relative_angular_orientation_rad[0] == pytest.approx(0.0)
    assert np.max(np.abs(np.diff(preview.relative_spatial_orientation_rad))) <= (
        0.5 * np.pi + 2.0e-15
    )
    assert np.max(np.abs(np.diff(preview.relative_angular_orientation_rad))) <= (
        0.5 * np.pi + 2.0e-15
    )

    for state, covariance in enumerate(preview.spatial_covariances_m2):
        values, vectors = np.linalg.eigh(covariance)
        major = vectors[:, np.argmax(values)]
        plotted = np.asarray(
            (
                np.cos(preview.spatial_orientation_rad[state]),
                np.sin(preview.spatial_orientation_rad[state]),
            )
        )
        assert abs(np.dot(major, plotted)) == pytest.approx(1.0, abs=2.0e-14)


def test_plot_is_explicitly_a_planning_preview_with_honest_units(preview):
    fig, axes = plot_astigmatic_planning_preview(preview)
    try:
        assert axes.shape == (3,)
        assert len(fig.axes) == 5  # three panels and two colour bars
        figure_text = "\n".join(text.get_text() for text in fig.texts)
        assert "no-commanded-tilt" in figure_text
        assert PLANNING_PREVIEW_DISCLAIMER in figure_text
        assert "not certified NEOARM DAC commands" in figure_text
        assert axes[0].get_xlabel() == "sample x (nm)"
        assert axes[0].get_ylabel() == "sample y (nm)"
        assert axes[1].get_xlabel() == "detector x (mm)"
        assert axes[1].get_ylabel() == "detector y (mm)"
        assert "native 256×256, 55 µm pixels" in axes[1].get_title()
        assert "modulo 180°" in axes[2].get_ylabel()
        panel_text = "\n".join(
            text.get_text() for axis in axes for text in axis.texts
        )
        assert "250 nm" in panel_text
        assert "2 mm" in panel_text
        assert "relative Gaussian geometry preview" in figure_text
    finally:
        plt.close(fig)


def test_export_metadata_cannot_be_mistaken_for_validation(preview, tmp_path):
    outputs = export_astigmatic_planning_preview(
        preview,
        tmp_path / "astigmatic_no_tilt_planning",
        formats=("svg",),
        dpi=90,
        close=True,
    )
    assert set(outputs) == {"svg", "metadata"}
    assert outputs["svg"].is_file()
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    payload = metadata["metadata"]
    assert payload["schema"] == PLANNING_PREVIEW_SCHEMA
    assert payload["status"] == PLANNING_PREVIEW_DISCLAIMER
    assert payload["evidence_class"] == "nominal observable-level planning only"
    assert payload["gpu_wave_validation"] is False
    assert payload["hardware_evidence"] is False
    assert payload["hardware_ready"] is False
    assert payload["selected_state_index"] == 5
    assert payload["detector_shape_yx"] == [256, 256]
    assert "no absolute specimen scattering" in payload["intensity_contract"]


def test_invalid_preview_resolution_fails_closed(design_audit_summaries):
    design, audit, summaries = design_audit_summaries
    with pytest.raises(
        AstigmaticPlanningPreviewError,
        match="odd integer of at least 129",
    ):
        build_astigmatic_planning_preview(
            design,
            audit=audit,
            summaries=summaries,
            realspace_grid_points=128,
        )
