"""Regression tests for the readable notebook's small support module."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from temgym_core.microscope_inversion import SamplePlaneState
from temgym_core.neoarm_calibration import (
    DetectorGeometry,
    FramePrediction,
    FrameRequest,
    GaussianRenderResult,
    IncompleteCheckpointError,
    extract_image_observables,
    hard_edge_column_from_config,
    render_frame_sequence_checkpointed,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "neoarm_review_workflow", ROOT / "examples/lens_inversion/neoarm_workflow.py"
)
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


@pytest.fixture
def checkpoint(tmp_path):
    config = tmp_path / "configuration.toml"
    config.write_text("# test configuration\n")
    detector = DetectorGeometry(shape_yx=(4, 6), pixel_size_yx_m=(2e-6, 3e-6), z_m=1.0)
    request = FrameRequest(
        frame_id=7,
        optical_state_index=0,
        optical_state_label="nominal_0",
        measurement="perforation",
        sample_state=SamplePlaneState(),
        raw_dac_values=(0.0,),
        interaction_route="vacuum",
    )

    def render(request):
        intensity = np.arange(24, dtype=float).reshape(4, 6) + 1
        result = GaussianRenderResult(
            field_yx=np.sqrt(intensity).astype(complex),
            intensity_yx=intensity,
            elapsed_s=0.0,
            estimate=None,
        )
        return FramePrediction(
            request=request,
            render=result,
            observables=extract_image_observables(intensity),
            interaction_route="vacuum",
        )

    directory = tmp_path / "checkpoint"
    render_frame_sequence_checkpointed(
        [request],
        detector,
        render,
        directory,
        render_signature="workflow-unit-test",
        render_metadata={
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()
        },
        storage_dtype="float32",
    )
    return directory, config


def test_load_validates_arrays_and_preserves_requests(checkpoint):
    directory, config = checkpoint
    run = workflow.load_saved_acquisition(directory, config)
    assert run.checkpoint.complete
    assert run.request(7).raw_dac_values == (0.0,)
    np.testing.assert_array_equal(run.image(7), np.arange(24).reshape(4, 6) + 1)


def test_changed_configuration_is_not_relabelled(checkpoint):
    directory, config = checkpoint
    config.write_text("# changed configuration\n")
    with pytest.raises(ValueError, match="different configuration"):
        workflow.load_saved_acquisition(directory, config)


@pytest.mark.parametrize("corruption", ["array", "request", "missing"])
def test_corrupt_artifacts_are_rejected(checkpoint, corruption):
    directory, config = checkpoint
    if corruption == "array":
        array_path = directory / "frames/frame_00000007_intensity.npy"
        array = np.load(array_path)
        array[0, 0] += 1
        np.save(array_path, array)
    elif corruption == "missing":
        (directory / "frames/frame_00000007_intensity.npy").unlink()
    else:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["payload"]["frame_requests"][0]["raw_dac_values"] = [0.1]
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, IncompleteCheckpointError)):
        workflow.load_saved_acquisition(directory, config)


def test_absolute_count_conversion_retains_transmission():
    detector = DetectorGeometry(
        shape_yx=(3, 5),
        pixel_size_yx_m=(2.0, 3.0),
        z_m=1.0,
        affine_xy=np.diag([2.0, 1.0]),
    )
    blank = np.ones((3, 5))
    counts = workflow.expected_detector_counts(blank, detector, 100)
    np.testing.assert_allclose(counts, 300)
    np.testing.assert_allclose(
        workflow.expected_detector_counts(0.4 * blank, detector, 100), 0.4 * counts
    )
    with pytest.raises(ValueError):
        workflow.expected_detector_counts(-blank, detector, 100)


def test_missing_fit_results_are_not_success():
    result = workflow.fit_status({})
    assert not result["optimizer_success"]
    assert not result["accepted_synthetic_fit"]
    assert not result["instrument_accepted"]
    assert result["held_out_rms_sigma"] is None


def test_optimizer_success_is_not_instrument_acceptance():
    result = workflow.fit_status({"fit_success": True, "converged_multistarts": 1})
    assert result["optimizer_success"]
    assert not result["accepted_synthetic_fit"]
    assert not result["instrument_accepted"]


@pytest.fixture
def column():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with (ROOT / "examples/lens_inversion/neoarm_inversion_config.toml").open(
        "rb"
    ) as stream:
        return hard_edge_column_from_config(tomllib.load(stream))


def test_dense_rays_match_endpoint_matrix_and_envelope(column):
    parameters = column.nominal_parameters
    controls = np.zeros(len(column.channel_names))
    trace = workflow.column_basis_trace(column, parameters, controls, 1e-6, 2e-5)
    basis = np.zeros((4, 2))
    basis[0, 0], basis[1, 1] = 1e-6, 2e-5
    final = np.asarray(column.matrix(parameters, controls)) @ basis
    rotation = -sum(
        float(bias) for _, _, _, bias, _ in column._decoded_lenses(parameters)
    )
    expected = np.array([np.cos(rotation), np.sin(rotation)]) @ final[[0, 2]]
    np.testing.assert_allclose(
        [trace.position[-1], trace.angle[-1]], expected, rtol=1e-10
    )
    np.testing.assert_allclose(trace.radius, np.hypot(trace.position, trace.angle))
    assert np.all(np.diff(trace.z) > 0)
    assert np.min(np.abs(trace.z - column.sample_z_m)) < 1e-12


def test_calibrated_sample_basis_is_not_rescaled_by_larmor_projection(column):
    parameters = column.nominal_parameters
    controls = np.zeros(len(column.channel_names))
    upstream = np.asarray(
        column.matrix(parameters, controls, segment="source_to_sample")
    )
    basis = np.zeros((4, 2))
    basis[0, 0], basis[1, 1] = 5e-9, 1e-6
    rotation = -sum(
        float(bias)
        for lens, _, _, bias, _ in column._decoded_lenses(parameters)
        if lens.exit_z <= column.sample_z_m
    )
    trace = workflow.column_basis_trace(
        column,
        parameters,
        controls,
        5e-9,
        1e-6,
        source_basis=np.linalg.solve(upstream, basis),
        frame_rotation_offset=-rotation,
    )
    index = np.argmin(np.abs(trace.z - column.sample_z_m))
    np.testing.assert_allclose(trace.position[index], 5e-9, rtol=1e-8)
    np.testing.assert_allclose(trace.angle[index], 0, atol=1e-16)


def test_sample_basis_helper_preserves_signed_inputs(column):
    parameters = column.nominal_parameters
    controls = np.zeros(len(column.channel_names))
    controls[0] = 0.5
    basis, offset = workflow.sample_basis_at_source(
        column, parameters, controls, 25e-9, 3e-5
    )
    trace = workflow.column_basis_trace(
        column,
        parameters,
        controls,
        25e-9,
        3e-5,
        source_basis=basis,
        frame_rotation_offset=offset,
    )
    index = np.argmin(np.abs(trace.z - column.sample_z_m))
    np.testing.assert_allclose(trace.position[index], 25e-9, rtol=1e-8)
    np.testing.assert_allclose(trace.radius[index], 25e-9, rtol=1e-8)


def test_gpu_forward_has_no_cpu_fallback(monkeypatch):
    import temgym_core.evaluate

    monkeypatch.setattr(temgym_core.evaluate, "gpu_kernel_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a working GPU"):
        workflow.reproduce_sample_image_gpu(None, 27, None, None)


def test_review_notebook_is_small_and_has_no_implicit_optimizer():
    notebook = json.loads(
        (
            ROOT / "examples/lens_inversion/neoarm_full_column_inversion.ipynb"
        ).read_text()
    )
    code = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert max(len(source.splitlines()) for source in code) <= 40
    for source in code:
        compile(source, "review-cell", "exec")
    assert "RERENDER_150K_ON_GPU = False" in "\n".join(code)
    assert "fit_nuisance_profiled_polytope(" not in "\n".join(code)
