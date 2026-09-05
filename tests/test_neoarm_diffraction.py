from pathlib import Path
import tomllib

import jax
import numpy as np
import pytest

from temgym_core.gaussian import make_gaussian
from temgym_core.microscope_inversion import SamplePlaneState, symplectic_defect
from temgym_core.neoarm_calibration import FrameRequest, hard_edge_column_from_config
from temgym_core.neoarm_diffraction import (
    NativeDiffractionInfeasibleError,
    bind_native_diffraction_requests,
    propagate_native_diffraction_gaussians,
    solve_native_diffraction_modes,
    verify_native_diffraction_mode,
)


jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def native_design():
    root = Path(__file__).resolve().parents[1]
    with (root / "tests/fixtures/neoarm_synthetic_column.toml").open("rb") as stream:
        column = hard_edge_column_from_config(tomllib.load(stream))
    targets = np.linspace(0.06, 0.16, 6)
    design = solve_native_diffraction_modes(
        column,
        column.nominal_parameters,
        targets,
        base_dac_values=np.zeros(len(column.channel_names)),
        adjusted_channel_names=("IL1", "PL1"),
        dac_bounds=(-30.0, 30.0),
        multistart_count=8,
    )
    return column, design


def test_six_native_diffraction_modes_cover_requested_camera_lengths(native_design):
    column, design = native_design
    targets = np.linspace(0.06, 0.16, 6)
    assert design.all_targets_feasible
    np.testing.assert_allclose(design.target_camera_lengths_m, targets)
    assert design.maximum_a_block_norm < 1e-9
    assert design.maximum_camera_length_relative_error < 1e-10

    upstream = [column.channel_names.index(name) for name in ("CL1", "CL3", "C-mini")]
    for target, mode in zip(targets, design.modes, strict=True):
        assert mode.adjusted_channel_names == ("IL1", "PL1")
        np.testing.assert_allclose(mode.raw_dac_values[upstream], 0.0, atol=0.0)
        assert np.all(mode.raw_dac_values[[3, 6]] >= -30.0)
        assert np.all(mode.raw_dac_values[[3, 6]] <= 30.0)
        np.testing.assert_allclose(mode.a_block, 0.0, atol=1e-9)
        np.testing.assert_allclose(
            mode.b_block,
            target * mode.diffraction_rotation_matrix,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            mode.diffraction_rotation_matrix.T @ mode.diffraction_rotation_matrix,
            np.eye(2),
            atol=2e-12,
        )
        assert np.linalg.det(mode.diffraction_rotation_matrix) == pytest.approx(1.0, abs=2e-12)
        np.testing.assert_allclose(
            np.asarray(symplectic_defect(mode.sample_to_detector_matrix)),
            0.0,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            verify_native_diffraction_mode(column, column.nominal_parameters, mode),
            mode.sample_to_detector_matrix,
            rtol=0.0,
            atol=0.0,
        )
        assert mode.round_structure_defect < 1e-10
        assert not mode.active_bound_mask.any()
        assert mode.model_signature.startswith("native-hard-edge-diffraction-v1:")


def test_native_mode_retains_projector_parameter_and_dac_sensitivity(native_design):
    column, design = native_design
    mode = design.modes[2]
    active = [column.channel_names.index(name) for name in mode.adjusted_channel_names]
    active_jacobian = mode.constraint_jacobian_wrt_dacs[:, active]
    assert np.linalg.matrix_rank(active_jacobian, tol=1e-10) == 2

    pl1_dac = column.channel_names.index("PL1")
    assert np.linalg.norm(mode.constraint_jacobian_wrt_dacs[:, pl1_dac]) > 1e-3
    for parameter_name in ("PL1.q_bias", "PL1.q_slope_per_dac"):
        parameter_index = column.parameter_names.index(parameter_name)
        assert (
            np.linalg.norm(mode.constraint_jacobian_wrt_parameters[:, parameter_index])
            > 1e-3
        )


def test_native_solver_fails_closed_when_dac_bounds_cannot_reach_mode(native_design):
    column, _design = native_design
    with pytest.raises(NativeDiffractionInfeasibleError) as caught:
        solve_native_diffraction_modes(
            column,
            column.nominal_parameters,
            (0.1,),
            base_dac_values=np.zeros(len(column.channel_names)),
            adjusted_channel_names=("IL1", "PL1"),
            dac_bounds=(-1e-8, 1e-8),
            multistart_count=1,
            maximum_function_evaluations=50,
        )
    mode = caught.value.design.modes[0]
    assert not mode.feasible
    assert "sample_conjugacy_A" in mode.failure_reasons


def test_binding_combines_upstream_commands_and_native_downstream_solution(native_design):
    column, _design = native_design
    base = np.asarray((0.21, -0.13, 0.08, 0.0, 0.02, -0.03, 0.0))
    approach_delta = np.asarray((-0.05,) * len(base))
    request = FrameRequest(
        frame_id=17,
        optical_state_index=4,
        optical_state_label="nominal_600k",
        measurement="indexed_diffraction",
        sample_state=SamplePlaneState(label="indexed_Si_diffraction"),
        raw_dac_values=tuple(base),
        dac_approach="forward",
        dac_approach_start_values=tuple(base + approach_delta),
        dac_ramp_steps=8,
        settling_time_s=2.0,
        feature_family="silicon_lattice_vectors",
        interaction_route="sample_plane_reexpansion",
        frozen_topology_id="silicon-lattice-native-diffraction-test",
    )
    (binding,) = bind_native_diffraction_requests(
        (request,),
        column,
        column.nominal_parameters,
        {17: 0.1},
        adjusted_channel_names=("IL1", "PL1"),
        dac_bounds=(-30.0, 30.0),
        multistart_count=6,
    )
    combined = np.asarray(binding.request.raw_dac_values)
    upstream = [column.channel_names.index(name) for name in ("CL1", "CL3", "C-mini")]
    untouched = [column.channel_names.index(name) for name in ("IL2", "IL3")]
    np.testing.assert_allclose(combined[upstream + untouched], base[upstream + untouched])
    assert not np.allclose(combined[[3, 6]], base[[3, 6]])
    np.testing.assert_allclose(
        np.asarray(binding.request.dac_approach_start_values) - combined,
        approach_delta,
    )
    metadata = binding.render_metadata
    assert metadata["model_scope"] == "neoarm_physical_column"
    assert metadata["renderer_family"] == "hard_edge_native_gaussian"
    assert metadata["uses_ideal_relay"] is False
    assert metadata["native_mode_signature"] == binding.mode.model_signature
    np.testing.assert_allclose(metadata["combined_raw_dac_values"], combined)


def test_native_gaussian_propagator_uses_solved_physical_matrix(native_design):
    column, design = native_design
    mode = design.modes[2]
    beam = make_gaussian(
        x=0.7e-9,
        y=-0.4e-9,
        dx=0.13e-3,
        dy=-0.09e-3,
        z=column.sample_z_m,
        voltage=200e3,
        waist_x=0.8e-9,
        waist_y=0.7e-9,
    )
    propagated = propagate_native_diffraction_gaussians(
        beam,
        column,
        column.nominal_parameters,
        mode,
        slices_per_lens=8,
    )
    initial_state = np.asarray((beam.x, beam.dx, beam.y, beam.dy), dtype=float)
    expected = mode.sample_to_detector_matrix @ initial_state
    np.testing.assert_allclose(
        np.asarray((propagated.x, propagated.dx, propagated.y, propagated.dy), dtype=float),
        expected,
        rtol=3e-10,
        atol=2e-12,
    )
    assert float(np.asarray(propagated.z)) == pytest.approx(column.detector_z_m)
