from pathlib import Path
import sys
import tomllib

import jax
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples/lens_inversion"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from temgym_core.constants import energy2wavelength
from temgym_core.neoarm_calibration import hard_edge_column_from_config
from neoarm_imaging_recipe import column_with_geometry_anchors
from neoarm_measured_initialization import initialize_measured_imaging
from neoarm_measured_ray_review import verified_basis_trace, label_overlap_audit


@pytest.fixture(scope="module")
def trace_and_evidence():
    column = column_with_geometry_anchors(hard_edge_column_from_config(tomllib.loads(
        (EXAMPLES / "neoarm_inversion_config.toml").read_text())),
        ROOT / "examples/microscope_models/measure_microscope_geometry.ipynb")
    initialized = initialize_measured_imaging(column, magnifications=[40000.])
    control = initialized.nominal_controls[0].copy()
    control[:3] = [(.285-.2)/.55, (.22522727272727272-.15)/.65, (.34984826301857863-.1)/.45]
    wavelength = float(energy2wavelength(200000.))
    return verified_basis_trace(initialized.column, initialized.parameters, control,
                                 60e-9, wavelength/(np.pi*60e-9), label="test")


def test_detector_basis_envelope_and_root_positions(trace_and_evidence):
    trace, evidence = trace_and_evidence
    assert abs(abs(trace.position[-1])-40000.*60e-9) < 1e-12
    assert abs(trace.angle[-1]) < 1e-14
    np.testing.assert_allclose(trace.radius, np.hypot(trace.position, trace.angle), rtol=1e-15)
    assert max(abs(value) for value in evidence["detector_basis_error_m"]) < 1e-14
    assert max(abs(event["position_residual_m"]) for event in evidence["events"]) < 1e-14
    assert all(np.isfinite(event["z_m"]) for event in evidence["events"])
    waists = [event for event in evidence["waists"] if event["stationary_bracketed"]]
    assert len(waists) >= 5
    # Callouts use the exact derivative roots, not the displaced finite-lens
    # piecewise-linear candidates retained in the diagnostic metadata.
    assert max(abs(event["stationary_derivative_m"]) for event in waists) < 1e-16
    assert max(abs(event["radius_error_m"]) for event in waists) < 1e-12


def test_text_overlap_audit_detects_collisions():
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    axis.text(.5, .5, "first")
    axis.text(.5, .5, "second")
    assert label_overlap_audit(figure)
    axis.texts[-1].set_position((.1, .1))
    assert not label_overlap_audit(figure)
    plt.close(figure)
