"""Verify physical-coordinate constraints and a profile with a known answer."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/microscope_models"))
jax.config.update("jax_enable_x64", True)

from condenser_aperture_experiment import named_aperture_parameters
from condenser_calibration_model import initial_parameters
from condenser_geometry_audit import (
    coordinate_geometry,
    from_coordinates,
    profile_native_quantity,
    profile_physical_quantity,
    to_coordinates,
)
from condenser_simplification_design import physical_geometry


def example_parameters():
    return np.r_[initial_parameters(), -3.0, 0.2, -0.15, 1.0, 0.0, 0.0]


def test_coordinate_round_trip_and_physical_positions():
    p = example_parameters()
    x = to_coordinates(p)
    np.testing.assert_allclose(from_coordinates(x), p, atol=1e-13)
    np.testing.assert_allclose(coordinate_geometry(x), physical_geometry(p), atol=1e-12)
    named = named_aperture_parameters(p[:22], "after")
    keys = [
        "source_z_mm",
        "CL1.z_mm",
        "CL3.z_mm",
        "Mini.z_mm",
        "Objective pre-field.z_mm",
        "CL1.length_mm",
        "CL3.length_mm",
        "Mini.length_mm",
        "Objective pre-field.length_mm",
        "aperture_z_mm",
        "aperture_x_um",
        "aperture_y_um",
    ]
    np.testing.assert_allclose(
        coordinate_geometry(x), [named[k] for k in keys], atol=1e-12
    )


@pytest.mark.parametrize(
    "profile_function", [profile_physical_quantity, profile_native_quantity]
)
def test_physical_profile_matches_analytic_gaussian_result(tmp_path, profile_function):
    p = example_parameters()
    centre = to_coordinates(p)
    scales = np.full(len(p), 0.1)
    scales[:9] = centre[:9] * 0.05

    def residual(v):
        x = v.at[:9].set(jnp.exp(v[:9]) * 1000)
        x = x.at[19].set(jax.nn.sigmoid(v[19]) * x[2])
        return (x - centre) / scales

    lower, upper = p - 0.5, p + 0.5
    lower[19], upper[19] = -7.0, 0.0
    target0 = float(coordinate_geometry(centre)[4])
    targets = [target0, target0 + 0.02]
    other = p.copy()
    other[:9] += np.random.default_rng(72).normal(0, 0.01, 9)
    points = profile_function(
        residual=jax.jit(residual),
        fitted_parameters=p,
        bounds_lower=lower,
        bounds_upper=upper,
        quantity="prefield_z_mm",
        targets=targets,
        starting_parameters=[p, other],
        output_directory=tmp_path,
    )
    variance = scales[4] ** 2 + 0.25 * scales[8] ** 2
    for target, point in zip(targets, points):
        assert all(t["success"] for t in point["trials"])
        np.testing.assert_allclose(
            point["best"]["delta_chi_squared"],
            (target - target0) ** 2 / variance,
            atol=1e-6,
            rtol=1e-6,
        )
        assert point["best"]["target_error"] < 1e-10


def test_profile_confidence_limits_bracket_the_analytic_answer(tmp_path, monkeypatch):
    import condenser_voltage_calibration
    import condenser_voltage_workflow
    from scipy.stats import chi2

    p = example_parameters()
    centre = to_coordinates(p)
    scales = np.full(len(p), 0.01)
    scales[:9] = centre[:9] * 0.01

    # Use the same exactly Gaussian physical-coordinate experiment as above.
    def errors(v):
        x = v.at[:9].set(jnp.exp(v[:9]) * 1000)
        x = x.at[19].set(jax.nn.sigmoid(v[19]) * x[2])
        return (x - centre) / scales

    residual = jax.jit(errors)
    lower, upper = p - 0.5, p + 0.5
    lower[19], upper[19] = -7.0, 0.0
    mean = float(coordinate_geometry(centre)[4])
    sigma = np.sqrt(scales[4] ** 2 + 0.25 * scales[8] ** 2)
    points = profile_native_quantity(
        residual=residual,
        fitted_parameters=p,
        bounds_lower=lower,
        bounds_upper=upper,
        quantity="prefield_z_mm",
        targets=mean + sigma * np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
        starting_parameters=[p, p + 0.001],
        output_directory=tmp_path / "profiles",
    )
    monkeypatch.setattr(
        condenser_voltage_calibration,
        "make_calibrated_voltage_residual",
        lambda **kwargs: residual,
    )
    monkeypatch.setattr(
        condenser_voltage_workflow,
        "geometry_report",
        lambda **kwargs: {
            "quantities": {"prefield_z_mm": {"local_sigma": sigma, "tolerance": 0.2}}
        },
    )
    result = condenser_voltage_workflow.finish_profile_intervals(
        measurements_directory=tmp_path,
        fitted={
            "best_fit": {"parameters": p.tolist()},
            "bounds_lower": lower.tolist(),
            "bounds_upper": upper.tolist(),
        },
        profiles={"prefield_z_mm": {"points": points}},
        reuse_completed=False,
    )["prefield_z_mm"]
    expected = mean + np.sqrt(chi2.ppf(0.95, 1)) * sigma * np.array([-1.0, 1.0])
    assert result["resolved"]
    assert result["interval"][0] <= expected[0]
    assert result["interval"][1] >= expected[1]
    np.testing.assert_allclose(result["interval"], expected, atol=0.011 * sigma)
