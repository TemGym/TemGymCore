import numpy as np
import pytest

from temgym_core.components import ElectromagneticLens, Plane
from temgym_core.projector_model import (
    build_projector_components,
    build_simplified_projector_components,
    export_projector_model_json,
    export_simplified_projector_model_json,
    fit_projector_model,
    load_projector_model_json,
    load_simplified_projector_model_json,
    parse_projector_hex_table,
    projector_metrics,
    realize_simplified_projector_setting,
    solve_simplified_projector_zoom,
)


RAW_HEX_SAMPLE = {
    "2000": {"IL1": "0x4b4b", "IL2": "0x4b2b", "IL3": "0xaf98", "PL1": "0xfa00"},
    "5000": {"IL1": "0x512f", "IL2": "0x2d1d", "IL3": "0xc719", "PL1": "0xfa00"},
    "8000": {"IL1": "0x6a31", "IL2": "0x607f", "IL3": "0xadd0", "PL1": "0xe2cb"},
    "15000": {"IL1": "0x7830", "IL2": "0x56ad", "IL3": "0xc026", "PL1": "0xf548"},
    "30000": {"IL1": "0x9f22", "IL2": "0x611e", "IL3": "0x9b35", "PL1": "0xfa00"},
    "120000": {"IL1": "0xa7d4", "IL2": "0x7d46", "IL3": "0x6c7f", "PL1": "0xfa00"},
    "500000": {"IL1": "0xa51d", "IL2": "0xb944", "IL3": "0x3268", "PL1": "0xfa00"},
    "1500000": {"IL1": "0xe300", "IL2": "0xec00", "IL3": "0x0", "PL1": "0xfa00"},
}


def test_parse_projector_hex_table_supports_mixed_keys_and_numeric_strings():
    raw = {
        "2000": {"IL1": "0x10", "IL2": "0x20", "IL3": "0x30", "PL1": "0x40"},
        "2500.0": {"IL1": "32", "IL2": "48", "IL3": "64", "PL1": "80"},
    }
    dataset = parse_projector_hex_table(raw)

    np.testing.assert_allclose(dataset.magnification, [2000.0, 2500.0])
    np.testing.assert_allclose(dataset.codes[0], [16.0, 32.0, 48.0, 64.0])
    np.testing.assert_allclose(dataset.codes[1], [32.0, 48.0, 64.0, 80.0])
    assert np.max(dataset.normalized_codes) <= 1.0
    assert np.min(dataset.normalized_codes) >= -1.0


@pytest.fixture(scope="module")
def fitted_projector_model():
    return fit_projector_model(
        RAW_HEX_SAMPLE,
        n_starts=1,
        max_nfev=200,
        random_seed=0,
    )


def test_projector_metrics_zero_rotation_correction(fitted_projector_model):
    model = fitted_projector_model
    no_corr = projector_metrics(model, magnification=30000.0, correct_zero_rotation=False)
    corr = projector_metrics(model, magnification=30000.0, correct_zero_rotation=True)

    assert corr["magnification_predicted"] > 0.0
    assert abs(corr["psi_total"]) < 1e-8
    assert abs(corr["psi_total"]) <= abs(no_corr["psi_total"]) + 1e-12


def test_build_components_and_json_round_trip(fitted_projector_model, tmp_path):
    model = fitted_projector_model
    components = build_projector_components(model, magnification=80000.0)

    assert len(components) == 6
    assert all(isinstance(c, ElectromagneticLens) for c in components[:-1])
    assert isinstance(components[-1], Plane)

    z_positions = [float(c.z) for c in components]
    assert all(z_positions[i + 1] > z_positions[i] for i in range(len(z_positions) - 1))
    assert float(components[0].Rc) == 0.0

    out = tmp_path / "projector_model.json"
    export_projector_model_json(model, out)
    loaded = load_projector_model_json(out)

    assert loaded.model_type == model.model_type
    assert loaded.schema_version == model.schema_version
    assert loaded.lens_names == model.lens_names
    np.testing.assert_allclose(loaded.dataset.magnification, model.dataset.magnification)
    np.testing.assert_allclose(
        loaded.fit_params.gc_var_lenses,
        model.fit_params.gc_var_lenses,
    )
    np.testing.assert_allclose(loaded.fit_params.beta, model.fit_params.beta)
    np.testing.assert_allclose(loaded.fit_params.alpha, model.fit_params.alpha)


def test_simplified_projector_zoom_solve_and_realize():
    model = solve_simplified_projector_zoom(
        target_magnifications=[50.0, 60.0],
        max_nfev=1200,
    )
    assert len(model.fit_table) == 2

    for row in model.fit_table:
        assert row["magnification_predicted"] > 0.0
        assert abs(row["psi_total"]) < 1e-9
        assert abs(row["B"]) < 1e-3

    realized = realize_simplified_projector_setting(
        model,
        magnification=55.0,
        mode="interpolate",
    )
    np.testing.assert_allclose(realized["magnification_predicted"], 55.0, rtol=1e-6)
    assert abs(realized["psi_total"]) < 1e-9
    assert abs(realized["B"]) < 1e-6


def test_simplified_projector_components_and_json_round_trip(tmp_path):
    model = solve_simplified_projector_zoom(
        target_magnifications=[50.0, 60.0],
        max_nfev=1200,
    )
    components = build_simplified_projector_components(model, magnification=55.0)

    assert len(components) == 6
    assert all(isinstance(c, ElectromagneticLens) for c in components[:-1])
    assert isinstance(components[-1], Plane)

    out = tmp_path / "simplified_projector_model.json"
    export_simplified_projector_model_json(model, out)
    loaded = load_simplified_projector_model_json(out)

    assert loaded.model_type == model.model_type
    assert loaded.schema_version == model.schema_version
    np.testing.assert_allclose(loaded.il_gc, model.il_gc)
    np.testing.assert_allclose(loaded.il_rotation_signs, model.il_rotation_signs)
    assert len(loaded.fit_table) == len(model.fit_table)


def test_simplified_projector_variable_pl1_and_ffp_constraint():
    model = solve_simplified_projector_zoom(
        target_magnifications=[50.0, 60.0, 80.0],
        variable_projector_current=True,
        projector_current_bounds_at=(1000.0, 12000.0),
        ffp_weight=1.0,
        max_nfev=1500,
    )
    assert model.variable_projector_current
    assert model.projector_current_bounds_at is not None

    for row in model.fit_table:
        assert "I0_PL1_at" in row
        assert "f_PL1_m" in row
        assert "ffp_error_m" in row
        assert model.projector_current_bounds_at[0] <= row["I0_PL1_at"] <= model.projector_current_bounds_at[1]

    realized = realize_simplified_projector_setting(model=model, magnification=70.0, mode="interpolate")
    assert "current_pl1_at" in realized
    assert "focal_pl1_m" in realized
    assert model.projector_current_bounds_at[0] <= realized["current_pl1_at"] <= model.projector_current_bounds_at[1]


def test_simplified_projector_components_use_realized_pl1_current():
    model = solve_simplified_projector_zoom(
        target_magnifications=[50.0, 60.0],
        variable_projector_current=True,
        projector_current_bounds_at=(1000.0, 12000.0),
        ffp_weight=0.5,
        max_nfev=1500,
    )
    mag = 55.0
    realized = realize_simplified_projector_setting(model=model, magnification=mag, mode="interpolate")
    components = build_simplified_projector_components(model=model, magnification=mag, mode="interpolate")
    # PL1 is the fifth lens in the simplified stack.
    np.testing.assert_allclose(float(components[4].I0), float(realized["current_pl1_at"]), rtol=1e-9, atol=0.0)


def test_simplified_projector_soft_balance_with_trend_and_focal_bounds():
    model = solve_simplified_projector_zoom(
        target_magnifications=[80.0, 100.0, 120.0, 150.0],
        exact_il_rotation_balance=False,
        psi_weight=0.5,
        psi_tolerance_rad=2.0,
        trend_band_c_magnification=(80.0, 150.0),
        trend_band_c_weight=0.2,
        projector_focal_bounds_m=(1.0e-3, 10.0e-3),
        projector_focal_bounds_weight=5.0,
        variable_projector_current=True,
        projector_current_bounds_at=(1000.0, 20000.0),
        max_nfev=1200,
    )
    assert model.solver_info["exact_il_rotation_balance"] is False
    assert "projector_focal_bounds_m" in model.solver_info

    for row in model.fit_table:
        assert "projector_focal_bounds_ok" in row

    realized = realize_simplified_projector_setting(model=model, magnification=100.0, mode="solve")
    assert "projector_focal_bounds_ok" in realized
    assert "psi_tolerance_ok" in realized
