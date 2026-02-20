"""Tests for the unified TEM model (tem_model.py)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from temgym_core.tem_model import (
    dac_hex_to_int,
    dac_int_to_hex,
    dac_to_normalised_current,
    focal_from_current,
    current_from_focal,
    focal_to_synthetic_dac,
    DialCurve,
    LensSystemGeometry,
    LensModel,
    LensSystem,
    TEMModel,
    export_tem_model_json,
    load_tem_model_json,
    parse_dac_table,
    RAW_DAC_HEX,
)


# ---------------------------------------------------------------------------
# DAC helpers
# ---------------------------------------------------------------------------

class TestDacHelpers:
    def test_hex_to_int_roundtrip(self):
        assert dac_hex_to_int("0x990a") == 0x990A
        assert dac_int_to_hex(0x990A) == "0x990a"

    def test_normalised_current_range(self):
        assert dac_to_normalised_current(0) == 0.0
        assert dac_to_normalised_current(65536) == pytest.approx(1.0)
        assert 0.0 < dac_to_normalised_current(32768) < 1.0


# ---------------------------------------------------------------------------
# Focal-length model
# ---------------------------------------------------------------------------

class TestFocalModel:
    def test_focal_from_current_linear(self):
        gc = 100.0
        I = 0.5
        expected = 1.0 / (gc * I**2)
        assert focal_from_current(I, gc) == pytest.approx(expected)

    def test_focal_from_current_nonlinear(self):
        gc, alpha = 100.0, 50.0
        I = 0.5
        expected = 1.0 / (gc * I**2 + alpha * I**4)
        assert focal_from_current(I, gc, alpha) == pytest.approx(expected)

    def test_current_from_focal_roundtrip_linear(self):
        gc = 200.0
        I_original = 0.4
        f = focal_from_current(I_original, gc, 0.0)
        I_recovered = current_from_focal(f, gc, 0.0)
        assert I_recovered == pytest.approx(I_original, rel=1e-10)

    def test_current_from_focal_roundtrip_nonlinear(self):
        gc, alpha = 200.0, -100.0
        I_original = 0.3
        f = focal_from_current(I_original, gc, alpha)
        I_recovered = current_from_focal(f, gc, alpha)
        assert I_recovered == pytest.approx(I_original, rel=1e-8)

    def test_focal_to_synthetic_dac(self):
        gc = 100.0
        I = 0.5
        f = focal_from_current(I, gc)
        dac = focal_to_synthetic_dac(f, gc, 0.0, i_min=0.0, i_max=1.0)
        # DAC should map I=0.5 → 0.5 * 65535 ≈ 32768
        assert abs(dac - 32768) < 2


# ---------------------------------------------------------------------------
# DialCurve
# ---------------------------------------------------------------------------

class TestDialCurve:
    def _make_simple_curve(self):
        return DialCurve(
            name="brightness",
            control_values=np.array([100.0, 200.0, 400.0]),
            control_unit="nm",
            dac_codes=np.array([
                [10000, 30000],
                [20000, 25000],
                [30000, 20000],
            ]),
        )

    def test_interpolate_at_exact(self):
        c = self._make_simple_curve()
        dac = c.interpolate_dac(200.0)
        np.testing.assert_array_equal(dac, [20000, 25000])

    def test_interpolate_between(self):
        c = self._make_simple_curve()
        dac = c.interpolate_dac(150.0)
        # Should be between the 100 nm and 200 nm values
        assert 10000 < dac[0] < 20000
        assert 25000 < dac[1] < 30000

    def test_json_roundtrip(self):
        c = self._make_simple_curve()
        d = c.as_json_dict()
        c2 = DialCurve.from_json_dict(d)
        assert c2.name == c.name
        assert c2.control_unit == c.control_unit
        np.testing.assert_array_almost_equal(c2.control_values, c.control_values)
        np.testing.assert_array_equal(c2.dac_codes, c.dac_codes)


# ---------------------------------------------------------------------------
# LensSystemGeometry
# ---------------------------------------------------------------------------

class TestLensSystemGeometry:
    def test_z_positions(self):
        geo = LensSystemGeometry(
            lens_names=("L1", "L2"),
            drift_distances_m=np.array([0.01, 0.02, 0.03]),
        )
        zpos = geo.z_positions(z0=0.0)
        assert zpos["L1"] == pytest.approx(0.01)
        assert zpos["L2"] == pytest.approx(0.03)
        assert zpos["_end"] == pytest.approx(0.06)

    def test_json_roundtrip(self):
        geo = LensSystemGeometry(
            lens_names=("A", "B", "C"),
            drift_distances_m=np.array([0.1, 0.2, 0.3, 0.4]),
        )
        geo2 = LensSystemGeometry.from_json_dict(geo.as_json_dict())
        assert geo2.lens_names == geo.lens_names
        np.testing.assert_array_almost_equal(
            geo2.drift_distances_m, geo.drift_distances_m,
        )


# ---------------------------------------------------------------------------
# LensModel
# ---------------------------------------------------------------------------

class TestLensModel:
    def test_focal_from_dac(self):
        gc = np.array([100.0, 200.0])
        alpha = np.zeros(2)
        lm = LensModel(
            gc=gc, alpha_nl=alpha,
            fixed_mask=np.array([False, False]),
            fixed_currents=np.zeros(2),
        )
        # DAC 32768 → normalised current 0.5
        f = lm.focal_from_dac(32768, 0)
        expected = 1.0 / (100.0 * 0.5**2)
        assert f == pytest.approx(expected)

    def test_fixed_mask(self):
        gc = np.array([100.0])
        alpha = np.zeros(1)
        I_fixed = 0.4
        lm = LensModel(
            gc=gc, alpha_nl=alpha,
            fixed_mask=np.array([True]),
            fixed_currents=np.array([I_fixed]),
        )
        # DAC code should be ignored when fixed_mask is True
        focals = lm.all_focals_from_dac(np.array([99999]))
        expected_f = 1.0 / (100.0 * I_fixed**2)
        assert focals[0] == pytest.approx(expected_f)

    def test_json_roundtrip(self):
        lm = LensModel(
            gc=np.array([1.0, 2.0]),
            alpha_nl=np.array([0.1, 0.2]),
            fixed_mask=np.array([True, False]),
            fixed_currents=np.array([0.5, 0.0]),
        )
        lm2 = LensModel.from_json_dict(lm.as_json_dict())
        np.testing.assert_array_almost_equal(lm2.gc, lm.gc)
        np.testing.assert_array_almost_equal(lm2.alpha_nl, lm.alpha_nl)
        np.testing.assert_array_equal(lm2.fixed_mask, lm.fixed_mask)
        np.testing.assert_array_almost_equal(lm2.fixed_currents, lm.fixed_currents)


# ---------------------------------------------------------------------------
# LensSystem
# ---------------------------------------------------------------------------

class TestLensSystem:
    def _make_system(self):
        geo = LensSystemGeometry(
            lens_names=("L1", "L2"),
            drift_distances_m=np.array([0.01, 0.02, 0.01]),
        )
        model = LensModel(
            gc=np.array([100.0, 200.0]),
            alpha_nl=np.zeros(2),
            fixed_mask=np.array([False, False]),
            fixed_currents=np.zeros(2),
        )
        curve = DialCurve(
            name="test",
            control_values=np.array([1.0, 2.0, 3.0]),
            control_unit="step",
            dac_codes=np.array([
                [10000, 20000],
                [20000, 30000],
                [30000, 40000],
            ]),
        )
        return LensSystem(
            name="test_system",
            geometry=geo,
            model=model,
            dial_curves={"test": curve},
        )

    def test_realize_keys(self):
        sys = self._make_system()
        r = sys.realize("test", 2.0)
        assert "focal_lengths_m" in r
        assert "dac_codes" in r
        assert "dac_hex" in r
        assert len(r["focal_lengths_m"]) == 2
        assert len(r["dac_hex"]) == 2

    def test_build_components_count(self):
        sys = self._make_system()
        comps = sys.build_components("test", 2.0, end_plane=True)
        # 2 lenses + 1 plane = 3 components
        assert len(comps) == 3

    def test_build_components_no_end_plane(self):
        sys = self._make_system()
        comps = sys.build_components("test", 2.0, end_plane=False)
        assert len(comps) == 2

    def test_json_roundtrip(self):
        sys = self._make_system()
        d = sys.as_json_dict()
        sys2 = LensSystem.from_json_dict(d)
        assert sys2.name == sys.name
        assert sys2.geometry.lens_names == sys.geometry.lens_names
        assert "test" in sys2.dial_curves


# ---------------------------------------------------------------------------
# TEMModel
# ---------------------------------------------------------------------------

class TestTEMModel:
    def _make_model(self):
        def _simple_system(name):
            geo = LensSystemGeometry(
                lens_names=("L1",),
                drift_distances_m=np.array([0.01, 0.01]),
            )
            model = LensModel(
                gc=np.array([100.0]),
                alpha_nl=np.zeros(1),
                fixed_mask=np.array([False]),
                fixed_currents=np.zeros(1),
            )
            curve_name = "brightness" if name == "illumination" else "magnification"
            curve = DialCurve(
                name=curve_name,
                control_values=np.array([100.0, 200.0]),
                control_unit="nm" if name == "illumination" else "x",
                dac_codes=np.array([[30000], [40000]]),
            )
            return LensSystem(
                name=name,
                geometry=geo,
                model=model,
                dial_curves={curve_name: curve},
            )

        return TEMModel(
            model_type="test_model",
            schema_version=1,
            voltage_v=200e3,
            illumination=_simple_system("illumination"),
            projection=_simple_system("projection"),
            sample_z_m=0.02,
        )

    def test_realize_full(self):
        m = self._make_model()
        state = m.realize_full(brightness_nm=150, magnification=150)
        assert "illumination" in state
        assert "projection" in state

    def test_json_file_roundtrip(self):
        m = self._make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.json"
            export_tem_model_json(m, p)
            assert p.exists()
            m2 = load_tem_model_json(p)
            assert m2.model_type == m.model_type
            assert m2.voltage_v == pytest.approx(m.voltage_v)
            assert m2.illumination.name == "illumination"
            assert m2.projection.name == "projection"


# ---------------------------------------------------------------------------
# DAC table parsing
# ---------------------------------------------------------------------------

class TestDACTable:
    def test_parse_dac_table_shape(self):
        data = parse_dac_table()
        assert len(data["mag"]) == len(RAW_DAC_HEX)
        assert data["IL1"].shape == data["mag"].shape
        assert data["mode"].shape == data["mag"].shape

    def test_mode_c_range(self):
        data = parse_dac_table()
        mode_c = data["mag"][data["mode"] == "C"]
        assert mode_c.min() >= 30_000
        assert mode_c.max() <= 600_000
