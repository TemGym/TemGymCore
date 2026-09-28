import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "neoarm_flc", ROOT / "examples/microscope_models/neoarm_flc.py"
)
flc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flc)


def table_file(tmp_path, raw):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"raw_dac_hex": raw}))
    return path


def raw_row():
    return {name: "0x0001" for name in flc.CONTROL_NAMES}


def test_recorded_table_preserves_actual_values_and_constant_controls():
    table = flc.read_flc_table()
    assert len(table["nominal_magnification"]) == 13
    np.testing.assert_array_equal(table["nominal_magnification"][[0, -1]], [40000, 600000])
    assert table["IL1"][0] == int("a336", 16)
    assert table["IL3"][-1] == int("2771", 16)
    for name, value in (("PL1", 64000), ("OLf", 39178), ("OLc", 45818)):
        assert np.all(table[name] == value)
    assert not table["IL1"].flags.writeable


def test_rows_are_sorted_by_magnification(tmp_path):
    table = flc.read_flc_table(table_file(tmp_path, {"80000": raw_row(), "40000": raw_row()}))
    np.testing.assert_array_equal(table["nominal_magnification"], [40000, 80000])


def test_invalid_raw_controls_and_empty_selection_are_rejected(tmp_path):
    row = raw_row()
    row["IL1"] = "0x10000"
    with pytest.raises(ValueError, match="16-bit"):
        flc.read_flc_table(table_file(tmp_path, {"40000": row}))
    with pytest.raises(ValueError, match="No recorded settings"):
        flc.read_flc_table(table_file(tmp_path, {"2000": raw_row()}))
    with pytest.raises(ValueError, match="ordered"):
        flc.read_flc_table("unused", minimum_mag=80000, maximum_mag=40000)
