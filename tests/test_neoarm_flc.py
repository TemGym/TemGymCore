import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "neoarm_flc", ROOT / "examples/lens_inversion/neoarm_flc.py"
)
flc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flc)


def notebook(tmp_path, code):
    path = tmp_path / "table.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": code}]}))
    return path


def raw_row():
    return {name: "0x0001" for name in flc.CONTROL_NAMES}


def test_recorded_table_preserves_actual_values_and_constant_controls():
    table = flc.read_flc_table(ROOT / "examples/lens_inversion/dac_lens_inversion.ipynb")
    assert len(table["nominal_magnification"]) == 13
    np.testing.assert_array_equal(table["nominal_magnification"][[0, -1]], [40000, 600000])
    assert table["IL1"][0] == int("a336", 16)
    assert table["IL3"][-1] == int("2771", 16)
    for name, value in (("PL1", 64000), ("OLf", 39178), ("OLc", 45818)):
        assert np.all(table[name] == value)
    assert not table["IL1"].flags.writeable


def test_parser_does_not_execute_code_and_sorts_rows(tmp_path):
    source = f'raise RuntimeError("Do not execute")\nRAW_DAC_HEX = {{"80000": {raw_row()!r}, "40000": {raw_row()!r}}}'
    table = flc.read_flc_table(notebook(tmp_path, source))
    np.testing.assert_array_equal(table["nominal_magnification"], [40000, 80000])


@pytest.mark.parametrize("source", [
    "x = 1", "RAW_DAC_HEX = dict()", "RAW_DAC_HEX = {}\nRAW_DAC_HEX = {}",
])
def test_missing_nonliteral_and_duplicate_assignments_are_rejected(tmp_path, source):
    with pytest.raises(ValueError):
        flc.read_flc_table(notebook(tmp_path, source))


def test_invalid_raw_controls_and_empty_selection_are_rejected(tmp_path):
    row = raw_row()
    row["IL1"] = "0x10000"
    with pytest.raises(ValueError, match="16-bit"):
        flc.read_flc_table(notebook(tmp_path, f'RAW_DAC_HEX = {{"40000": {row!r}}}'))
    with pytest.raises(ValueError, match="No recorded settings"):
        flc.read_flc_table(notebook(tmp_path, f'RAW_DAC_HEX = {{"2000": {raw_row()!r}}}'))
    with pytest.raises(ValueError, match="ordered"):
        flc.read_flc_table("unused", minimum_mag=80000, maximum_mag=40000)
