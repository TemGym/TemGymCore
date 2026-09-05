"""Safe numeric snapshots used by the small recovery driver."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from temgym_core.neoarm_calibration import DetectorGeometry


spec = importlib.util.spec_from_file_location(
    "neoarm_problem_io", Path(__file__).parents[1]
    / "examples/lens_inversion/neoarm_problem_io.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


def test_snapshot_roundtrip_and_checksum(tmp_path):
    path = tmp_path / "problem.json"
    detector = DetectorGeometry(shape_yx=(13, 21), pixel_size_yx_m=(2e-6, 3e-6))
    codec.save_problem(path, detector=detector, arrays=(np.arange(4),), metadata={"seed": 4})
    loaded = codec.load_problem(path)
    assert loaded["detector"].shape_yx == (13, 21)
    np.testing.assert_array_equal(loaded["arrays"][0], np.arange(4))
    document = json.loads(path.read_text())
    document["payload"]["metadata"]["seed"] = 5
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="checksum"):
        codec.load_problem(path)


@pytest.mark.parametrize("array", [np.array([np.nan]), np.array([1j]), np.array(["x"])])
def test_snapshot_rejects_nonreal_or_nonfinite_arrays(tmp_path, array):
    with pytest.raises(ValueError, match="finite real"):
        codec.save_problem(tmp_path / "bad.json", array=array)


def test_snapshot_rejects_executable_objects():
    with pytest.raises(ValueError, match="not permitted"):
        codec._decode({"class": "subprocess:Popen", "fields": {}})
    with pytest.raises(TypeError, match="Unsupported"):
        codec._encode(lambda: None)
