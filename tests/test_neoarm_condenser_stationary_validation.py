import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/lens_inversion"))
from neoarm_condenser_stationary_validation import freeze_selection, digest, verify_sources


def test_frozen_selection_is_absolute_and_exactly_reproducible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selection = {"parameters": [1., 2.], "training_only": True}
    path, checksum = freeze_selection("new/selection.json", selection)
    assert path.is_absolute()
    assert json.loads(path.read_text()) == selection
    assert freeze_selection(path, selection) == (path, checksum)


def test_different_training_selection_cannot_replace_existing_heldout_choice(tmp_path):
    path, checksum = freeze_selection(tmp_path/"selection.json", {"p": [1.]})
    with pytest.raises(ValueError, match="different training selection"):
        freeze_selection(path, {"p": [2.]})
    assert digest(path) == checksum


def test_nonfinite_selection_rejected_before_write(tmp_path):
    path = tmp_path/"selection.json"
    with pytest.raises(ValueError):
        freeze_selection(path, {"p": [float("nan")]})
    assert not path.exists()


def test_modified_report_dependency_rejected(tmp_path):
    path, checksum = freeze_selection(tmp_path/"input.json", {"p": [1.]})
    verify_sources({"source_hashes": {str(path): checksum}})
    with pytest.raises(ValueError, match="Changed computational input"):
        verify_sources({"source_hashes": {str(path): "0"*64}})
