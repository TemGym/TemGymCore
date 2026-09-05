from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "examples"
    / "lens_inversion"
    / "neoarm_condenser_branch_b_design.py"
)
PAYLOAD = SCRIPT.with_suffix(".json")


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "neoarm_condenser_branch_b_design", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_branch_b_design_is_hash_bound_safe_and_frame_exact():
    verifier = _load_verifier()
    result = verifier.load_and_verify(PAYLOAD, repository_root=ROOT)

    assert result["schema"] == "temgym.neoarm.condenser-branch-b-design/v1"
    assert len(result["payload_sha256"]) == 64
    assert result["designs"]["design_248"] == {
        "frames": 248,
        "states": 36,
        "rank": 12,
        "max_scaled_standard_uncertainty": pytest.approx(
            0.0094959008166699
        ),
    }
    assert result["designs"]["design_296"] == {
        "frames": 296,
        "states": 39,
        "rank": 12,
        "max_scaled_standard_uncertainty": pytest.approx(
            0.009495143439037717
        ),
    }


def test_payload_cannot_be_mistaken_for_certified_hardware_commands():
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))

    assert "not certified hardware DAC" in payload["coordinate_status"]
    assert payload["hardware_admission"]["ready"] is False
    assert payload["hardware_admission"]["preferred_design"] == "design_296"
    assert payload["designs"]["design_296"]["state_counts"] == {
        "edge": 4,
        "tomography": 32,
        "deflector": 3,
    }


def test_verifier_fails_closed_when_a_bound_source_hash_changes(tmp_path):
    verifier = _load_verifier()
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    payload["provenance"]["config"]["sha256"] = "0" * 64
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        verifier.DesignVerificationError,
        match="config.*hash differs",
    ):
        verifier.load_and_verify(corrupted, repository_root=ROOT)
