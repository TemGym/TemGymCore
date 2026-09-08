"""Acquisition reuse must preserve known inputs without copying fitted answers."""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from temgym_core import calibration
from temgym_core.calibration.workflow import input_signature
from temgym_core.calibration.workflow import _record_step


def recorded_experiment(root):
    """Small file-level fixture with the same acquisition contract as the notebook."""
    root.mkdir()
    paths = ["experiment.json", "condenser/alignment/alignment_constraint.json"]
    for block in ("projector", "projector_predictions", "condenser"):
        paths += [f"{block}/experiment.json", f"{block}/scoring_truth.json"]
        for index in (0, 1):
            prefix = f"{block}/voltage_{index}"
            names = (
                (
                    "images.npz",
                    "counts.npz",
                    "template.npz",
                    "detector.json",
                    "aperture.json",
                    "prediction/images.npz",
                    "prediction/current_counts.npz",
                    "prediction/template.npz",
                    "prediction/detector.json",
                    "gaussian_specimen.npz",
                    "moment_specimen_local.npz",
                )
                if block == "condenser"
                else (
                    "images.npz",
                    "tilt_images_3mrad.npz",
                    "beam_shift_images_10nm.npz",
                    "specimen_template.npz",
                    "known_settings.json",
                    "scoring_truth.json",
                )
            )
            paths += [f"{prefix}/{name}" for name in (*names, "moment_specimen.npz")]
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            np.savez(path, values=np.arange(3.0))
        else:
            path.write_text(json.dumps({"image_model": "coherent_gaussian"}))
    (root / "condenser/analysis").mkdir()
    (root / "condenser/analysis/full_fit.json").write_text('{"answer": 123}')
    return paths


def test_copy_preserves_known_specimen_but_excludes_old_fits(tmp_path):
    source, output = tmp_path / "source", tmp_path / "copy"
    paths = recorded_experiment(source)
    assert (
        calibration.copy_recorded_experiment(
            source_directory=source, output_directory=output
        )
        == output
    )
    for name in paths:
        assert (output / name).read_bytes() == (source / name).read_bytes()
    assert not (output / "condenser/analysis/full_fit.json").exists()
    # A repeat copy preserves work done on the new acquisition.
    (output / "analysis.txt").write_text("new fit")
    calibration.copy_recorded_experiment(
        source_directory=source, output_directory=output
    )
    assert (output / "analysis.txt").read_text() == "new fit"
    (source / "projector/voltage_0/moment_specimen.npz").unlink()
    with pytest.raises(FileNotFoundError, match="moment_specimen"):
        calibration.copy_recorded_experiment(
            source_directory=source, output_directory=tmp_path / "incomplete"
        )
    assert not (tmp_path / "incomplete").exists()


def test_copy_rejects_overlapping_directories(tmp_path):
    for output in (tmp_path, tmp_path / "nested", tmp_path.parent):
        with pytest.raises(ValueError, match="non-overlapping"):
            calibration.copy_recorded_experiment(
                source_directory=tmp_path, output_directory=output
            )


def test_stage_reuse_checks_inputs_and_required_outputs(tmp_path):
    output = tmp_path / "result.txt"
    calls = []

    def run():
        calls.append(1)
        output.write_text("result")

    def stage(signature):
        _record_step(
            directory=tmp_path,
            name="fitting",
            signature=signature,
            required=[output],
            function=run,
            reuse_completed=True,
        )

    stage("first")
    stage("first")
    assert len(calls) == 1
    stage("changed-input")
    assert len(calls) == 2
    output.unlink()
    stage("changed-input")
    assert len(calls) == 3
    output.unlink()
    with pytest.raises(RuntimeError, match="required outputs"):
        _record_step(
            directory=tmp_path,
            name="fitting",
            signature="failed",
            required=[output],
            function=lambda: None,
            reuse_completed=False,
        )
    assert not (tmp_path / "fitting_manifest.json").exists()


def test_fingerprint_uses_array_contents_and_presets_are_installed(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, x=[1, 2])
    np.savez_compressed(b, x=[1, 2])
    signature = lambda p: input_signature(paths=[p], settings={})
    assert signature(a) == signature(b)
    np.savez(b, x=[1, 3])
    assert signature(a) != signature(b)
    for function in (
        calibration.fit_projector,
        calibration.measure_condenser,
        calibration.fit_microscope,
        calibration.check_full_microscope,
    ):
        for parameter in inspect.signature(function).parameters.values():
            if isinstance(parameter.default, Path):
                assert parameter.default.is_file()
                assert isinstance(json.loads(parameter.default.read_text()), dict)
