"""The benchmark helpers: configuration overrides and the regression comparison."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
compare = pytest.importorskip("compare", reason="benchmarks/ is not in the source distribution")
common = pytest.importorskip("common")

from temgym_core.render import RenderConfig, Tiles  # noqa: E402

parse_config = common.parse_config


def test_parse_config_builds_a_render_config():
    config = parse_config(["cutoff=1e-9", "rows=32x4x2", "class_edges=30,60", "n_classes=3",
                           "items=1024", "coordinates=metre"])
    assert config == RenderConfig(cutoff=1e-9, rows=Tiles(32, 4, 2), class_edges=(30.0, 60.0),
                                  n_classes=3, items=1024, coordinates="metre")
    assert parse_config(None) == RenderConfig()
    assert parse_config(["class_edges="]).class_edges == ()
    with pytest.raises(ValueError):
        parse_config(["rows=24x4x1"])


def result(**timings):
    return {"cases": {"a": dict(timings), "b": {"fwd_ms": 10.0, "note": "text"}}}


def test_compare_flags_only_timings_slower_than_the_factor():
    stored = result(fwd_ms=10.0, grad_ms=20.0)
    rows, failures = compare.compare(stored, result(fwd_ms=29.0, grad_ms=61.0), 3.0)
    assert {r[1] for r in rows} == {"fwd_ms", "grad_ms"}
    assert [(f[0], f[1]) for f in failures] == [("a", "grad_ms")]
    assert not compare.compare(stored, result(fwd_ms=1.0, grad_ms=2.0), 3.0)[1]
    # cases and timings missing from the new file are not compared
    assert compare.compare(stored, {"cases": {"c": {"fwd_ms": 1.0}}}, 3.0)[0] == []


def test_compare_exit_status(tmp_path):
    stored, fast, slow = (tmp_path / n for n in ("stored.json", "fast.json", "slow.json"))
    stored.write_text(json.dumps(result(fwd_ms=10.0)))
    fast.write_text(json.dumps(result(fwd_ms=12.0)))
    slow.write_text(json.dumps(result(fwd_ms=40.0)))
    assert compare.main([str(stored), str(fast)]) == 0
    assert compare.main([str(stored), str(slow)]) == 1
    assert compare.main([str(stored), str(slow), "--factor", "5"]) == 0


def test_compare_divides_out_a_slower_machine():
    stored = {"header": {"ruler_ms": 10.0}, "cases": {"a": {"fwd_ms": 10.0}}}
    busy = {"header": {"ruler_ms": 40.0}, "cases": {"a": {"fwd_ms": 50.0}}}      # 4x busy machine
    rows, failures = compare.compare(stored, busy, 3.0)
    assert rows[0][4] == pytest.approx(1.25) and not failures
    slow = {"header": {"ruler_ms": 10.0}, "cases": {"a": {"fwd_ms": 50.0}}}      # a real regression
    assert compare.compare(stored, slow, 3.0)[1]
    assert compare.machine_factor({}, busy) == 1.0
