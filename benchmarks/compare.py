"""Compare a benchmark result with the stored one and fail on a regression.

Every timing (a key ending in ``_ms``) of every case present in both files is compared; the run
fails (exit status 1) if any new time exceeds the stored minimum by more than ``--factor``
(default 3, generous because CI GPUs are shared).  Both files carry a ``ruler_ms`` (a fixed
workload independent of the renderer); when present, the new timings are divided by the ratio of
the two rulers first, so a GPU that another job slows down does not fail the comparison.  The
nightly GPU job runs::

    python benchmarks/render.py --out render_nightly
    python benchmarks/compare.py benchmarks/results/render.json benchmarks/results/render_nightly.json
    python benchmarks/hologram_gpu.py --out hologram_gpu_nightly
    python benchmarks/compare.py benchmarks/results/hologram_gpu.json \\
        benchmarks/results/hologram_gpu_nightly.json

Store a new baseline by copying a run made on an idle GPU over the stored file.
"""

import argparse
import json
import sys
from pathlib import Path


def load(path):
    """Read a result file written by ``common.write_results``."""
    return json.loads(Path(path).read_text())


def machine_factor(stored, new):
    """Speed of the new run's GPU relative to the stored run's, from their rulers (1 if absent)."""
    old = stored.get("header", {}).get("ruler_ms")
    current = new.get("header", {}).get("ruler_ms")
    return current / old if old and current else 1.0


def compare(stored, new, factor):
    """Return ``(rows, failures)`` comparing every shared ``*_ms`` timing.

    Parameters
    ----------
    stored, new : dict
        Result files (``{"header": {"ruler_ms": ...}, "cases": {case: {timing_ms: value}}}``).
    factor : float
        Largest allowed ratio of the new to the stored time, after dividing by the ratio of the
        two files' ``ruler_ms`` (the slowdown of the machine itself).

    Returns
    -------
    tuple of list
        Rows ``(case, timing, stored_ms, new_ms, ratio)`` with ``ratio`` corrected for the
        machine, and the subset that exceeded ``factor``.
    """
    speed = machine_factor(stored, new)
    rows = []
    for case, old_row in stored["cases"].items():
        new_row = new["cases"].get(case)
        if new_row is None:
            continue
        for key, old_value in old_row.items():
            if key.endswith("_ms") and key in new_row:
                rows.append((case, key, old_value, new_row[key], new_row[key] / old_value / speed))
    return rows, [r for r in rows if r[4] > factor]


def main(argv=None):
    """Command-line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stored", help="stored baseline JSON")
    parser.add_argument("new", help="new result JSON")
    parser.add_argument("--factor", type=float, default=3.0)
    args = parser.parse_args(argv)
    stored, new = load(args.stored), load(args.new)
    rows, failures = compare(stored, new, args.factor)
    if not rows:
        print("no shared timings to compare")
        return 1
    print(f"machine speed relative to the stored run (ruler): {machine_factor(stored, new):.2f}x slower")
    print(f"{'case':<28s}{'timing':<14s}{'stored ms':>12s}{'new ms':>12s}{'ratio':>8s}")
    for case, key, old, now, ratio in rows:
        flag = "  <-- REGRESSION" if ratio > args.factor else ""
        print(f"{case:<28s}{key:<14s}{old:12.2f}{now:12.2f}{ratio:8.2f}{flag}")
    if failures:
        print(f"\n{len(failures)} timing(s) slower than {args.factor:g}x the stored minimum")
        return 1
    print(f"\nall {len(rows)} timings within {args.factor:g}x of the stored minimum")
    return 0


if __name__ == "__main__":
    sys.exit(main())
