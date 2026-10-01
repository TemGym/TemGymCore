"""Fail when a wave benchmark timing exceeds three times the stored minimum.

    python benchmarks/compare_wave.py benchmarks/results/wave_now.json
    python benchmarks/compare_wave.py NEW.json --baseline benchmarks/results/wave_gh200.json --factor 3

For every ``(case, n)`` in the new file that also exists in the baseline, the new ``min_ms`` (or the
median with ``--stat median``) must be at most ``factor`` times the baseline ``min_ms``. The best of
N runs ignores the noise of a shared GPU, and the factor absorbs the rest; a real regression (an
extra transform, a lost FFT path) is much larger than 3x. Cases missing from either file, and cases
that ran with another process on the GPU, are reported and skipped unless ``--strict`` is given.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_BASELINE = Path(__file__).resolve().parent / "results" / "wave_gh200.json"


def load(path):
    """Return ``{(case, n): record}`` from a benchmark JSON file."""
    data = json.loads(Path(path).read_text())
    return {(r["case"], r["n"]): r for r in data["results"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new", help="JSON written by benchmarks/wave.py")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--factor", type=float, default=3.0)
    parser.add_argument("--stat", choices=("min", "median"), default="min", help="statistic of the new run")
    parser.add_argument("--strict", action="store_true", help="also fail on missing or contended cases")
    args = parser.parse_args(argv)
    new, baseline = load(args.new), load(args.baseline)
    failures, skipped = [], []
    print(f"{'case':<26}{'n':>6}{'new ' + args.stat + ' ms':>14}{'stored min ms':>15}{'ratio':>8}")
    for key in sorted(baseline, key=lambda k: (k[1], k[0])):
        if key not in new:
            skipped.append(f"{key[0]} n={key[1]}: not measured")
            continue
        record = new[key]
        value = record["min_ms" if args.stat == "min" else "median_ms"]
        ratio = value / baseline[key]["min_ms"]
        contended = bool(record.get("other_processes_before") or record.get("other_processes_after"))
        flag = ""
        if ratio > args.factor:
            flag = "  FAIL (contended GPU)" if contended and not args.strict else "  FAIL"
            if not (contended and not args.strict):
                failures.append(f"{key[0]} n={key[1]}: {value:.3f} ms is {ratio:.1f}x the stored minimum")
            else:
                skipped.append(f"{key[0]} n={key[1]}: {ratio:.1f}x but another process was on the GPU")
        print(f"{key[0]:<26}{key[1]:>6}{value:>14.3f}{baseline[key]['min_ms']:>15.3f}{ratio:>8.2f}{flag}")
    for note in skipped:
        print("skipped:", note)
    if args.strict and skipped:
        failures.extend(skipped)
    if failures:
        print("\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print(f"\nall timings within {args.factor:g}x of the stored minimum")
    return 0


if __name__ == "__main__":
    sys.exit(main())
