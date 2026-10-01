"""Synthetic benchmarks of the differentiable GPU renderer (pixel-unit bundles and mixed radii).

Bundles are in pixel units (pitch 1, ``k = 1``); ``waist`` is the 1/e amplitude radius in pixels
and the cutoff-1e-12 footprint half-width is about ``5.25 * waist``.  Each case reports the
minimum of ``--repeats`` runs of the forward field, ``grad`` of an intensity loss, ``jvp`` (one
direction) and value + ``jacfwd`` (seven directions), plus their same-process ratios to the
forward.  ``--mixed`` times 100,000 narrow beamlets with one broad beamlet on 512 x 512.

Run on an idle GPU::

    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python benchmarks/render.py --out render                # all cases
    python benchmarks/render.py --case narrow251k_w4.5 --set rows=16x4x1 --set n_classes=6
    python benchmarks/render.py --mixed --legacy                # compare with the old renderer

The result is written to ``benchmarks/results/<out>.json``.
"""

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import header, parse_config, print_table, timeit, transforms, write_results  # noqa: E402
from render_bundles import pixel_bundle  # noqa: E402

from temgym_core.render import evaluate_packed  # noqa: E402

CASES = {
    "narrow20k_256_w2.5": dict(n=20_000, shape=(256, 256), waist=2.5),
    "narrow251k_w2.5": dict(n=251_000, shape=(1024, 1024), waist=2.5),
    "narrow251k_w4.5": dict(n=251_000, shape=(1024, 1024), waist=4.5),
    "narrow594k_w4.5": dict(n=594_000, shape=(1024, 1024), waist=4.5),
    "broad20k_w15": dict(n=20_000, shape=(1024, 1024), waist=15.0),
}
MIXED = dict(n=100_000, shape=(512, 512), narrow=2.5, broad=(None, 8.0, 15.0, 30.0))
COLUMNS = ("fwd_ms", "grad_ms", "jvp_ms", "jac7_ms", "legacy_fwd_ms")


def bundle(n, shape, waist, broad=None, seed=0):
    """Random pixel-unit bundle; beamlet 0 is replaced by a beamlet of waist ``broad`` if given."""
    rng = np.random.default_rng(seed)
    packed, _, unpacked = pixel_bundle(rng, n, shape, waist_px=waist)
    if broad is not None:
        q = unpacked[4].copy()
        q[0] = 2j * np.diag([1 / broad**2] * 2)
        from temgym_core.render import pack_gaussian_params

        packed = np.asarray(pack_gaussian_params(*(jnp.asarray(a) for a in (
            unpacked[0], unpacked[1], unpacked[2], unpacked[3], q, unpacked[5]))))
    return jnp.asarray(packed), unpacked, q if broad is not None else unpacked[4]


def legacy_forward(unpacked, q, shape, repeats):
    """Time the frozen historical spatial renderer (forward only) on the same bundle."""
    from legacy_spatial import _evaluate_spatial

    x, y = np.meshgrid(np.arange(shape[1], dtype=float), np.arange(shape[0], dtype=float))
    coords = jnp.asarray(np.column_stack([x.ravel(), y.ravel()]))
    r, dr, amp, path, _, k = (jnp.asarray(a) for a in unpacked)
    return timeit(lambda: _evaluate_spatial(r, dr, amp, path, jnp.asarray(q), k, coords,
                                            cutoff=1e-12, bins_per_axis=64, tile_beams=32),
                  repeats)[0]


def run_case(name, spec, config, repeats, what, legacy):
    """Time one synthetic case; returns the row of timings."""
    print(f"[{name}] N={spec['n']:,} shape={spec['shape']} waist={spec['waist']}", flush=True)
    packed, unpacked, q = bundle(spec["n"], spec["shape"], spec["waist"], spec.get("broad"))
    shape = spec["shape"]

    def evaluate(p):
        return evaluate_packed(p, shape=shape, config=config)

    row = transforms(packed, evaluate, shape[0] * shape[1], what, repeats)
    if legacy:
        row["legacy_fwd_ms"] = legacy_forward(unpacked, q, shape, repeats)
    print("   ", {k: round(v, 2) for k, v in row.items()}, flush=True)
    return row


def main():
    """Command-line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="default: all")
    parser.add_argument("--mixed", action="store_true", help="the mixed-radius case only")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="RenderConfig override, e.g. rows=16x4x1 class_edges=30,60,120")
    parser.add_argument("--what", default="fwd_ms,grad_ms,jvp_ms,jac7_ms")
    parser.add_argument("--legacy", action="store_true", help="also time the old spatial forward")
    parser.add_argument("--out", default=None, help="result file name under benchmarks/results")
    args = parser.parse_args()
    config = parse_config(args.set)
    what = tuple(args.what.split(","))
    results = dict(header=header(config), cases={})
    if args.mixed:
        for broad in MIXED["broad"]:
            spec = dict(n=MIXED["n"], shape=MIXED["shape"], waist=MIXED["narrow"], broad=broad)
            name = "mixed_all_narrow" if broad is None else f"mixed_broad_w{broad:g}"
            results["cases"][name] = run_case(name, spec, config, args.repeats, what, args.legacy)
        base = results["cases"]["mixed_all_narrow"]["fwd_ms"]
        for name, row in results["cases"].items():
            row["fwd_vs_all_narrow"] = row["fwd_ms"] / base
    else:
        for name in args.case or CASES:
            results["cases"][name] = run_case(name, CASES[name], config, args.repeats, what,
                                              args.legacy)
    columns = tuple(c for c in COLUMNS if any(c in r for r in results["cases"].values()))
    print_table("minimum of %d runs, ms" % args.repeats, results["cases"], columns)
    if args.mixed:
        for name, row in results["cases"].items():
            print(f"{name}: forward {row['fwd_vs_all_narrow']:.2f}x the all-narrow case")
    if args.out:
        print("wrote", write_results(args.out, results))


if __name__ == "__main__":
    main()
