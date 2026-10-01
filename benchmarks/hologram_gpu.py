"""The renderer on the real hologram bundle: 160,801 beamlets on a 1024 x 1024 camera.

The bundle (``benchmarks/hologram_bundle.py``) is the workload the GPU renderer is sized for:
90% of the beamlets that touch the sensor share one width (footprint half-width 169 px), 7% are
intermediate and 401 are extremely broad, and only about 12,500 of the 160,801 reach the sensor.
The whole bundle is rendered, with no host-side culling.

Reports the minimum of ``--repeats`` runs of the forward field, ``grad``, ``jvp`` (one
direction) and value + ``jacfwd`` (seven directions), the end-to-end ``evaluate_gaussians``
forward, and the class occupancy of the plan.  ``--accuracy`` also compares with the exact dense
sum and with the frozen historical dense kernel (both take seconds)::

    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python benchmarks/hologram_gpu.py --out hologram_gpu --accuracy
    python benchmarks/hologram_gpu.py --set n_classes=6 --set items=65536 --what fwd_ms,grad_ms

The result is written to ``benchmarks/results/<out>.json``.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from common import header, parse_config, print_table, timeit, transforms, write_results  # noqa: E402
from hologram_bundle import hologram_bundle  # noqa: E402

from temgym_core.evaluate import (  # noqa: E402
    _prepare_gaussian_params,
    describe_plan,
    evaluate_gaussians,
)
from temgym_core.render import evaluate_packed  # noqa: E402
from temgym_core.render.packing import (  # noqa: E402
    affine_from_grid,
    pack_gaussian_params,
    to_index_space,
)

COLUMNS = ("fwd_ms", "grad_ms", "jvp_ms", "jac7_ms", "e2e_fwd_ms")


def peak_error(a, b):
    """Largest difference of two fields relative to the peak amplitude of ``b``."""
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def accuracy(beam, detector, config):
    """Errors relative to the peak field of the renderer against the dense references."""
    index = dataclasses.replace(config, coordinates="index")
    metre = dataclasses.replace(config, coordinates="metre")
    sparse_index = evaluate_gaussians(beam, detector, method="sparse", config=index)
    sparse_metre = evaluate_gaussians(beam, detector, method="sparse", config=metre)
    dense_index = evaluate_gaussians(beam, detector, method="dense", config=index)
    dense_metre = evaluate_gaussians(beam, detector, method="dense", config=metre)
    from legacy_dense_kernel import legacy_dense_kernel

    legacy = legacy_dense_kernel(*_prepare_gaussian_params(beam), detector.coords,
                                 tile_pixels=16, tile_beams=16).reshape(detector.shape)
    return dict(
        sparse_index_vs_dense_index=peak_error(sparse_index, dense_index),
        sparse_metre_vs_dense_metre=peak_error(sparse_metre, dense_metre),
        dense_metre_vs_legacy_dense=peak_error(dense_metre, legacy),
        sparse_metre_vs_legacy_dense=peak_error(sparse_metre, legacy),
        dense_index_vs_legacy_dense=peak_error(dense_index, legacy),
        sparse_index_vs_legacy_dense=peak_error(sparse_index, legacy),
        cutoff_bound_over_peak=float(config.cutoff * jnp.sum(jnp.abs(beam.amplitude))
                                     / jnp.max(jnp.abs(legacy))),
    )


def main():
    """Command-line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="RenderConfig override, e.g. rows=16x4x1 class_edges=190,260,512")
    parser.add_argument("--what", default="fwd_ms,grad_ms,jvp_ms,jac7_ms,e2e_fwd_ms")
    parser.add_argument("--metre", action="store_true", help="also time coordinates='metre'")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--out", default=None, help="result file name under benchmarks/results")
    args = parser.parse_args()
    config = parse_config(args.set)
    what = tuple(args.what.split(","))

    beam, detector = hologram_bundle()
    shape = tuple(detector.shape)
    n_pix = shape[0] * shape[1]
    plan = describe_plan(beam, detector, config)
    print(f"{beam.x.shape[0]:,} beamlets; classes {plan.class_counts}, dense {plan.dense}, "
          f"inactive {plan.inactive}", flush=True)
    results = dict(header=header(config), plan=plan._asdict(), cases={})

    def run(name, cfg):
        packed = pack_gaussian_params(*_prepare_gaussian_params(beam))
        j, o = affine_from_grid(detector)
        if cfg.coordinates == "index":
            packed = to_index_space(packed, j, o)
            kw = dict(shape=shape)
        else:
            kw = dict(coords=detector.coords, shape=shape,
                      affine=jnp.concatenate([j.ravel(), o]))

        def evaluate(p):
            return evaluate_packed(p, config=cfg, **kw)

        row = transforms(packed, evaluate, n_pix, tuple(w for w in what if w != "e2e_fwd_ms"),
                         args.repeats)
        if "e2e_fwd_ms" in what:
            forward = jax.jit(lambda b: evaluate_gaussians(b, detector, method="sparse", config=cfg))
            row["e2e_fwd_ms"] = timeit(lambda: forward(beam), args.repeats)[0]
        results["cases"][name] = row
        print(name, {k: round(v, 2) for k, v in row.items()}, flush=True)

    run("index", dataclasses.replace(config, coordinates="index"))
    if args.metre:
        run("metre", dataclasses.replace(config, coordinates="metre"))
    if args.accuracy:
        results["accuracy"] = accuracy(beam, detector, config)
        for key, value in results["accuracy"].items():
            print(f"  {key:<34s} {value:.3e}")
    columns = tuple(c for c in COLUMNS if any(c in r for r in results["cases"].values()))
    print_table("minimum of %d runs, ms (whole bundle, no culling)" % args.repeats,
                results["cases"], columns)
    if args.out:
        print("wrote", write_results(args.out, results))


if __name__ == "__main__":
    main()
