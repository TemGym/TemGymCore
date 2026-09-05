"""Benchmark the MAGICAL notebook; optionally test a fixed-neighbour renderer.

Run from the repository root:
    .venv/bin/python examples/microscope_models/benchmark_gaussian_forward.py \
        --sparse --output /tmp/gaussian-forward.json

The sparse experiment is valid only for the beam geometry used to construct its
index. Rebuild the index if centres or widths change. It is not a production
replacement for evaluate_gaussians or a general differentiable optics renderer.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import nbformat
import numpy as np
from scipy.spatial import cKDTree

from temgym_core.evaluate import evaluate_gaussians, gpu_kernel_available
from temgym_core.run import run_to_end
from temgym_core.source import circular_input_wave

jax.config.update("jax_enable_x64", True)


def notebook_setup(path):
    """Execute this trusted example's setup cells, stopping before propagation."""
    notebook = nbformat.read(path, as_version=4)
    namespace = {}
    stopped = False
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if cell.source.startswith("apply_sample ="):
                stopped = True
                break
            source = cell.source.replace("%matplotlib inline", "")
            exec(compile(source, str(path), "exec"), namespace)
    if not stopped:
        raise ValueError("Notebook setup boundary changed; review the benchmark loader.")
    return namespace


def timed(fn, value, repeats):
    jax.block_until_ready(value)
    result = jax.block_until_ready(fn(value))  # compile and warm this variant
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = jax.block_until_ready(fn(value))
        samples.append(1000 * (perf_counter() - start))
    return result, {
        "median_ms": float(np.median(samples)),
        "min_ms": min(samples), "max_ms": max(samples),
    }


def relative_l2(actual, reference):
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def neighbour_index(beam, coords, cutoff=1e-12):
    """Conservative circular support from the weakest envelope precision axis.

    Outside radius sqrt(2 log(1/cutoff) / lambda_min(k Im Q)), a
    beam's envelope is at most cutoff. The sum of omitted field magnitudes
    is therefore at most cutoff * sum(abs(amplitude)) for these beams.
    """
    q_imag = np.asarray(beam.Q_inv).imag
    q_imag = 0.5 * (q_imag + q_imag.swapaxes(-1, -2))
    smallest = np.linalg.eigvalsh(np.asarray(beam.k)[:, None, None] * q_imag)[:, 0]
    if not np.all(np.isfinite(smallest)) or np.any(smallest <= 0):
        raise ValueError("Sparse support requires positive definite envelope precision.")
    radius = float(np.sqrt(2 * np.log(1 / cutoff) / smallest).max())
    tree = cKDTree(np.asarray(beam.r_xy))
    neighbours = tree.query_ball_point(np.asarray(coords), radius, workers=1,
                                      return_sorted=False)
    counts = np.array([len(row) for row in neighbours])
    indices = np.full((len(neighbours), max(1, int(counts.max()))), -1, np.int32)
    for i, row in enumerate(neighbours):
        indices[i, :len(row)] = row
    return jnp.asarray(indices), {
        "cutoff": cutoff, "radius_m": radius,
        "candidate_pairs": int(counts.sum()),
        "padded_pairs": int(indices.size),
        "max_neighbours": int(counts.max()),
        "omitted_field_absolute_bound": float(cutoff * np.abs(np.asarray(beam.amplitude)).sum()),
    }


@jax.jit
def sparse_field(beam, indices, coords, k):
    """Experimental coherent gather-and-sum with a fixed, externally built index.

    Pass k explicitly, as the public dense evaluator does: recomputing it inside
    this JIT can change rounding, magnified by metre-scale electron path phases.
    """
    valid = indices >= 0
    ids = jnp.maximum(indices, 0)
    q = 0.5 * (beam.Q_inv + jnp.swapaxes(beam.Q_inv, -1, -2))
    amplitude = beam.amplitude * jnp.exp(1j * k * beam.pathlength)
    linear = k[:, None] * beam.d_xy
    quadratic = 0.5 * k[:, None, None] * q
    delta = coords[:, None, :] - beam.r_xy[ids]
    phase = (delta * linear[ids]).sum(-1)
    phase += jnp.einsum("pki,pkij,pkj->pk", delta, quadratic[ids].real, delta)
    decay = jnp.einsum("pki,pkij,pkj->pk", delta, quadratic[ids].imag, delta)
    values = amplitude[ids] * jnp.exp(-decay) * (jnp.cos(phase) + 1j * jnp.sin(phase))
    return jnp.sum(jnp.where(valid, values, 0.), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--waist-nm", type=float, default=1.)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--gpu-spatial", action="store_true",
                        help="Benchmark device-side bin rebuilding and spatial rendering")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.waist_nm <= 0:
        parser.error("repeats and waist must be positive")
    path = Path(__file__).with_name("magical_gaussian_imaging.ipynb")
    ns = notebook_setup(path)
    if not gpu_kernel_available():
        raise RuntimeError("This benchmark requires the verified Pallas GPU evaluator.")
    sample, column, grid = ns["sample"], ns["column"], ns["image_grid"]
    beam = circular_input_wave(aperture_radius=ns["beam_radius"],
                               waist=args.waist_nm * 1e-9, voltage=ns["voltage"],
                               z0=ns["z_sample"], overlap_factor=1.5)
    apply_sample = jax.jit(lambda b: sample(b))
    propagate = jax.jit(lambda b: run_to_end(b, column))
    exited = jax.block_until_ready(apply_sample(beam))
    out = jax.block_until_ready(propagate(exited))
    rows = {}
    metadata = {
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "precision": "float64/complex128", "beam_count": int(beam.x.size),
        "shape": list(grid.shape), "repeats": args.repeats,
        "dense_pair_count": int(beam.x.size * np.prod(grid.shape)),
        "waist_nm": args.waist_nm,
    }
    print(json.dumps(metadata), flush=True)

    def record(name, fn, value, reference=None):
        result, stats = timed(fn, value, args.repeats)
        if reference is not None:
            stats["relative_l2"] = relative_l2(np.asarray(result), reference)
        rows[name] = stats
        print(name, json.dumps(stats), flush=True)
        return result

    record("sample", apply_sample, beam)
    record("optics", propagate, exited)
    reference = np.asarray(record("render_16x16", lambda b: evaluate_gaussians(b, grid), out))
    for tp, tb in ((16, 32), (16, 64), (32, 32), (8, 64)):
        render = lambda b: evaluate_gaussians(b, grid, tile_pixels=tp, tile_beams=tb)
        field = record(f"render_{tp}x{tb}", render, out, reference)
        assert relative_l2(np.asarray(field), reference) < 1e-12
    for tb in (16, 32):
        render = lambda b: evaluate_gaussians(b, grid, tile_beams=tb)
        record(f"render_outer_jit_16x{tb}", jax.jit(render), out, reference)
        forward = lambda b: jnp.abs(render(propagate(apply_sample(b))))**2
        record(f"forward_16x{tb}", forward, beam, np.abs(reference)**2)
        fused = jax.jit(lambda b: jnp.abs(render(run_to_end(sample(b), column)))**2)
        record(f"forward_fused_16x{tb}", fused, beam, np.abs(reference)**2)
    # Exactly the same pixel coordinates as a subset of the full reference.
    subset = grid.coords.reshape(*grid.shape, 2)[::2, ::2]
    preview_grid = SimpleNamespace(shape=subset.shape[:2], coords=subset.reshape(-1, 2))
    record("render_128_subset_16x32", lambda b: evaluate_gaussians(b, preview_grid, tile_beams=32),
           out, reference[::2, ::2])

    if args.gpu_spatial:
        from temgym_core.spatial_evaluate import _build_bins

        build = jax.jit(lambda r, q, k: _build_bins(
            r, q, k, cutoff=1e-12, bins_per_axis=64))
        _, stats = timed(lambda values: build(*values), (out.r_xy, out.Q_inv, out.k), args.repeats)
        rows["gpu_bin_build"] = stats
        print("gpu_bin_build", json.dumps(stats), flush=True)
        render = lambda b: evaluate_gaussians(b, grid, method="spatial", tile_beams=32)
        field = np.asarray(record("render_gpu_spatial_rebuild", render, out, reference))
        assert relative_l2(field, reference) < 1e-9
        bound = 1e-12 * np.abs(np.asarray(out.amplitude)).sum()
        assert np.max(abs(field - reference)) < bound + 1e-14
        record("forward_gpu_spatial_rebuild",
               lambda b: jnp.abs(render(propagate(apply_sample(b))))**2,
               beam, np.abs(reference)**2)

    if args.sparse:
        start = perf_counter()
        indices, details = neighbour_index(out, grid.coords)
        jax.block_until_ready(indices)
        details["index_build_ms"] = 1000 * (perf_counter() - start)
        metadata["sparse"] = details
        print("sparse_index", json.dumps(details), flush=True)
        render = lambda b: sparse_field(b, indices, grid.coords, b.k).reshape(grid.shape)
        field = np.asarray(record("render_sparse_fixed_index", render, out, reference))
        rows["render_sparse_fixed_index"]["max_absolute_error"] = float(np.max(abs(field - reference)))
        assert relative_l2(field, reference) < 1e-9
        assert relative_l2(abs(field)**2, abs(reference)**2) < 1e-9
        assert np.max(abs(field - reference)) < details["omitted_field_absolute_bound"] + 1e-14
        # This reuse is valid because the timed calls retain the same geometry.
        record("forward_sparse_fixed_index",
               lambda b: jnp.abs(render(propagate(apply_sample(b))))**2,
               beam, np.abs(reference)**2)

    if args.output:
        args.output.write_text(json.dumps({"metadata": metadata, "timings": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
