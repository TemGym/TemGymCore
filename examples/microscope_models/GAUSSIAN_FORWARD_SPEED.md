# GPU spatial Gaussian evaluation

The public evaluator now supports an explicit spatial backend:

```python
field = evaluate_gaussians(
    beam_out, image_grid, method="spatial",
    tile_beams=32, spatial_cutoff=1e-12, spatial_bins=64,
)
intensity = jnp.abs(field)**2
```

The MAGICAL notebook selects this backend through `evaluation_method`. The
library's `auto` choice remains the dense GPU evaluator (or scan on CPU).

## What runs on the GPU

Every call computes conservative Gaussian support radii from the smallest
eigenvalue of `k * Im(Q_inv)`, assigns beam centres to spatial bins, sorts beam
indices, and computes bin counts and prefix sums. A Pallas/Triton kernel visits
the nine neighbouring bins for each pixel and sums the complex Gaussian fields.
The bin size covers the largest support radius. All beam geometry, index building,
sorting and evaluation stay on the GPU; the implementation uses no SciPy tree,
host-side geometry decisions or cached neighbour lists.

Bin ranges have no fixed capacity: a crowded bin can span arbitrarily many
beam tiles. Each invocation rebuilds the bins from the current beam positions
and curvatures. Shift, rotation, magnification and defocus changes therefore do
not depend on a stale index. The same TemGym sample and `run_to_end` pipeline
still perform the wave propagation.

For broad/strongly overlapping beams, a device-side conditional selects the
original dense kernel. Nonfinite inputs and invalid/near-singular Gaussian
precision matrices also use that path. On CPU, the public spatial option uses
the exact portable scan evaluator. No GPU speedup is promised for those cases.

## Accuracy and differentiation

The backend retains float64/complex128 and coherent summation. The per-pixel
absolute omitted-field bound is

```
spatial_cutoff * sum(abs(beam.amplitude))
```

apart from floating-point roundoff. This is an absolute field bound, not a
relative intensity bound near destructive-interference zeros. Neighbour bins
include extra beams beyond the circular support, so actual errors can be much
smaller than the bound. Very anisotropic envelopes use a conservative circular
radius; they may be less efficient than an ellipse-aware index.

Dense and spatial kernels share the same phase-coefficient packing. Computing
wavenumbers outside the inner compiled renderer preserves the existing handling
of large electron path phases. Putting an additional JIT around an entire pipeline
can change floating-point rounding, as it can for the existing dense renderer.

**This is a forward-only Pallas backend. Automatic differentiation is not
implemented; use `method="scan"` for gradient-based calculations.** Device-side
bin rebuilding supports changing inputs under JIT, but it does not supply an
autodiff rule through the sparse renderer or its changing membership.

## Measurements

Measured on 2026-09-05, NVIDIA GH200 144G HBM3e, JAX 0.10.2,
float64/complex128. The cleaned MAGICAL scene uses 159,044 Gaussians (1 nm
waist), five electromagnetic lenses and a 256 × 256 observation grid.

| Operation | Warm median |
|---|---:|
| Original dense complete forward, 16 × 16 tiles | 159.6 ms |
| Tuned dense complete forward, 16 × 32 tiles | 141.6 ms |
| GPU bin construction alone | 0.26 ms |
| Spatial rendering, including bin construction | 5.3 ms |
| **Spatial complete forward, including bin construction** | **7.3 ms** |

This is about **22× faster than the original complete forward**, or **19× faster
than the tuned dense path**. These are medians of five synchronized warm calls,
not asynchronous launch timings. The complete forward includes the sample,
`run_to_end`, index rebuilding, field reconstruction and GPU intensity formation.
Compilation, source/atlas construction, reference comparisons, NumPy output
copies and plotting are excluded. Component timings need not sum to the complete
forward because their separate synchronization/dispatch overhead differs.

For this scene the relative complex-field difference from the dense evaluator
was approximately **5.5e-16**; relative intensity difference was about **9.7e-16**.
The notebook's independent comparison with direct sample transmission still
has about **0.26%** image error, dominated by the sample's finite Gaussian
approximation, rather than the sparse evaluator.

The earlier CPU-neighbour-list prototype took about 613 ms to build its index
and about 5.4 ms for a complete forward with that index cached. The new backend
removes that CPU construction cost and rebuilds its own bins every time.
Its sorted-bin representation visits more candidates than the exact per-pixel
CPU neighbour list, which explains why its render alone can be slower than that
prototype's cached gather-and-sum. The dense pair count is about 10.4 billion;
the acceleration comes from avoiding distant beam/pixel pairs.

## Reproduce and validate

From the repository root:

```bash
.venv/bin/python examples/microscope_models/benchmark_gaussian_forward.py \
    --gpu-spatial --output /tmp/gaussian-gpu-spatial.json
```

The script executes the trusted notebook's setup cells, stopping before the
image-verification loop. It prints timing ranges and complex-field/intensity
errors, and asserts the spatial field comparison and tail bound. Use
`--waist-nm 2` for the coarse source; `--sparse` additionally runs the older
CPU-index prototype for comparison.

The normal root `conftest.py` forces JAX onto CPU. To execute the kernel tests
on GPU explicitly:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python -m pytest --noconftest \
    tests/test_spatial_evaluate.py tests/test_gaussian_improvements.py -q
```

The tests compare against an independent NumPy coherent sum and the dense
kernel. They cover complex interference, per-beam voltage, rotated anisotropic
curvatures, repeated JIT calls with moving/broadening beams, hot bins spanning
multiple beam tiles, dense fallback, bin boundaries, distant pixels, nonfinite
inputs, empty grids, parameter validation and CPU fallback.

Further optimisations can target the per-pixel kernel launch layout or
anisotropic support bounds. Lower output resolution and coarser beamlets remain
explicit accuracy/sampling tradeoffs: the existing 2 nm-waist source has 39,761
beamlets and about 0.92% sample-image error.
