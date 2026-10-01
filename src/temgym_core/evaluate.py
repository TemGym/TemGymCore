"""Coherent Gaussian fields and detector images.

Use ``evaluate_gaussians`` for complex fields and ``gaussian_image`` for pixel intensities.
``scan`` is the portable renderer and supports every JAX transformation; ``sparse`` (alias
``spatial``) and ``dense`` (alias ``gpu_kernel``) are the GPU renderers of
:mod:`temgym_core.render`, which are first-order differentiable with ``jit``, ``grad``, ``jvp``,
``jacfwd``, ``jacrev``, ``vmap`` and ``checkpoint``.  ``sparse`` sums the pairs whose envelope
exceeds a cutoff, ``dense`` every pair.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .config import require_x64
from .grid import Grid
from .render import RenderConfig, Tiles, evaluate_packed
from .render.packing import (
    affine_from_grid,
    pack_gaussian_params,
    sanitise_beam_fields,
    to_index_space,
)

METHODS = (
    "auto", "scan", "sparse", "spatial", "dense", "gpu_kernel", "gpu_kernel_required", "loop",
)


def evaluate_gaussians(
    gaussian_ray,
    grid: Grid,
    *,
    method: str = "auto",
    batch_size: int | None = 128,
    tile_pixels: int | None = None,
    tile_beams: int | None = None,
    spatial_cutoff: float | None = None,
    spatial_bins: int | None = None,
    config: RenderConfig | None = None,
):
    """Evaluate a Gaussian bundle on a detector grid with a stable backend dispatcher.

    Parameters
    ----------
    gaussian_ray : GaussianBeam
        The bundle at the plane of ``grid``.
    grid : Grid
        Detector or any object with ``shape`` and ``coords`` (pixel centres in metres); grids
        with ``pixels_to_metres_mat`` use the fast raster renderers.
    method : str
        ``"auto"`` is ``"sparse"`` on a GPU and ``"scan"`` elsewhere.  ``"sparse"`` (alias
        ``"spatial"``) sums, for every beamlet, the pixels where its envelope exceeds
        ``config.cutoff``; beamlets are binned by footprint radius on the device and the
        result is the truncated sum, with the omitted field bounded by
        ``cutoff * sum(abs(amplitude))``.  ``"dense"`` (alias ``"gpu_kernel"``) sums every pair
        exactly.  ``"gpu_kernel_required"`` is ``"dense"`` but raises unless a probed Pallas GPU
        backend is available.  ``"scan"`` is the portable JAX scan, ``"loop"`` a plain Python
        loop kept as a reference.  Without a usable GPU ``"sparse"`` and ``"dense"`` fall back
        to ``"scan"``.
    batch_size : int, optional
        Beamlets per step of the ``scan`` renderer (also the fallback of the GPU renderers).
    tile_pixels, tile_beams : int, optional
        Override the pixel and beam extent of the pixel-major GPU kernels (powers of two).
        Defaults are tuned; see :class:`~temgym_core.render.RenderConfig`.
    spatial_cutoff : float, optional
        Truncation level, overriding ``config.cutoff`` (default ``1e-12``).
    spatial_bins : int, optional
        Bins per axis of the sparse plan, overriding ``config.bins_per_axis`` (default 256).
    config : RenderConfig, optional
        Full options of the GPU renderers: cutoff, coordinate mode, radius classes and launch
        shapes.

    Returns
    -------
    array, shape ``grid.shape``
        Complex128 field.

    Notes
    -----
    The GPU renderers are first-order differentiable with respect to every beamlet field and,
    with the default ``coordinates="index"``, the detector pose; second derivatives raise
    ``NotImplementedError``.  A beamlet with any non-finite field contributes zero and
    receives a zero gradient.  Beamlets with a non-positive-definite envelope are summed in
    full by every pixel (slow but exact).
    """
    require_x64("evaluate_gaussians")
    if method not in METHODS:
        raise ValueError(f"Unknown Gaussian evaluation method {method!r}.")
    if method == "auto":
        method = "sparse" if gpu_kernel_available() else "scan"
    if method in ("sparse", "spatial"):
        return _evaluate_gaussians_sparse(
            gaussian_ray, grid, config=_render_config(
                config, tile_pixels, tile_beams, spatial_cutoff, spatial_bins),
            batch_size=batch_size,
        )
    if method == "scan":
        return evaluate_gaussians_jax_scan(
            gaussian_ray,
            grid,
            batch_size=batch_size,
        )
    if method in ("dense", "gpu_kernel"):
        return evaluate_gaussians_gpu(
            gaussian_ray,
            grid,
            tile_pixels=tile_pixels,
            tile_beams=tile_beams,
            fallback_batch_size=batch_size,
            config=config,
        )
    if method == "gpu_kernel_required":
        if not gpu_kernel_available():
            raise RuntimeError(
                "Gaussian evaluation method 'gpu_kernel_required' requires "
                "a working JAX Pallas GPU backend; CPU fallback is disabled."
            )
        return evaluate_gaussians_gpu_kernel_wrapper(
            gaussian_ray,
            grid,
            tile_pixels=tile_pixels,
            tile_beams=tile_beams,
            config=config,
        )
    return evaluate_gaussians_for(gaussian_ray, grid)


def _render_config(config, tile_pixels, tile_beams, cutoff, bins):
    """Merge the legacy keyword options of ``evaluate_gaussians`` into a ``RenderConfig``."""
    config = RenderConfig() if config is None else config
    changes = {}
    if cutoff is not None:
        changes["cutoff"] = float(cutoff)
    if bins is not None:
        changes["bins_per_axis"] = operator.index(bins)
    if tile_pixels is not None or tile_beams is not None:
        for name in ("forward", "tangent", "adjoint", "rows"):
            tiles = getattr(config, name)
            changes[name] = Tiles.sized(
                tile_pixels or tiles.pixels, tile_beams or tiles.beams)
    return replace(config, **changes) if changes else config


def gaussian_image(
    beam, detector, *, samples_per_pixel=2, blur_sigma_m=None,
    padding_pixels=4, method="scan", tile_beams=None,
):
    """Integrate coherent Gaussian intensity over a native Detector's pixels.

    The beam must already be at the detector plane. Fields interfere before
    taking their squared magnitude. Optional isotropic Gaussian blur represents
    source angular spread and/or detector PSF; its pixel integral is analytic.
    Refine quadrature and padding separately to check convergence. The result
    retains absolute flux, including losses outside the detector.
    """
    from jax.scipy.special import ndtr

    if not isinstance(samples_per_pixel, int) or samples_per_pixel < 1:
        raise ValueError("samples_per_pixel must be a positive integer.")
    if not isinstance(padding_pixels, int) or padding_pixels < 0:
        raise ValueError("padding_pixels must be a nonnegative integer.")
    if len(detector.shape) != 2 or any(int(n) != n or n <= 0 for n in detector.shape):
        raise ValueError("Detector shape must contain two positive integers.")
    n = samples_per_pixel
    ny, nx = detector.shape
    dy, dx = detector.pixel_size
    pad = padding_pixels * n if blur_sigma_m is not None else 0
    grid = replace(
        detector,
        shape=(ny * n + 2 * pad, nx * n + 2 * pad),
        pixel_size=(dy / n, dx / n),
    )
    field = evaluate_gaussians(beam, grid, method=method, tile_beams=tile_beams)
    intensity = jnp.abs(field) ** 2
    if blur_sigma_m is None:
        return intensity.reshape(ny, n, nx, n).mean(axis=(1, 3)) * dy * dx

    # Work along the detector axes, so rectangular pixels, pose and flips use
    # the same Grid transform as ordinary TemGym field evaluation.
    sigma = jnp.maximum(blur_sigma_m, 1e-30)

    def response(count, pitch):
        positions = (
            jnp.arange(count * n + 2 * pad) + 0.5 - (count * n + 2 * pad) / 2
        ) * pitch / n
        edges = (jnp.arange(count + 1) - count / 2) * pitch
        values = ndtr((edges[:, None] - positions[None, :]) / sigma)
        return values[1:] - values[:-1]

    return response(ny, dy) @ intensity @ response(nx, dx).T * (dy * dx / n**2)


def gaussian_intensity_image(
    centres,
    variance,
    weights,
    dose=1.0,
    background=0.0,
    *,
    shape_yx,
    psf_sigma_pixels=0.0,
):
    """Fast intensity approximation: integrate independent round Gaussian spots.

    Centres, variance and PSF use pixel units. Weights are integrated spot
    intensities. Off-detector flux is lost; no field interference is included.
    """
    ny, nx = shape_yx
    xe, ye = jnp.arange(nx + 1) - nx / 2, jnp.arange(ny + 1) - ny / 2
    width = jnp.sqrt(variance + psf_sigma_pixels**2)
    px = normal_interval_probability(
        (xe[:-1, None] - centres[:, 0]) / width,
        (xe[1:, None] - centres[:, 0]) / width,
    )
    py = normal_interval_probability(
        (ye[:-1, None] - centres[:, 1]) / width,
        (ye[1:, None] - centres[:, 1]) / width,
    )
    return dose * ((py * weights) @ px.T) + background


def normal_interval_probability(lower, upper):
    """Stable standard-normal probability on an ordered interval."""
    from jax.scipy.special import ndtr

    # Positive tails use symmetry to avoid subtracting two numbers near one.
    return jnp.where(lower > 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))


def zero_phase(u, idx_x, idx_y):
    """Remove the uniform phase at array index (idx_x, idx_y)."""
    u_centre = u[idx_x, idx_y]
    phase_difference = 0 - jnp.angle(u_centre)
    u_new = u * jnp.exp(1j * phase_difference)
    return u_new


_pack_gaussian_params = pack_gaussian_params


def _kernel_frame(gaussian_ray, grid, config):
    """Packed bundle and pixel description in the frame the renderer works in.

    Returns ``(packed, coords, shape, affine)``: ``coords`` is ``None`` in the pixel-index
    frame, ``shape`` the raster shape or ``None`` for grids without a pixel-to-metre map,
    ``affine`` the flattened ``[J, o]`` or ``None``.
    """
    params = sanitise_beam_fields(*_prepare_gaussian_params(gaussian_ray))
    packed = pack_gaussian_params(*params)
    affine = affine_from_grid(grid)
    if affine is None:
        # Only pixel positions are known: dense kernels on the metre coordinates.
        return packed, grid.coords, None, None
    shape = tuple(int(n) for n in grid.shape)
    if config.coordinates == "index":
        return to_index_space(packed, *affine), None, shape, None
    return packed, grid.coords, shape, jnp.concatenate([affine[0].ravel(), affine[1]])


def _render_field(gaussian_ray, grid, config, *, exact):
    """Render a bundle on ``grid`` with the differentiable GPU renderer (or its reference)."""
    packed, coords, shape, affine = _kernel_frame(gaussian_ray, grid, config)
    field = evaluate_packed(packed, coords, shape=shape, affine=affine, config=config,
                            exact=exact)
    return field.reshape(grid.shape)


class PlanSummary(NamedTuple):
    """How the sparse renderer sorts a bundle (see :func:`describe_plan`).

    Attributes
    ----------
    edges : tuple of float
        Upper footprint radius (pixels) of each sparse radius class.
    class_counts : tuple of int
        Beamlets in each sparse class.
    dense : int
        Beamlets in the dense class (footprint beyond the last edge, or a non-positive-definite
        envelope): every pixel sums over all of them.
    inactive : int
        Beamlets whose footprint box contains no pixel centre of the detector (or that are
        non-finite): never visited.
    cell_sizes : tuple of float
        Bin cell size (pixels) of each sparse class.
    largest_bin : int
        Most beamlets in any bin of any sparse class.
    """

    edges: tuple
    class_counts: tuple
    dense: int
    inactive: int
    cell_sizes: tuple
    largest_bin: int


def describe_plan(gaussian_ray, grid, config: RenderConfig | None = None) -> PlanSummary:
    """Report how the sparse renderer would classify a bundle on a detector.

    Runs the device-side planning step and copies the class occupancy back to the host, so it
    is a diagnostic (not jittable): use it to check that a bundle is dominated by narrow
    classes and how many beamlets fall in the slow dense class.

    Parameters
    ----------
    gaussian_ray : GaussianBeam
        The bundle at the plane of ``grid``.
    grid : Grid
        A detector with a pixel-to-metre map.
    config : RenderConfig, optional
        Rendering options.

    Returns
    -------
    PlanSummary
        Class occupancy of the plan.

    Raises
    ------
    ValueError
        If ``grid`` has no pixel-to-metre map (the sparse renderer needs a raster).
    """
    from .render.config import KernelSpec
    from .render.plan import build_plan, n_classes

    config = RenderConfig() if config is None else config
    packed, coords, shape, affine = _kernel_frame(gaussian_ray, grid, config)
    if shape is None:
        raise ValueError("describe_plan needs a grid with a pixel-to-metre map")
    spec = KernelSpec(config, shape[0] * shape[1], shape,
                      "index" if coords is None else "array", sparse=True)
    plan = build_plan(packed, jnp.zeros(6) if affine is None else affine, spec)
    m, bins = n_classes(spec), config.bins_per_axis
    counts = np.asarray(plan.counts).reshape(m + 2, bins * bins)
    return PlanSummary(
        edges=config.edges_for(shape),
        class_counts=tuple(int(c) for c in counts[:m].sum(axis=1)),
        dense=int(counts[m].sum()),
        inactive=int(counts[m + 1].sum()),
        cell_sizes=tuple(float(c) for c in np.asarray(plan.cell)),
        largest_bin=int(counts[:m].max(initial=0)),
    )


def _evaluate_gaussians_gpu_kernel_impl(
    r_centre: jnp.ndarray,      # (N,2) float64
    dr: jnp.ndarray,            # (N,2) float64
    amplitude: jnp.ndarray,     # (N,) complex128
    pathlength: jnp.ndarray,    # (N,) float64
    Q_inv: jnp.ndarray,         # (N,2,2) complex128
    k: jnp.ndarray,             # (N,) float64
    r2: jnp.ndarray,            # (P,2) float64
    *,
    tile_pixels: int = 16,
    tile_beams: int = 16,
):
    """
    Evaluate sum_b amp_b * exp(i*k_b*(path_b + d·dr_b + 0.5*d^T Q_inv_b d))
    where d = r2 - r_centre_b, with the dense Pallas/Triton kernel in metre space.

    All per-beam quantities are packed into a single (N, 12) array on the device
    with k folded into every phase coefficient, so the inner kernel loop
    contains no per-beam k multiplications.
    """
    params = sanitise_beam_fields(r_centre, dr, amplitude, pathlength, Q_inv, k)
    config = RenderConfig(forward=Tiles.sized(tile_pixels, tile_beams),
                          tangent=Tiles.sized(tile_pixels, tile_beams))
    return evaluate_packed(pack_gaussian_params(*params), r2, config=config, exact=True)


def evaluate_gaussians_gpu_kernel_wrapper(
    gaussian_ray,
    grid,
    *,
    tile_pixels: int | None = None,
    tile_beams: int | None = None,
    config: RenderConfig | None = None,
):
    """Sum every beamlet-pixel pair with the dense GPU kernel (differentiable).

    Parameters
    ----------
    gaussian_ray : GaussianBeam
        The bundle at the plane of ``grid``.
    grid : Grid
        Detector or object with ``shape`` and ``coords``.
    tile_pixels, tile_beams : int, optional
        Pixel and beam extent of the kernel tile (powers of two).
    config : RenderConfig, optional
        Coordinate mode and launch shapes; the exact sum ignores ``config.cutoff``.

    Returns
    -------
    array, shape ``grid.shape``
        Complex128 field.
    """
    config = _render_config(config, tile_pixels, tile_beams, None, None)
    return _render_field(gaussian_ray, grid, config, exact=True)


_evaluate_gaussians_gpu_kernel_impl = jax.jit(
    _evaluate_gaussians_gpu_kernel_impl,
    static_argnames=["tile_pixels", "tile_beams"],
)


# Public low-level entry point used for direct array evaluation.
evaluate_gaussians_gpu_kernel = _evaluate_gaussians_gpu_kernel_impl


def _prepare_gaussian_params(gaussian_ray):
    """Use the native beam's batching, then set the evaluator's float64 dtypes."""
    beam = gaussian_ray.to_vector()
    return (
        jnp.asarray(beam.r_xy, jnp.float64),
        jnp.asarray(beam.d_xy, jnp.float64),
        jnp.asarray(beam.amplitude, jnp.complex128),
        jnp.asarray(beam.pathlength, jnp.float64),
        jnp.asarray(beam.Q_inv, jnp.complex128),
        jnp.asarray(beam.k, jnp.float64),
    )


def _beam_field(r, dr, amplitude, pathlength, Q_inv, k, det_xy):
    delta = det_xy - r
    linear = jnp.sum(delta * dr, axis=-1)
    quadratic = jnp.sum((delta @ Q_inv) * delta, axis=-1)
    S_tot = pathlength + linear + 0.5 * quadratic
    return amplitude * jnp.exp(1j * k * S_tot)


def evaluate_gaussians_jax_scan(
    gaussian_ray,
    grid: Grid,
    *,
    batch_size: int | None = 128,
):
    """Sum every beamlet at every pixel with a ``lax.scan`` over beamlets (portable, exact).

    Supports every JAX transformation, on any platform; it is the CPU renderer and the
    reference the GPU renderers are checked against.

    Parameters
    ----------
    gaussian_ray : GaussianBeam
        The bundle at the plane of ``grid``.
    grid : Grid
        Detector or object with ``shape`` and ``coords``.
    batch_size : int, optional
        Beamlets evaluated together per scan step (``None``: one at a time).

    Returns
    -------
    array, shape ``grid.shape``
        Complex128 field.
    """
    # Extract grid properties before JIT boundary
    coords = grid.coords
    shape = grid.shape
    return _evaluate_gaussians_jax_scan_impl(
        gaussian_ray, coords, shape, batch_size=batch_size
    )


def _evaluate_gaussians_jax_scan_impl(
    gaussian_ray,
    coords: jnp.ndarray,
    shape: tuple,
    *,
    batch_size: int | None = 128,
):
    r, dr, amp, pathlength, Q_inv, k = _prepare_gaussian_params(gaussian_ray)
    r2 = coords
    P = r2.shape[0]
    init = jnp.zeros((P,), dtype=jnp.complex128)

    xs = (r, dr, amp, pathlength, Q_inv, k)

    @jax.jit
    def f_element(x):
        r_c, dr_c, amp_c, pathlength_c, Q_inv_c, k_c = x
        return _beam_field(r_c, dr_c, amp_c, pathlength_c, Q_inv_c, k_c, det_xy=r2)

    out = map_reduce(f_element, jnp.add, init, xs, batch_size=batch_size)
    return out.reshape(shape)


def evaluate_gaussians_for(
    gaussian_ray,
    grid: Grid,
):
    """Sum the beamlets with a plain Python loop: the simplest possible reference evaluator.

    Parameters
    ----------
    gaussian_ray : GaussianBeam
        The bundle at the plane of ``grid``.
    grid : Grid
        Detector or object with ``shape`` and ``coords``.

    Returns
    -------
    array, shape ``grid.shape``
        Complex128 field.
    """
    r, dr, amp, pathlength, Q_inv, k = _prepare_gaussian_params(gaussian_ray)
    r2 = grid.coords
    n = r.shape[0]
    total_field = jnp.zeros((r2.shape[0],), dtype=jnp.complex128)
    for i in range(n):
        field = _beam_field(
            r[i],
            dr[i],
            amp[i],
            pathlength[i],
            Q_inv[i],
            k[i],
            r2
        )
        total_field = total_field + field
    return total_field.reshape(grid.shape)


_evaluate_gaussians_jax_scan_impl = jax.jit(
    _evaluate_gaussians_jax_scan_impl,
    static_argnames=("batch_size", "shape")
)


def map_reduce(f, reducer, init, xs, *, batch_size: int | None = None):
    """Reduce ``f`` applied to the leading axis of ``xs`` with ``lax.scan``, ``batch_size`` at a time.

    Parameters
    ----------
    f : callable
        Maps one element of ``xs`` (a pytree slice) to a value ``reducer`` can accumulate.
    reducer : callable
        ``reducer(accumulator, value) -> accumulator``.
    init : object
        Initial accumulator.
    xs : pytree
        Arrays sharing their leading length.
    batch_size : int, optional
        Elements mapped together with ``vmap`` per scan step; ``None`` scans one at a time.

    Returns
    -------
    object
        The final accumulator.
    """
    def scan_fn(acc_inner, x):
        return reducer(acc_inner, f(x)), None

    if batch_size is None:
        acc, _ = lax.scan(scan_fn, init, xs)
        return acc

    if batch_size <= 0:
        raise ValueError("batch_size must be positive or None.")

    leaves = jax.tree_util.tree_leaves(xs)
    if not leaves:
        raise ValueError("xs must contain at least one array leaf.")
    n = leaves[0].shape[0]
    full = (n // batch_size) * batch_size
    acc = init

    def reduce_chunk(acc_inner, chunk):
        elements = jax.vmap(f)(chunk)

        def reduce_one(accumulator, value):
            return reducer(accumulator, value), None

        return lax.scan(reduce_one, acc_inner, elements)

    if full:
        chunked = jax.tree_util.tree_map(
            lambda x: x[:full].reshape((-1, batch_size) + x.shape[1:]),
            xs,
        )
        acc, _ = lax.scan(reduce_chunk, acc, chunked)

    if full < n:
        remainder = jax.tree_util.tree_map(lambda x: x[full:], xs)
        acc, _ = lax.scan(scan_fn, acc, remainder)
    return acc


def evaluate_gaussians_gpu(
    gaussian_ray,
    grid: Grid,
    *,
    tile_pixels: int | None = None,
    tile_beams: int | None = None,
    fallback_batch_size: int | None = 128,
    config: RenderConfig | None = None,
):
    """Evaluate with the dense Pallas GPU kernel, falling back to JAX scan on CPU."""
    if (config is None or config.impl == "auto") and not gpu_kernel_available():
        return evaluate_gaussians_jax_scan(
            gaussian_ray,
            grid,
            batch_size=fallback_batch_size,
        )
    return evaluate_gaussians_gpu_kernel_wrapper(
        gaussian_ray,
        grid,
        tile_pixels=tile_pixels,
        tile_beams=tile_beams,
        config=config,
    )


def _evaluate_gaussians_sparse(gaussian_ray, grid, *, config, batch_size=128):
    """Cutoff-truncated GPU rendering (see ``evaluate_gaussians``); scan without a GPU."""
    if config.impl == "auto" and not gpu_kernel_available():
        return evaluate_gaussians_jax_scan(gaussian_ray, grid, batch_size=batch_size)
    return _render_field(gaussian_ray, grid, config, exact=False)


@lru_cache(maxsize=1)
def gpu_kernel_available() -> bool:
    """Return whether the Pallas Gaussian kernel is usable on a real GPU.

    This is a numerical execution probe rather than a package-presence check.
    It requires JAX x64 on a real GPU, exercises the production 16x16 tiles
    with both beam and pixel padding, and compares the result with an
    independent NumPy evaluation before caching success.  The probe runs in a
    worker thread, whose JAX trace state is clean, so it executes eagerly even
    when first called while JAX is tracing (inside ``jit`` or ``grad``); an answer
    is never cached from a failed conversion of a tracer.  Call
    ``gpu_kernel_available.cache_clear()`` after changing JAX backend
    configuration in a long-lived process.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_probe_gpu_kernel).result()


def _probe_gpu_kernel() -> bool:
    try:
        if jax.default_backend() != "gpu" or not bool(jax.config.x64_enabled):
            return False
        gpu_devices = tuple(jax.devices("gpu"))
        if not gpu_devices or any(
            device.platform != "gpu" for device in gpu_devices
        ):
            return False

        n_beams = 17
        n_pixels = 19
        beam_index = np.arange(n_beams, dtype=np.float64)
        pixel_index = np.arange(n_pixels, dtype=np.float64)
        r_centre = np.column_stack(
            (
                0.025 * (beam_index - 8.0),
                0.018 * (np.mod(5.0 * beam_index, 17.0) - 8.0),
            )
        )
        dr = np.column_stack(
            (
                0.015 * np.sin(0.37 * beam_index),
                -0.012 * np.cos(0.29 * beam_index),
            )
        )
        amplitude = (0.4 + 0.03 * beam_index) * np.exp(
            1j * (0.17 * beam_index - 0.4)
        )
        pathlength = 0.007 * np.sin(0.23 * beam_index)
        k = 1.7 + 0.025 * beam_index
        q_inv = np.zeros((n_beams, 2, 2), dtype=np.complex128)
        q_inv[:, 0, 0] = (
            0.08 + 0.01 * np.cos(0.13 * beam_index)
            + 1j * (0.20 + 0.01 * beam_index)
        )
        q_inv[:, 1, 1] = (
            -0.05 + 0.008 * np.sin(0.19 * beam_index)
            + 1j * (0.24 + 0.006 * beam_index)
        )
        q_cross = (
            0.025 * np.sin(0.11 * beam_index)
            + 1j * 0.015 * np.cos(0.07 * beam_index)
        )
        q_inv[:, 0, 1] = q_cross
        q_inv[:, 1, 0] = q_cross
        coords = np.column_stack(
            (
                0.031 * (pixel_index - 9.0),
                0.027 * (np.mod(7.0 * pixel_index, 19.0) - 9.0),
            )
        )

        field = _evaluate_gaussians_gpu_kernel_impl(
            jnp.asarray(r_centre),
            jnp.asarray(dr),
            jnp.asarray(amplitude),
            jnp.asarray(pathlength),
            jnp.asarray(q_inv),
            jnp.asarray(k),
            jnp.asarray(coords),
            tile_pixels=16,
            tile_beams=16,
        )
        actual = np.asarray(jax.block_until_ready(field))

        # Keep the reference independent of the Pallas packing and reduction.
        delta = coords[:, None, :] - r_centre[None, :, :]
        linear = np.einsum("pbi,bi->pb", delta, dr)
        quadratic = np.einsum("pbi,bij,pbj->pb", delta, q_inv, delta)
        expected = np.sum(
            amplitude[None, :]
            * np.exp(
                1j
                * k[None, :]
                * (pathlength[None, :] + linear + 0.5 * quadratic)
            ),
            axis=1,
        )
        if (
            actual.shape != (n_pixels,)
            or actual.dtype != np.dtype(np.complex128)
            or not np.all(np.isfinite(actual))
            or not np.allclose(actual, expected, rtol=2e-11, atol=2e-12)
        ):
            return False
    except Exception:
        return False
    return True


__all__ = [
    "gaussian_image",
    "gaussian_intensity_image",
    "zero_phase",
    "evaluate_gaussians",
    "evaluate_gaussians_for",
    "evaluate_gaussians_gpu",
    "evaluate_gaussians_gpu_kernel",
    "evaluate_gaussians_gpu_kernel_wrapper",
    "evaluate_gaussians_jax_scan",
    "describe_plan",
    "gpu_kernel_available",
    "map_reduce",
    "PlanSummary",
]
