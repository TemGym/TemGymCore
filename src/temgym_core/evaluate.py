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
    affine = affine_from_grid(grid)
    if affine is None:
        # Only pixel positions are known: dense kernels on the metre coordinates.
        return _packed(gaussian_ray, None), grid.coords, None, None
    shape = tuple(int(n) for n in grid.shape)
    if config.coordinates == "index":
        return _packed(gaussian_ray, affine), None, shape, None
    packed = _packed(gaussian_ray, None)
    return packed, grid.coords, shape, jnp.concatenate([affine[0].ravel(), affine[1]])


@jax.jit
def _packed(gaussian_ray, affine):
    """The packed bundle, in the pixel-index frame of ``affine`` when it is given.

    Packing is some forty small array operations. Run one at a time they took 46 ms for a
    256 x 256 image whose kernel takes 1.1 ms (GH200, 2026-11-22); compiled together, well under
    a millisecond.
    """
    packed = pack_gaussian_params(*sanitise_beam_fields(*_prepare_gaussian_params(gaussian_ray)))
    if affine is None:
        return packed
    return to_index_space(packed, *affine)


def _render_field(gaussian_ray, grid, config, *, exact):
    """Render a bundle on ``grid`` with the differentiable GPU renderer (or its reference)."""
    packed, coords, shape, affine = _kernel_frame(gaussian_ray, grid, config)
    field = evaluate_packed(packed, coords, shape=shape, affine=affine, config=config,
                            exact=exact)
    return field.reshape(grid.shape)



class ConvergedImage(NamedTuple):
    """The result of :func:`converged_gaussian_image`.

    Attributes
    ----------
    field : array, shape ``detector.shape``
        Complex field of the finest decomposition.
    num_across : int
        Beamlets across the decomposition that gave ``field`` (``num_across**2`` in all).
    change : float
        Largest change of the intensity between the last two decompositions, as a fraction of
        the largest intensity: an estimate of the error of the coarser of the two.
    converged : bool
        Whether ``change`` fell to ``tolerance`` before ``max_across`` was reached.
    history : tuple of (int, float)
        ``(num_across, change)`` for each refinement, the first with ``change = inf``.
    """

    field: jnp.ndarray
    num_across: int
    change: float
    converged: bool
    history: tuple


def converged_gaussian_image(beam, components, detector, *, num_across=31, tolerance=1e-2,
                             max_across=511, extent_rms=6.0, overlap=1.1, method="auto"):
    """Render one Gaussian beam through ``components`` on ``detector``, refining until it converges.

    The beam is decomposed into ``num_across**2`` beamlets
    (:func:`~temgym_core.source.decompose_gaussian`), traced through ``components`` to
    ``detector`` and summed. Then the beamlets across are doubled (``2 * num_across + 1``), and again, until the intensity changes by at most
    ``tolerance`` of its peak between two decompositions, or ``max_across`` is passed. A specimen
    acts on each beamlet through its local slope and curvature, so finer beamlets follow finer
    detail; the change between two levels estimates the error of the coarser.

    Parameters
    ----------
    beam : GaussianBeam
        One round Gaussian, at a plane before ``components``.
    components : sequence of Component
        The column from the beam to the detector, without the detector.
    detector : Detector
        Where the image forms.
    num_across : int
        Beamlets across the first decomposition (at least 8).
    tolerance : float
        Largest change of the intensity, as a fraction of its peak, at which to stop.
    max_across : int
        Largest number of beamlets across; the result says whether it converged before it.
    extent_rms, overlap : float
        As in :func:`~temgym_core.source.decompose_gaussian`. The extent sets the area the
        beamlets cover, in intensity RMS units of ``beam``: cover the image, not much more.
    method : str
        Renderer, as in :func:`evaluate_gaussians`.

    Returns
    -------
    ConvergedImage
    """
    from .run import run_to_end
    from .source import decompose_gaussian

    if int(num_across) < 8:
        raise ValueError("num_across must be at least 8.")
    if num_across > max_across:
        raise ValueError(f"num_across {num_across} is above max_across {max_across}.")
    history, previous, field, change = [], None, None, float("inf")
    while num_across <= max_across:
        bundle = decompose_gaussian(beam, num_across=num_across, extent_rms=extent_rms,
                                    overlap=overlap)
        at_detector = run_to_end(bundle, [*components, detector])
        field = evaluate_gaussians(at_detector, detector, method=method)
        intensity = jnp.abs(field) ** 2
        if not float(jnp.max(intensity)) > 0:
            raise ValueError("No light reaches the detector: nothing to converge.")
        if previous is not None:
            change = float(jnp.max(jnp.abs(intensity - previous)) / jnp.max(intensity))
        history.append((num_across, change))
        if change <= tolerance:
            return ConvergedImage(field, num_across, change, True, tuple(history))
        previous = intensity
        num_across = 2 * num_across + 1
    num_across = history[-1][0]
    return ConvergedImage(field, num_across, change, False, tuple(history))


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
    "ray_image",
    "converged_gaussian_image",
    "ConvergedImage",
    "PlanSummary",
]


@lru_cache(maxsize=4)
def _ray_binner(shape):
    """Trace rays through components, weight them by the components' intensity transmission and
    bin them on a detector of ``shape`` (compiled once per shape)."""
    from .components import GaussianActionComponent
    from .components._base import _uses_default_log_transmission
    from .run import run_iter

    @jax.jit
    def binned(components, rays, detector):
        weights = jnp.ones_like(rays.x)
        for component, at in run_iter(rays, components):
            if isinstance(component, GaussianActionComponent) and not _uses_default_log_transmission(component):
                # Rays carry no amplitude: apertures and specimens weight them by intensity.
                weights *= jnp.exp(2 * component.log_transmission(at.r_xy))
        iy, ix = replace(detector, z=at.z).metres_to_pixels((at.x, at.y))
        inside = (iy >= 0) & (iy < shape[0]) & (ix >= 0) & (ix < shape[1])
        index = jnp.where(inside, iy * shape[1] + ix, 0)
        return jnp.bincount(index, weights=jnp.where(inside, weights, 0),
                            length=shape[0] * shape[1]).reshape(shape) / weights.size

    return binned


def ray_image(rays, components, detector, *, samples_per_pixel=1, psf_sigma_pixels=0.0):
    """The image of rays on a detector: each ray traced through ``components``, weighted by the
    intensity transmission of apertures and specimens on the way, and counted in its pixel.

    The ray counterpart of :func:`gaussian_image` (geometric optics: no interference, no
    diffraction). Rays are binned on a grid ``samples_per_pixel`` times finer than the pixels,
    blurred by a Gaussian of ``psf_sigma_pixels`` (the detector's point-spread function, in
    pixels), then summed to pixels; the fine grid keeps small spots from locking to pixel centres.

    Parameters
    ----------
    rays : Ray
        The rays where they start (e.g. from :func:`temgym_core.source.sample_rays`).
    components : tuple
        Components ending at the detector plane.
    detector : Detector
        Pixel layout (shape, pixel size, pose); its plane is where the rays end.
    samples_per_pixel : int, optional
        Sub-samples per pixel axis.
    psf_sigma_pixels : float, optional
        Detector blur (rms, pixels).

    Returns
    -------
    numpy.ndarray
        Fraction of the rays per pixel (sums to the transmitted fraction that lands on the
        detector).
    """
    from scipy.signal import fftconvolve

    n = int(samples_per_pixel)
    height, width = detector.shape
    dy, dx = detector.pixel_size
    fine = replace(detector, shape=(height * n, width * n), pixel_size=(dy / n, dx / n))
    image = np.asarray(_ray_binner(fine.shape)(tuple(components), rays, fine))
    # Blur only the occupied rows and columns plus the whole PSF support, on pixel boundaries.
    occupied = [np.flatnonzero(image.any(axis=axis)) for axis in (1, 0)]
    result = np.zeros((height, width))
    if all(len(indices) for indices in occupied):
        halo = int(np.ceil(8 * psf_sigma_pixels)) + 1
        (y0, y1), (x0, x1) = [(max(0, indices[0] // n - halo), min(size, indices[-1] // n + halo + 1))
                              for indices, size in zip(occupied, (height, width))]
        crop = image[y0 * n:y1 * n, x0 * n:x1 * n]
        sigma = psf_sigma_pixels * n
        if sigma > 0:                      # separable; eight sigma (four biases high-dose widths)
            radius = int(8 * sigma + 0.5)
            kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
            kernel /= kernel.sum()
            crop = fftconvolve(crop, kernel[:, None], mode="same", axes=0)
            crop = np.maximum(fftconvolve(crop, kernel[None, :], mode="same", axes=1), 0)
        result[y0:y1, x0:x1] = crop.reshape(y1 - y0, n, x1 - x0, n).sum(axis=(1, 3))
    return result
