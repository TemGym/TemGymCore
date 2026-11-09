"""Coherent Gaussian fields and detector images.

``evaluate_gaussians`` sums a bundle's complex fields on a grid; ``gaussian_image`` integrates
the coherent intensity over a detector's pixels; ``gaussian_intensity_image`` is the fast
incoherent alternative (independent round spots). ``method="scan"`` is plain JAX: CPU or GPU,
every JAX transformation. ``method="gpu"`` is a dense Pallas kernel (every beamlet at every
pixel, forward only, no gradients); the full GPU renderer with gradients and sparse sums is
parked on branch ``park/gpu-gaussian-renderer``.
"""

from dataclasses import replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .config import require_x64
from .grid import Grid


def evaluate_gaussians(gaussian_ray, grid: Grid, *, method: str = "scan", batch_size: int | None = 128):
    """Sum every beamlet's complex field at every pixel of ``grid``.

    Parameters
    ----------
    gaussian_ray : GaussianBeam
        The bundle at the plane of ``grid``.
    grid : Grid
        Detector or any object with ``shape`` and ``coords`` (pixel centres in metres).
    method : {"scan", "gpu", "loop"}
        ``"scan"``: a ``lax.scan`` over beamlets, ``batch_size`` at a time (exact, portable,
        differentiable). ``"gpu"``: the dense Pallas kernel (exact, fast on a GPU, interpreted
        elsewhere; no gradients). ``"loop"``: a plain Python loop, kept as a reference.
    batch_size : int, optional
        Beamlets evaluated together per scan step (``None``: one at a time).

    Returns
    -------
    array, shape ``grid.shape``
        Complex128 field.
    """
    require_x64("evaluate_gaussians")
    if method == "scan":
        return _evaluate_gaussians_jax_scan_impl(gaussian_ray, grid.coords, grid.shape, batch_size=batch_size)
    if method == "gpu":
        field = _pallas_dense_field_jit(*_prepare_gaussian_params(gaussian_ray), jnp.asarray(grid.coords))
        return field.reshape(grid.shape)
    if method == "loop":
        return evaluate_gaussians_for(gaussian_ray, grid)
    raise ValueError(f"Unknown Gaussian evaluation method {method!r} (use 'scan', 'gpu' or 'loop').")


def gaussian_image(
    beam, detector, *, samples_per_pixel=2, blur_sigma_m=None,
    padding_pixels=4,
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
    field = evaluate_gaussians(beam, grid)
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


# ---------------------------------------------------------------- the dense Pallas kernel
def _pack_gaussian_params(r_centre, dr, amplitude, pathlength, Q_inv, k):
    """Pack phase coefficients identically for dense and spatial GPU kernels."""
    f = jnp.float64

    # Symmetrize Q_inv: after this Qxy == Qyx
    Q = jnp.asarray(Q_inv, jnp.complex128)
    Q = 0.5 * (Q + jnp.swapaxes(Q, -1, -2))

    kf = jnp.asarray(k, f)
    k_half = 0.5 * kf

    # Fold pathlength into amplitude: amp * exp(i*k*path)
    amp = jnp.asarray(amplitude, jnp.complex128) * jnp.exp(
        1j * kf * jnp.asarray(pathlength, f)
    )

    # Pack all per-beam scalars into (N, 12), pre-multiplying k into every
    # phase coefficient. After symmetrization the cross-term is 2*Qxy;
    # combined with the 0.5 prefactor: 0.5*k*2*Qxy = k*Qxy.
    rc = jnp.asarray(r_centre, f)
    drf = jnp.asarray(dr, f)
    return jnp.column_stack([
        rc[:, 0],                           # 0  mx
        rc[:, 1],                           # 1  my
        kf * drf[:, 0],                     # 2  k·drx
        kf * drf[:, 1],                     # 3  k·dry
        k_half * jnp.real(Q[:, 0, 0]),      # 4  ½k·Re(Qxx)
        k_half * jnp.imag(Q[:, 0, 0]),      # 5  ½k·Im(Qxx)
        kf * jnp.real(Q[:, 0, 1]),          # 6  k·Re(Qxy)  (= ½k·2Re(Qxy))
        kf * jnp.imag(Q[:, 0, 1]),          # 7  k·Im(Qxy)
        k_half * jnp.real(Q[:, 1, 1]),      # 8  ½k·Re(Qyy)
        k_half * jnp.imag(Q[:, 1, 1]),      # 9  ½k·Im(Qyy)
        jnp.real(amp),                      # 10 Re(amp)
        jnp.imag(amp),                      # 11 Im(amp)
    ])


def _pallas_dense_field(
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
    where d = r2 - r_centre_b, using a tiled Pallas/Triton kernel.

    All per-beam quantities are packed into a single (N, 12) array on the device
    with k folded into every phase coefficient, so the inner kernel loop
    contains no per-beam k multiplications. On a GPU it compiles through Triton; elsewhere it
    runs in Pallas interpret mode (same result, slowly).
    """
    from jax.experimental import pallas as pl

    try:
        from jax.experimental.pallas import triton as plgpu
    except ImportError:
        plgpu = None

    N, P = r_centre.shape[0], r2.shape[0]
    f = jnp.float64
    beams = _pack_gaussian_params(r_centre, dr, amplitude, pathlength, Q_inv, k)

    # Column indices
    MX, MY = 0, 1
    KDX, KDY = 2, 3
    QR_XX, QI_XX, QR_XY, QI_XY, QR_YY, QI_YY = 4, 5, 6, 7, 8, 9
    AR, AI = 10, 11

    # Tiling & padding
    T_P, T_B = int(tile_pixels), int(tile_beams)
    grid_n = -(-P // T_P)
    n_btiles = -(-N // T_B)

    def _pad(x, n):
        d = n - x.shape[0]
        return jnp.pad(x, ((0, d),) + ((0, 0),) * (x.ndim - 1)) if d > 0 else x

    det = _pad(jnp.asarray(r2, f), grid_n * T_P)
    beams = _pad(beams, n_btiles * T_B)

    out_shape = (
        jax.ShapeDtypeStruct((grid_n * T_P,), f),
        jax.ShapeDtypeStruct((grid_n * T_P,), f),
    )

    # ---- Pallas kernel ----
    def kernel(det_ref, beams_ref, out_re_ref, out_im_ref):
        pid = pl.program_id(axis=0)
        p = pid * T_P + jnp.arange(T_P, dtype=jnp.int32)
        x, y = det_ref[p, 0], det_ref[p, 1]

        def body(t, acc):
            re, im = acc
            b = t * T_B + jnp.arange(T_B, dtype=jnp.int32)

            dx = x[:, None] - beams_ref[b, MX][None, :]
            dy = y[:, None] - beams_ref[b, MY][None, :]
            dx2, dy2, dxdy = dx * dx, dy * dy, dx * dy

            # Phase — k already folded into all coefficients before the kernel
            phase = (dx   * beams_ref[b, KDX][None, :]
                   + dy   * beams_ref[b, KDY][None, :]
                   + dx2  * beams_ref[b, QR_XX][None, :]
                   + dxdy * beams_ref[b, QR_XY][None, :]
                   + dy2  * beams_ref[b, QR_YY][None, :])

            # Gaussian envelope (imaginary part of quadratic form)
            atten = jnp.exp(-(dx2  * beams_ref[b, QI_XX][None, :]
                            + dxdy * beams_ref[b, QI_XY][None, :]
                            + dy2  * beams_ref[b, QI_YY][None, :]))

            s, c = jnp.sin(phase), jnp.cos(phase)
            ar = beams_ref[b, AR][None, :]
            ai = beams_ref[b, AI][None, :]

            re = re + jnp.sum(atten * (ar * c - ai * s), axis=1)
            im = im + jnp.sum(atten * (ar * s + ai * c), axis=1)
            return re, im

        re, im = lax.fori_loop(
            0, n_btiles, body,
            (jnp.zeros((T_P,), f), jnp.zeros((T_P,), f)),
        )
        out_re_ref[p] = re
        out_im_ref[p] = im

    # ---- Launch ----
    has_gpu = any(d.platform in ("gpu", "cuda", "rocm") for d in jax.devices())
    kw = dict(out_shape=out_shape, grid=(grid_n,))
    if has_gpu:
        if plgpu is not None:
            kw["compiler_params"] = plgpu.CompilerParams()
        else:
            # Compatibility with older Pallas releases.
            kw["backend"] = "triton"
    else:
        kw["interpret"] = True

    out_re, out_im = pl.pallas_call(kernel, **kw)(det, beams)

    return out_re[:P] + 1j * out_im[:P]


_pallas_dense_field_jit = jax.jit(_pallas_dense_field, static_argnames=["tile_pixels", "tile_beams"])


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


# ---------------------------------------------------------------- ray images
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
