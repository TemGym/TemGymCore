import jax.numpy as jnp
import numpy as np
from numba import njit
from ase import units
from skimage.restoration import unwrap_phase


def custom_jacobian_matrix(ray_jac):
    """Convert a nested Ray-Jacobian Pytree to a 5×5 homogeneous matrix.

    Parameters
    ----------
    ray_jac : nested structure
        Result of `jax.jacobian` applied to a function returning `Ray`.

    Returns
    -------
    J : jnp.ndarray, shape (5, 5)
        Matrix with rows corresponding to [x, y, dx, dy, 1] derivatives.

    Notes
    -----
    Pure and JIT-friendly; structure mirrors `temgym_core.ray.Ray`. The final
    row keeps the homogeneous variable constant.

    Examples
    --------
    >>> import jax
    >>> from temgym_core.ray import Ray
    >>> f = lambda r: r
    >>> J = custom_jacobian_matrix(jax.jacobian(f)(Ray.origin()))
    >>> J.shape
    (5, 5)
    """
    return jnp.stack(
        [
            jnp.stack([ray_jac.x.x, ray_jac.x.y, ray_jac.x.dx, ray_jac.x.dy, ray_jac.x._one], axis=-1),
            jnp.stack([ray_jac.y.x, ray_jac.y.y, ray_jac.y.dx, ray_jac.y.dy, ray_jac.y._one], axis=-1),
            jnp.stack([ray_jac.dx.x, ray_jac.dx.y, ray_jac.dx.dx, ray_jac.dx.dy, ray_jac.dx._one], axis=-1),
            jnp.stack([ray_jac.dy.x, ray_jac.dy.y, ray_jac.dy.dx, ray_jac.dy.dy, ray_jac.dy._one], axis=-1),
            jnp.stack([ray_jac._one.x, ray_jac._one.y, ray_jac._one.dx, ray_jac._one.dy, ray_jac._one._one], axis=-1),
        ],
        axis=-2
    )


@njit
def multi_cumsum_inplace(values, partitions, start):
    """Compute multiple cumulative sums in-place with resets per partition.

    Parameters
    ----------
    values : numpy.ndarray, shape (K,)
        Values to be cumulatively summed. Mutated in-place.
    partitions : numpy.ndarray, shape (R,)
        Number of elements in each partition.
    start : float
        Initial value for each partition.

    Returns
    -------
    None

    Notes
    -----
    Numba-compiled; operates in-place on NumPy arrays. Partitions must sum to
    len(values).
    """
    part_idx = 0
    current_part_len = partitions[part_idx]
    part_count = 0
    values[0] = start
    for v_idx in range(1, values.size):
        if current_part_len == part_count:
            part_count = 0
            part_idx += 1
            current_part_len = partitions[part_idx]
            values[v_idx] = start
        else:
            values[v_idx] += values[v_idx - 1]
            part_count += 1


@njit
def inplace_sum(px_y, px_x, mask, frame, buffer):
    """Accumulate values into a 2D buffer at integer coordinates in-place.

    Parameters
    ----------
    px_y : numpy.ndarray, int
        Row indices.
    px_x : numpy.ndarray, int
        Column indices.
    mask : numpy.ndarray, bool
        Mask selecting valid entries to add.
    frame : numpy.ndarray, float32
        Values to add.
    buffer : numpy.ndarray, float32 or int
        Target accumulator; mutated in-place.

    Returns
    -------
    None

    Notes
    -----
    Numba-compiled; does bounds checking and mask filtering.
    """
    h, w = buffer.shape
    n = px_y.size
    for i in range(n):
        py = px_y[i]
        px = px_x[i]
        if mask[i] and (0 <= px_y[i] < h) and (0 <= px_x[i] < w):
            buffer[py, px] += frame[i]


def concentric_rings(
    num_points_approx: int,
    radius: float,
) -> np.ndarray:
    """Generate approximately uniform samples on concentric rings.

    Parameters
    ----------
    num_points_approx : int
        Approximate number of points to generate.
    radius : float
        Maximum radius in metres (or radians for angular distributions).

    Returns
    -------
    points : numpy.ndarray, shape (N, 2), float64
        Coordinates (y, x) within the disc of given radius.

    Notes
    -----
    Deterministic layout approximating uniform density.
    """
    num_rings = max(
        1, int(np.floor((-1 + np.sqrt(1 + 4 * num_points_approx / np.pi)) / 2))
    )

    # Calculate the circumference of each ring
    num_points_kth_ring = np.round(2 * np.pi * np.arange(1, num_rings + 1)).astype(int)
    num_rings = num_points_kth_ring.size
    points_per_unit = num_points_approx / num_points_kth_ring.sum()
    points_per_ring = np.round(num_points_kth_ring * points_per_unit).astype(int)

    # Make get the radii for the number of circles of rays we need
    radii = np.linspace(
        0,
        radius,
        num_rings + 1,
        endpoint=True,
    )[1:]
    div_angle = 2 * np.pi / points_per_ring

    params = np.stack((radii, div_angle), axis=0)

    # Cupy gave an error here saying that points_per_ring must not be an array
    repeats = points_per_ring.tolist()

    all_params = np.repeat(params, repeats, axis=-1)
    multi_cumsum_inplace(all_params[1, :], points_per_ring, 0.0)

    all_radii = all_params[0, :]
    all_angles = all_params[1, :]

    return np.stack(
        (
            all_radii * np.sin(all_angles),
            all_radii * np.cos(all_angles),
        ),
        axis=-1,
    )


def random_coords(num: int) -> np.ndarray:
    """Draw uniformly random points within a unit-radius disc.

    Parameters
    ----------
    num : int
        Number of points to draw (approximate; uses rejection sampling).

    Returns
    -------
    yx : numpy.ndarray, shape (N, 2), float64
        Coordinates (y, x) in [-1, 1] within the unit disc.
    """
    yx = np.random.uniform(
        -1,
        1,
        size=(int(num * 1.28), 2),  # 4 / np.pi
    )
    radii = np.sqrt((yx**2).sum(axis=1))
    mask = radii < 1
    yx = yx[mask, :]
    return yx


def try_ravel(val):
    """Ravel an array-like if possible, otherwise return as-is.

    Parameters
    ----------
    val : Any
        Input array-like or scalar.

    Returns
    -------
    out : Any
        Flattened array or original object.
    """
    try:
        return val.ravel()
    except AttributeError:
        return val


def try_reshape(val, maybe_has_shape):
    """Reshape a value to match a reference's shape if possible.

    Parameters
    ----------
    val : Any
        Array-like to reshape.
    maybe_has_shape : Any
        Reference object providing target `.shape` if present.

    Returns
    -------
    out : Any
        Reshaped array or original value.
    """
    try:
        return val.reshape(maybe_has_shape.shape)
    except AttributeError:
        return val


def FresnelPropagator(u1, L, wavelength, z, xp=np, prefactor=None):
    """Paraxial Fresnel propagation via transfer function (frequency-domain).

    Parameters
    ----------
    u1 : array_like, shape (M, N)
        Complex field at the source plane.
    L : float
        Physical side length of the (square) simulation window (metres).
    wavelength : float
        Wavelength (metres).
    z : float
        Propagation distance (metres).
    xp : module, optional
        Array module providing FFT routines (defaults to numpy).
    prefactor : complex or array_like, optional
        Global multiplicative factor applied to the propagated field.  This can
        be used to match a specific propagation gauge (e.g. the Gaussian beam
        Gouy phase prefactor det(A + B S₂)^(-1/2)).
    """

    M, N = u1.shape
    dx = L / M

    # build frequency coordinates without manual shifting
    fx = xp.fft.fftfreq(N, d=dx)
    fy = xp.fft.fftfreq(M, d=dx)
    FX, FY = xp.meshgrid(fx, fy)

    # transfer function in unshifted FFT domain
    H = xp.exp(-1j * xp.pi * wavelength * z * (FX**2 + FY**2))

    # forward FFT, multiply, and inverse FFT
    U1 = xp.fft.fft2(u1)
    U2 = H * U1
    u2 = xp.fft.ifft2(U2)
    u2 *= xp.exp(1j * 2 * xp.pi * z / wavelength)        # e^{ikz}
    return u2


# ---------- Fresnel FFT 2D ----------
def fresnel_fft_2d(X, Y, U, wavelength, z):
    k = 2*np.pi/wavelength
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    Ny, Nx = X.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    H = np.exp(1j*k*z) * np.exp(-1j*np.pi*wavelength*z*(FX**2 + FY**2))
    return np.fft.ifft2(np.fft.fft2(U) * H)



def FresnelPropagator1D(u1, L, wavelength, z, xp=np, prefactor=None):
    """1D Fresnel propagation via transfer function (frequency-domain).

    Parameters
    ----------
    u1 : array_like, shape (N,)
        Complex field at the source line.
    L : float
        Physical length of the simulation window (metres).
    wavelength : float
        Wavelength (metres).
    z : float
        Propagation distance (metres).
    xp : module, optional
        Array module providing FFT routines (defaults to numpy).

    Returns
    -------
    u2 : array_like, shape (N,)
        Propagated 1D field at distance z.
    """
    u1 = xp.asarray(u1)
    if u1.ndim != 1:
        raise ValueError("u1 must be a 1D array.")

    N = u1.shape[0]
    dx = L / N

    # frequency coordinate (cycles per metre)
    f = xp.fft.fftfreq(N, d=dx)

    # 1D Fresnel transfer function in unshifted FFT domain
    H = xp.exp(-1j * xp.pi * wavelength * z * (f**2))

    # forward FFT, multiply by transfer function, and inverse FFT
    U1 = xp.fft.fft(u1)
    U2 = H * U1
    u2 = xp.fft.ifft(U2)
    u2 *= xp.exp(1j * 2 * xp.pi * z / wavelength)        # e^{ikz}
    if prefactor is not None:
        u2 *= prefactor
    return u2


def AngularSpectrumPropagator(u1, L, wavelength, z, xp=np):
    """Propagate a complex field by distance z using the angular spectrum method.

    Parameters
    ----------
    u1 : array_like, shape (M, N)
        Complex field at the source plane.
    L : float
        Side length of the simulation window (metres).
    wavelength : float
        Wavelength (metres).
    z : float
        Propagation distance (metres).
    xp : module, optional
        Array module providing FFT routines (defaults to numpy).

    Returns
    -------
    u2 : array_like, shape (M, N)
        Propagated field at distance z.
    """
    M, N = u1.shape
    dx = L / M

    fx = xp.fft.fftfreq(N, d=dx)
    fy = xp.fft.fftfreq(M, d=dx)
    FX, FY = xp.meshgrid(fx, fy)

    k = 2 * xp.pi / wavelength
    kxky_sq = (2 * xp.pi) ** 2 * (FX ** 2 + FY ** 2)
    kz_sq = (k ** 2) - kxky_sq
    kz = xp.sqrt(kz_sq + 0j)

    U1 = xp.fft.fft2(u1)
    H = xp.exp(1j * kz * z)
    U2 = H * U1
    u2 = xp.fft.ifft2(U2)
    return u2


def fresnel_lens_imaging_solution(E0, Y, X, ps, lambda0, z1, f, z2):
    k = 2*np.pi/lambda0
    L = E0.shape[0]*ps

    # 1) propagate to lens
    E_lens = FresnelPropagator(E0, L, lambda0, z1)  # should include e^{ik z1}/(i λ z1) & chirps

    # 2) thin-lens phase (no amplitude)
    E_lens *= np.exp((-1j*k)/(2*f) * (X**2 + Y**2))

    # 3) propagate lens -> image plane
    E_final = FresnelPropagator(E_lens, L, lambda0, z2)

    return E_final


def zero_phase(u, idx_x, idx_y):
    u_centre = u[idx_x, idx_y]
    phase_difference = 0 - jnp.angle(u_centre)
    u_new = u * jnp.exp(1j * phase_difference)
    return u_new


def make_aperture(X, Y, aperture_ratio=0.1):
    """
    Generate a circular detector aperture mask.
    X, Y           : meshgrid coordinates (m)
    aperture_ratio : fraction of the detector diagonal to use as radius
    """
    detector_radius_x = np.max(np.abs(X))
    aperture_radius = detector_radius_x * aperture_ratio
    r2 = X**2 + Y**2
    return r2 < aperture_radius**2


def fibonacci_spiral(
    nb_samples: int,
    radius: float,
    alpha=2,
    **_,
):
    # From https://github.com/matt77hias/fibpy/blob/master/src/sampling.py
    # Fibonacci spiral sampling in a unit circle
    # Alpha parameter determines smoothness of boundary - default of 2 means a smooth boundary
    # 0 for a rough boundary.
    # Returns a tuple of x, y coordinates of the samples

    ga = np.pi * (3.0 - np.sqrt(5.0))

    # Boundary points
    np_boundary = np.round(alpha * np.sqrt(nb_samples))

    ii = np.arange(nb_samples)
    rr = np.where(
        ii > nb_samples - (np_boundary + 1),
        radius,
        radius * np.sqrt((ii + 0.5) / (nb_samples - 0.5 * (np_boundary + 1)))
    )
    rr[0] = 0.
    phi = ii * ga

    x = rr * np.cos(phi)
    y = rr * np.sin(phi)

    return x, y


def uniform_disk(
    nb_samples: int,
    radius: float,
    *,
    waist_radius: float | None = None,
    overlap_factor: float | None = None,
    max_iterations: int = 20,
    **_,
):
    if nb_samples < 0:
        raise ValueError("nb_samples must be non-negative.")
    if radius <= 0.0 and nb_samples > 0:
        raise ValueError("radius must be positive for non-zero samples.")
    if nb_samples == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty
    if nb_samples == 1:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float)

    if overlap_factor is not None and overlap_factor <= 0.0:
        raise ValueError("overlap_factor must be positive when provided.")

    area = np.pi * radius**2
    if area == 0.0:
        return np.zeros(nb_samples, dtype=float), np.zeros(nb_samples, dtype=float)

    if waist_radius is not None and overlap_factor is not None:
        base_spacing = waist_radius / overlap_factor
    else:
        base_spacing = np.sqrt(area / nb_samples)
    if base_spacing <= 0.0:
        raise ValueError("Computed spacing must be positive.")

    tol = 1e-12

    def build_rings(spacing: float) -> list[np.ndarray]:
        rings: list[np.ndarray] = []
        rings.append(np.array([[0.0, 0.0]], dtype=float))
        total = 1
        ring_idx = 1
        while True:
            r = ring_idx * spacing
            if r > radius + tol:
                break
            circumference = 2.0 * np.pi * r
            points_on_ring = max(1, int(np.round(circumference / spacing)))
            angles = np.linspace(0.0, 2.0 * np.pi, points_on_ring, endpoint=False, dtype=float)
            coords = np.stack((r * np.cos(angles), r * np.sin(angles)), axis=1)
            rings.append(coords)
            total += points_on_ring
            if total >= nb_samples:
                break
            ring_idx += 1
        return rings

    spacing = base_spacing
    for _ in range(max_iterations):
        rings = build_rings(spacing)
        counts = [ring.shape[0] for ring in rings]
        total_points = sum(counts)

        if total_points < nb_samples:
            ratio = total_points / nb_samples if total_points > 0 else 0.5
            spacing *= max(0.2, np.sqrt(max(ratio, 1e-6)))
            continue

        excess = total_points - nb_samples
        if excess > 0:
            last_ring = rings[-1]
            last_count = last_ring.shape[0]
            if excess >= last_count:
                rings.pop()
                total_points -= last_count
                excess = total_points - nb_samples
                if excess > 0 and rings:
                    last_ring = rings[-1]
                    last_count = last_ring.shape[0]
            if rings and excess > 0:
                keep = last_count - excess
                if keep <= 0:
                    rings.pop()
                else:
                    idx = np.linspace(0, last_count - 1, keep, endpoint=False, dtype=float)
                    idx = np.clip(np.round(idx).astype(int), 0, last_count - 1)
                    rings[-1] = last_ring[idx]
        coords = np.concatenate(rings, axis=0)
        if coords.shape[0] == nb_samples:
            coords = _snap_to_spacing(coords, base_spacing, radius, tol)
            return coords[:, 0], coords[:, 1]
        if coords.shape[0] > nb_samples:
            coords = coords[:nb_samples]
            coords = _snap_to_spacing(coords, base_spacing, radius, tol)
            return coords[:, 0], coords[:, 1]
        spacing *= 0.9

    raise RuntimeError("Could not construct uniform disk sampling with the requested parameters.")


def _snap_to_spacing(coords: np.ndarray, spacing: float, radius: float, tol: float) -> np.ndarray:
    radii = np.linalg.norm(coords, axis=1)
    snapped = coords.copy()
    mask = radii > tol
    if not np.any(mask):
        return snapped

    snap_spacing = np.round(spacing, decimals=12)
    snap_spacing = snap_spacing if snap_spacing > 0 else spacing
    target = np.round(radii[mask] / snap_spacing) * snap_spacing
    max_multiple = np.floor(radius / snap_spacing)
    target = np.clip(target, 0.0, max_multiple * snap_spacing)
    scale = np.where(radii[mask] > 0.0, target / radii[mask], 1.0)
    snapped[mask] *= scale[:, None]
    return snapped


def wavelength2energy(wavelength: float) -> float:
    """
    Calculate energy from relativistic de Broglie wavelength.

    Parameters
    ----------
    wavelength: float
        Relativistic de Broglie wavelength [Å].

    Returns
    -------
    float
        Energy [eV].
    """
    # Constants
    h = units._hplanck
    c = units._c
    m_e = units._me
    e = units._e

    term1 = (h * c / wavelength) ** 2
    term2 = (m_e * c**2) ** 2

    energy_joules = np.sqrt(term1 + term2) - m_e * c**2

    return energy_joules / e


def energy2wavelength(energy: float):
    """
    Calculate relativistic de Broglie wavelength from energy with higher precision options.

    Parameters
    ----------
    energy : float
        Kinetic energy [eV].
    """
    h = units._hplanck
    c = units._c
    m_e = units._me
    e = units._e

    E_j = energy * e
    mc2 = m_e * c * c
    E_total = E_j + mc2
    rad = E_total * E_total - mc2 * mc2
    p = jnp.sqrt(rad) / c
    wavelength_m = h / p
    return wavelength_m


def uniform_amp_from_overlaps(points: jnp.ndarray, waist: float) -> float:
    # points: (N,2) centers (your x0,y0)
    diffs = points[:, None, :] - points[None, :, :]
    d2 = jnp.sum(diffs**2, axis=-1)
    K = jnp.exp(-d2 / (2.0 * waist**2))
    return points.shape[0] / jnp.sum(K)


def uniform_amp_from_area(num_gaussians: int, waist: float, area: float) -> float:
    # area = (2*extent)^2 if you tile a square [-extent, extent]^2
    return area / (num_gaussians * jnp.pi * waist**2)


def grid_line_area(extent: float, n_cells: int, line_width: float) -> float:
    """
    Exact area of the union of n_lines vertical + n_lines horizontal strips
    of full width `line_width` inside the square [-extent, extent]^2.
    """
    n_lines = n_cells + 1
    L = 2.0 * extent
    if n_lines <= 1 or line_width <= 0.0:
        return 0.0
    s = L / (n_lines - 1)  # grid spacing
    w = line_width

    # Total covered measure along one axis by the union of equally spaced strips
    cover_1D = jnp.minimum(L, w + (n_lines - 1) * jnp.minimum(w, s))

    # Inclusion–exclusion for the 2D union (vertical ∪ horizontal)
    A = L * cover_1D + L * cover_1D - cover_1D * cover_1D
    return float(A)


def total_grid_length(extent: float, n_cells: int) -> float:
    """
    Total curve length of all grid lines (useful if you think in 'length × width').
    """
    n_lines = n_cells + 1
    L = 2.0 * extent
    return 2.0 * n_lines * L  # vertical + horizontal lengths


def find_sideband_center(mag, exclude_radius=15):
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist2 = (Y - cy)**2 + (X - cx)**2
    search = mag.copy()
    search[dist2 <= exclude_radius**2] = 0.0
    iy, ix = np.unravel_index(np.argmax(search), mag.shape)
    return (iy, ix)

def gaussian_sideband_filter(fft_shifted, center, sigma):
    h, w = fft_shifted.shape
    Y, X = np.ogrid[:h, :w]
    mask = np.exp(-(((X - center[1])**2 + (Y - center[0])**2) / (2 * sigma**2)))
    filtered_fft = fft_shifted * mask
    recon = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    return recon, mask, filtered_fft


def _fit_plane(phase, mask=None):
    """
    Fit z = ax + by + c (least-squares) to 'phase' over 'mask' (True=use).
    Returns (a, b, c).
    """
    H, W = phase.shape
    y, x = np.mgrid[0:H, 0:W]
    if mask is None:
        mask = np.isfinite(phase)
    m = mask & np.isfinite(phase)
    X = np.column_stack([x[m], y[m], np.ones(np.count_nonzero(m))])
    coeff, *_ = np.linalg.lstsq(X, phase[m].ravel(), rcond=None)
    a, b, c = coeff
    return a, b, c

def remove_ramp(phase, mask=None):
    """
    Subtract best-fit plane from 'phase'. Returns phase_detrended and (a,b,c).
    """
    a, b, c = _fit_plane(phase, mask)
    H, W = phase.shape
    y, x = np.mgrid[0:H, 0:W]
    plane = a*x + b*y + c
    return phase - plane, (a, b, c)


def reconstruct_complex(
    hologram_intensity: np.ndarray,
    *,
    exclude_radius: int = 5,
    sigma_pixels: float = 5.0,
    sideband_center: tuple | None = None,
    recenter_to_dc: bool = True,
    normalize_amplitude: bool = True,
):
    """
    Reconstruct the complex sideband field from an off-axis hologram.

    Parameters
    ----------
    hologram_intensity : 2D float array
        Raw intensity image of the hologram.
    exclude_radius : int
        Radius around DC to exclude when auto-picking a sideband.
    sigma_pixels : float
        Gaussian mask sigma (in pixels) for sideband filtering.
    sideband_center : (row, col) or None
        If None, auto-detect via 'find_sideband_center'.
    recenter_to_dc : bool
        If True, shift the filtered sideband back to the origin (carrier removal).
    normalize_amplitude : bool
        If True, divide by mean amplitude to make amplitudes ~ O(1).

    Returns
    -------
    dict with keys:
      complex: complex field
      amp: amplitude map
      phase_wrapped: wrapped phase (radians)
      phase_unwrapped: unwrapped phase (radians)
      sideband_center: chosen sideband center (r,c)
      gaussian_mask: mask used in Fourier domain
      fft: original centered FFT (complex)
    """
    # FFT of intensity pattern
    fft_img = np.fft.fftshift(np.fft.fft2(hologram_intensity.astype(np.float64)))
    fft_mag = np.abs(fft_img)

    # Auto-detect sideband if needed
    if sideband_center is None:
        sideband_center = find_sideband_center(fft_mag, exclude_radius=exclude_radius)

    # Bandpass the sideband with your helper
    recon_sideband, gaussian_mask, sideband_fft = gaussian_sideband_filter(
        fft_img, sideband_center, sigma_pixels
    )

    # Optionally recenter the carrier (depends on how your filter returns it)
    if recenter_to_dc:
        # Shift by the detected carrier vector (negative shift to move to DC)
        H, W = hologram_intensity.shape
        r0, c0 = sideband_center
        ky = (r0 - H//2) / H  # fractional cycles per pixel
        kx = (c0 - W//2) / W
        y, x = np.mgrid[0:H, 0:W]
        carrier = np.exp(-2j * np.pi * (kx * x + ky * y))
        recon_sideband = recon_sideband * carrier

    # Complex field, amplitude, and phase
    amp = np.abs(recon_sideband)
    if normalize_amplitude:
        m = np.isfinite(amp)
        scale = np.mean(amp[m]) if np.any(m) else 1.0
        if scale > 0:
            amp = amp / scale
            recon_sideband = recon_sideband / scale

    phase_wrapped = np.angle(recon_sideband)
    phase_unwrapped = unwrap_phase(phase_wrapped)

    return {
        "complex": recon_sideband,
        "amp": amp,
        "phase_wrapped": phase_wrapped,
        "phase_unwrapped": phase_unwrapped,
        "sideband_center": tuple(sideband_center),
        "gaussian_mask": gaussian_mask,
        "fft": fft_img,
    }


def remove_background_with_reference(
    sample_complex: np.ndarray,
    reference_complex: np.ndarray,
    *,
    amp_threshold: float = 0.05,
    unwrap: bool = True,
    detrend_plane: bool = True,
):
    """
    Remove instrument/background phase using a separate reference by complex division.

    C = S * conj(R) / (|R| + eps)
    phase_wrapped = angle(C)
    phase_unwrapped = unwrap_phase(phase_wrapped)
    (then optional plane/ramp removal)

    Parameters
    ----------
    sample_complex : 2D complex array
    reference_complex : 2D complex array
    amp_threshold : float
        Mask out pixels where |R| < amp_threshold * median(|R|).
    unwrap : bool
        If True, unwrap the differenced phase.
    detrend_plane : bool
        If True, subtract a best-fit plane from the (un)wrapped phase.

    Returns
    -------
    dict with keys:
      complex_corrected: complex background-corrected field
      amp: amplitude of corrected field
      phase_wrapped: wrapped phase difference
      phase_unwrapped: unwrapped phase difference (if unwrap=True)
      phase_final: after optional plane removal
      mask: valid pixels used
    """
    # Build mask from reference amplitude
    ref_amp = np.abs(reference_complex)
    med = np.median(ref_amp[np.isfinite(ref_amp)])
    eps = 1e-12
    mask = ref_amp >= (amp_threshold * max(med, eps))

    # Complex division (stabilized)
    C = np.zeros_like(sample_complex, dtype=np.complex128)
    denom = ref_amp + eps
    C[mask] = sample_complex[mask] * np.conj(reference_complex[mask]) / denom[mask]

    amp = np.abs(C)
    phase_wrapped = np.angle(C)

    if unwrap:
        phase_u = unwrap_phase(phase_wrapped)
    else:
        phase_u = phase_wrapped.copy()

    # Optional plane/ramp removal
    if detrend_plane:
        phase_final, _ = remove_ramp(phase_u, mask=mask)
    else:
        phase_final = phase_u

    return {
        "complex_corrected": C,
        "amp": amp,
        "phase_wrapped": phase_wrapped,
        "phase_unwrapped": phase_u,
        "phase_final": phase_final,
        "mask": mask,
    }


def reconstruct_with_background_removal(
    reference_hologram_intensity: np.ndarray,
    sample_hologram_intensity: np.ndarray,
    *,
    exclude_radius: int = 5,
    sigma_pixels: float = 5.0,
    ref_sideband_center: tuple | None = None,
    samp_sideband_center: tuple | None = None,
    amp_threshold: float = 0.05,
    detrend_plane: bool = True,
):
    """
    Full pipeline:
      1) reconstruct complex waves for reference & sample
      2) complex-divide (sample ÷ reference)
      3) unwrap once
      4) optional plane removal
    Returns dict with all intermediate & final results.
    """
    ref = reconstruct_complex(
        reference_hologram_intensity,
        exclude_radius=exclude_radius,
        sigma_pixels=sigma_pixels,
        sideband_center=ref_sideband_center,
    )

    samp = reconstruct_complex(
        sample_hologram_intensity,
        exclude_radius=exclude_radius,
        sigma_pixels=sigma_pixels,
        sideband_center=samp_sideband_center,
    )

    bg = remove_background_with_reference(
        samp["complex"], ref["complex"],
        amp_threshold=amp_threshold,
        unwrap=True,
        detrend_plane=detrend_plane,
    )

    return {
        "reference": ref,
        "sample": samp,
        "background_removed": bg,
    }
