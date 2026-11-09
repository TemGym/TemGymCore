"""Stage 2 (draft): the two specimen measurements from their frames -- |A| from the cross-grating
image (whole-image lattice fit) and |B| from the gold rings.
Uses TemGym: the gold ring spacings and the electron wavelength only."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from stage2 import stage1_fit as F
from temgym_core.components.specimens import AU_RINGS


def _fft_lattice(image, min_cycles=1.5, pad=16):
    """Two perpendicular fundamental wave vectors (cycles per px) from the zero-padded FFT."""
    n = image.shape[0]
    window = np.outer(np.hanning(n), np.hanning(n))
    f = np.abs(np.fft.fftshift(np.fft.fft2((image - image.mean()) * window, s=(n * pad, n * pad))))
    k = (np.arange(n * pad) - n * pad // 2) / (n * pad)
    KX, KY = np.meshgrid(k, k)
    f[np.hypot(KX, KY) < min_cycles / n] = 0
    f[KY < 0] = 0                                                # half plane (Hermitian)
    vectors = []
    for _ in range(2):
        i, j = np.unravel_index(np.argmax(f), f.shape)
        vectors.append(np.array([k[j], k[i]]))
        angle = np.abs(np.angle(np.exp(1j * (np.arctan2(KY, KX) - np.arctan2(k[i], k[j])))))
        f[(angle < np.pi / 4) | (angle > 3 * np.pi / 4)] = 0     # keep only the perpendicular one
    return np.array(vectors)


def grating_period_px(image):
    """Lattice period (px) of a cross-grating image, from a fit of the whole image.

    Model of every pixel: ``exp(P(r)) * (b + (1 - b) * hole(s) * hole(t))``. Here ``(s, t)`` are
    lattice coordinates, ``r = r0 + s a1 + t a2``; ``hole(f)`` is 1 inside a square hole of side
    ``h`` periods and falls to 0 over a soft edge ``e``; ``b`` is the bar level; ``P`` is a quadratic
    polynomial, the beam profile. The profile is fitted, so an uneven beam does not bias the
    lattice, and squares cut by the frame edge still count. Start: lattice vectors from the FFT
    peaks, the origin from a scan over one cell. Returns the mean period, the two lattice vectors
    (px), and the fitted parameters with the fraction of the image the fit explains. Raises
    ``ValueError`` if that fraction is below 0.9 (too few squares lit).
    """
    from scipy.ndimage import gaussian_filter
    n = image.shape[0]
    x = np.arange(n) - (n - 1) / 2
    X, Y = np.meshgrid(x, x)
    data = np.asarray(image, float) / np.max(image)
    xs, ys, d = jnp.asarray(X.ravel()), jnp.asarray(Y.ravel()), jnp.asarray(data.ravel())

    def model(q):
        lattice = jnp.array([[q[0], q[2]], [q[1], q[3]]])            # columns a1, a2
        s, t = jnp.linalg.solve(lattice, jnp.stack([xs - q[4], ys - q[5]]))
        h, e, b = q[6], q[7], q[8]
        hole = (jax.nn.sigmoid((h / 2 - jnp.abs(s - jnp.round(s))) / e)
                * jax.nn.sigmoid((h / 2 - jnp.abs(t - jnp.round(t))) / e))
        u, v = xs / n, ys / n
        profile = jnp.exp(q[9] + q[10] * u + q[11] * v + q[12] * u * u + q[13] * u * v + q[14] * v * v)
        return profile * (b + (1 - b) * hole)

    residual = jax.jit(lambda q: model(q) - d)
    jacobian = jax.jit(jax.jacfwd(residual))
    # start: lattice vectors from the FFT, the beam profile from a heavy blur, levels from percentiles
    k = _fft_lattice(image)
    lattice = np.linalg.inv(k)                                     # columns: real-space lattice vectors
    blur = np.log(np.maximum(gaussian_filter(data, 0.15 * n), 1e-6))
    u, v = X.ravel() / n, Y.ravel() / n
    design = np.c_[np.ones_like(u), u, v, u * u, u * v, v * v]
    profile, *_ = np.linalg.lstsq(design, blur.ravel(), rcond=None)
    flat = data / np.exp((design @ profile).reshape(n, n))
    b0 = float(np.clip(np.percentile(flat, 10) / np.percentile(flat, 90), 0.02, 0.9))
    best = None
    for fx in np.arange(8) / 8:                                    # origin: scan one cell
        for fy in np.arange(8) / 8:
            r0 = lattice @ np.array([fx, fy])
            q = np.r_[lattice[:, 0], lattice[:, 1], r0, 0.6, 0.03, b0, profile]
            cost = float(np.sum(np.asarray(residual(jnp.asarray(q))) ** 2))
            if best is None or cost < best[0]:
                best = (cost, q)
    out = least_squares(lambda q: np.asarray(residual(jnp.asarray(q))), best[1],
                        jac=lambda q: np.asarray(jacobian(jnp.asarray(q))), method="lm", xtol=1e-12, ftol=1e-12)
    a1, a2 = out.x[0:2], out.x[2:4]
    explained = 1 - np.sum(out.fun ** 2) / np.sum((data - data.mean()) ** 2)
    if explained < 0.9:                       # a good fit explains >= 96 % of the image
        raise ValueError(f"the lattice fit explains only {explained:.0%} of the image: too few squares lit, "
                         "or not a cross grating. Spread the beam over the frame and try again.")
    return float((np.hypot(*a1) + np.hypot(*a2)) / 2), (a1, a2), dict(parameters=out.x, explained=float(explained))


def ring_radii_px(image, camera, rings=("111", "200"), half_width_px=5.0, direct_mask_px=8.0):
    """Ring radii (px) of a gold pattern.

    The centre is the centroid of the direct beam. The radial intensity profile (direct beam
    masked) gives a first guess of the 111 radius; the other rings are placed by their known
    d-spacing ratios. Each radius is then the intensity-weighted mean radius within
    +/- ``half_width_px`` of its guess (biased outward by about s^2 / 2R for spots of rms s:
    0.03 % for 2 px spots at 80 px). Returns the radii and the profile.
    """
    x = camera.coordinates
    X, Y = np.meshgrid(x, x)
    j = np.unravel_index(np.argmax(image), image.shape)
    near = np.hypot(X - x[j[1]], Y - x[j[0]]) < direct_mask_px
    centre = complex((image * X)[near].sum() / image[near].sum(), (image * Y)[near].sum() / image[near].sum())
    r = np.hypot(X - centre.real, Y - centre.imag)
    rest = np.where(r > direct_mask_px, image, 0.0)
    bins = np.arange(0, camera.n / 2, 0.5)
    profile = np.bincount(np.digitize(r.ravel(), bins), weights=rest.ravel(), minlength=len(bins) + 1)[1:len(bins) + 1]
    profile = profile / np.maximum(np.bincount(np.digitize(r.ravel(), bins), minlength=len(bins) + 1)[1:len(bins) + 1], 1)
    ratio = {k: AU_RINGS["111"] / AU_RINGS[k] for k in rings}
    # the 111 guess that puts the most intensity on all the rings
    candidates = bins[(bins > direct_mask_px + half_width_px)]
    score = [sum(np.interp(g * ratio[k], bins, profile) for k in rings) for g in candidates]
    guess = candidates[int(np.argmax(score))]
    radii = {}
    for k in rings:
        window = np.abs(r - guess * ratio[k]) < half_width_px
        radii[k] = float((rest * r)[window].sum() / rest[window].sum())
    return radii, dict(centre=centre, bins=bins, profile=profile)


def magnification_and_camera_length(*, grating_frame, gold_frame, period_m, pitch_m, ht_v, dark=0.0):
    """The two specimen measurements from their frames: the magnification |A| at the 40k row (grating image,
    certified ``period_m``) and the camera length |B| (m/rad) at the diffraction setting (gold
    rings). Returns a dict with both and the measured lattice period and ring radii (px)."""
    from temgym_core.constants import energy2wavelength

    period_px, _, fit = grating_period_px(np.asarray(grating_frame, float) - dark)
    gold = np.asarray(gold_frame, float) - dark
    radii, _ = ring_radii_px(gold, F.Camera(n=gold.shape[0], pitch=pitch_m))
    lam = energy2wavelength(ht_v)
    lengths = {k: r * pitch_m / (lam / AU_RINGS[k]) for k, r in radii.items()}
    return dict(magnification=period_px * pitch_m / period_m, camera_length_m=float(np.mean(list(lengths.values()))),
                period_px=period_px, lattice_fit_explained=fit["explained"], ring_radii_px=radii,
                camera_length_per_ring_m=lengths)
