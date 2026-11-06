"""Specimens for the virtual microscope: a cross grating and a polycrystalline gold film.

Both sit at the specimen plane (``z = 0``) of the virtual column.

* :class:`CrossGrating` absorbs on its bars. It is a TemGym component; the ray renderer weights
  every ray by its intensity transmission, which gives the in-focus bright-field image.
* :class:`PolycrystallineGold` diffracts: every grain sends a beamlet at a Bragg angle. Each
  beamlet is a TemGym Gaussian beam carried through the projector and integrated over the camera
  pixels; the direct beam stays a ray image. The rings come out at radius |B| * wavelength / d.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax_dataclasses as jdc
import numpy as np

from ..components import GaussianActionComponent
from ..constants import energy2wavelength

#: Gold lattice constant (m) and the first three ring spacings with their relative strengths
#: (multiplicity times a falling structure factor).
AU_LATTICE = 0.40782e-9
AU_RINGS = {"111": AU_LATTICE / np.sqrt(3), "200": AU_LATTICE / 2, "220": AU_LATTICE / np.sqrt(8)}
AU_WEIGHTS = {"111": 8 * 1.0, "200": 6 * 0.8, "220": 12 * 0.5}


@jdc.pytree_dataclass(kw_only=True)
class CrossGrating(GaussianActionComponent):
    """Square holes in an absorbing film, on a square lattice.

    Parameters
    ----------
    period : float
        Lattice period (m).
    hole_fraction : float
        Side of a hole as a fraction of the period.
    bar_transmission : float
        Intensity transmission of the bars (holes transmit 1).
    edge_width : float
        Width of the soft bar edges (m).
    rotation : float
        Angle of the lattice in the specimen plane (rad).
    x0, y0 : float
        A hole centre (m).
    z : float
        Axial position (the specimen plane).
    """

    period: float = 230e-9
    hole_fraction: float = 0.6
    bar_transmission: float = 0.2
    edge_width: float = 5e-9
    rotation: float = 0.3
    x0: float = 0.0
    y0: float = 0.0
    z: float = 0.0

    def _call_ray(self, ray):
        return ray                              # absorbs only; the ray renderer applies the weight

    def phase_shift(self, xy):
        """No optical path: the grating only absorbs."""
        return 0.0

    def log_transmission(self, xy):
        """Log of the amplitude transmission: 0 in a hole, ``log(sqrt(bar_transmission))`` on a bar."""
        c, s = jnp.cos(self.rotation), jnp.sin(self.rotation)
        x, y = xy[..., 0] - self.x0, xy[..., 1] - self.y0
        u, v = c * x + s * y, -s * x + c * y
        half = 0.5 * self.hole_fraction * self.period

        def inside(t):                          # 1 inside a hole's extent along one axis, smooth edges
            distance = jnp.abs(t - self.period * jnp.round(t / self.period))
            return jnn.sigmoid((half - distance) / self.edge_width)

        hole = inside(u) * inside(v)
        intensity = self.bar_transmission + (1 - self.bar_transmission) * hole
        return 0.5 * jnp.log(intensity)         # amplitude: the square root of the intensity


class PolycrystallineGold:
    """A gold film of small, randomly oriented grains, each diffracting into one reflection.

    Grains sit at fixed places on a jittered grid of ``grain_spacing`` over ``extent``; each is
    given one reflection (111, 200 or 220, by :data:`AU_WEIGHTS`) at a random azimuth. A lit grain
    sends a Gaussian beamlet of rms size ``grain_rms`` at the Bragg angle wavelength / d.

    Parameters
    ----------
    scattered_fraction : float
        Fraction of the incident electrons the film scatters into the rings.
    grain_spacing, grain_rms, extent : float
        Grain grid spacing, grain size (rms) and film size (m).
    seed : int
        Seed of the grain orientations.
    """

    def __init__(self, *, scattered_fraction=0.3, grain_spacing=5e-9, grain_rms=2.0e-9, extent=2e-6, seed=1):
        rng = np.random.default_rng(seed)
        n = int(extent / grain_spacing)
        grid = (np.arange(n) - (n - 1) / 2) * grain_spacing
        gx, gy = np.meshgrid(grid, grid)
        self.positions = np.stack([gx.ravel(), gy.ravel()], -1) + rng.uniform(-0.3, 0.3, (n * n, 2)) * grain_spacing
        names = list(AU_RINGS)
        weights = np.array([AU_WEIGHTS[k] for k in names])
        pick = rng.choice(len(names), n * n, p=weights / weights.sum())
        self.inverse_d = np.array([1 / AU_RINGS[k] for k in names])[pick]
        self.azimuth = rng.uniform(0, 2 * np.pi, n * n)
        self.scattered_fraction = float(scattered_fraction)
        self.grain_rms = float(grain_rms)

    def beamlets(self, mean, covariance, voltage_v, *, reach=3.5, max_grains=20000):
        """Gaussian beamlets of the lit grains at the specimen plane, and their weights.

        Parameters
        ----------
        mean, covariance : array_like
            Incident beam moments at the specimen plane, ``[x, y, theta_x, theta_y]`` (4 and 4x4).
        voltage_v : float
            High tension.
        reach : float
            Grains farther than this many rms from the beam centre are not lit.

        Returns
        -------
        beams : GaussianBeam
            One beamlet per lit grain (batched).
        weights : numpy.ndarray
            Fraction of the incident electrons each beamlet carries (they sum to ``scattered_fraction``).
        """
        from ..source import gaussian_waist

        mean, covariance = np.asarray(mean, float), np.asarray(covariance, float)
        cxx = covariance[:2, :2]
        d = self.positions - mean[:2]
        r2 = np.einsum("ni,ij,nj->n", d, np.linalg.inv(cxx), d)
        lit = np.flatnonzero(r2 < reach ** 2)
        if len(lit) > max_grains:
            lit = lit[np.argsort(r2[lit])[:max_grains]]
        weights = np.exp(-0.5 * r2[lit])
        weights = self.scattered_fraction * weights / weights.sum()
        # incident direction at each grain: the mean tilt plus the beam's position-angle correlation
        slope = mean[2:] + (d[lit] @ np.linalg.solve(cxx, covariance[:2, 2:]))
        bragg = energy2wavelength(voltage_v) * self.inverse_d[lit]
        slope = slope + bragg[:, None] * np.stack([np.cos(self.azimuth[lit]), np.sin(self.azimuth[lit])], -1)
        x, y = self.positions[lit, 0], self.positions[lit, 1]
        beams = jax.vmap(lambda x, y, tx, ty: gaussian_waist(z=0.0, voltage=voltage_v, rms_size=self.grain_rms,
                                                             x=x, y=y, theta_x=tx, theta_y=ty))(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(slope[:, 0]), jnp.asarray(slope[:, 1]))
        return beams, weights


def render_beamlets(beams, weights, components, *, shape_yx, pixel_pitch_m, psf_sigma_pixels):
    """Carry Gaussian beamlets through ``components`` to the camera and integrate their intensity
    over the pixels (independent spots, no interference). Returns electrons per incident electron."""
    from ..components import Detector
    from ..evaluate import gaussian_intensity_image
    from ..run import run_to_end

    out = jax.vmap(lambda b: run_to_end(b, components))(beams)
    k = 2 * np.pi / energy2wavelength(float(np.asarray(beams.voltage).ravel()[0]))
    variance = 1 / (2 * k * np.asarray(out.Q_inv)[:, 0, 0].imag)            # intensity variance (m^2)
    detector = Detector(z=float(np.asarray(out.z).ravel()[0]), shape=shape_yx, pixel_size=(pixel_pitch_m, pixel_pitch_m))
    iy, ix = detector.metres_to_pixels((out.x, out.y), cast=False)
    ny, nx = shape_yx
    centres = jnp.stack([jnp.asarray(ix) - nx / 2 + 0.5, jnp.asarray(iy) - ny / 2 + 0.5], -1)
    return np.asarray(gaussian_intensity_image(centres, jnp.asarray(variance / pixel_pitch_m ** 2), jnp.asarray(weights),
                                               shape_yx=shape_yx, psf_sigma_pixels=psf_sigma_pixels))
