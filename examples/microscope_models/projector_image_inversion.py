"""Small, reproducible Gaussian-image inversion of fixed projector controls.

The specimen is an analytic asymmetric multiscale exit wave. Geometry,
objective excitation, specimen coordinates, dose, and detector calibration
are known in this conditional synthetic experiment. Four unknown excitation
gains act on IL1, IL2, IL3, and PL1, changing both focusing and rotation.
"""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from temgym_core.components import Plane
from temgym_core.evaluate import evaluate_gaussians, gpu_kernel_available
from temgym_core.gaussian import make_gaussian
from temgym_core.ray import Ray
from temgym_core.run import run_to_end


GAIN_NAMES = ("IL1", "IL2", "IL3", "PL1")


def calibration_exit_wave(z, voltage):
    """A fixed analytic specimen with features visible across 40k–600k.

    Widths are field 1/e radii. Normalize using the exact pairwise Gaussian
    overlap integral, so the integrated exit-wave intensity is one before
    detector clipping. This is a synthetic target, not a MAGICAL certificate.
    """
    centres = np.array([
        [-85, -55], [65, -65], [-60, 75], [80, 50], [5, 105],
        [-25, -12], [20, -20], [-18, 25], [28, 15],
        [-6, -4], [5, -5], [-4, 6], [6, 4], [0, 0],
    ]) * 1e-9
    widths = np.array([9, 11, 8, 10, 7, 3, 2.5, 3.5, 2.8, .9, 1.1, .8, 1., .7]) * 1e-9
    weights = np.array([1., .85, .7, 1.15, .6, .8, 1.1, .65, .9, 1., .75, 1.2, .8, .55])
    phases = np.array([0., .2, -.3, .1, .4, -.2, .1, .3, -.1, .4, -.2, .3, .1, -.3])
    amplitude = weights / widths * np.exp(1j * phases)
    distance2 = np.sum((centres[:, None] - centres[None, :])**2, axis=-1)
    overlap = (np.pi / (widths[:, None]**-2 + widths[None, :]**-2)
               * np.exp(-distance2 / (widths[:, None]**2 + widths[None, :]**2)))
    norm = np.real(np.sum(amplitude[:, None] * amplitude[None, :].conj() * overlap))
    beam = make_gaussian(
        x=centres[:, 0], y=centres[:, 1], z=z, voltage=voltage,
        waist_x=widths, waist_y=widths,
        amp=np.abs(amplitude) / np.sqrt(norm), phase=phases,
    )
    return beam, {"centres_m": centres, "field_radii_m": widths, "phases_rad": phases}


def make_experiment(model, magnifications=(40_000, 100_000, 300_000), step=.015):
    """Record nominal and individual lens sweeps, then independent mixed states.

    The focus solver is used once to choose reference controls. All returned
    currents are frozen before any unknown gain is applied or fitted.
    """
    start = model.auxiliary["sample"]["z"]
    end = model.auxiliary["detector"]["z"]

    def baseline(mag):
        return np.array([float(c.current) for c in model.build_components("magnification", mag)
                         if start < float(c.z) < end])

    controls, labels = [], []
    for mag in magnifications:
        base = baseline(mag)
        controls.append(base)
        labels.append(f"{mag // 1000}k reference")
        for index, name in enumerate(GAIN_NAMES, start=1):
            for sign in (-1, 1):
                perturbed = base.copy()
                perturbed[index] *= 1 + sign * step
                controls.append(perturbed)
                labels.append(f"{mag // 1000}k {name} {sign * step:+.1%}")
    held_controls, held_labels = [], []
    for mag in (80_000, 200_000, 600_000):
        for sign in (-1, 1):
            row = baseline(mag)
            row[1:] *= 1 + sign * np.array([.006, -.008, .01, -.005])
            held_controls.append(row)
            held_labels.append(f"{mag // 1000}k mixed {sign:+}")
    return np.array(controls), labels, np.array(held_controls), held_labels


class ProjectorImageForward:
    """Propagate the coherent exit wave through the executable microscope.

    GPU Pallas evaluates every forward image. Uniform subpixel quadrature
    integrates intensity before Poisson/read noise. The 128² detector uses
    110 µm pitch (2× binning of the configured detector), preserving its FOV.
    """

    def __init__(self, model, *, shape=(128, 128), subpixels=3, dose=2e6, background=2.):
        if not gpu_kernel_available():
            raise RuntimeError("This notebook requires a working Gaussian Pallas GPU kernel.")
        self.model = model
        self.shape = tuple(shape)
        self.subpixels = int(subpixels)
        self.dose = float(dose)
        self.background = float(background)
        self.start = model.auxiliary["sample"]["z"]
        self.end = model.auxiliary["detector"]["z"]
        det = model.auxiliary["detector"]
        self.pitch = float(det["pixel_size"]) * det["shape"][1] / self.shape[1]
        if self.shape[0] != self.shape[1] or self.subpixels < 1:
            raise ValueError("The pilot uses a square detector and positive subpixel count.")
        self.beam, self.specimen = calibration_exit_wave(self.start, model.voltage)
        self.lenses = tuple(c for c in model.build_components("magnification", 40_000)
                            if self.start < float(c.z) < self.end)
        fine_shape = tuple(n * self.subpixels for n in self.shape)
        x = (np.arange(fine_shape[1]) - (fine_shape[1] - 1) / 2) * self.pitch / self.subpixels
        y = (np.arange(fine_shape[0]) - (fine_shape[0] - 1) / 2) * self.pitch / self.subpixels
        xx, yy = np.meshgrid(x, y)
        self.grid = SimpleNamespace(shape=fine_shape, coords=jnp.asarray(np.stack((xx, yy), -1).reshape(-1, 2)))

        def propagate(log_gains, controls):
            excitation = controls * jnp.r_[1., jnp.exp(log_gains)]
            column = [replace(lens, current=excitation[i]) for i, lens in enumerate(self.lenses)]
            return run_to_end(self.beam, column + [Plane(z=self.end)])

        self.propagate = jax.jit(propagate)

        def render(log_gains, controls):
            out = propagate(log_gains, controls)
            # Removing one common phase leaves intensity unchanged and avoids
            # evaluating the enormous common free-space propagation phase.
            out = out.derive(pathlength=out.pathlength - out.pathlength[0])
            field = evaluate_gaussians(out, self.grid, method="gpu_kernel_required")
            fine_intensity = jnp.abs(field)**2
            pixel_intensity = fine_intensity.reshape(
                self.shape[0], self.subpixels, self.shape[1], self.subpixels,
            ).mean(axis=(1, 3))
            return self.dose * self.pitch**2 * pixel_intensity + self.background

        self.render = jax.jit(render)

    def images(self, log_gains, controls):
        return np.stack([np.asarray(self.render(jnp.asarray(log_gains), jnp.asarray(row))) for row in controls])

    def ray_matrix(self, log_gains, controls):
        excitation = np.asarray(controls) * np.r_[1., np.exp(log_gains)]
        column = [replace(lens, current=excitation[i]) for i, lens in enumerate(self.lenses)]

        def trace(state):
            out = run_to_end(Ray(x=state[0], y=state[1], dx=state[2], dy=state[3],
                                z=self.start, pathlength=0., voltage=self.model.voltage),
                             column + [Plane(z=self.end)])
            return jnp.array((out.x, out.y, out.dx, out.dy))

        return np.asarray(jax.jacfwd(trace)(jnp.zeros(4)))


def noisy_observations(expected, rng, read_noise=2.):
    """Raw pixels and fixed weights from an independent synthetic dose reference."""
    observed = rng.poisson(expected).astype(float) + rng.normal(0., read_noise, expected.shape)
    dose_reference = rng.poisson(20 * expected) / 20
    variance = np.maximum(dose_reference, 1.) + read_noise**2
    return observed, variance


def fit_images(forward, controls, observed, variance, initial, *, maximum_evaluations=80):
    """Fit four excitation gains to pixels, without access to generating gains."""
    sigma = np.sqrt(variance)

    def residual(parameters):
        return ((forward.images(parameters, controls) - observed) / sigma).ravel()

    # Absolute central differences are stable near zero log gain. A 2e-5 step
    # also avoids resolving the last bits of accumulated optical path length.
    def jacobian(parameters):
        step = 2e-5
        eye = np.eye(4) * step
        return np.column_stack([(residual(parameters + e) - residual(parameters - e)) / (2 * step) for e in eye])

    return least_squares(residual, np.asarray(initial), jac=jacobian,
                         bounds=(-.04, .04), x_scale="jac", max_nfev=maximum_evaluations,
                         ftol=1e-9, xtol=1e-9, gtol=1e-5)
