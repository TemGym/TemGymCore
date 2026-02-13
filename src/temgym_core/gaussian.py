from __future__ import annotations

import warnings
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from interpax import Interpolator2D

from ._gaussian_core import (
    BaseGaussianPropagator,
    FreeSpacePropagator,
    GaussianBeam,
    Propagator,
    apply_action_delta,
    make_gaussian,
    scalar_grad_hess_complex,
    taylor_expand,
)
from .components import (
    ABCDTransfer,
    AtomicPotential,
    Component,
    ConstantAmplitudeShift,
    ConstantPhaseShift,
    DeflectionBiprism,
    Detector,
    DistortedLens,
    ElectromagneticLens,
    FourierTransform,
    InterpolatedFields3D,
    InterpolatedSample2D,
    KrivanekLens,
    Lens,
    LinearAmplitudeShift,
    LinearPhaseShift,
    MagneticPhaseSample,
    PhaseBiprism,
    QuadraticAmplitudeShift,
    QuadraticPhaseShift,
    RandomPhaseSample,
    SeidelLens,
    SigmoidAperture,
)
from .run import passthrough_transform
from .run import run_iter as _run_iter
from .utils import fibonacci_spiral, uniform_amp_from_area, uniform_disk


@jdc.pytree_dataclass(kw_only=True)
class Biprism(PhaseBiprism):
    def __post_init__(self):
        warnings.warn(
            "`gaussian.Biprism` now aliases `PhaseBiprism`. "
            "Use `temgym_core.components.PhaseBiprism` for canonical imports.",
            DeprecationWarning,
            stacklevel=2,
        )


TransformT = Callable[[Any], Callable[[Any], Tuple[Any, Any]]]


def run_iter(
    ray: GaussianBeam,
    components: Sequence[Any],
    transform: TransformT = passthrough_transform,
    propagator: BaseGaussianPropagator = FreeSpacePropagator(),
):
    return tuple(
        out_ray for _, out_ray in _run_iter(
            ray,
            components,
            transform=transform,
            propagator=propagator,
        )
    )


def run_to_end(
    ray: GaussianBeam,
    components: Sequence[Any],
    transform: TransformT = passthrough_transform,
    propagator: BaseGaussianPropagator = FreeSpacePropagator(),
) -> GaussianBeam:
    out = ray
    for _, out in _run_iter(
        ray,
        components,
        transform=transform,
        propagator=propagator,
    ):
        pass
    return out


run_iter_vmapped = jax.jit(jax.vmap(run_iter, in_axes=(0, None)), static_argnums=(3))
run_to_end_vmapped = jax.jit(jax.vmap(run_to_end, in_axes=(0, None)), static_argnums=(3))


def circular_input_wave(
    aperture_radius: float,
    waist: float,
    voltage: float,
    *,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    overlap_factor: float = 2.0,
    sampling: str = "fibonacci",
    offset_xy: Tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    d = waist / overlap_factor
    area = jnp.pi * aperture_radius**2
    num_rays = int(jnp.ceil(area / (d * d)))

    if sampling.lower() == "fibonacci":
        x0, y0 = fibonacci_spiral(num_rays, aperture_radius)
    else:
        x0, y0 = uniform_disk(num_rays, aperture_radius)

    x0 = x0 + offset_xy[0]
    y0 = y0 + offset_xy[1]

    amp_per_ray = uniform_amp_from_area(num_rays, waist, area)

    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=jnp.ones_like(x0) * amp_per_ray,
        phase=jnp.ones_like(x0) * phase,
        waist_x=jnp.ones_like(x0) * waist,
        waist_y=jnp.ones_like(y0) * waist,
        rcurv_x=jnp.ones_like(x0) * jnp.inf,
        rcurv_y=jnp.ones_like(y0) * jnp.inf,
        z=jnp.ones_like(x0) * z0,
        voltage=jnp.ones_like(x0) * voltage,
        wavelength_unit=wavelength_unit,
    )
    return beam


def square_input_wave(
    aperture_length: float,
    waist: float,
    voltage: float,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    overlap_factor: float = 2.0,
    centre_xy: Tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    d = waist / overlap_factor
    Nx = int(jnp.ceil(aperture_length / d))
    Ny = int(jnp.ceil(aperture_length / d))

    xs = (jnp.arange(Nx) - 0.5 * (Nx - 1)) * d
    ys = (jnp.arange(Ny) - 0.5 * (Ny - 1)) * d
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    x0 = X.ravel()
    y0 = Y.ravel()

    amp_norm = overlap_factor * 2 * jnp.pi

    x0 = x0 + centre_xy[0]
    y0 = y0 + centre_xy[1]

    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=jnp.ones_like(x0) * amp / amp_norm,
        phase=jnp.zeros_like(y0) + phase,
        waist_x=jnp.ones_like(x0) * waist,
        waist_y=jnp.ones_like(y0) * waist,
        rcurv_x=jnp.ones_like(x0) * jnp.inf,
        rcurv_y=jnp.ones_like(y0) * jnp.inf,
        z=jnp.ones_like(x0) * z0,
        voltage=jnp.ones_like(x0) * voltage,
        wavelength_unit=wavelength_unit,
    )
    return beam


def rectangular_input_wave(
    aperture_width: float,
    aperture_height: float,
    waist: float,
    voltage: float,
    amp: float = 1.0,
    phase: float = 0.0,
    z0: float = 0.0,
    overlap_factor: float = 2.0,
    centre_xy: Tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    area = aperture_width * aperture_height
    d = waist / overlap_factor

    Nx = int(jnp.ceil(aperture_width / d)) + 1
    Ny = int(jnp.ceil(aperture_height / d)) + 1
    num_rays = Nx * Ny

    xs = (jnp.arange(Nx) - 0.5 * (Nx - 1)) * d
    ys = (jnp.arange(Ny) - 0.5 * (Ny - 1)) * d
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    x0 = X.ravel()
    y0 = Y.ravel()

    amp_ray = uniform_amp_from_area(num_rays, waist, area)

    x0 = x0 + centre_xy[0]
    y0 = y0 + centre_xy[1]
    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=jnp.ones_like(x0) * amp_ray,
        phase=jnp.zeros_like(y0) + phase,
        waist_x=jnp.ones_like(x0) * waist,
        waist_y=jnp.ones_like(y0) * waist,
        rcurv_x=jnp.ones_like(x0) * jnp.inf,
        rcurv_y=jnp.ones_like(y0) * jnp.inf,
        z=jnp.ones_like(x0) * z0,
        voltage=jnp.ones_like(x0) * voltage,
        wavelength_unit=wavelength_unit,
    )
    return beam


def sample_input_wave(
    aperture_length: float,
    waist: float,
    voltage: float,
    amp_interpolator: Interpolator2D,
    phase_interpolator: Interpolator2D,
    z0: float = 0.0,
    overlap_factor: float = 2.0,
    centre_xy: Tuple[float, float] = (0.0, 0.0),
    wavelength_unit: str = "m",
) -> GaussianBeam:
    d = waist / overlap_factor
    Nx = int(jnp.ceil(aperture_length / d))
    Ny = int(jnp.ceil(aperture_length / d))

    xs = (jnp.arange(Nx) - 0.5 * (Nx - 1)) * d
    ys = (jnp.arange(Ny) - 0.5 * (Ny - 1)) * d
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    x0 = X.ravel()
    y0 = Y.ravel()

    amp_norm = overlap_factor * 2 * jnp.pi

    x0 = x0 + centre_xy[0]
    y0 = y0 + centre_xy[1]

    amplitude = amp_interpolator(x0, y0)
    phase = phase_interpolator(x0, y0)

    beam = make_gaussian(
        x=x0,
        y=y0,
        dx=jnp.zeros_like(x0),
        dy=jnp.zeros_like(y0),
        amp=amplitude / amp_norm,
        phase=phase,
        waist_x=jnp.ones_like(x0) * waist,
        waist_y=jnp.ones_like(y0) * waist,
        rcurv_x=jnp.ones_like(x0) * jnp.inf,
        rcurv_y=jnp.ones_like(y0) * jnp.inf,
        z=jnp.ones_like(x0) * z0,
        voltage=jnp.ones_like(x0) * voltage,
        wavelength_unit=wavelength_unit,
    )
    return beam


__all__ = [
    "GaussianBeam",
    "make_gaussian",
    "scalar_grad_hess_complex",
    "taylor_expand",
    "apply_action_delta",
    "Propagator",
    "BaseGaussianPropagator",
    "FreeSpacePropagator",
    "Component",
    "Lens",
    "KrivanekLens",
    "SeidelLens",
    "DistortedLens",
    "ElectromagneticLens",
    "ABCDTransfer",
    "SigmoidAperture",
    "Biprism",
    "PhaseBiprism",
    "DeflectionBiprism",
    "ConstantPhaseShift",
    "LinearPhaseShift",
    "QuadraticPhaseShift",
    "ConstantAmplitudeShift",
    "LinearAmplitudeShift",
    "QuadraticAmplitudeShift",
    "MagneticPhaseSample",
    "RandomPhaseSample",
    "InterpolatedSample2D",
    "InterpolatedFields3D",
    "AtomicPotential",
    "FourierTransform",
    "Detector",
    "run_iter",
    "run_to_end",
    "run_iter_vmapped",
    "run_to_end_vmapped",
    "circular_input_wave",
    "square_input_wave",
    "rectangular_input_wave",
    "sample_input_wave",
]
