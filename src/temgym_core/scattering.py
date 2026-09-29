"""Small, explicit approximations for coherent surface scattering.

These are not dynamical RHEED or total electron-backscatter models.
"""
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from .constants import ELECTRON_MASS, ELEMENTARY_CHARGE, SPEED_OF_LIGHT

_REST_ENERGY_EV = ELECTRON_MASS * SPEED_OF_LIGHT**2 / ELEMENTARY_CHARGE


@jdc.pytree_dataclass(kw_only=True)
class PotentialStepResponse:
    """Scalar electron reflection amplitude at an abrupt attractive potential step.

    ``inner_potential_v`` is positive for an attractive electron potential;
    the potential energy inside is -e*(V + i W). Nonnegative ``absorption_ev``
    selects the decaying transmitted branch. Zero absorption is the default:
    W is an adjustable optical-potential parameter, not inferred material data.

    The scalar dispersion includes the relativistic energy correction. This
    approximates the smooth coherent specular envelope, not Bragg resonances,
    diffuse/inelastic backscatter, or the yield of all reflected electrons.
    Redistributing its flux over crystal orders is a separate approximation.
    """
    inner_potential_v: float
    absorption_ev: float = 0.0

    def __post_init__(self):
        for name in ('inner_potential_v', 'absorption_ev'):
            try:
                value = np.asarray(getattr(self, name))
            except jax.errors.TracerArrayConversionError:
                continue
            if value.ndim or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a finite nonnegative scalar.')

    def amplitude(self, grazing_angle_rad, voltage):
        """Return complex field amplitude; angles are measured from the surface.

        Supports scalar/array inputs and JAX differentiation. Voltage must be
        positive. Angles outside [0, pi/2] represent no incident illumination.
        """
        angle, energy = jnp.asarray(grazing_angle_rad), jnp.asarray(voltage)
        valid = (angle >= 0) & (angle <= jnp.pi/2) & (energy > 0)
        energy = jnp.maximum(energy, jnp.finfo(float).tiny)
        a = jnp.clip(angle, 0., jnp.pi/2)
        vacuum = jnp.sqrt(energy*(1+energy/(2*_REST_ENERGY_EV)))*jnp.sin(a)
        potential = self.inner_potential_v + 1j*self.absorption_ev
        shift = potential*(1+energy/_REST_ENERGY_EV) + potential**2/(2*_REST_ENERGY_EV)
        material = jnp.sqrt(vacuum**2 + shift + 0j)
        denominator = vacuum + material
        safe = jnp.where(jnp.abs(denominator)>0, denominator, 1.+0j)
        # Rationalized difference avoids cancellation for small potentials.
        reflection = -shift/safe**2
        return jnp.where(valid, reflection, 0.+0j)

    def reflectivity(self, grazing_angle_rad, voltage):
        """Fractional smooth-step coherent reflected intensity, |r|²."""
        return jnp.abs(self.amplitude(grazing_angle_rad, voltage))**2

    def complex_action(self, grazing_angle_rad, voltage, k):
        """Action increment ΔS=-i log(r)/k, so exp(i k ΔS)=r.

        k uses the inverse beam length unit. Exact zero reflection has an
        infinite attenuating action; multiply amplitudes directly at zeros.
        This is an angular-spectrum multiplier, not a spatial absorption map.
        """
        return -1j*jnp.log(self.amplitude(grazing_angle_rad, voltage))/k


def copper_scattering_factor(spatial_frequency_inv_angstrom):
    """Neutral Cu elastic electron factor in Å, with frequency |q|/(2π).

    Lobato & Van Dyck (2014), Acta Cryst. A70, 636–649.
    Numerical Cu parameters: abTEM parametrizations/data/lobato.json
    https://github.com/abTEM/abTEM/blob/main/abtem/parametrizations/data/lobato.json
    Input is cycles/Å, NOT q in radians/Å or sin(theta)/lambda.
    The relativistic multiplicative correction cancels in relative weights.
    """
    a = jnp.array([1.75207145212145, -43.0410523492124, 44.0705915543573,
                   .0186876154088144, .00020172732479162])
    b = jnp.array([6.18750497986187, 1.00266263628976, .98538431135303,
                   .0302984703916117, .00255855598748879])
    x = jnp.asarray(spatial_frequency_inv_angstrom)[...,None]**2*b
    return jnp.sum(a*(2+x)/(1+x)**2, axis=-1)
