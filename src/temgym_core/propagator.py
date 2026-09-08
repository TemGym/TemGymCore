"""Free-space propagation of rays and sampled complex fields."""

from typing import NamedTuple, TYPE_CHECKING
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .ray import Ray


class BasePropagator:
    """Abstract base for ray propagators.

    Implement `__call__(ray, distance)` in subclasses to return a new ray.
    """
    def __call__(self, ray: "Ray", distance: float) -> "Ray":
        raise NotImplementedError

    def with_distance(self, distance: float) -> "Propagator":
        return Propagator(
            distance, self
        )


class Propagator(NamedTuple):
    """Callable pair of (distance, propagator) that acts on a ray.

    Parameters
    ----------
    distance : float
        Distance to propagate, metres.
    propagator : BasePropagator
        The underlying propagation model to use.

    Notes
    -----
    Calling `Propagator(ray)` returns the propagated ray.
    """
    distance: float
    propagator: BasePropagator

    def __call__(self, ray: "Ray") -> "Ray":
        return self.propagator(ray, self.distance)


class FreeSpaceParaxial(BasePropagator):
    """Paraxial free-space propagation with constant slopes.

    Notes
    -----
    Updates: x += dx*d, y += dy*d, z += d, pathlength += d. Pure and
    differentiable.
    """
    @staticmethod
    def propagate(ray: "Ray", distance: float):
        """Propagate a ray by `distance` assuming small angles.

        Parameters
        ----------
        ray : Ray
            Input ray.
        distance : float
            Propagation distance, metres.

        Returns
        -------
        ray : Ray
            Propagated ray.
        """
        return ray.derive(
            x=ray.x + ray.dx * distance,
            y=ray.y + ray.dy * distance,
            z=ray.z + distance,
            pathlength=ray.pathlength + distance,
        )

    @classmethod
    def __call__(cls, ray: "Ray", distance: float):
        return cls.propagate(ray, distance)


class FreeSpaceDirCosine(BasePropagator):
    """Free-space propagation using direction cosines (L, M, N).

    Notes
    -----
    Handles larger angles than paraxial model. Updates: x += (L/N)*d,
    y += (M/N)*d, z += d, pathlength += N*d. Not currently integrated with
    ABCD-based matrices in this repo.
    """
    @staticmethod
    def propagate(ray: "Ray", distance: float):
        """Propagate using direction cosines.

        Parameters
        ----------
        ray : Ray
            Input ray.
        distance : float
            Propagation distance, metres.

        Returns
        -------
        ray : Ray
            Propagated ray with direction-cosine model.
        """
        N = np.sqrt(1 + ray.dx**2 + ray.dy**2)
        L = ray.dx / N
        M = ray.dy / N
        return ray.derive(
            x=ray.x + L / N * distance,
            y=ray.y + M / N * distance,
            z=ray.z * ray._one + distance,
            pathlength=ray.pathlength + distance * N,
        )

    @classmethod
    def __call__(cls, ray: "Ray", distance: float):
        return cls.propagate(ray, distance)


def _field_frequencies(shape_yx, pixel_size_yx, xp):
    ny, nx = shape_yx
    dy, dx = pixel_size_yx
    return xp.meshgrid(xp.fft.fftfreq(nx, d=dx), xp.fft.fftfreq(ny, d=dy))


def _fresnel_transfer(shape_yx, pixel_size_yx, wavelength, z, xp):
    """Paraxial transfer function; uniform axial phase is applied by the caller."""
    fx, fy = _field_frequencies(shape_yx, pixel_size_yx, xp)
    return xp.exp(-1j * xp.pi * wavelength * z * (fx**2 + fy**2))


def FresnelPropagator(u1, L, wavelength, z):
    """Propagate a 2D field with JAX; pixel spacing in both axes is L/height.

    L is the full sampled height, including one pixel interval per row.
    Includes the uniform axial phase exp(ikz); z may be negative or zero.
    """
    spacing = L / u1.shape[0]
    transfer = _fresnel_transfer(u1.shape, (spacing, spacing), wavelength, z, jnp)
    field = jnp.fft.ifft2(transfer * jnp.fft.fft2(u1))
    return field * jnp.exp(1j * 2 * jnp.pi * z / wavelength)


def fresnel_fft_2d(X, Y, U, wavelength, z):
    """NumPy adapter for a field on uniform meshgrid X/Y coordinates.

    Uses the same Fresnel kernel, with independent x/y pixel spacing.
    Retains the axial-phase evaluation used by the NumPy reference images.
    """
    dx, dy = X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0]
    transfer = _fresnel_transfer(U.shape, (dy, dx), wavelength, z, np)
    k = 2 * np.pi / wavelength
    transfer = np.exp(1j * k * z) * transfer
    return np.fft.ifft2(np.fft.fft2(U) * transfer)


def AngularSpectrumPropagator(u1, L, wavelength, z, xp=np):
    """Full angular-spectrum propagation; spacing is L/height in both axes.

    The complex square root retains evanescent-wave decay for positive z.
    Supply xp=jax.numpy for JAX evaluation and derivatives.
    """
    spacing = L / u1.shape[0]
    fx, fy = _field_frequencies(u1.shape, (spacing, spacing), xp)
    k = 2 * xp.pi / wavelength
    kz = xp.sqrt(k**2 - (2 * xp.pi)**2 * (fx**2 + fy**2) + 0j)
    return xp.fft.ifft2(xp.exp(1j * kz * z) * xp.fft.fft2(u1))
