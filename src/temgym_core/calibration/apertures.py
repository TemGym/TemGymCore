"""Native aperture transmission, diffraction and per-hole centres."""

import jax
import jax.numpy as jnp
import numpy as np

from .condenser_model import condenser_at_voltage, equivalent_parameters, source_moments


def make_aperture_moments(
    *,
    radius_m,
    edge_width_m,
    voltage_v,
    radial_order=192,
    angular_order=64,
    diffraction=True,
):
    """Predict specimen moments and flux with the native aperture inserted.

    Integrate the aperture transmission and its amplitude gradient at its own
    plane, then transport the resulting moments through the remaining lenses.
    This includes diffraction and requires no Gaussian shape after the aperture.
    Set diffraction=False only to match geometric ray images: the aperture then
    weights incident rays without adding an angular diffraction contribution.
    """
    from temgym_core.constants import energy2wavelength
    from temgym_core.moments import SampledTransmission
    from temgym_core.run import column_matrix

    from .condenser_model import condenser_with_aperture

    nodes, weights = np.polynomial.legendre.leggauss(radial_order)
    limit = radius_m + 16 * edge_width_m
    radius = (nodes + 1) * limit / 2
    angle = np.arange(angular_order) * (2 * np.pi / angular_order)
    xy = jnp.asarray(
        np.stack(
            (radius[:, None] * np.cos(angle), radius[:, None] * np.sin(angle)), axis=-1
        ).reshape(-1, 2)
    )
    areas = jnp.asarray(
        np.repeat(
            weights * limit / 2 * radius * (2 * np.pi / angular_order), angular_order
        )
    )
    wavelength = energy2wavelength(voltage_v)

    def predict(parameters, controls, centre_m):
        column = condenser_with_aperture(
            parameters,
            controls,
            radius_m=radius_m,
            edge_width_m=edge_width_m,
            centre_m=centre_m,
        )
        aperture = column[2]
        upstream = column_matrix(
            column[:3], source_z=-jnp.sum(jnp.exp(parameters[:9])), voltage=voltage_v
        )[:4, :4]
        mean, covariance = source_moments(parameters, voltage_v)
        mean, covariance = upstream @ mean, upstream @ covariance @ upstream.T
        # Follow a narrow crossover instead of missing it between aperture-grid
        # points. Broad beams use the aperture support; both integrals converge
        # to the same physical moments as the quadrature is refined.
        sigma = jnp.sqrt(covariance[0, 0])
        narrow = sigma < jnp.minimum(radius_m / 4, 2 * edge_width_m)
        scale = jnp.where(narrow, 10 * sigma / limit, 1.0)
        points = xy * scale + jnp.where(narrow, mean[:2], centre_m)
        integration_areas = areas * scale**2
        sample = SampledTransmission.from_component(
            aperture, points, quadrature_weights=integration_areas
        )
        exit_mean, exit_covariance = sample.gaussian_moments(
            mean, covariance, wavelength if diffraction else 0.0
        )
        delta = points - mean[:2]
        spatial = covariance[:2, :2]
        log_density = -0.5 * jnp.einsum(
            "ni,ij,nj->n", delta, jnp.linalg.inv(spatial), delta
        )
        flux = jnp.sum(jnp.exp(log_density) * sample.intensity * integration_areas) / (
            2 * jnp.pi * jnp.sqrt(jnp.linalg.det(spatial))
        )
        downstream = column_matrix(column[3:], source_z=aperture.z, voltage=voltage_v)[
            :4, :4
        ]
        return downstream @ exit_mean, downstream @ exit_covariance @ downstream.T, flux

    return jax.jit(predict)


def aperture_centres(parameters, number_of_apertures):
    """Micrometres; first centre is p[20:22], additional centres follow source two."""
    p = jnp.asarray(parameters)
    if number_of_apertures == 1:
        return p[20:22][None, :]
    return jnp.concatenate(
        [p[20:22][None, :], p[29:].reshape(number_of_apertures - 1, 2)]
    )


def effective_parameters(parameters, voltage, reference, source_index):
    p = jnp.asarray(parameters)
    effective = condenser_at_voltage(
        p,
        voltage,
        reference_voltage_v=reference,
        source_parameters=p[22:29] if source_index else None,
    )
    return p.at[:22].set(effective)


def equivalent_aperture_model(parameters, *, objective_strength):
    """Exact surviving ambiguity: translate the upstream assembly and rotate it.

    The aperture keeps its fractional location beside CL3. Its unknown transverse
    centre rotates with the unknown source. Every circular-aperture transmission
    and every specimen beam remain unchanged.
    """
    p = np.asarray(parameters)
    other = np.r_[
        equivalent_parameters(p[:19], objective_strength=objective_strength), p[19:]
    ]
    angle = objective_strength + np.exp(p[12])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    other[20:22] = rotation @ p[20:22]
    return other
