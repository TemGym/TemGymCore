"""Physical condenser and source parameters in TemGym ray coordinates."""

import jax
import jax.numpy as jnp
from scipy.constants import c, electron_mass, elementary_charge

from temgym_core.components import solenoid_column
from temgym_core.run import column_matrix


def condenser_components(parameters, controls):
    """Native lenses from the source waist to the specimen at z=0."""
    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    strengths = jnp.exp(p[9:13]) * jnp.r_[jnp.asarray(controls), -1.0]
    source_z = -jnp.sum(gaps) - jnp.sum(lengths)
    return solenoid_column(gaps, lengths, strengths, source_z=source_z)


def condenser_with_aperture(parameters, controls, *, radius_m, edge_width_m, centre_m):
    """Native source-to-specimen column with the aperture just after CL3."""
    from temgym_core.components import SigmoidAperture

    column = condenser_components(parameters, controls)
    z = (
        column[1].z
        + column[1].length
        + jax.nn.sigmoid(parameters[19]) * jnp.exp(parameters[2])
    )
    aperture = SigmoidAperture(
        z=z, radius=radius_m, edge_width=edge_width_m, x0=centre_m[0], y0=centre_m[1]
    )
    return (*column[:2], aperture, *column[2:])


def condenser_with_steering(
    column, specimen_kick, *, voltage_v, gap_fractions=(0.3, 0.7)
):
    """Align a native DoubleDeflector and set its calibrated specimen response.

    Illustrative positions lie in the free gap AFTER the aperture and before
    the mini lens. They are not measured or fitted hardware locations. Changing
    these positions and recalibrating the kicks preserves the specimen response.
    This equivalence requires unchanged aperture illumination; upstream steering
    that changes clipping needs a different acquisition/forward model.
    """
    from temgym_core.components import DoubleDeflector, SigmoidAperture
    from temgym_core.ray import Ray

    if not 0 < gap_fractions[0] < gap_fractions[1] < 1:
        raise ValueError("Place two ordered deflectors strictly inside the free gap.")
    index = next(i for i, c in enumerate(column) if isinstance(c, SigmoidAperture))
    aperture, downstream = column[index], column[index + 1 :]
    z1, z2 = aperture.z + jnp.asarray(gap_fractions) * (downstream[0].z - aperture.z)
    transfer = column_matrix(downstream, source_z=z2, voltage=voltage_v)[:4, :4]
    reference = Ray.origin().derive(z=z1, voltage=voltage_v)

    def steering(commands, balances):
        return DoubleDeflector(
            z=z1,
            spacing=z2 - z1,
            shift_x=commands[0],
            shift_y=commands[1],
            tilt_x=commands[2],
            tilt_y=commands[3],
            shift_balance_x=balances[0],
            shift_balance_y=balances[1],
            tilt_balance_x=balances[2],
            tilt_balance_y=balances[3],
        )

    def response(balances):
        def trace(commands):
            ray = steering(commands, balances)(reference)
            return transfer @ jnp.stack((ray.x, ray.y, ray.dx, ray.dy))

        return jax.jacfwd(trace)(jnp.zeros(4))

    def alignment_error(balances):
        matrix = response(balances)
        # Shift must not tilt; tilt must not move the beam at the specimen.
        # Divide position by spacing to put both errors in angular units.
        return 0.5 * (
            jnp.sum(matrix[2:, :2] ** 2) + jnp.sum((matrix[:2, 2:] / (z2 - z1)) ** 2)
        )

    # Newton–Raphson on the alignment gradient. The error is quadratic in
    # these four ratios, so one Newton step aligns this linear column.
    gradient = jax.grad(alignment_error)
    initial = jnp.ones(4)
    balances = initial - jnp.linalg.solve(
        jax.jacfwd(gradient)(initial), gradient(initial)
    )
    commands = jnp.linalg.solve(response(balances), jnp.asarray(specimen_kick))
    return (
        *column[: index + 1],
        steering(commands, balances),
        *downstream,
    )


def matrix5(parameters, controls):
    """Differentiate ray propagation through the same native condenser lenses."""
    column = condenser_components(parameters, controls)
    source_z = -jnp.sum(jnp.exp(jnp.asarray(parameters)[:9]))
    return column_matrix(column, source_z=source_z)


def source_moments(parameters, voltage_v=200000.0):
    """Gaussian waist plus incoherent tilt spread; the diffraction floor is exact."""
    from temgym_core.constants import energy2wavelength

    p = jnp.asarray(parameters)
    mean = jnp.r_[p[13:15] * 1e-8, p[15:17] * 1e-4]
    angular_variance = (
        jnp.exp(2 * p[18])
        + (energy2wavelength(voltage_v) / (4 * jnp.pi * jnp.exp(p[17]))) ** 2
    )
    covariance = jnp.diag(
        jnp.r_[jnp.repeat(jnp.exp(2 * p[17]), 2), jnp.repeat(angular_variance, 2)]
    )
    return mean, covariance


def magnetic_strength_ratio(voltage_v, reference_voltage_v=200000.0):
    """p(reference)/p(voltage); fixed magnetic fields have q proportional to 1/p.

    Relativistic kinetic energy gives (pc)^2 = eV(eV + 2mc^2).
    Voltages are positive kinetic energies per electron charge, in volts.
    """
    rest_voltage = electron_mass * c**2 / elementary_charge
    voltage = jnp.asarray(voltage_v)
    reference = jnp.asarray(reference_voltage_v)
    return jnp.sqrt(
        reference
        * (reference + 2 * rest_voltage)
        / (voltage * (voltage + 2 * rest_voltage))
    )


def condenser_at_voltage(
    parameters, voltage_v, *, reference_voltage_v=200000.0, source_parameters=None
):
    """Return 22 condenser/aperture parameters at the recorded voltage.

    Optional source_parameters = [log(source-to-CL1 gap), source x, y,
    theta_x, theta_y, log(size), log(divergence)], in the model's native units.
    Each voltage may have an independent source waist and all six beam values.
    Lens geometry and aperture coordinates remain shared.
    """
    result = jnp.asarray(parameters[:22])
    if source_parameters is not None:
        source = jnp.asarray(source_parameters)
        result = result.at[0].set(source[0]).at[13:19].set(source[1:])
    ratio = magnetic_strength_ratio(voltage_v, reference_voltage_v)
    return result.at[9:13].add(jnp.log(ratio))


