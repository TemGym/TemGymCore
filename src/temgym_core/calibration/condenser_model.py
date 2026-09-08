"""Physical condenser and source parameters in TemGym ray coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import c, electron_mass, elementary_charge

from temgym_core.solenoid import solenoid_column_matrix

from .projector_model import matrix5 as projector_matrix5

QUANTITIES = (
    "source_z_mm",
    "CL1_z_mm",
    "CL3_z_mm",
    "mini_z_mm",
    "prefield_z_mm",
    "CL1_length_mm",
    "CL3_length_mm",
    "mini_length_mm",
    "prefield_length_mm",
    "aperture_z_mm",
    "aperture_x_um",
    "aperture_y_um",
)


def physical_geometry(parameters):
    """Source and lens positions, field lengths and aperture geometry in mm/µm."""
    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    source = -jnp.sum(gaps) - jnp.sum(lengths)
    centres = (
        source
        + jnp.cumsum(gaps[:-1])
        + jnp.r_[0.0, jnp.cumsum(lengths[:-1])]
        + lengths / 2
    )
    aperture = centres[1] + lengths[1] / 2 + jax.nn.sigmoid(p[19]) * gaps[2]
    return jnp.r_[
        source * 1000, centres * 1000, lengths * 1000, aperture * 1000, p[20:22]
    ]


def beam_to_moments(beam):
    size2 = jnp.exp(2 * beam[4])
    h = beam[5] * 1e4
    spread2 = jnp.exp(2 * beam[6])
    return jnp.r_[
        beam[:4], size2 / 1e-14, h * size2 / 1e-10, (h * h * size2 + spread2) / 1e-6
    ]


def model_moments(parameters, controls):
    mean, cov = specimen_moments(parameters, controls)
    return jnp.r_[
        mean[:2] / 1e-7,
        mean[2:] / 1e-3,
        cov[0, 0] / 1e-14,
        cov[0, 2] / 1e-10,
        cov[2, 2] / 1e-6,
    ]


def physical_quantity(parameters, quantity):
    """A shared geometry quantity or the second voltage's independent source waist."""
    p = jnp.asarray(parameters)
    if quantity == "source_z_second_voltage_mm":
        return physical_geometry(p)[0] + 1000 * (jnp.exp(p[0]) - jnp.exp(p[22]))
    return physical_geometry(p)[QUANTITIES.index(quantity)]


def initial_parameters():
    return np.r_[
        np.log([0.195, 0.04, 0.05, 0.025, 0.0025]),
        np.log([0.005, 0.005, 0.003, 0.001]),
        np.log([0.18, 0.35, 0.38, 0.45]),
        [0.1, -0.1],
        [0.01, -0.01],
        np.log(10e-9),
        np.log(60e-6),
    ]


def generating_parameters():
    return np.r_[
        np.log([0.1975, 0.035, 0.056, 0.0205, 0.002]),
        np.log([0.005, 0.005, 0.003, 0.001]),
        np.log([0.20, 0.37, 0.40, 0.48]),
        [0.2, -0.3],
        [0.02, -0.01],
        np.log(8e-9),
        np.log(80e-6),
    ]


def matrix5(parameters, controls):
    """Map an unknown source waist to the specimen through four finite lenses."""
    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    strengths = jnp.exp(p[9:13]) * jnp.r_[jnp.asarray(controls), -1.0]
    return solenoid_column_matrix(gaps, lengths, strengths)


def source_moments(parameters):
    p = jnp.asarray(parameters)
    mean = jnp.r_[p[13:15] * 1e-8, p[15:17] * 1e-4]
    covariance = jnp.diag(
        jnp.r_[jnp.repeat(jnp.exp(2 * p[17]), 2), jnp.repeat(jnp.exp(2 * p[18]), 2)]
    )
    return mean, covariance


def specimen_moments(parameters, controls):
    transfer = matrix5(parameters, controls)[:4, :4]
    mean, covariance = source_moments(parameters)
    return transfer @ mean, transfer @ covariance @ transfer.T


def equivalent_parameters(parameters, *, upstream_scale=1.0, objective_strength=None):
    """An exact two-parameter family with identical specimen means/covariances.

    Scale upstream axial distances, rescale the source's size and divergence
    inversely, and adjust the static final lens plus its two surrounding gaps.
    A change of its constant Larmor rotation is absorbed into the unknown
    source mean orientation. This is an identifiability diagnostic, not data.
    """
    p = np.asarray(parameters, dtype=float)
    scale = float(upstream_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Upstream scale must be positive.")
    old_q = -np.exp(p[12])
    new_q = old_q if objective_strength is None else float(objective_strength)
    if not np.isfinite(new_q) or not -1.2 < new_q < 0:
        raise ValueError("Use the declared negative objective-strength branch.")
    gaps, lengths = np.exp(p[:5]), np.exp(p[5:9])
    c = -old_q * np.sin(old_q) / lengths[3]
    a = np.cos(old_q) + gaps[4] * c
    d = np.cos(old_q) + gaps[3] * c
    root = np.sqrt(scale)
    new_c = c / root
    new_length = -new_q * np.sin(new_q) / new_c
    new_gap_after = (a / root - np.cos(new_q)) / new_c
    new_gap_before = (d * root - np.cos(new_q)) / new_c
    if min(new_length, new_gap_after, new_gap_before) <= 0:
        raise ValueError("This family member has a nonpositive length or gap.")
    result = p.copy()
    result[:3] += np.log(scale)
    result[5:8] += np.log(scale)
    result[3:5] = np.log([new_gap_before, new_gap_after])
    result[8] = np.log(new_length)
    result[12] = np.log(-new_q)
    angle = new_q - old_q
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    result[13:15] = root * (rotation @ p[13:15])
    result[15:17] = (rotation @ p[15:17]) / root
    result[17] += np.log(root)
    result[18] -= np.log(root)
    return result


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


def projector_at_voltage(
    parameters, controls, voltage_v, *, reference_voltage_v=200000.0
):
    """Known 5x5 projector map; every magnetic lens sees the new momentum.

    Controls are physical relative currents. Four downstream controls imply
    fixed objective current of one. This does not assume a new objective-current
    response. Calibration validity at the second voltage must be tested separately.
    """
    ratio = magnetic_strength_ratio(voltage_v, reference_voltage_v)
    p = jnp.asarray(parameters).at[11].multiply(ratio).at[12:16].add(jnp.log(ratio))
    return projector_matrix5(p, controls)
