"""Physical condenser and source parameters in TemGym ray coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import c, electron_mass, elementary_charge

from temgym_core.components import solenoid_column
from temgym_core.config import require_x64
from temgym_core.run import column_matrix

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


def condenser_components(parameters, controls):
    """Native lenses from the source waist to the specimen at z=0."""
    p = jnp.asarray(parameters)
    gaps, lengths = jnp.exp(p[:5]), jnp.exp(p[5:9])
    strengths = jnp.exp(p[9:13]) * jnp.r_[jnp.asarray(controls), -1.0]
    source_z = -jnp.sum(gaps) - jnp.sum(lengths)
    return solenoid_column(gaps, lengths, strengths, source_z=source_z)


def design_condenser_curve(
    parameters,
    *,
    image_magnifications,
    mini_control=1.5,
    initial_controls=(3.5, 1.5),
    control_bounds=(0.05, 7.0),
):
    """Design CL1/CL3 settings with one fixed source-image plane before mini.

    Parameters use the same geometry/excitation law as ``condenser_components``.
    Mini and objective stay fixed. Their combined front focal plane sets the
    image plane: a point there leaves the downstream optics parallel (D=0).
    At that plane solve A=m*R and B=0, allowing the native Larmor rotation R.
    Magnifications refer to the source image, not the illuminated specimen area.

    Return the image-plane z [m], requested magnifications, native 5x5 image
    matrices, and rows of [CL1, CL3, mini] relative currents. These are synthetic
    design settings, not amperes/DACs or additional observations for inference.
    Interpolated currents are only guesses; solve again for an exact condition.
    """
    from temgym_core.components import Plane
    from temgym_core.optimization import least_squares

    require_x64("design_condenser_curve")
    targets = np.asarray(image_magnifications, dtype=float)
    if (
        targets.ndim != 1
        or targets.size == 0
        or not np.all(np.isfinite(targets) & (targets > 0))
    ):
        raise ValueError(
            "Supply a nonempty vector of positive source-image magnifications."
        )
    if not np.isfinite(mini_control) or mini_control <= 0:
        raise ValueError("The fixed mini control must be positive and finite.")
    p = jnp.asarray(parameters)
    column = condenser_components(p, jnp.array([1.0, 1.0, mini_control]))
    source_z = -jnp.sum(jnp.exp(p[:9]))
    downstream = np.asarray(column_matrix(column[2:], source_z=column[2].z))
    cc, dd = downstream[2:4, :2], downstream[2:4, 2:4]
    distance = -np.sum(cc * dd) / np.sum(cc * cc)
    image_z = float(column[2].z) - distance
    if not float(column[1].z + column[1].length) < image_z < float(column[2].z):
        raise ValueError(
            "Choose a mini control with a real front focal plane between CL3 and mini."
        )

    def image_matrix(controls):
        lenses = condenser_components(p, jnp.r_[controls, mini_control])
        return column_matrix((*lenses[:2], Plane(z=image_z)), source_z=source_z)

    def residual(controls, target):
        matrix = image_matrix(controls)
        angle = jnp.dot(jnp.exp(p[9:11]), controls)
        rotation = jnp.array(
            [[jnp.cos(angle), jnp.sin(angle)], [-jnp.sin(angle), jnp.cos(angle)]]
        )
        return jnp.r_[
            (matrix[:2, :2] / target - rotation).ravel(),
            (matrix[:2, 2:4] / (image_z - source_z)).ravel(),
        ]

    current = np.asarray(initial_controls, dtype=float)
    controls, matrices = [], []
    for target in targets:
        fit = least_squares(
            residual,
            current,
            args=(target,),
            bounds=control_bounds,
            x_scale="jac",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-12,
            max_iter=300,
        )
        if not np.all(np.isfinite(fit.residual)) or np.max(np.abs(fit.residual)) > 1e-9:
            raise ValueError(
                f"No accurate condenser design at source-image magnification {target:g}."
            )
        current = np.asarray(fit.x)
        controls.append(np.r_[current, mini_control])
        matrices.append(np.asarray(image_matrix(current)))
    return {
        "image_plane_z_m": image_z,
        "image_magnifications": targets,
        "controls": np.asarray(controls),
        "image_matrices": np.asarray(matrices),
    }


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


def first_crossover_fraction(parameters, controls):
    """Source-image distance after CL1, divided by the free gap to CL3.

    B+tD=0 gives the source conjugate in a drift after the native first lens.
    Values between zero and one put it in the gap. This is an optical image
    plane, not a threshold on the finite beam radius or an aperture measurement.
    Use the first excitation branch (CL1 q < pi/2) for a first crossover.
    """
    p = jnp.asarray(parameters)
    first = condenser_components(p, controls)[:1]
    matrix = column_matrix(first, source_z=-jnp.sum(jnp.exp(p[:9])))
    bb, dd = matrix[:2, 2:4], matrix[2:4, 2:4]
    distance = -jnp.sum(bb * dd) / jnp.maximum(jnp.sum(dd * dd), 1e-24)
    return distance / jnp.exp(p[1])


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
            include_axial_path=False,
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


def specimen_moments(parameters, controls, voltage_v=200000.0):
    transfer = matrix5(parameters, controls)[:4, :4]
    mean, covariance = source_moments(parameters, voltage_v)
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


def condenser_starting_guesses(
    *, starting_guess_path, number_of_broad_starts=4, number_of_apertures=1
):
    """Four engineering guesses and broad starts; no earlier fit or truth is loaded."""
    import json
    from pathlib import Path

    from scipy.stats import qmc

    from .apertures import equivalent_aperture_model
    from .condenser_model import initial_parameters

    settings = json.loads(Path(starting_guess_path).read_text())
    model = settings["model"]
    fraction = model["aperture_gap_fraction"]
    p = np.r_[
        np.log(np.asarray(model["gaps_mm"]) / 1000),
        np.log(np.asarray(model["lens_lengths_mm"]) / 1000),
        np.log(model["strength_magnitudes"]),
        np.asarray(model["source_position_nm"]) / 10,
        np.asarray(model["source_angle_mrad"]) * 10,
        np.log(model["source_rms_nm"] * 1e-9),
        np.log(model["source_incoherent_angular_rms_mrad"] * 1e-3),
        np.log(fraction / (1 - fraction)),
        model["aperture_centre_um"],
    ]

    def extend(base):
        return np.r_[
            base, base[0], base[13:19], np.tile(base[20:22], number_of_apertures - 1)
        ]

    starts = [extend(p), extend(np.r_[initial_parameters(), -3.0, 0.1, -0.1])]
    starts += [
        extend(equivalent_aperture_model(p, objective_strength=-q))
        for q in [0.35, 0.65]
    ]
    lower, upper = extend(settings["bounds_lower"]), extend(settings["bounds_upper"])
    if number_of_broad_starts:
        sample = qmc.LatinHypercube(d=len(lower), seed=713).random(
            number_of_broad_starts
        )
        starts += list(qmc.scale(sample, lower + 1e-7, upper - 1e-7))
    return starts, lower, upper
