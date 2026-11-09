"""Physical constants and relativistic electron-optics helpers (SI units)."""

import math

import jax.numpy as jnp

# Fundamental constants in SI units. These are the CODATA 2014 values the package
# previously took from ``ase.units``; they are kept exactly so results do not change.
ELEMENTARY_CHARGE = 1.6021766208e-19    # C
ELECTRON_MASS = 9.10938356e-31          # kg
SPEED_OF_LIGHT = 299792458.0            # m / s
PLANCK = 6.62607004e-34                 # J s
HBAR = PLANCK / (2 * math.pi)           # J s
VACUUM_PERMEABILITY = 1.25663706212e-6  # H / m (CODATA 2018)

# Factors that convert SI numbers to the atomic unit system of ``ase.units``
# (length A, energy eV, mass u, charge e). Only the interaction constant ``sigma``
# uses them; it is meaningful for beams expressed in angstrom.
_ATOMIC_MASS_UNIT = 1.660539040e-27     # kg
_KG = 1.0 / _ATOMIC_MASS_UNIT
_COULOMB = 1.0 / ELEMENTARY_CHARGE
_JOULE = 1.0 / ELEMENTARY_CHARGE
_SECOND = 1e10 * math.sqrt(ELEMENTARY_CHARGE / _ATOMIC_MASS_UNIT)

_E_OVER_2M0C2 = ELEMENTARY_CHARGE / (2.0 * ELECTRON_MASS * SPEED_OF_LIGHT**2)


def wavelength2energy(wavelength: float) -> float:
    """
    Calculate energy from relativistic de Broglie wavelength.

    Parameters
    ----------
    wavelength: float
        Relativistic de Broglie wavelength [m].

    Returns
    -------
    float
        Energy [eV].
    """
    # Constants
    h = PLANCK
    c = SPEED_OF_LIGHT
    m_e = ELECTRON_MASS
    e = ELEMENTARY_CHARGE

    term1 = (h * c / wavelength) ** 2
    term2 = (m_e * c**2) ** 2

    energy_joules = jnp.sqrt(term1 + term2) - m_e * c**2

    return energy_joules / e


def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy with higher precision options.

    Parameters
    ----------
    energy : float
        Kinetic energy [eV].
    """
    h = PLANCK
    c = SPEED_OF_LIGHT
    m_e = ELECTRON_MASS
    e = ELEMENTARY_CHARGE

    E_j = energy * e
    mc2 = m_e * c * c
    E_total = E_j + mc2
    rad = E_total * E_total - mc2 * mc2
    p = jnp.sqrt(rad) / c
    wavelength_m = h / p
    return wavelength_m


def interaction_constant_CE(energy: float) -> float:
    """
    Electrostatic phase interaction constant C_E [rad / (V·m)]
    as used in electron holography.

    Parameters
    ----------
    energy : float
        Kinetic energy [eV].
    """

    m_e = ELECTRON_MASS
    c = SPEED_OF_LIGHT
    e = ELEMENTARY_CHARGE
    hbar = HBAR

    gamma = 1.0 + energy / (m_e * c**2 / e)
    lam = energy2wavelength(energy)

    return gamma * m_e * e * lam / (2 * jnp.pi * hbar**2)


def e_over_hbar() -> float:
    """
    Return e/ħ in SI units: rad / (T·m)
    Typically used as the magnetic prefactor in the A·dl integral.
    """
    return ELEMENTARY_CHARGE / HBAR


def relativistic_mass_correction(energy: float) -> float:
    e, m_e, c = ELEMENTARY_CHARGE, ELECTRON_MASS, SPEED_OF_LIGHT
    return 1 + e * energy / (m_e * c**2)


def relativistic_voltage_factor(voltage: float) -> float:
    """
    Return the relativistic accelerating-voltage factor V*/V.

    Notes
    -----
    V* / V = 1 + eV / (2 m0 c^2)
    """
    return 1.0 + _E_OVER_2M0C2 * voltage


def effective_accelerating_potential(voltage: float) -> float:
    """
    Return the effective relativistic accelerating potential V* [V].

    Notes
    -----
    V* = V * (V*/V)
    """
    return voltage * relativistic_voltage_factor(voltage)


def voltage_scaling_ratio(voltage: float, reference_voltage: float) -> float:
    """
    Return V*_ref / V* for scaling voltage-dependent lens constants.
    """
    return (
        effective_accelerating_potential(reference_voltage)
        / effective_accelerating_potential(voltage)
    )


def magnetic_strength_ratio(voltage, reference_voltage=200e3):
    """Return p(reference) / p(voltage): how a magnetic lens's strength scales with the high
    tension at a fixed field (strength proportional to 1/p). The square root of
    :func:`voltage_scaling_ratio`."""
    import jax.numpy as jnp

    return jnp.sqrt(voltage_scaling_ratio(voltage, reference_voltage))


def relativistic_voltage_correction(voltage: float) -> float:
    """
    Backward-compatible alias of :func:`relativistic_voltage_factor`.
    """
    return relativistic_voltage_factor(voltage)


def energy2sigma(energy: float) -> float:
    lam = energy2wavelength(energy)
    mass = relativistic_mass_correction(energy) * ELECTRON_MASS
    return (
        2 * jnp.pi * mass * _KG * ELEMENTARY_CHARGE * _COULOMB * lam
        / (PLANCK * _SECOND * _JOULE) ** 2
    )


def compute_Rc_from_voltage(U_accel: float) -> float:
    """
    Calculate rotation constant Rc from accelerating voltage.

    For electromagnetic lenses in the Glaser bell model, the image rotation
    angle is ψ = Rc·I₀, where I₀ is the excitation current in ampere-turns.

    Parameters
    ----------
    U_accel : float
        Accelerating voltage [V].

    Returns
    -------
    float
        Rotation constant Rc[rad/AT].

    Notes
    -----
    Formula: Rc = (e·μ₀)/(2·mₑ·v)
    where v is the relativistic electron velocity.

    References
    ----------
    Glaser bell model for electromagnetic lenses.
    (Worked example: archived/examples/lens_inversion/n_lens_inversion.ipynb, removed in
    the 2026-11-06 cleanup; Git revision b2d873e retains it.)
    """
    e = ELEMENTARY_CHARGE
    m_e = ELECTRON_MASS
    c = SPEED_OF_LIGHT
    mu_0 = VACUUM_PERMEABILITY

    # Relativistic gamma factor
    gamma_rel = 1.0 + e * U_accel / (m_e * c**2)

    # Relativistic electron velocity
    v_electron = c * jnp.sqrt(1.0 - 1.0 / gamma_rel**2)

    # Rotation constant
    K_rot = e * mu_0 / (2.0 * m_e * v_electron)

    return K_rot


def compute_Kv_from_voltage(U_accel: float) -> float:
    """
    Backward-compatible alias of :func:`compute_Rc_from_voltage`.
    """
    return compute_Rc_from_voltage(U_accel)


def compute_NI_Gc_from_lens_parameters(
    focal_length: float, rotation_angle: float, Rc: float
) -> tuple[float, float]:
    """
    Compute excitation current NI and geometry constant Gc from target optics.

    Given desired focal length and image rotation, compute the electromagnetic
    lens parameters needed. This is the inverse of the Glaser model formulas.

    Parameters
    ----------
    focal_length : float
        Desired focal length [m].
    rotation_angle : float
        Desired image rotation angle [rad].
    Rc: float
        Rotation constant [rad/AT], computed from voltage via
        `compute_Kv_from_voltage()`.

    Returns
    -------
    NI : float
        Required Ampere Turns.
    Gc: float
        Required geometry constant [1/(AT²·m)].

    Notes
    -----
    **Inverse formulas:**

    - NI = ψ / Rc
    - Gc = Rc² / (f·ψ²)

    Derived from the Glaser bell model:

    - f = 1/(Gc·NI²)
    - ψ = Rc·NI

    **Physical interpretation:**

    NI is the controllable parameter (coil current × turns).
    Gc encodes fixed geometry (bore radius, gap, pole pieces).

    This function tells you what NI to set and what Gc the lens must have
    to achieve the target focal length and rotation simultaneously.

    Examples
    --------
    >>> from temgym_core.constants import compute_Kv_from_voltage
    >>> from temgym_core.constants import compute_NI_Gc_from_lens_parameters
    >>>
    >>> Rc= compute_Kv_from_voltage(200e3)  # 200 kV
    >>> NI, Gc = compute_NI_Gc_from_lens_parameters(
    ...     focal_length=0.005,    # 5 mm
    ...     rotation_angle=1.0,     # 1 radian
    ...     Rc=Rc
    ... )
    >>> print(f"NI = {NI:.1f} AT, Gc= {Gc:.2e} [1/(AT²·m)]")
    """
    if focal_length <= 0:
        raise ValueError("focal_length must be > 0.")
    if Rc == 0:
        raise ValueError("Rc must be non-zero to compute NI from rotation.")

    # From ψ = Rc·NI
    NI = rotation_angle / Rc

    # From f = 1/(Gc·NI²), rearrange to Gc = 1/(f·NI²)
    # Or equivalently: Gc = Rc²/(f·ψ²)
    Gc = 1.0 / (focal_length * NI**2)

    return NI, Gc


def compute_rotation_angle_Gc_from_NI_focal_length(
    NI: float, focal_length: float, Rc: float
) -> tuple[float, float]:
    """
    Compute rotation angle and geometry constant Gc from NI and focal length.

    Given the excitation current (NI) and desired focal length, compute the
    resulting image rotation angle and required geometry constant.

    Parameters
    ----------
    NI : float
        Ampere Turns (excitation current).
    focal_length : float
        Desired focal length [m].
    Rc : float
        Rotation constant [rad/AT], computed from voltage via
        `compute_Kv_from_voltage()`.

    Returns
    -------
    rotation_angle : float
        Resulting image rotation angle [rad].
    Gc : float
        Required geometry constant [1/(AT²·m)].

    Notes
    -----
    **Forward formulas (from Glaser bell model):**

    - ψ = Rc·NI
    - Gc = 1/(f·NI²)

    This is the inverse operation of `compute_NI_Gc_from_lens_parameters()`.

    Examples
    --------
    >>> from temgym_core.constants import compute_Kv_from_voltage
    >>> from temgym_core.constants import compute_rotation_angle_Gc_from_NI_focal_length
    >>>
    >>> Rc = compute_Kv_from_voltage(200e3)  # 200 kV
    >>> rotation_angle, Gc = compute_rotation_angle_Gc_from_NI_focal_length(
    ...     NI=10.0,                # 10 AT
    ...     focal_length=0.005,     # 5 mm
    ...     Rc=Rc
    ... )
    >>> print(f"ψ = {rotation_angle:.4f} rad, Gc = {Gc:.2e} [1/(AT²·m)]")
    """
    if focal_length <= 0:
        raise ValueError("focal_length must be > 0.")

    # From ψ = Rc·NI
    rotation_angle = Rc * NI

    # From f = 1/(Gc·NI²), rearrange to Gc = 1/(f·NI²)
    Gc = 1.0 / (focal_length * NI**2)

    return rotation_angle, Gc


def deflection_from_excitation(Dc: float, NI: float, voltage: float) -> float:
    """
    Compute deflection angle from deflection constant and excitation.

    Parameters
    ----------
    Dc : float
        Deflection constant (geometry) [rad·V^{1/2}/AT].
    NI : float
        Excitation in ampere-turns [AT].
    voltage : float
        Accelerating voltage [V].

    Returns
    -------
    float
        Deflection angle [rad].

    Notes
    -----
    Formula: alpha = Dc * NI / sqrt(V*)

    Analogous to the lens formula f = V* / (Gc * NI^2), but linear
    in excitation rather than quadratic.
    """
    V_star = effective_accelerating_potential(voltage)
    return Dc * NI / V_star ** 0.5
