import warnings

from ase import units
import jax.numpy as jnp

_E_OVER_2M0C2 = units._e / (2.0 * units._me * units._c**2)


def wavelength2energy(wavelength: float) -> float:
    """
    Calculate energy from relativistic de Broglie wavelength.

    Parameters
    ----------
    wavelength: float
        Relativistic de Broglie wavelength [Å].

    Returns
    -------
    float
        Energy [eV].
    """
    # Constants
    h = units._hplanck
    c = units._c
    m_e = units._me
    e = units._e

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
    h = units._hplanck
    c = units._c
    m_e = units._me
    e = units._e

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

    m_e = units._me
    c = units._c
    e = units._e
    hbar = units._hbar

    gamma = 1.0 + energy / (m_e * c**2 / e)
    lam = energy2wavelength(energy)

    return gamma * m_e * e * lam / (2 * jnp.pi * hbar**2)


def e_over_hbar() -> float:
    """
    Return e/ħ in SI units: rad / (T·m)
    Typically used as the magnetic prefactor in the A·dl integral.
    """
    e = units._e
    hbar = units._hbar
    return e / hbar


def relativistic_mass_correction(energy: float) -> float:
    e = units._e
    m_e = units._me
    c = units._c
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


def relativistic_voltage_correction(voltage: float) -> float:
    """
    Backward-compatible alias of :func:`relativistic_voltage_factor`.
    """
    return relativistic_voltage_factor(voltage)


def energy2sigma(energy: float) -> float:
    lam = energy2wavelength(energy)
    mass = relativistic_mass_correction(energy) * units._me
    return (
        2 * jnp.pi * mass * units.kg * units._e * units.C * lam
        / (units._hplanck * units.s * units.J) ** 2
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
    See examples/lens_inversion/n_lens_inversion.ipynb.
    """
    e = units._e       # elementary charge [C]
    m_e = units._me    # electron rest mass [kg]
    c = units._c       # speed of light [m/s]
    mu_0 = 1.25663706212e-6  # vacuum permeability [H/m]

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
