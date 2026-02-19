import warnings

from ase import units
import jax.numpy as jnp


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


def compute_I0_Gc_from_focal_rotation(
    focal_length: float, rotation_angle: float, Rc: float
) -> tuple[float, float]:
    """
    Compute excitation current I₀ and geometry constant Gc from target optics.

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
    I0 : float
        Required excitation current [AT].
    Gc: float
        Required geometry constant [1/(AT²·m)].

    Notes
    -----
    **Inverse formulas:**

    - I₀ = ψ / Rc
    - Gc = Rc² / (f·ψ²)

    Derived from the Glaser bell model:

    - f = 1/(Gc·I₀²)
    - ψ = Rc·I₀

    **Physical interpretation:**

    I₀ is the controllable parameter (coil current × turns).
    Gc encodes fixed geometry (bore radius, gap, pole pieces).

    This function tells you what I₀ to set and what Gc the lens must have
    to achieve the target focal length and rotation simultaneously.

    Examples
    --------
    >>> from temgym_core.constants import compute_Kv_from_voltage
    >>> from temgym_core.constants import compute_I0_Gc_from_focal_rotation
    >>>
    >>> Rc= compute_Kv_from_voltage(200e3)  # 200 kV
    >>> I0, Gc = compute_I0_Gc_from_focal_rotation(
    ...     focal_length=0.005,    # 5 mm
    ...     rotation_angle=1.0,     # 1 radian
    ...     Rc=Rc
    ... )
    >>> print(f"I0 = {I0:.1f} AT, Gc= {Gc:.2e} [1/(AT²·m)]")
    """
    if focal_length <= 0:
        raise ValueError("focal_length must be > 0.")
    if Rc == 0:
        raise ValueError("Rc must be non-zero to compute I0 from rotation.")

    # From ψ = Rc·I₀
    I0 = rotation_angle / Rc

    # From f = 1/(Gc·I₀²), rearrange to Gc = 1/(f·I₀²)
    # Or equivalently: Gc = Rc²/(f·ψ²)
    Gc = 1.0 / (focal_length * I0**2)

    return I0, Gc


def compute_I0_Cf_from_focal_rotation(
    focal_length: float, rotation_angle: float, Rc: float
) -> tuple[float, float]:
    """
    Backward-compatible alias for :func:`compute_I0_Gc_from_focal_rotation`.

    Deprecated
    ----------
    Use ``compute_I0_Gc_from_focal_rotation`` and ``Gc/Rc`` naming instead of
    ``Cf/Kv``.
    """
    warnings.warn(
        "`compute_I0_Cf_from_focal_rotation` is deprecated; "
        "use `compute_I0_Gc_from_focal_rotation`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_I0_Gc_from_focal_rotation(
        focal_length=focal_length,
        rotation_angle=rotation_angle,
        Rc=Rc,
    )


def compute_I0_from_focal_length(focal_length: float, Gc: float) -> float:
    """
    Compute excitation current I₀ from target focal length and known Gc.

    For a physical lens with fixed geometry constant Gc, compute the
    excitation current needed to achieve a desired focal length.

    Parameters
    ----------
    focal_length : float
        Desired focal length [m].
    Gc: float
        Lens geometry constant [1/(AT²·m)], fixed by hardware.

    Returns
    -------
    float
        Required excitation current I₀ [AT].

    Notes
    -----
    Formula: I₀ = 1/√(Gc·f)

    Derived from: f = 1/(Gc·I₀²)

    The resulting image rotation will be ψ = Rc·I₀ (not controllable
    independently when Gc is fixed).

    Examples
    --------
    >>> I0 = compute_I0_from_focal_length(
    ...     focal_length=0.005,  # 5 mm
    ...     Gc=5e-6              # 1/(AT²·m)
    ... )
    >>> print(f"Set I0 = {I0:.1f} AT")
    """
    if focal_length <= 0:
        raise ValueError("focal_length must be > 0.")
    if Gc <= 0:
        raise ValueError("Gc must be > 0.")
    return 1.0 / jnp.sqrt(Gc * focal_length)


def compute_I0_from_rotation(rotation_angle: float, Rc: float) -> float:
    """
    Compute excitation current I₀ from target image rotation.

    Compute the excitation current needed to achieve a desired image rotation.

    Parameters
    ----------
    rotation_angle : float
        Desired image rotation angle [rad].
    Rc: float
        Rotation constant [rad/AT], from `compute_Kv_from_voltage()`.

    Returns
    -------
    float
        Required excitation current I₀ [AT].

    Notes
    -----
    Formula: I₀ = ψ/Rc

    Derived from: ψ = Rc·I₀

    The resulting focal length will be f = 1/(Gc·I₀²) (depends on
    the fixed geometry constant Gc of the lens).

    Examples
    --------
    >>> from temgym_core.constants import compute_Kv_from_voltage
    >>> Rc= compute_Kv_from_voltage(200e3)
    >>> I0 = compute_I0_from_rotation(
    ...     rotation_angle=1.0,  # 1 radian
    ...     Rc=Rc
    ... )
    >>> print(f"Set I0 = {I0:.1f} AT")
    """
    if Rc == 0:
        raise ValueError("Rc must be non-zero.")
    return rotation_angle / Rc
