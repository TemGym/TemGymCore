"""First-order, DAC-native tools for microscope system identification.

The functions in this module deliberately use microscope control values directly.
They do not assume that a DAC value is a current, an ampere-turn excitation, or a
magnetic field.  A control law maps a raw DAC value to an *optical* parameter such
as focal power or integrated Larmor strength.

The canonical ray order throughout is ``[x, theta_x, y, theta_y]``.  Matrices act
from the left, so a sequence encountered in the order ``M1, M2`` has total matrix
``M2 @ M1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares


ArrayLike = Any
MatrixFunction = Callable[[jnp.ndarray], jnp.ndarray]

STATE_ORDER = ("x", "theta_x", "y", "theta_y")
DEFAULT_LENS_CHANNELS = (
    "CL1",
    "CL3",
    "C-mini",
    "IL1",
    "IL2",
    "IL3",
    "PL1",
)

POSITION_TILT_OBSERVABLE_NAMES = (
    "x_out_per_x",
    "x_out_per_y",
    "y_out_per_x",
    "y_out_per_y",
    "x_out_per_theta_x",
    "x_out_per_theta_y",
    "y_out_per_theta_x",
    "y_out_per_theta_y",
)

FULL_RESPONSE_OBSERVABLE_NAMES = POSITION_TILT_OBSERVABLE_NAMES + (
    "theta_x_out_per_x",
    "theta_x_out_per_y",
    "theta_y_out_per_x",
    "theta_y_out_per_y",
    "theta_x_out_per_theta_x",
    "theta_x_out_per_theta_y",
    "theta_y_out_per_theta_x",
    "theta_y_out_per_theta_y",
)

ILLUMINATION_IMAGE_MOMENT_NAMES = (
    "centroid_u",
    "centroid_v",
    "covariance_uu",
    "covariance_uv",
    "covariance_vv",
)

PHASE_SPACE_COVARIANCE_NAMES = tuple(
    f"covariance_{left}_{right}"
    for left_index, left in enumerate(STATE_ORDER)
    for right in STATE_ORDER[left_index:]
)


def mathematical_sinc(x: ArrayLike) -> jnp.ndarray:
    """Return ``sin(x) / x`` with a stable value and derivative at zero.

    ``jax.numpy.sinc`` uses the normalized convention ``sin(pi*x)/(pi*x)``;
    rescaling its argument gives the mathematical convention used here.
    """

    value = jnp.asarray(x)
    return jnp.sinc(value / jnp.pi)


def canonical_symplectic_form(dtype: Any | None = None) -> jnp.ndarray:
    """Return the canonical 4-D symplectic form for :data:`STATE_ORDER`."""

    dtype = jnp.asarray(0.0).dtype if dtype is None else dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    return jnp.asarray(
        [
            [zero, one, zero, zero],
            [-one, zero, zero, zero],
            [zero, zero, zero, one],
            [zero, zero, -one, zero],
        ]
    )


def symplectic_defect(matrix: ArrayLike) -> jnp.ndarray:
    """Return ``M.T @ J @ M - J`` for a 4-by-4 ray matrix."""

    matrix = jnp.asarray(matrix)
    form = canonical_symplectic_form(matrix.dtype)
    return matrix.T @ form @ matrix - form


def drift_matrix(distance: ArrayLike) -> jnp.ndarray:
    """First-order free-space matrix for a signed distance."""

    distance = jnp.asarray(distance)
    zero = jnp.zeros_like(distance)
    one = jnp.ones_like(distance)
    return jnp.stack(
        (
            jnp.stack((one, distance, zero, zero)),
            jnp.stack((zero, one, zero, zero)),
            jnp.stack((zero, zero, one, distance)),
            jnp.stack((zero, zero, zero, one)),
        )
    )


def larmor_rotation_4x4(angle: ArrayLike) -> jnp.ndarray:
    """Rotate positions and slopes through ``angle`` in the transverse plane.

    Positive angles are active counter-clockwise rotations.  For a hard-edge
    solenoid with signed integrated strength ``q = K*L``, the electron-optical
    Larmor rotation used by :func:`hard_edge_solenoid_matrix` is ``-q``.
    """

    angle = jnp.asarray(angle)
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    zero = jnp.zeros_like(angle)
    return jnp.stack(
        (
            jnp.stack((cosine, zero, -sine, zero)),
            jnp.stack((zero, cosine, zero, -sine)),
            jnp.stack((sine, zero, cosine, zero)),
            jnp.stack((zero, sine, zero, cosine)),
        )
    )


def _embed_round_radial_matrix(radial: ArrayLike) -> jnp.ndarray:
    """Embed one 2-by-2 radial map identically in x and y."""

    radial = jnp.asarray(radial)
    if radial.shape != (2, 2):
        raise ValueError("radial must have shape (2, 2).")
    zero = jnp.zeros_like(radial[0, 0])
    return jnp.stack(
        (
            jnp.stack((radial[0, 0], radial[0, 1], zero, zero)),
            jnp.stack((radial[1, 0], radial[1, 1], zero, zero)),
            jnp.stack((zero, zero, radial[0, 0], radial[0, 1])),
            jnp.stack((zero, zero, radial[1, 0], radial[1, 1])),
        )
    )


def thin_lens_matrix(power: ArrayLike, rotation: ArrayLike = 0.0) -> jnp.ndarray:
    """Return a rotating thin round-lens matrix.

    Parameters
    ----------
    power
        Signed focal power ``1/f`` in inverse length.
    rotation
        Independently calibrated transverse rotation in radians.
    """

    power = jnp.asarray(power)
    zero = jnp.zeros_like(power)
    one = jnp.ones_like(power)
    radial = jnp.stack(
        (jnp.stack((one, zero)), jnp.stack((-power, one)))
    )
    return larmor_rotation_4x4(rotation) @ _embed_round_radial_matrix(radial)


def hard_edge_solenoid_matrix(q: ArrayLike, length: ArrayLike) -> jnp.ndarray:
    """Exact paraxial matrix for an ideal hard-edge solenoid.

    ``q = K*length`` is the signed, dimensionless integrated Larmor strength.
    The map includes propagation through the finite body of the lens and a
    physical transverse rotation of ``-q``.  It is exact for a uniform axial
    field with abrupt edges, not for an arbitrary real pole-piece field.
    """

    q = jnp.asarray(q)
    length = jnp.asarray(length)
    cosine = jnp.cos(q)
    sine = jnp.sin(q)
    radial = jnp.stack(
        (
            jnp.stack((cosine, length * mathematical_sinc(q))),
            jnp.stack((-(q / length) * sine, cosine)),
        )
    )
    return larmor_rotation_4x4(-q) @ _embed_round_radial_matrix(radial)


def evaluate_dac_polynomial(
    dac: ArrayLike,
    coefficients: ArrayLike,
    reference_dac: ArrayLike = 0.0,
) -> jnp.ndarray:
    """Evaluate an ascending-order polynomial in a raw DAC displacement.

    ``coefficients=(c0, c1, c2)`` represents
    ``c0 + c1*(dac-reference_dac) + c2*(dac-reference_dac)**2``.
    """

    coefficients = jnp.asarray(coefficients)
    if coefficients.ndim != 1 or coefficients.shape[0] == 0:
        raise ValueError("coefficients must be a non-empty 1-D array.")
    displacement = jnp.asarray(dac) - jnp.asarray(reference_dac)
    powers = jnp.arange(coefficients.shape[0])
    return jnp.sum(coefficients * displacement**powers)


def dac_thin_lens_matrix(
    dac: ArrayLike,
    power_coefficients: ArrayLike,
    rotation_coefficients: ArrayLike = (0.0,),
    reference_dac: ArrayLike = 0.0,
) -> jnp.ndarray:
    """Thin-lens matrix whose optical parameters are functions of raw DAC."""

    power = evaluate_dac_polynomial(dac, power_coefficients, reference_dac)
    rotation = evaluate_dac_polynomial(dac, rotation_coefficients, reference_dac)
    return thin_lens_matrix(power, rotation)


def dac_hard_edge_solenoid_matrix(
    dac: ArrayLike,
    length: ArrayLike,
    q_coefficients: ArrayLike,
    reference_dac: ArrayLike = 0.0,
) -> jnp.ndarray:
    """Hard-edge solenoid with ``q`` calibrated directly in raw DAC units."""

    q = evaluate_dac_polynomial(dac, q_coefficients, reference_dac)
    return hard_edge_solenoid_matrix(q, length)


def compose_matrices(*matrices: ArrayLike) -> jnp.ndarray:
    """Compose matrices in the order in which a ray encounters them."""

    if not matrices:
        return jnp.eye(4)
    total = jnp.eye(4, dtype=jnp.asarray(matrices[0]).dtype)
    for matrix in matrices:
        total = jnp.asarray(matrix) @ total
    return total


def smooth_field_solenoid_matrix(
    z_edges: ArrayLike,
    alpha_midpoints: ArrayLike,
) -> jnp.ndarray:
    """Symplectically integrate an arbitrary smooth round-lens field.

    Parameters
    ----------
    z_edges
        Monotonically increasing slice edges with shape ``(n + 1,)``.
    alpha_midpoints
        Signed Larmor rate ``e*B(z)/(2*p)`` at slice midpoints, shape ``(n,)``.

    Each interval is advanced by its exact constant-field map.  The product is
    symplectic to numerical precision and differentiable with respect to the
    field samples and edge locations.
    """

    z_edges = jnp.asarray(z_edges)
    alpha_midpoints = jnp.asarray(alpha_midpoints)
    if z_edges.ndim != 1 or alpha_midpoints.ndim != 1:
        raise ValueError("z_edges and alpha_midpoints must be 1-D arrays.")
    if z_edges.shape[0] != alpha_midpoints.shape[0] + 1:
        raise ValueError("len(z_edges) must equal len(alpha_midpoints) + 1.")
    distances = jnp.diff(z_edges)

    def advance(total: jnp.ndarray, values: tuple[jnp.ndarray, jnp.ndarray]):
        distance, alpha = values
        step = hard_edge_solenoid_matrix(alpha * distance, distance)
        return step @ total, None

    initial = jnp.eye(4, dtype=jnp.result_type(z_edges, alpha_midpoints))
    total, _ = jax.lax.scan(advance, initial, (distances, alpha_midpoints))
    return total


def normalized_profile_solenoid_matrix(
    q: ArrayLike,
    z_edges: ArrayLike,
    profile_midpoints: ArrayLike,
) -> jnp.ndarray:
    """Integrate a profile normalized so its discrete integral equals ``q``."""

    z_edges = jnp.asarray(z_edges)
    profile_midpoints = jnp.asarray(profile_midpoints)
    if z_edges.shape[0] != profile_midpoints.shape[0] + 1:
        raise ValueError("len(z_edges) must equal len(profile_midpoints) + 1.")
    distances = jnp.diff(z_edges)
    normalization = jnp.sum(profile_midpoints * distances)
    alpha = jnp.asarray(q) * profile_midpoints / normalization
    return smooth_field_solenoid_matrix(z_edges, alpha)


def glaser_solenoid_matrix(
    q: ArrayLike,
    center: ArrayLike,
    width: ArrayLike,
    z_min: ArrayLike,
    z_max: ArrayLike,
    n_slices: int = 128,
) -> jnp.ndarray:
    """Integrate a normalized Glaser bell field over finite model boundaries.

    The sampled profile is proportional to
    ``1 / (1 + ((z-center)/width)**2)`` and is renormalized on
    ``[z_min, z_max]`` so that the accumulated Larmor strength is exactly ``q``.
    """

    if int(n_slices) != n_slices or n_slices < 1:
        raise ValueError("n_slices must be a positive integer.")
    edges = jnp.linspace(z_min, z_max, int(n_slices) + 1)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    profile = 1.0 / (1.0 + ((midpoints - center) / width) ** 2)
    return normalized_profile_solenoid_matrix(q, edges, profile)


def smooth_basis_solenoid_matrix(
    q: ArrayLike,
    z_edges: ArrayLike,
    basis_values: ArrayLike,
    coefficients: ArrayLike,
    positive: bool = True,
) -> jnp.ndarray:
    """Integrate a low-dimensional smooth axial-field basis.

    ``basis_values`` has shape ``(n_basis, n_slices)``.  When ``positive`` is
    true, a softplus link makes the normalized profile non-negative while the
    sign of the field remains controlled by ``q``.
    """

    basis_values = jnp.asarray(basis_values)
    coefficients = jnp.asarray(coefficients)
    if basis_values.ndim != 2 or coefficients.ndim != 1:
        raise ValueError("basis_values must be 2-D and coefficients must be 1-D.")
    if basis_values.shape[0] != coefficients.shape[0]:
        raise ValueError("One coefficient is required per basis function.")
    raw_profile = coefficients @ basis_values
    profile = jax.nn.softplus(raw_profile) if positive else raw_profile
    return normalized_profile_solenoid_matrix(q, z_edges, profile)


def generic_round_symplectic_matrix(
    phase: ArrayLike,
    log_magnification: ArrayLike,
    shear: ArrayLike,
    rotation: ArrayLike = 0.0,
    length_scale: ArrayLike = 1.0,
) -> jnp.ndarray:
    """Return a general axisymmetric first-order symplectic section.

    The radial 2-by-2 map uses an Iwasawa factorization: phase advance, positive
    squeeze, and upper shear.  ``length_scale`` nondimensionalizes position and
    angle and should be near the characteristic propagation length.  A separate
    Larmor rotation supplies the fourth round-section degree of freedom.
    """

    phase = jnp.asarray(phase)
    log_magnification = jnp.asarray(log_magnification)
    shear = jnp.asarray(shear)
    length_scale = jnp.asarray(length_scale)
    cosine = jnp.cos(phase)
    sine = jnp.sin(phase)
    zero = jnp.zeros_like(phase)
    one = jnp.ones_like(phase)

    phase_map = jnp.stack(
        (
            jnp.stack((cosine, length_scale * sine)),
            jnp.stack((-sine / length_scale, cosine)),
        )
    )
    squeeze = jnp.stack(
        (
            jnp.stack((jnp.exp(log_magnification), zero)),
            jnp.stack((zero, jnp.exp(-log_magnification))),
        )
    )
    upper_shear = jnp.stack(
        (
            jnp.stack((one, length_scale * shear)),
            jnp.stack((zero, one)),
        )
    )
    radial = phase_map @ squeeze @ upper_shear
    return larmor_rotation_4x4(rotation) @ _embed_round_radial_matrix(radial)


@dataclass(frozen=True)
class DACPolynomial:
    """Immutable ascending-order optical control law in raw DAC units."""

    coefficients: tuple[float, ...]
    reference_dac: float = 0.0

    def __post_init__(self) -> None:
        coefficients = tuple(float(value) for value in self.coefficients)
        if not coefficients or not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must be non-empty and finite.")
        if not np.isfinite(self.reference_dac):
            raise ValueError("reference_dac must be finite.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "reference_dac", float(self.reference_dac))

    def __call__(self, dac: ArrayLike) -> jnp.ndarray:
        return evaluate_dac_polynomial(dac, self.coefficients, self.reference_dac)


@dataclass(frozen=True)
class DACThinLens:
    """Thin round lens driven by raw DAC control."""

    name: str
    z: float
    power: DACPolynomial
    rotation: DACPolynomial = field(default_factory=lambda: DACPolynomial((0.0,)))

    def matrix(self, dac: ArrayLike) -> jnp.ndarray:
        return thin_lens_matrix(self.power(dac), self.rotation(dac))


@dataclass(frozen=True)
class DACHardEdgeLens:
    """Finite hard-edge round lens driven by raw DAC control."""

    name: str
    center_z: float
    length: float
    q: DACPolynomial

    def __post_init__(self) -> None:
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and positive.")

    @property
    def entrance_z(self) -> float:
        return self.center_z - 0.5 * self.length

    @property
    def exit_z(self) -> float:
        return self.center_z + 0.5 * self.length

    def matrix(self, dac: ArrayLike) -> jnp.ndarray:
        return hard_edge_solenoid_matrix(self.q(dac), self.length)


@dataclass(frozen=True)
class DACGlaserLens:
    """Normalized Glaser-profile lens driven by raw DAC control."""

    name: str
    center_z: float
    width: float
    z_min: float
    z_max: float
    q: DACPolynomial
    n_slices: int = 128

    def __post_init__(self) -> None:
        if not np.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("width must be finite and positive.")
        if not self.z_min < self.z_max:
            raise ValueError("z_min must be smaller than z_max.")
        if int(self.n_slices) != self.n_slices or self.n_slices < 1:
            raise ValueError("n_slices must be a positive integer.")

    def matrix(self, dac: ArrayLike) -> jnp.ndarray:
        return glaser_solenoid_matrix(
            self.q(dac),
            self.center_z,
            self.width,
            self.z_min,
            self.z_max,
            self.n_slices,
        )


@dataclass(frozen=True)
class DACRoundSymplecticSection:
    """Generic round symplectic section with DAC-polynomial coordinates."""

    name: str
    phase: DACPolynomial
    log_magnification: DACPolynomial
    shear: DACPolynomial
    rotation: DACPolynomial = field(default_factory=lambda: DACPolynomial((0.0,)))
    length_scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.length_scale) or self.length_scale <= 0.0:
            raise ValueError("length_scale must be finite and positive.")

    def matrix(self, dac: ArrayLike) -> jnp.ndarray:
        return generic_round_symplectic_matrix(
            self.phase(dac),
            self.log_magnification(dac),
            self.shear(dac),
            self.rotation(dac),
            self.length_scale,
        )


@dataclass(frozen=True)
class SamplePlaneState:
    """A calibrated paraxial state at the specimen plane."""

    x: float = 0.0
    theta_x: float = 0.0
    y: float = 0.0
    theta_y: float = 0.0
    label: str = "sample_state"

    def __post_init__(self) -> None:
        values = np.asarray((self.x, self.theta_x, self.y, self.theta_y), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Sample-plane coordinates and tilts must be finite.")

    def as_array(self) -> np.ndarray:
        return np.asarray((self.x, self.theta_x, self.y, self.theta_y), dtype=float)

    def as_jax(self) -> jnp.ndarray:
        return jnp.asarray((self.x, self.theta_x, self.y, self.theta_y))


@dataclass(frozen=True)
class CalibratedSourceState:
    """A measured entrance-plane centroid and phase-space covariance.

    This state is deliberately defined *before* the condenser system.  A
    calibrated state at the specimen is not interchangeable with it: treating
    specimen coordinates as known inputs removes the source-to-specimen map
    from the prediction and therefore supplies no information about condenser
    lenses.

    ``covariance`` follows :data:`STATE_ORDER` and may be positive
    semidefinite.  Its entries describe the source distribution used for one
    detector exposure, rather than the uncertainty of the calibration itself.
    Calibration uncertainty belongs in the corresponding observation block.
    """

    mean_state: ArrayLike
    covariance: ArrayLike
    label: str = "source_on_axis"

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_state, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if not self.label:
            raise ValueError("A calibrated source state requires a non-empty label.")
        if mean.shape != (4,) or not np.all(np.isfinite(mean)):
            raise ValueError("mean_state must be a finite vector with shape (4,).")
        if covariance.shape != (4, 4) or not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be a finite array with shape (4, 4).")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-15):
            raise ValueError("covariance must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(covariance)
        tolerance = 1e-10 * max(
            float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny
        )
        if float(eigenvalues[0]) < -tolerance:
            raise ValueError("covariance must be positive semidefinite.")
        mean = mean.copy()
        covariance = covariance.copy()
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mean_state", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class IlluminationProjection2D:
    """One calibrated two-coordinate image of the specimen-plane beam.

    ``position_map`` has shape ``(2, 4)`` and maps a specimen-plane ray state
    to the two calibrated coordinates measured in an image.  Examples include
    a specimen-conjugate image, a known defocus, and a diffraction-plane image.
    It must describe a physical screen: its two rows must be independent and
    form a commuting (Lagrangian) coordinate pair.  The map should be obtained
    from the already calibrated objective/projector column, not fitted again as
    part of the condenser experiment.
    """

    label: str
    position_map: ArrayLike
    coordinate_unit: str = "m"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.position_map, dtype=float)
        if not self.label:
            raise ValueError("An illumination projection requires a non-empty label.")
        if not self.coordinate_unit:
            raise ValueError("coordinate_unit must be non-empty.")
        if matrix.shape != (2, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("position_map must be a finite array with shape (2, 4).")
        if np.linalg.matrix_rank(matrix) != 2:
            raise ValueError("position_map must contain two independent coordinates.")
        commutator = matrix @ np.asarray(canonical_symplectic_form()) @ matrix.T
        scale = max(float(np.linalg.norm(matrix, ord=2) ** 2), np.finfo(float).tiny)
        if float(np.linalg.norm(commutator, ord=2)) > 1e-10 * scale:
            raise ValueError(
                "position_map rows must be a commuting physical screen-coordinate pair."
            )
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "position_map", matrix)

    @classmethod
    def from_transfer_matrix(
        cls,
        label: str,
        sample_to_screen_matrix: ArrayLike,
        *,
        detector_affine_xy: ArrayLike | None = None,
        coordinate_unit: str = "m",
    ) -> IlluminationProjection2D:
        """Construct a projection from a calibrated downstream ray matrix.

        ``detector_affine_xy`` optionally applies the measured two-dimensional
        detector-coordinate calibration after selecting the screen-position
        rows.  This is the preferred instrument path: calibrate a small set of
        downstream imaging modes, freeze those matrices, and use their position
        rows as the condenser-measurement projections.
        """

        matrix = np.asarray(sample_to_screen_matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                "sample_to_screen_matrix must be a finite array with shape (4, 4)."
            )
        position_map = matrix[np.asarray((0, 2)), :]
        if detector_affine_xy is not None:
            affine = np.asarray(detector_affine_xy, dtype=float)
            if affine.shape != (2, 2) or not np.all(np.isfinite(affine)):
                raise ValueError(
                    "detector_affine_xy must be a finite array with shape (2, 2)."
                )
            position_map = affine @ position_map
        return cls(label, position_map, coordinate_unit)


@dataclass(frozen=True)
class IlluminationMomentFrame:
    """A detector exposure summarized by centroid and central covariance.

    The five predicted values are listed in
    :data:`ILLUMINATION_IMAGE_MOMENT_NAMES`.  They are directly extractable
    from an unnormalised beam image; total intensity can be retained as an
    additional observation, but a lossless first-order transfer matrix cannot
    predict aperture clipping and therefore does not include it here.
    """

    label: str
    source_state: CalibratedSourceState
    projection: IlluminationProjection2D

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("An illumination-moment frame requires a non-empty label.")

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(f"{self.label}.{name}" for name in ILLUMINATION_IMAGE_MOMENT_NAMES)


@dataclass(frozen=True)
class IlluminationMomentExperiment:
    """A physical image-moment experiment for one condenser DAC setting.

    Each frame refers to a calibrated entrance-plane state and a calibrated
    downstream projection.  Predictions are moments of detector images, not
    synthetic entries copied directly out of a transfer matrix.
    """

    frames: tuple[IlluminationMomentFrame, ...]

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("At least one illumination-moment frame is required.")
        labels = [frame.label for frame in frames]
        if len(set(labels)) != len(labels):
            raise ValueError("Illumination-moment frame labels must be unique.")
        object.__setattr__(self, "frames", frames)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(
            name for frame in self.frames for name in frame.observable_names
        )


@dataclass(frozen=True)
class AcquisitionState:
    """One microscope lens setting in an inversion acquisition plan."""

    label: str
    dac_values: tuple[float, ...]
    role: str
    frame_count: int
    perturbed_channel: str | None = None
    target_magnification: float | None = None

    def __post_init__(self) -> None:
        allowed_roles = {"nominal", "single_lens", "d_optimal", "held_out"}
        if self.role not in allowed_roles:
            raise ValueError(f"role must be one of {sorted(allowed_roles)}.")
        values = tuple(float(value) for value in self.dac_values)
        if not values or not np.all(np.isfinite(values)):
            raise ValueError("dac_values must be non-empty and finite.")
        if int(self.frame_count) != self.frame_count or self.frame_count < 1:
            raise ValueError("frame_count must be a positive integer.")
        object.__setattr__(self, "dac_values", values)
        object.__setattr__(self, "frame_count", int(self.frame_count))
        if self.target_magnification is not None:
            target = float(self.target_magnification)
            if not np.isfinite(target) or target <= 0.0:
                raise ValueError("target_magnification must be finite and positive.")
            object.__setattr__(self, "target_magnification", target)


@dataclass(frozen=True)
class AcquisitionPlan:
    """A set of optical states and its explicit raw-frame budget."""

    channel_names: tuple[str, ...]
    states: tuple[AcquisitionState, ...]
    reference_dacs: tuple[float, ...]
    position_amplitude: float = 1.0e-6
    tilt_amplitude: float = 1.0e-3

    def __post_init__(self) -> None:
        channels = tuple(str(name) for name in self.channel_names)
        references = tuple(float(value) for value in self.reference_dacs)
        states = tuple(self.states)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("channel_names must be non-empty and unique.")
        if len(references) != len(channels):
            raise ValueError("reference_dacs must have one value per channel.")
        if not np.all(np.isfinite(references)):
            raise ValueError("reference_dacs must be finite.")
        if not states:
            raise ValueError("states must be non-empty.")
        labels = [state.label for state in states]
        if len(set(labels)) != len(labels):
            raise ValueError("Acquisition-state labels must be unique.")
        for state in states:
            if len(state.dac_values) != len(channels):
                raise ValueError(
                    f"State '{state.label}' has {len(state.dac_values)} DAC values; "
                    f"expected {len(channels)}."
                )
            if (
                state.perturbed_channel is not None
                and state.perturbed_channel not in channels
            ):
                raise ValueError(
                    f"Unknown perturbed channel '{state.perturbed_channel}'."
                )
        if not np.isfinite(self.position_amplitude) or self.position_amplitude <= 0:
            raise ValueError("position_amplitude must be finite and positive.")
        if not np.isfinite(self.tilt_amplitude) or self.tilt_amplitude <= 0:
            raise ValueError("tilt_amplitude must be finite and positive.")
        object.__setattr__(self, "channel_names", channels)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "reference_dacs", references)

    def __len__(self) -> int:
        return len(self.states)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    @property
    def total_frames(self) -> int:
        return sum(state.frame_count for state in self.states)

    @property
    def training_mask(self) -> np.ndarray:
        return np.asarray([state.role != "held_out" for state in self.states])

    @property
    def held_out_mask(self) -> np.ndarray:
        return ~self.training_mask

    def control_array(self) -> np.ndarray:
        return np.asarray([state.dac_values for state in self.states], dtype=float)

    def role_counts(self) -> dict[str, int]:
        roles = ("nominal", "single_lens", "d_optimal", "held_out")
        return {role: sum(state.role == role for state in self.states) for role in roles}

    def frame_counts(self) -> dict[str, int]:
        roles = ("nominal", "single_lens", "d_optimal", "held_out")
        return {
            role: sum(
                state.frame_count for state in self.states if state.role == role
            )
            for role in roles
        }


def _design_block(
    name: str,
    values: ArrayLike | None,
    expected_rows: int,
    n_channels: int,
    default: np.ndarray,
) -> np.ndarray:
    if values is None:
        return np.asarray(default, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_rows, n_channels):
        raise ValueError(
            f"{name} must have shape ({expected_rows}, {n_channels}); "
            f"received {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def compact_30_state_plan(
    channel_names: Sequence[str] = DEFAULT_LENS_CHANNELS,
    reference_dacs: ArrayLike | None = None,
    perturbations: ArrayLike | None = None,
    position_amplitude: float = 1.0e-6,
    tilt_amplitude: float = 1.0e-3,
    nominal_dacs: ArrayLike | None = None,
    target_magnifications: ArrayLike | None = None,
    d_optimal_dacs: ArrayLike | None = None,
    held_out_dacs: ArrayLike | None = None,
) -> AcquisitionPlan:
    """Build the deterministic 30-state JEOL inversion plan.

    The plan contains exactly five nominal states, two single-channel
    perturbations for each of seven free-lens controls, eight mixed D-optimal
    states, and three held-out states.  Defaults are deterministic local designs
    around ``reference_dacs``.  After a pilot Jacobian is available, callers can
    pass selected ``d_optimal_dacs`` and real nominal microscope modes.

    Frame-count metadata uses 12 frames per nominal state, 7 per single-lens
    state, 6 per mixed-information state, and 8 per held-out state: 230 frames.
    This leaves enough frames to repeat every nominal and held-out state from
    the opposite DAC approach direction.  These are acquisition-budget
    defaults, not an assumption about detector noise.
    """

    channels = tuple(str(name) for name in channel_names)
    if len(channels) != 7:
        raise ValueError("compact_30_state_plan requires exactly seven channels.")
    n_channels = len(channels)
    references = (
        np.zeros(n_channels, dtype=float)
        if reference_dacs is None
        else np.asarray(reference_dacs, dtype=float)
    )
    if references.shape != (n_channels,) or not np.all(np.isfinite(references)):
        raise ValueError("reference_dacs must be a finite vector of length seven.")

    deltas = (
        np.ones(n_channels, dtype=float)
        if perturbations is None
        else np.asarray(perturbations, dtype=float)
    )
    if deltas.ndim == 0:
        deltas = np.full(n_channels, float(deltas))
    if deltas.shape != (n_channels,) or not np.all(np.isfinite(deltas)):
        raise ValueError("perturbations must be a finite scalar or length-seven vector.")
    if np.any(deltas <= 0.0):
        raise ValueError("perturbations must be strictly positive raw-DAC steps.")

    alternating = np.where(np.arange(n_channels) % 2 == 0, 1.0, -1.0)
    default_nominal = np.stack(
        (
            references,
            references + 0.25 * deltas,
            references - 0.25 * deltas,
            references + 0.25 * deltas * alternating,
            references - 0.25 * deltas * alternating,
        )
    )
    nominal = _design_block(
        "nominal_dacs", nominal_dacs, 5, n_channels, default_nominal
    )
    if target_magnifications is None:
        nominal_targets: tuple[float | None, ...] = (None,) * 5
    else:
        target_array = np.asarray(target_magnifications, dtype=float)
        if (
            target_array.shape != (5,)
            or not np.all(np.isfinite(target_array))
            or np.any(target_array <= 0.0)
        ):
            raise ValueError("target_magnifications must be five finite positive values.")
        nominal_targets = tuple(float(value) for value in target_array)

    hadamard = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1],
        ],
        dtype=float,
    )
    default_d_optimal = references + 0.65 * deltas * hadamard[:, :n_channels]
    mixed = _design_block(
        "d_optimal_dacs", d_optimal_dacs, 8, n_channels, default_d_optimal
    )

    held_patterns = np.asarray(
        (
            (1, 0, -1, 0, 1, 0, -1),
            (0, 1, 0, -1, 0, 1, 0),
            (1, -1, 1, -1, 1, -1, 1),
        ),
        dtype=float,
    )
    default_held_out = references + 0.43 * deltas * held_patterns
    held_out = _design_block(
        "held_out_dacs", held_out_dacs, 3, n_channels, default_held_out
    )

    states: list[AcquisitionState] = []
    for index, (values, target) in enumerate(zip(nominal, nominal_targets, strict=True)):
        states.append(
            AcquisitionState(
                label=f"nominal_{index}",
                dac_values=tuple(values),
                role="nominal",
                frame_count=12,
                target_magnification=target,
            )
        )

    for channel_index, channel in enumerate(channels):
        for sign, suffix in ((-1.0, "minus"), (1.0, "plus")):
            values = references.copy()
            values[channel_index] += sign * deltas[channel_index]
            states.append(
                AcquisitionState(
                    label=f"single_{channel}_{suffix}",
                    dac_values=tuple(values),
                    role="single_lens",
                    frame_count=7,
                    perturbed_channel=channel,
                )
            )

    for index, values in enumerate(mixed):
        states.append(
            AcquisitionState(
                label=f"d_optimal_{index}",
                dac_values=tuple(values),
                role="d_optimal",
                frame_count=6,
            )
        )

    for index, values in enumerate(held_out):
        states.append(
            AcquisitionState(
                label=f"held_out_{index}",
                dac_values=tuple(values),
                role="held_out",
                frame_count=8,
            )
        )

    return AcquisitionPlan(
        channel_names=channels,
        states=tuple(states),
        reference_dacs=tuple(references),
        position_amplitude=position_amplitude,
        tilt_amplitude=tilt_amplitude,
    )


class PositionTiltResponse(NamedTuple):
    """Four canonical 2-by-2 blocks of a first-order ray matrix."""

    position_from_position: jnp.ndarray
    position_from_tilt: jnp.ndarray
    tilt_from_position: jnp.ndarray
    tilt_from_tilt: jnp.ndarray


def extract_position_tilt_response(matrix: ArrayLike) -> PositionTiltResponse:
    """Split a 4-by-4 map into position/tilt input and output blocks."""

    matrix = jnp.asarray(matrix)
    if matrix.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4).")
    position = jnp.asarray((0, 2))
    tilt = jnp.asarray((1, 3))
    return PositionTiltResponse(
        position_from_position=matrix[jnp.ix_(position, position)],
        position_from_tilt=matrix[jnp.ix_(position, tilt)],
        tilt_from_position=matrix[jnp.ix_(tilt, position)],
        tilt_from_tilt=matrix[jnp.ix_(tilt, tilt)],
    )


def response_observable_vector(
    matrix: ArrayLike,
    include_output_tilt: bool = False,
) -> jnp.ndarray:
    """Return detector response observables in documented row-major order."""

    blocks = extract_position_tilt_response(matrix)
    values = (
        blocks.position_from_position.reshape(-1),
        blocks.position_from_tilt.reshape(-1),
    )
    if include_output_tilt:
        values += (
            blocks.tilt_from_position.reshape(-1),
            blocks.tilt_from_tilt.reshape(-1),
        )
    return jnp.concatenate(values)


def propagate_sample_state(
    matrix: ArrayLike,
    state: SamplePlaneState | ArrayLike,
) -> jnp.ndarray:
    """Propagate one sample-plane state through a 4-by-4 matrix."""

    vector = state.as_jax() if isinstance(state, SamplePlaneState) else jnp.asarray(state)
    if vector.shape != (4,):
        raise ValueError("state must have shape (4,).")
    return jnp.asarray(matrix) @ vector


def propagate_phase_space_moments(
    matrix: ArrayLike,
    mean_state: ArrayLike,
    covariance: ArrayLike,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate first and second central moments through a linear ray map."""

    matrix = jnp.asarray(matrix)
    mean = jnp.asarray(mean_state)
    covariance = jnp.asarray(covariance)
    if matrix.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4).")
    if mean.shape != (4,):
        raise ValueError("mean_state must have shape (4,).")
    if covariance.shape != (4, 4):
        raise ValueError("covariance must have shape (4, 4).")
    return matrix @ mean, matrix @ covariance @ matrix.T


def project_illumination_moments(
    sample_mean_state: ArrayLike,
    sample_covariance: ArrayLike,
    projection: IlluminationProjection2D | ArrayLike,
) -> jnp.ndarray:
    """Predict image centroid and covariance for one calibrated projection.

    The returned vector is ``(u, v, covariance_uu, covariance_uv,
    covariance_vv)``.  These quantities can be estimated, with their joint
    uncertainty, from an actual vacuum-probe or known-object probe-retrieval
    image and placed directly in a :class:`FrameObservationBlock`.
    """

    mean = jnp.asarray(sample_mean_state)
    covariance = jnp.asarray(sample_covariance)
    position_map = jnp.asarray(
        projection.position_map
        if isinstance(projection, IlluminationProjection2D)
        else projection
    )
    if mean.shape != (4,):
        raise ValueError("sample_mean_state must have shape (4,).")
    if covariance.shape != (4, 4):
        raise ValueError("sample_covariance must have shape (4, 4).")
    if position_map.shape != (2, 4):
        raise ValueError("projection must have shape (2, 4).")
    projected_mean = position_map @ mean
    projected_covariance = position_map @ covariance @ position_map.T
    return jnp.stack(
        (
            projected_mean[0],
            projected_mean[1],
            projected_covariance[0, 0],
            projected_covariance[0, 1],
            projected_covariance[1, 1],
        )
    )


def predict_illumination_frame(
    source_to_sample_matrix: ArrayLike,
    frame: IlluminationMomentFrame,
) -> jnp.ndarray:
    """Predict five image moments for one physical illumination frame."""

    mean, covariance = propagate_phase_space_moments(
        source_to_sample_matrix,
        frame.source_state.mean_state,
        frame.source_state.covariance,
    )
    return project_illumination_moments(mean, covariance, frame.projection)


def predict_illumination_experiment(
    source_to_sample_matrix: ArrayLike,
    experiment: IlluminationMomentExperiment,
) -> jnp.ndarray:
    """Predict ``(n_frames, 5)`` measured moments for one condenser state."""

    return jnp.stack(
        tuple(
            predict_illumination_frame(source_to_sample_matrix, frame)
            for frame in experiment.frames
        )
    )


def calibrated_source_basis(
    covariance: ArrayLike,
    *,
    position_step_m: float,
    tilt_step_rad: float,
    symmetric: bool = True,
) -> tuple[CalibratedSourceState, ...]:
    """Build an on-axis and Cartesian entrance-plane calibration basis.

    The returned means are independently calibrated source-plane excitations,
    not nominal gun/deflector command values.  Positive and negative states are
    included by default so image drift and intercepts can be cancelled.  If
    such an entrance-plane basis is unavailable, condenser matrices cannot be
    inferred from calibrated *specimen-plane* inputs by relabelling them.
    """

    if not np.isfinite(position_step_m) or position_step_m <= 0.0:
        raise ValueError("position_step_m must be finite and positive.")
    if not np.isfinite(tilt_step_rad) or tilt_step_rad <= 0.0:
        raise ValueError("tilt_step_rad must be finite and positive.")
    covariance = np.asarray(covariance, dtype=float)
    steps = np.asarray(
        (position_step_m, tilt_step_rad, position_step_m, tilt_step_rad),
        dtype=float,
    )
    labels = ("x", "theta_x", "y", "theta_y")
    states = [
        CalibratedSourceState(
            np.zeros(4, dtype=float), covariance, label="source_on_axis"
        )
    ]
    signs = (-1.0, 1.0) if symmetric else (1.0,)
    for index, coordinate in enumerate(labels):
        for sign in signs:
            mean = np.zeros(4, dtype=float)
            mean[index] = sign * steps[index]
            suffix = "minus" if sign < 0.0 else "plus"
            states.append(
                CalibratedSourceState(
                    mean,
                    covariance,
                    label=f"source_{coordinate}_{suffix}",
                )
            )
    return tuple(states)


def standard_illumination_projections(
    defocus_distance_m: float,
    diffraction_length_m: float,
    *,
    coupled_length_m: float | None = None,
) -> tuple[IlluminationProjection2D, ...]:
    """Return calibrated screen maps for illumination phase-space tomography.

    The basic set contains a specimen image, equal negative/positive defocus
    views, and a diffraction view.  Their covariance design has rank nine, not
    ten: ordinary round through-focus images cannot distinguish the
    antisymmetric position--angle covariance associated with canonical angular
    momentum.  Supplying ``coupled_length_m`` adds a physical mixed-quadrature
    screen ``(x + L*theta_y, y + L*theta_x)`` and makes the covariance design
    full rank.  That screen must correspond to a separately calibrated coupled
    projector state; known-object complex-probe retrieval is an experimental
    alternative when the column cannot make it.
    """

    if not np.isfinite(defocus_distance_m) or defocus_distance_m <= 0.0:
        raise ValueError("defocus_distance_m must be finite and positive.")
    if not np.isfinite(diffraction_length_m) or diffraction_length_m <= 0.0:
        raise ValueError("diffraction_length_m must be finite and positive.")
    distance = float(defocus_distance_m)
    camera_length = float(diffraction_length_m)

    def drift_projection(value: float) -> np.ndarray:
        return np.asarray(
            ((1.0, value, 0.0, 0.0), (0.0, 0.0, 1.0, value)), dtype=float
        )

    projections = [
        IlluminationProjection2D("sample_image", drift_projection(0.0)),
        IlluminationProjection2D("defocus_minus", drift_projection(-distance)),
        IlluminationProjection2D("defocus_plus", drift_projection(distance)),
        IlluminationProjection2D(
            "diffraction",
            np.asarray(
                (
                    (0.0, camera_length, 0.0, 0.0),
                    (0.0, 0.0, 0.0, camera_length),
                ),
                dtype=float,
            ),
        ),
    ]
    if coupled_length_m is not None:
        if not np.isfinite(coupled_length_m) or coupled_length_m == 0.0:
            raise ValueError("coupled_length_m must be finite and non-zero.")
        coupling = float(coupled_length_m)
        projections.append(
            IlluminationProjection2D(
                "coupled_quadrature",
                np.asarray(
                    (
                        (1.0, 0.0, 0.0, coupling),
                        (0.0, coupling, 1.0, 0.0),
                    ),
                    dtype=float,
                ),
            )
        )
    return tuple(projections)


def illumination_covariance_design_matrix(
    projections: Sequence[IlluminationProjection2D | ArrayLike],
) -> np.ndarray:
    """Map the ten unique 4-D beam covariances to image covariances.

    The result has three rows per projection (``uu, uv, vv``) and columns in
    :data:`PHASE_SPACE_COVARIANCE_NAMES` order.  Its numerical rank is the
    number of second-moment combinations the proposed image set can observe.
    """

    projections = tuple(projections)
    if not projections:
        raise ValueError("At least one illumination projection is required.")
    upper_indices = tuple(
        (row, column) for row in range(4) for column in range(row, 4)
    )
    rows: list[np.ndarray] = []
    for projection in projections:
        position_map = np.asarray(
            projection.position_map
            if isinstance(projection, IlluminationProjection2D)
            else projection,
            dtype=float,
        )
        if position_map.shape != (2, 4) or not np.all(np.isfinite(position_map)):
            raise ValueError("Every projection must be a finite (2, 4) array.")
        for output_row, output_column in ((0, 0), (0, 1), (1, 1)):
            coefficients = []
            for state_row, state_column in upper_indices:
                coefficient = (
                    position_map[output_row, state_row]
                    * position_map[output_column, state_column]
                )
                if state_row != state_column:
                    coefficient += (
                        position_map[output_row, state_column]
                        * position_map[output_column, state_row]
                    )
                coefficients.append(coefficient)
            rows.append(np.asarray(coefficients, dtype=float))
    return np.stack(rows)


def compact_illumination_moment_experiment(
    source_states: Sequence[CalibratedSourceState],
    projections: Sequence[IlluminationProjection2D],
    *,
    centroid_projection_labels: Sequence[str] = ("sample_image", "diffraction"),
) -> IlluminationMomentExperiment:
    """Use all projections on-axis and two projections for each basis state.

    With the states from :func:`calibrated_source_basis`, on-axis images supply
    beam-envelope tomography while specimen and diffraction images of the
    non-zero basis states measure the position and angle response.  The default
    symmetric basis requires 21 detector frames per selected condenser DAC
    setting; the positive-only basis requires 13.  This explicit budget avoids
    counting inaccessible transfer-matrix rows as experimental data.
    """

    sources = tuple(source_states)
    projections = tuple(projections)
    if not sources or not projections:
        raise ValueError("source_states and projections must both be non-empty.")
    if sources[0].label != "source_on_axis":
        raise ValueError("The first source state must be labelled 'source_on_axis'.")
    projection_by_label = {projection.label: projection for projection in projections}
    if len(projection_by_label) != len(projections):
        raise ValueError("Illumination projection labels must be unique.")
    centroid_labels = tuple(centroid_projection_labels)
    if len(set(centroid_labels)) != len(centroid_labels):
        raise ValueError("centroid_projection_labels must be unique.")
    missing = tuple(label for label in centroid_labels if label not in projection_by_label)
    if missing:
        raise ValueError(f"Unknown centroid projection labels: {missing}.")

    frames = [
        IlluminationMomentFrame(
            f"{sources[0].label}.{projection.label}", sources[0], projection
        )
        for projection in projections
    ]
    for source in sources[1:]:
        for label in centroid_labels:
            projection = projection_by_label[label]
            frames.append(
                IlluminationMomentFrame(
                    f"{source.label}.{projection.label}", source, projection
                )
            )
    return IlluminationMomentExperiment(tuple(frames))


def predict_acquisition_plan(
    plan: AcquisitionPlan,
    matrix_function: MatrixFunction,
    include_output_tilt: bool = False,
) -> jnp.ndarray:
    """Predict response observables for every optical state in a plan.

    ``matrix_function`` receives a length-seven raw-DAC vector and must return
    a 4-by-4 sample-to-detector matrix.  The output has shape ``(n_states, 8)``
    or ``(n_states, 16)`` when output-angle responses are requested.
    """

    controls = jnp.asarray(plan.control_array())
    matrices = jax.vmap(matrix_function)(controls)
    return jax.vmap(
        lambda matrix: response_observable_vector(matrix, include_output_tilt)
    )(matrices)


@dataclass(frozen=True)
class InversionDataset:
    """Measured response observables and uncertainties for an acquisition plan."""

    plan: AcquisitionPlan
    values: ArrayLike
    standard_deviations: ArrayLike = 1.0
    observable_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.plan.n_states:
            raise ValueError(
                "values must have shape (plan.n_states, n_observables)."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("values must contain only finite measurements.")
        try:
            deviations = np.broadcast_to(
                np.asarray(self.standard_deviations, dtype=float), values.shape
            ).copy()
        except ValueError as error:
            raise ValueError(
                "standard_deviations must broadcast to values.shape."
            ) from error
        if not np.all(np.isfinite(deviations)) or np.any(deviations <= 0.0):
            raise ValueError("standard_deviations must be finite and positive.")

        names = tuple(self.observable_names)
        if not names:
            if values.shape[1] == len(POSITION_TILT_OBSERVABLE_NAMES):
                names = POSITION_TILT_OBSERVABLE_NAMES
            elif values.shape[1] == len(FULL_RESPONSE_OBSERVABLE_NAMES):
                names = FULL_RESPONSE_OBSERVABLE_NAMES
            else:
                names = tuple(f"observable_{i}" for i in range(values.shape[1]))
        if len(names) != values.shape[1] or len(set(names)) != len(names):
            raise ValueError("observable_names must be unique and match values.shape[1].")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "standard_deviations", deviations)
        object.__setattr__(self, "observable_names", names)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def whitened_residuals(
        self,
        predicted: ArrayLike,
        include_held_out: bool = False,
    ) -> np.ndarray:
        predicted = np.asarray(predicted, dtype=float)
        if predicted.shape != self.values.shape:
            raise ValueError("predicted must have the same shape as values.")
        mask = np.ones(self.plan.n_states, dtype=bool)
        if not include_held_out:
            mask = self.plan.training_mask
        return (
            (predicted[mask] - self.values[mask])
            / self.standard_deviations[mask]
        ).reshape(-1)


@dataclass(frozen=True)
class FrameObservationBlock:
    """A heterogeneous observable vector and covariance for one detector frame."""

    frame_id: int
    optical_state_index: int
    values: ArrayLike
    covariance: ArrayLike
    observable_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if self.frame_id < 0 or self.optical_state_index < 0:
            raise ValueError("Frame and optical-state indices must be non-negative.")
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Frame values must be a finite non-empty vector.")
        if covariance.shape != (values.size, values.size):
            raise ValueError("Frame covariance must be square and match its values.")
        if not np.all(np.isfinite(covariance)) or not np.allclose(
            covariance, covariance.T, rtol=1e-10, atol=1e-15
        ):
            raise ValueError("Frame covariance must be finite and symmetric.")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("Frame covariance must be positive definite.") from error
        names = tuple(self.observable_names) or tuple(
            f"observable_{index}" for index in range(values.size)
        )
        if len(names) != values.size or len(set(names)) != len(names):
            raise ValueError("Frame observable names must be unique and match values.")
        values = values.copy()
        covariance = covariance.copy()
        values.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "observable_names", names)


@dataclass(frozen=True)
class HeterogeneousInversionDataset:
    """Variable-size, covariance-aware observable blocks for individual frames."""

    plan: AcquisitionPlan
    blocks: tuple[FrameObservationBlock, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if not blocks:
            raise ValueError("At least one frame observation block is required.")
        frame_ids = [block.frame_id for block in blocks]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Frame observation IDs must be unique.")
        if any(block.optical_state_index >= self.plan.n_states for block in blocks):
            raise ValueError("A frame observation references an unknown optical state.")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_observations(self) -> int:
        return sum(block.values.size for block in self.blocks)

    def jax_whitened_residuals(
        self,
        predicted_blocks: Sequence[ArrayLike],
        *,
        include_held_out: bool = False,
    ) -> jnp.ndarray:
        """Whiten variable blocks using their full declared covariances."""

        if len(predicted_blocks) != len(self.blocks):
            raise ValueError("predicted_blocks must contain one vector per frame block.")
        residuals = []
        for prediction, block in zip(predicted_blocks, self.blocks, strict=True):
            if (
                not include_held_out
                and self.plan.states[block.optical_state_index].role == "held_out"
            ):
                continue
            predicted = jnp.asarray(prediction)
            if predicted.shape != block.values.shape:
                raise ValueError(
                    f"Prediction for frame {block.frame_id} has shape {predicted.shape}; "
                    f"expected {block.values.shape}."
                )
            inverse_cholesky = np.linalg.inv(np.linalg.cholesky(block.covariance))
            residuals.append(
                jnp.asarray(inverse_cholesky)
                @ (predicted - jnp.asarray(block.values))
            )
        if not residuals:
            return jnp.empty((0,))
        return jnp.concatenate(residuals)


@dataclass(frozen=True)
class JacobianDiagnostics:
    """Numerical identifiability diagnostics for a scaled Jacobian."""

    jacobian: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    rank: int
    nullity: int
    condition_number: float
    smallest_to_largest: float
    threshold: float


def jax_jacobian(
    function: Callable[[jnp.ndarray], ArrayLike],
    parameters: ArrayLike,
    mode: str = "forward",
) -> jnp.ndarray:
    """Differentiate and flatten an arbitrary vector-valued JAX prediction."""

    flattened = lambda value: jnp.ravel(function(value))
    parameters = jnp.asarray(parameters)
    if mode == "forward":
        return jax.jacfwd(flattened)(parameters)
    if mode == "reverse":
        return jax.jacrev(flattened)(parameters)
    raise ValueError("mode must be 'forward' or 'reverse'.")


def jacobian_svd(
    jacobian: ArrayLike,
    rtol: float | None = None,
) -> JacobianDiagnostics:
    """Compute SVD rank, nullity, condition, and right singular vectors."""

    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a 2-D array.")
    _, singular_values, right_vectors = np.linalg.svd(jacobian, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    if rtol is None:
        rtol = np.finfo(float).eps * max(jacobian.shape)
    if rtol < 0.0:
        raise ValueError("rtol must be non-negative.")
    threshold = float(rtol * largest)
    rank = int(np.sum(singular_values > threshold))
    n_parameters = jacobian.shape[1]
    nullity = n_parameters - rank
    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    if rank < n_parameters:
        ratio = 0.0
        condition = np.inf
    else:
        ratio = smallest / largest if largest > 0.0 else 0.0
        condition = largest / smallest if smallest > 0.0 else np.inf
    return JacobianDiagnostics(
        jacobian=jacobian,
        singular_values=singular_values,
        right_singular_vectors=right_vectors,
        rank=rank,
        nullity=nullity,
        condition_number=float(condition),
        smallest_to_largest=float(ratio),
        threshold=threshold,
    )


@dataclass(frozen=True)
class IdentifiableParameterSubspace:
    """Scaled local parameter directions resolved by an observable Jacobian.

    Columns of ``active_basis_scaled`` and ``null_basis_scaled`` are orthonormal
    in dimensionless coordinates ``(parameters-reference_parameters)/scales``.
    Their physical counterparts already include ``parameter_scales``.  Thus an
    active coordinate vector is embedded without ever pretending that a mixed
    singular direction is one independently measured lens parameter.
    """

    reference_parameters: np.ndarray
    parameter_scales: np.ndarray
    parameter_names: tuple[str, ...]
    active_basis_scaled: np.ndarray
    null_basis_scaled: np.ndarray
    active_basis_physical: np.ndarray
    null_basis_physical: np.ndarray
    singular_values: np.ndarray
    rank: int
    threshold: float
    rank_rtol: float

    @property
    def n_parameters(self) -> int:
        return int(self.reference_parameters.size)

    @property
    def nullity(self) -> int:
        return int(self.null_basis_scaled.shape[1])

    def parameters_from_active(self, coordinates: ArrayLike) -> jnp.ndarray:
        """Embed dimensionless active coordinates into physical parameters."""

        values = jnp.asarray(coordinates)
        if values.shape != (self.rank,):
            raise ValueError(f"coordinates must have shape ({self.rank},).")
        return jnp.asarray(self.reference_parameters) + jnp.asarray(
            self.active_basis_physical
        ) @ values


def scaled_identifiable_subspace(
    jacobian: ArrayLike,
    reference_parameters: ArrayLike,
    parameter_scales: ArrayLike,
    *,
    parameter_names: Sequence[str] = (),
    rank_rtol: float,
) -> IdentifiableParameterSubspace:
    """Resolve active and null directions after explicit parameter scaling.

    ``jacobian`` must already describe the residual vector used by the fit,
    including whitening by the declared experimental covariance.  The explicit
    ``rank_rtol`` is intentionally required: numerical machine rank is not a
    defensible experimental identifiability threshold.
    """

    matrix = np.asarray(jacobian, dtype=float)
    reference = np.asarray(reference_parameters, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array.")
    if reference.shape != (matrix.shape[1],) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_parameters must be finite and match jacobian columns.")
    if scales.shape != reference.shape or not np.all(np.isfinite(scales)) or np.any(
        scales <= 0.0
    ):
        raise ValueError("parameter_scales must be finite, positive, and match parameters.")
    tolerance = float(rank_rtol)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("rank_rtol must be finite and lie strictly between zero and one.")
    names = tuple(parameter_names) or tuple(
        f"parameter_{index}" for index in range(reference.size)
    )
    if len(names) != reference.size or len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique and match parameters.")

    diagnostics = jacobian_svd(matrix * scales[None, :], rtol=tolerance)
    active_scaled = diagnostics.right_singular_vectors[: diagnostics.rank, :].T
    null_scaled = diagnostics.right_singular_vectors[diagnostics.rank :, :].T
    active_physical = scales[:, None] * active_scaled
    null_physical = scales[:, None] * null_scaled
    return IdentifiableParameterSubspace(
        reference_parameters=reference.copy(),
        parameter_scales=scales.copy(),
        parameter_names=names,
        active_basis_scaled=active_scaled,
        null_basis_scaled=null_scaled,
        active_basis_physical=active_physical,
        null_basis_physical=null_physical,
        singular_values=diagnostics.singular_values,
        rank=diagnostics.rank,
        threshold=diagnostics.threshold,
        rank_rtol=tolerance,
    )


def fisher_information(
    jacobian: ArrayLike,
    prior_precision: ArrayLike | None = None,
) -> np.ndarray:
    """Return ``J.T @ J`` with an optional prior precision matrix."""

    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim == 1:
        jacobian = jacobian[None, :]
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be one- or two-dimensional.")
    information = jacobian.T @ jacobian
    if prior_precision is not None:
        prior = np.asarray(prior_precision, dtype=float)
        if prior.ndim == 0:
            prior = float(prior) * np.eye(information.shape[0])
        if prior.shape != information.shape:
            raise ValueError("prior_precision has the wrong shape.")
        information = information + prior
    return information


def _prior_precision_spectrum(
    prior_precision: ArrayLike | None,
    n_parameters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a numerically cleaned eigendecomposition of a PSD prior.

    Precision matrices are positive semidefinite by definition.  Decomposing a
    scaled, symmetrised copy avoids losing a small regularizer merely because a
    Jacobian contribution is many orders of magnitude larger.  Tiny negative
    eigenvalues at the round-off level are clipped; a materially indefinite
    matrix is rejected.
    """

    if prior_precision is None:
        return np.zeros(n_parameters), np.eye(n_parameters)

    prior = np.asarray(prior_precision, dtype=float)
    if prior.ndim == 0:
        prior = float(prior) * np.eye(n_parameters)
    if prior.shape != (n_parameters, n_parameters):
        raise ValueError("prior_precision has the wrong shape.")
    if not np.all(np.isfinite(prior)):
        raise ValueError("prior_precision must contain only finite values.")

    scale = float(np.max(np.abs(prior), initial=0.0))
    if scale == 0.0:
        return np.zeros(n_parameters), np.eye(n_parameters)

    symmetry_tolerance = 64.0 * np.finfo(float).eps * max(n_parameters, 1) * scale
    if np.max(np.abs(prior - prior.T), initial=0.0) > symmetry_tolerance:
        raise ValueError("prior_precision must be symmetric.")

    eigenvalues_scaled, eigenvectors = np.linalg.eigh(
        (prior + prior.T) / (2.0 * scale)
    )
    negative_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(n_parameters, 1)
        * max(float(np.max(np.abs(eigenvalues_scaled), initial=0.0)), 1.0)
    )
    if float(np.min(eigenvalues_scaled, initial=0.0)) < -negative_tolerance:
        raise ValueError("prior_precision must be positive semidefinite.")
    eigenvalues = np.maximum(eigenvalues_scaled, 0.0) * scale
    return eigenvalues, eigenvectors


def _stable_psd_logdet(
    jacobian: np.ndarray,
    prior_eigenvalues: np.ndarray,
    prior_eigenvectors: np.ndarray,
    ridge: float,
) -> float:
    """Evaluate ``log det(J.T J + P + ridge I)`` without a Gram determinant.

    For a positive-definite baseline ``B = P + ridge I``, the matrix
    determinant lemma gives

    ``log det(B) + log det(I + (J B**-1/2).T (J B**-1/2))``.

    The latter term is evaluated from singular values, so rank-deficient
    Jacobians never acquire a negative determinant through cancellation.  A
    scalar rescaling before the SVD prevents overflow and is exactly removed in
    log space.  With a singular baseline (possible only when ``ridge == 0``),
    the same calculation falls back to the singular values of a stacked factor
    of ``J.T J + P``.
    """

    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim == 1:
        jacobian = jacobian[None, :]
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be one- or two-dimensional.")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("jacobian must contain only finite values.")

    n_parameters = jacobian.shape[1]
    if n_parameters == 0:
        return 0.0

    baseline_eigenvalues = prior_eigenvalues + ridge
    if np.all(baseline_eigenvalues > 0.0):
        logdet_baseline = float(np.sum(np.log(baseline_eigenvalues)))
        rotated = jacobian @ prior_eigenvectors
        whitened = rotated / np.sqrt(baseline_eigenvalues)[None, :]
        scale = float(np.max(np.abs(whitened), initial=0.0))
        if scale == 0.0:
            return logdet_baseline
        singular_values = np.linalg.svd(whitened / scale, compute_uv=False)
        if singular_values.size == 0:
            return logdet_baseline
        threshold = (
            np.finfo(float).eps
            * max(whitened.shape)
            * float(singular_values[0])
        )
        resolved = singular_values[singular_values > threshold]
        log_singular_values = np.log(resolved) + np.log(scale)
        increments = np.logaddexp(0.0, 2.0 * log_singular_values)
        return logdet_baseline + float(np.sum(increments))

    positive = prior_eigenvalues > 0.0
    prior_factor = (
        np.sqrt(prior_eigenvalues[positive])[:, None]
        * prior_eigenvectors[:, positive].T
    )
    factors = np.concatenate((jacobian, prior_factor), axis=0)
    scale = float(np.max(np.abs(factors), initial=0.0))
    if scale == 0.0:
        return -np.inf
    singular_values = np.linalg.svd(factors / scale, compute_uv=False)
    threshold = (
        np.finfo(float).eps * max(factors.shape) * float(singular_values[0])
        if singular_values.size
        else 0.0
    )
    resolved = singular_values[singular_values > threshold]
    if resolved.size < n_parameters:
        return -np.inf
    return float(2.0 * (np.sum(np.log(resolved)) + n_parameters * np.log(scale)))


def d_optimal_logdet(
    jacobian: ArrayLike,
    prior_precision: ArrayLike | None = None,
    ridge: float = 1.0e-12,
) -> float:
    """Return a stable regularized D-optimal Fisher log-determinant.

    The score is evaluated from a whitened Jacobian rather than by forming a
    potentially ill-conditioned Gram determinant.  Consequently it remains
    finite for rank-deficient Jacobians whenever the prior or ``ridge`` makes
    the information matrix positive definite.
    """

    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative.")
    jacobian_array = np.asarray(jacobian, dtype=float)
    if jacobian_array.ndim == 1:
        jacobian_array = jacobian_array[None, :]
    if jacobian_array.ndim != 2:
        raise ValueError("jacobian must be one- or two-dimensional.")
    prior_eigenvalues, prior_eigenvectors = _prior_precision_spectrum(
        prior_precision, jacobian_array.shape[1]
    )
    return _stable_psd_logdet(
        jacobian_array,
        prior_eigenvalues,
        prior_eigenvectors,
        float(ridge),
    )


@dataclass(frozen=True)
class DOptimalSelection:
    """Result of greedy selection from candidate Jacobian blocks."""

    indices: tuple[int, ...]
    logdet_history: tuple[float, ...]
    information: np.ndarray


def greedy_d_optimal_selection(
    candidate_jacobians: ArrayLike,
    n_select: int,
    prior_precision: ArrayLike | None = None,
    ridge: float = 1.0e-12,
) -> DOptimalSelection:
    """Greedily select candidate experiments by Fisher-information logdet.

    Candidate shape is ``(n_candidates, n_observables, n_parameters)``.  A
    two-dimensional input is treated as one scalar observable per candidate.
    Ties are resolved by the lower candidate index, making selection deterministic.
    """

    candidates = np.asarray(candidate_jacobians, dtype=float)
    if candidates.ndim == 2:
        candidates = candidates[:, None, :]
    if candidates.ndim != 3:
        raise ValueError("candidate_jacobians must be a 2-D or 3-D array.")
    n_candidates, _, n_parameters = candidates.shape
    if int(n_select) != n_select or not 0 < n_select <= n_candidates:
        raise ValueError("n_select must be between one and n_candidates.")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative.")

    prior_eigenvalues, prior_eigenvectors = _prior_precision_spectrum(
        prior_precision, n_parameters
    )
    prior = (
        prior_eigenvectors
        @ np.diag(prior_eigenvalues)
        @ prior_eigenvectors.T
    )
    information = prior + ridge * np.eye(n_parameters)

    selected: list[int] = []
    selected_blocks: list[np.ndarray] = []
    history: list[float] = []
    remaining = set(range(n_candidates))
    for _ in range(int(n_select)):
        best_index = -1
        best_score = -np.inf
        for index in sorted(remaining):
            block = candidates[index]
            trial_jacobian = np.concatenate((*selected_blocks, block), axis=0)
            score = _stable_psd_logdet(
                trial_jacobian,
                prior_eigenvalues,
                prior_eigenvectors,
                float(ridge),
            )
            if best_index < 0 or score > best_score:
                best_index = index
                best_score = score
        selected.append(best_index)
        history.append(best_score)
        block = candidates[best_index]
        selected_blocks.append(block)
        information = information + block.T @ block
        remaining.remove(best_index)

    return DOptimalSelection(tuple(selected), tuple(history), information)


@dataclass(frozen=True)
class InversionResult:
    """Best result and local uncertainty diagnostics from multistart fitting."""

    parameters: np.ndarray
    parameter_names: tuple[str, ...]
    success: bool
    cost: float
    residuals: np.ndarray
    jacobian: np.ndarray
    singular_values: np.ndarray
    covariance: np.ndarray
    identifiable_covariance: np.ndarray
    parameter_standard_errors: np.ndarray
    null_space_basis: np.ndarray
    jacobian_rank: int
    degrees_of_freedom: int
    message: str
    nfev: int
    njev: int | None
    start_parameters: np.ndarray
    start_costs: np.ndarray
    solution_parameters: np.ndarray
    start_success: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=bool)
    )
    start_gradients: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    start_active_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=int)
    )

    @property
    def weighted_rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))

    @property
    def parameter_mapping(self) -> dict[str, float]:
        return dict(zip(self.parameter_names, self.parameters, strict=True))


@dataclass(frozen=True)
class EquivalentSolutionFamily:
    """Local physical-parameter directions left unconstrained by the data."""

    parameters: np.ndarray
    parameter_names: tuple[str, ...]
    basis: np.ndarray
    singular_values: np.ndarray
    rank: int
    threshold: float

    @property
    def nullity(self) -> int:
        return int(self.basis.shape[1])

    def displaced(self, coordinates: ArrayLike) -> np.ndarray:
        """Map null-family coordinates to physical parameter vectors."""

        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape[-1] != self.nullity:
            raise ValueError(f"coordinates must end in dimension {self.nullity}.")
        return self.parameters + np.einsum("pn,...n->...p", self.basis, coordinates)


def equivalent_solution_family(
    jacobian: ArrayLike,
    parameters: ArrayLike,
    *,
    parameter_names: Sequence[str] = (),
    parameter_scales: ArrayLike | None = None,
    rtol: float | None = None,
) -> EquivalentSolutionFamily:
    """Return the scaled local null space of an observable Jacobian."""

    jacobian = np.asarray(jacobian, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    if jacobian.ndim != 2 or parameters.shape != (jacobian.shape[1],):
        raise ValueError("jacobian must be (m,n) and parameters must have shape (n,).")
    names = tuple(parameter_names) or tuple(
        f"parameter_{index}" for index in range(parameters.size)
    )
    if len(names) != parameters.size or len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique and match parameters.")
    if parameter_scales is None:
        scales = np.maximum(np.abs(parameters), 1.0)
    else:
        scales = np.broadcast_to(np.asarray(parameter_scales, dtype=float), parameters.shape)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("parameter_scales must be finite and positive.")
    scaled = jacobian * scales[None, :]
    diagnostics = jacobian_svd(scaled, rtol=rtol)
    null_vectors_scaled = diagnostics.right_singular_vectors[diagnostics.rank :, :].T
    physical_basis = scales[:, None] * null_vectors_scaled
    return EquivalentSolutionFamily(
        parameters=parameters.copy(),
        parameter_names=names,
        basis=physical_basis,
        singular_values=diagnostics.singular_values,
        rank=diagnostics.rank,
        threshold=diagnostics.threshold,
    )


@dataclass(frozen=True)
class EquivalentSolutions:
    """Distinct multistart optima with statistically equivalent costs."""

    parameters: np.ndarray
    costs: np.ndarray
    distances_from_best: np.ndarray


@dataclass(frozen=True)
class MultistartBasin:
    """One sampled basin among statistically competitive multistart fits.

    The representative is the lowest-cost member of the basin.  Parameter
    distances and ``scaled_radius`` are measured after division by the caller's
    parameter scales.
    """

    label: int
    parameter_names: tuple[str, ...]
    representative_start_index: int
    member_start_indices: tuple[int, ...]
    representative_parameters: np.ndarray
    representative_cost: float
    maximum_delta_cost: float
    scaled_radius: float
    representative_bound_hits: tuple[str, ...]
    any_bound_hits: tuple[str, ...]
    representative_gradient_norm: float

    @property
    def size(self) -> int:
        return len(self.member_start_indices)

    @property
    def representative_parameter_mapping(self) -> dict[str, float]:
        """Return the representative coordinates keyed by physical name."""

        return dict(
            zip(
                self.parameter_names,
                self.representative_parameters,
                strict=True,
            )
        )


@dataclass(frozen=True)
class MultistartBasinDiagnostics:
    """Cost-gated basin diagnostics for one multistart optimization.

    Only finite, successful starts within ``delta_cost_threshold`` of the best
    finite successful cost are clustered.  A successful but appreciably worse
    local minimum is labelled ``"high_cost"`` and therefore cannot by itself be
    mistaken for an equivalent solution or a sampled uniqueness failure.
    """

    parameter_names: tuple[str, ...]
    parameter_scales: np.ndarray
    delta_cost_threshold: float
    scaled_distance_threshold: float
    best_start_index: int | None
    best_cost: float
    classifications: tuple[str, ...]
    finite_successful_mask: np.ndarray
    delta_costs: np.ndarray
    basin_labels: np.ndarray
    bound_status: np.ndarray
    scaled_gradient_norms: np.ndarray
    basins: tuple[MultistartBasin, ...]

    @property
    def near_best_start_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.classifications) == "near_best")

    @property
    def high_cost_start_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.classifications) == "high_cost")

    @property
    def n_near_best_basins(self) -> int:
        return len(self.basins)

    @property
    def single_sampled_near_best_basin(self) -> bool:
        """Whether all sampled near-best solutions occupy one cluster.

        This is deliberately not called a proof of uniqueness: multistart
        sampling cannot exclude an unsampled basin.
        """

        return bool(self.basins) and len(self.basins) == 1


def equivalent_solutions(
    result: InversionResult,
    *,
    delta_cost: float = 1.0,
    scaled_distance_min: float = 1.0e-3,
    parameter_scales: ArrayLike | None = None,
) -> EquivalentSolutions:
    """Select distinct converged multistart solutions near the best cost."""

    if delta_cost < 0.0 or scaled_distance_min < 0.0:
        raise ValueError("delta_cost and scaled_distance_min must be non-negative.")
    candidates = np.asarray(result.solution_parameters, dtype=float)
    costs = np.asarray(result.start_costs, dtype=float)
    success = np.asarray(result.start_success, dtype=bool)
    if success.size == 0:
        # Compatibility with results serialized before per-start convergence
        # was retained.  Newly produced results always carry this mask.
        success = np.ones(costs.shape, dtype=bool)
    if success.shape != costs.shape:
        raise ValueError("result.start_success must match result.start_costs.")
    if parameter_scales is None:
        scales = np.maximum(np.abs(result.parameters), 1.0)
    else:
        scales = np.broadcast_to(np.asarray(parameter_scales, dtype=float), result.parameters.shape)
        if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
            raise ValueError("parameter_scales must be finite and positive.")
    order = np.argsort(costs)
    accepted_parameters: list[np.ndarray] = []
    accepted_costs: list[float] = []
    for index in order:
        if not success[index]:
            continue
        if costs[index] > result.cost + delta_cost:
            continue
        candidate = candidates[index]
        if any(
            np.linalg.norm((candidate - previous) / scales) < scaled_distance_min
            for previous in accepted_parameters
        ):
            continue
        accepted_parameters.append(candidate)
        accepted_costs.append(float(costs[index]))
    parameters = np.asarray(accepted_parameters, dtype=float).reshape((-1, result.parameters.size))
    distances = (
        np.linalg.norm((parameters - result.parameters) / scales, axis=1)
        if parameters.size
        else np.empty((0,), dtype=float)
    )
    return EquivalentSolutions(parameters, np.asarray(accepted_costs), distances)


def diagnose_multistart_basins(
    result: InversionResult,
    *,
    delta_cost: float = 1.0,
    scaled_distance_threshold: float = 1.0e-2,
    parameter_scales: ArrayLike | None = None,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    bound_rtol: float = 1.0e-7,
    bound_atol: float = 0.0,
) -> MultistartBasinDiagnostics:
    """Classify multistart outcomes and cluster only competitive solutions.

    Parameters
    ----------
    result
        Result returned by :func:`fit_least_squares` (directly or through one
        of the dataset fitting helpers).
    delta_cost
        Maximum robust least-squares cost increase above the best finite,
        successful start admitted to the near-best set.
    scaled_distance_threshold
        Maximum Euclidean distance from a basin's lowest-cost representative,
        after division by ``parameter_scales``.  Starts are visited in
        increasing-cost order, so the clustering is deterministic and a chain
        of intermediate points cannot merge two distant representatives.
    parameter_scales
        Positive physical scale for every named parameter.  Supplying the same
        scales used by the fit is strongly recommended for mixed-unit models.
    bounds
        Optional physical lower and upper bounds.  When supplied, coordinates
        within ``bound_atol + bound_rtol * parameter_scale`` of a bound are
        reported in addition to SciPy's retained active-bound masks.

    Notes
    -----
    ``single_sampled_near_best_basin`` means exactly what it says: it is
    evidence from the sampled starts, not a mathematical proof of global
    uniqueness.  Successful finite starts outside the explicit cost gate are
    labelled ``"high_cost"`` and do not count as equivalent basins.
    """

    if not np.isfinite(delta_cost) or delta_cost < 0.0:
        raise ValueError("delta_cost must be finite and non-negative.")
    if (
        not np.isfinite(scaled_distance_threshold)
        or scaled_distance_threshold < 0.0
    ):
        raise ValueError(
            "scaled_distance_threshold must be finite and non-negative."
        )
    if not np.isfinite(bound_rtol) or bound_rtol < 0.0:
        raise ValueError("bound_rtol must be finite and non-negative.")
    if not np.isfinite(bound_atol) or bound_atol < 0.0:
        raise ValueError("bound_atol must be finite and non-negative.")

    costs = np.asarray(result.start_costs, dtype=float)
    solutions = np.asarray(result.solution_parameters, dtype=float)
    if costs.ndim != 1:
        raise ValueError("result.start_costs must be one-dimensional.")
    n_starts = costs.size
    n_parameters = np.asarray(result.parameters).size
    if solutions.shape != (n_starts, n_parameters):
        raise ValueError(
            "result.solution_parameters must have shape "
            "(n_starts, n_parameters)."
        )

    names = tuple(result.parameter_names)
    if len(names) != n_parameters or len(set(names)) != len(names):
        raise ValueError(
            "result.parameter_names must be unique and match n_parameters."
        )
    if parameter_scales is None:
        scales = np.maximum(np.abs(np.asarray(result.parameters, dtype=float)), 1.0)
    else:
        scales = np.broadcast_to(
            np.asarray(parameter_scales, dtype=float), (n_parameters,)
        ).copy()
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("parameter_scales must be finite and positive.")

    success = np.asarray(result.start_success, dtype=bool)
    if success.size == 0:
        # Compatibility with results serialized before per-start convergence
        # was retained.
        success = np.ones((n_starts,), dtype=bool)
    if success.shape != (n_starts,):
        raise ValueError("result.start_success must match result.start_costs.")
    finite = np.isfinite(costs) & np.all(np.isfinite(solutions), axis=1)
    finite_successful = success & finite

    classifications = np.full((n_starts,), "nonfinite", dtype="<U10")
    classifications[~success] = "failed"
    classifications[finite_successful] = "high_cost"
    delta_costs = np.full((n_starts,), np.nan, dtype=float)
    eligible = np.flatnonzero(finite_successful)
    if eligible.size:
        best_start_index: int | None = int(
            eligible[np.argmin(costs[eligible])]
        )
        best_cost = float(costs[best_start_index])
        delta_costs[finite_successful] = costs[finite_successful] - best_cost
        near_best = finite_successful & (delta_costs <= float(delta_cost))
        classifications[near_best] = "near_best"
    else:
        best_start_index = None
        best_cost = float("nan")
        near_best = np.zeros((n_starts,), dtype=bool)

    active_mask = np.asarray(result.start_active_mask, dtype=int)
    if active_mask.size == 0:
        bound_status = np.zeros((n_starts, n_parameters), dtype=int)
    elif active_mask.shape == (n_starts, n_parameters):
        bound_status = np.sign(active_mask).astype(int, copy=True)
    else:
        raise ValueError(
            "result.start_active_mask must be empty or have shape "
            "(n_starts, n_parameters)."
        )
    if bounds is not None:
        lower, upper = _normalize_bounds(bounds, n_parameters)
        tolerance = float(bound_atol) + float(bound_rtol) * scales
        lower_hit = (
            np.isfinite(lower)[None, :]
            & np.isfinite(solutions)
            & (np.abs(solutions - lower[None, :]) <= tolerance[None, :])
        )
        upper_hit = (
            np.isfinite(upper)[None, :]
            & np.isfinite(solutions)
            & (np.abs(solutions - upper[None, :]) <= tolerance[None, :])
        )
        inferred_status = np.where(lower_hit, -1, np.where(upper_hit, 1, 0))
        bound_status = np.where(bound_status != 0, bound_status, inferred_status)

    gradients = np.asarray(result.start_gradients, dtype=float)
    if gradients.size == 0:
        scaled_gradient_norms = np.full((n_starts,), np.nan, dtype=float)
    elif gradients.shape == (n_starts, n_parameters):
        # If u = p / scale, then d(cost)/d(u) = d(cost)/d(p) * scale.
        scaled_gradient_norms = np.max(
            np.abs(gradients * scales[None, :]), axis=1
        )
    else:
        raise ValueError(
            "result.start_gradients must be empty or have shape "
            "(n_starts, n_parameters)."
        )

    basin_labels = np.full((n_starts,), -1, dtype=int)
    representatives: list[int] = []
    members: list[list[int]] = []
    near_indices = sorted(
        np.flatnonzero(near_best).tolist(), key=lambda index: (costs[index], index)
    )
    for index in near_indices:
        if representatives:
            distances = np.asarray(
                [
                    np.linalg.norm(
                        (solutions[index] - solutions[representative]) / scales
                    )
                    for representative in representatives
                ],
                dtype=float,
            )
            closest = int(np.argmin(distances))
        else:
            distances = np.empty((0,), dtype=float)
            closest = -1
        if distances.size and distances[closest] <= scaled_distance_threshold:
            label = closest
            members[label].append(index)
        else:
            label = len(representatives)
            representatives.append(index)
            members.append([index])
        basin_labels[index] = label

    def hit_names(rows: Sequence[int]) -> tuple[str, ...]:
        statuses = bound_status[np.asarray(tuple(rows), dtype=int)]
        if statuses.ndim == 1:
            statuses = statuses[None, :]
        hits: list[str] = []
        for parameter_index, name in enumerate(names):
            if np.any(statuses[:, parameter_index] < 0):
                hits.append(f"{name}:lower")
            if np.any(statuses[:, parameter_index] > 0):
                hits.append(f"{name}:upper")
        return tuple(hits)

    basins: list[MultistartBasin] = []
    for label, (representative, basin_members) in enumerate(
        zip(representatives, members, strict=True)
    ):
        member_array = np.asarray(basin_members, dtype=int)
        radii = np.linalg.norm(
            (solutions[member_array] - solutions[representative])
            / scales[None, :],
            axis=1,
        )
        basins.append(
            MultistartBasin(
                label=label,
                parameter_names=names,
                representative_start_index=representative,
                member_start_indices=tuple(basin_members),
                representative_parameters=solutions[representative].copy(),
                representative_cost=float(costs[representative]),
                maximum_delta_cost=float(np.max(delta_costs[member_array])),
                scaled_radius=float(np.max(radii)),
                representative_bound_hits=hit_names((representative,)),
                any_bound_hits=hit_names(basin_members),
                representative_gradient_norm=float(
                    scaled_gradient_norms[representative]
                ),
            )
        )

    return MultistartBasinDiagnostics(
        parameter_names=names,
        parameter_scales=scales,
        delta_cost_threshold=float(delta_cost),
        scaled_distance_threshold=float(scaled_distance_threshold),
        best_start_index=best_start_index,
        best_cost=best_cost,
        classifications=tuple(str(value) for value in classifications),
        finite_successful_mask=finite_successful,
        delta_costs=delta_costs,
        basin_labels=basin_labels,
        bound_status=bound_status,
        scaled_gradient_norms=scaled_gradient_norms,
        basins=tuple(basins),
    )


@dataclass(frozen=True)
class ProfileLikelihood:
    """Conditional optimum and cost for a fixed-parameter grid."""

    parameter_index: int
    parameter_name: str
    grid: np.ndarray
    costs: np.ndarray
    conditional_parameters: np.ndarray
    success: np.ndarray

    @property
    def delta_cost(self) -> np.ndarray:
        return self.costs - np.nanmin(self.costs)


def profile_likelihood(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    optimum: ArrayLike,
    parameter_index: int,
    grid: ArrayLike,
    *,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    parameter_names: Sequence[str] = (),
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int | None = None,
) -> ProfileLikelihood:
    """Compute a one-dimensional profile while refitting all other parameters."""

    optimum = np.asarray(optimum, dtype=float)
    values = np.asarray(grid, dtype=float)
    if optimum.ndim != 1 or optimum.size < 2 or not np.all(np.isfinite(optimum)):
        raise ValueError("optimum must be a finite vector with at least two parameters.")
    index = int(parameter_index)
    if not 0 <= index < optimum.size:
        raise IndexError("parameter_index is outside optimum.")
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("grid must be a finite vector with at least two values.")
    names = tuple(parameter_names) or tuple(f"parameter_{i}" for i in range(optimum.size))
    if len(names) != optimum.size:
        raise ValueError("parameter_names must match optimum.")
    lower, upper = _normalize_bounds(bounds, optimum.size)
    if np.any(values < lower[index]) or np.any(values > upper[index]):
        raise ValueError("profile grid lies outside the selected parameter bounds.")
    free = np.asarray([i for i in range(optimum.size) if i != index], dtype=int)
    conditional = []
    costs = []
    successes = []
    previous = optimum.copy()
    for fixed_value in values:
        template = previous.copy()
        template[index] = fixed_value

        def reduced_residual(free_parameters: jnp.ndarray) -> jnp.ndarray:
            full = jnp.asarray(template).at[jnp.asarray(free)].set(free_parameters)
            return jnp.ravel(residual_function(full))

        fit = fit_least_squares(
            reduced_residual,
            template[free],
            bounds=(lower[free], upper[free]),
            parameter_names=tuple(names[i] for i in free),
            loss=loss,
            f_scale=f_scale,
            multistart=1,
            max_nfev=max_nfev,
        )
        full = template.copy()
        full[free] = fit.parameters
        previous = full
        conditional.append(full)
        costs.append(fit.cost)
        successes.append(fit.success)
    return ProfileLikelihood(
        parameter_index=index,
        parameter_name=names[index],
        grid=values.copy(),
        costs=np.asarray(costs),
        conditional_parameters=np.asarray(conditional),
        success=np.asarray(successes, dtype=bool),
    )


def _normalize_bounds(
    bounds: tuple[ArrayLike, ArrayLike],
    n_parameters: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (n_parameters,)).copy()
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (n_parameters,)).copy()
    if np.any(lower >= upper):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    return lower, upper


def _multistart_points(
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    multistart: int,
    seed: int,
    jitter_scale: float,
) -> np.ndarray:
    if int(multistart) != multistart or multistart < 1:
        raise ValueError("multistart must be a positive integer.")
    if jitter_scale <= 0.0:
        raise ValueError("jitter_scale must be positive.")
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError("x0 must lie within bounds.")
    starts = [x0.copy()]
    if multistart == 1:
        return np.stack(starts)

    generator = np.random.default_rng(seed)
    natural_scale = np.maximum(np.abs(x0), 1.0) * jitter_scale
    for _ in range(1, int(multistart)):
        draw = x0 + generator.normal(size=x0.shape) * natural_scale
        both_finite = np.isfinite(lower) & np.isfinite(upper)
        if np.any(both_finite):
            draw[both_finite] = generator.uniform(
                lower[both_finite], upper[both_finite]
            )
        draw = np.maximum(draw, lower)
        draw = np.minimum(draw, upper)
        starts.append(draw)
    return np.stack(starts)


def _local_covariance(
    jacobian: np.ndarray,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return rank-aware local covariance and explicit null information.

    The finite ``identifiable_covariance`` is the Moore--Penrose covariance of
    identifiable combinations.  ``covariance`` marks rows/columns touched by
    exact local null directions as NaN, and their marginal standard errors as
    infinity; a null direction must never masquerade as zero uncertainty.
    """

    jacobian = np.asarray(jacobian, dtype=float)
    n_observations, n_parameters = jacobian.shape
    _left, singular_values, right_t = np.linalg.svd(jacobian, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = np.finfo(float).eps * max(jacobian.shape) * largest
    rank = int(np.sum(singular_values > threshold))
    degrees_of_freedom = max(n_observations - rank, 1)
    residual_variance = 2.0 * float(cost) / degrees_of_freedom
    if rank:
        identifiable_vectors = right_t[:rank].T
        inverse_squares = 1.0 / singular_values[:rank] ** 2
        identifiable_covariance = (
            identifiable_vectors
            @ np.diag(inverse_squares)
            @ identifiable_vectors.T
            * residual_variance
        )
    else:
        identifiable_covariance = np.zeros((n_parameters, n_parameters), dtype=float)
    null_basis = right_t[rank:].T
    covariance = identifiable_covariance.copy()
    standard_errors = np.sqrt(np.maximum(np.diag(identifiable_covariance), 0.0))
    if null_basis.size:
        null_participation = np.sum(null_basis**2, axis=1)
        affected = null_participation > 100.0 * np.finfo(float).eps
        covariance[affected, :] = np.nan
        covariance[:, affected] = np.nan
        standard_errors[affected] = np.inf
    return (
        covariance,
        identifiable_covariance,
        standard_errors,
        null_basis,
        rank,
        degrees_of_freedom,
    )


def fit_least_squares(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    x0: ArrayLike,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    parameter_names: Sequence[str] = (),
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    multistart: int = 1,
    seed: int = 0,
    jitter_scale: float = 0.25,
    jacobian_mode: str = "forward",
    max_nfev: int | None = None,
) -> InversionResult:
    """Fit a JAX residual function with robust bounded multistart least squares."""

    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1 or x0.size == 0 or not np.all(np.isfinite(x0)):
        raise ValueError("x0 must be a non-empty finite 1-D array.")
    lower, upper = _normalize_bounds(bounds, x0.size)
    starts = _multistart_points(
        x0, lower, upper, multistart, seed, jitter_scale
    )
    names = tuple(parameter_names) or tuple(f"parameter_{i}" for i in range(x0.size))
    if len(names) != x0.size or len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique and match len(x0).")
    if jacobian_mode not in {"forward", "reverse"}:
        raise ValueError("jacobian_mode must be 'forward' or 'reverse'.")

    flattened = lambda value: jnp.ravel(residual_function(value))
    derivative = jax.jacfwd(flattened) if jacobian_mode == "forward" else jax.jacrev(flattened)

    def scipy_residual(value: np.ndarray) -> np.ndarray:
        result = np.asarray(flattened(jnp.asarray(value)), dtype=float)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("residual_function returned non-finite values.")
        return result

    def scipy_jacobian(value: np.ndarray) -> np.ndarray:
        result = np.asarray(derivative(jnp.asarray(value)), dtype=float)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("JAX Jacobian contains non-finite values.")
        return result

    fits = []
    start_costs = []
    for start in starts:
        fit = least_squares(
            scipy_residual,
            start,
            jac=scipy_jacobian,
            bounds=(lower, upper),
            loss=loss,
            f_scale=f_scale,
            x_scale="jac",
            max_nfev=max_nfev,
        )
        fits.append(fit)
        start_costs.append(float(fit.cost))

    start_success = np.asarray([fit.success for fit in fits], dtype=bool)
    eligible = np.flatnonzero(start_success)
    best_index = int(
        eligible[np.argmin(np.asarray(start_costs)[eligible])]
        if eligible.size
        else np.argmin(start_costs)
    )
    best = fits[best_index]
    final_jacobian = scipy_jacobian(best.x)
    diagnostics = jacobian_svd(final_jacobian)
    # SciPy's returned Jacobian includes the robust-loss curvature scaling;
    # retain the raw JAX Jacobian for identifiability and use the scaled one for
    # local uncertainty.
    (
        covariance,
        identifiable_covariance,
        parameter_standard_errors,
        null_space_basis,
        covariance_rank,
        degrees_of_freedom,
    ) = _local_covariance(np.asarray(best.jac, dtype=float), float(best.cost))
    return InversionResult(
        parameters=np.asarray(best.x, dtype=float),
        parameter_names=names,
        success=bool(best.success),
        cost=float(best.cost),
        residuals=np.asarray(best.fun, dtype=float),
        jacobian=final_jacobian,
        singular_values=diagnostics.singular_values,
        covariance=covariance,
        identifiable_covariance=identifiable_covariance,
        parameter_standard_errors=parameter_standard_errors,
        null_space_basis=null_space_basis,
        jacobian_rank=covariance_rank,
        degrees_of_freedom=degrees_of_freedom,
        message=str(best.message),
        nfev=int(best.nfev),
        njev=None if best.njev is None else int(best.njev),
        start_parameters=starts,
        start_costs=np.asarray(start_costs),
        solution_parameters=np.asarray([fit.x for fit in fits], dtype=float),
        start_success=start_success,
        start_gradients=np.asarray([fit.grad for fit in fits], dtype=float),
        start_active_mask=np.asarray([fit.active_mask for fit in fits], dtype=int),
    )


def _conservative_linear_coordinate_bounds(
    base_parameters: np.ndarray,
    physical_basis: np.ndarray,
    full_bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return an axis-aligned coordinate box wholly inside physical bounds.

    A rotated SVD basis turns physical box bounds into a polytope.  SciPy's
    least-squares interface accepts only coordinate-wise bounds, so each free
    coordinate receives an equal share of every physical parameter's available
    positive and negative margin.  This is conservative, but every point the
    optimizer can evaluate is guaranteed to satisfy the original bounds.
    """

    base = np.asarray(base_parameters, dtype=float)
    basis = np.asarray(physical_basis, dtype=float)
    lower, upper = (np.asarray(value, dtype=float) for value in full_bounds)
    if basis.ndim != 2 or basis.shape[0] != base.size:
        raise ValueError("physical_basis must have shape (n_parameters, n_coordinates).")
    if lower.shape != base.shape or upper.shape != base.shape:
        raise ValueError("full_bounds must match base_parameters.")
    tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(base))
    if np.any(base < lower - tolerance) or np.any(base > upper + tolerance):
        raise ValueError("base_parameters lie outside the physical bounds.")
    count = basis.shape[1]
    coordinate_lower = np.full(count, -np.inf)
    coordinate_upper = np.full(count, np.inf)
    if count == 0:
        return coordinate_lower, coordinate_upper
    negative_margin = (lower - base) / count
    positive_margin = (upper - base) / count
    for coordinate in range(count):
        coefficients = basis[:, coordinate]
        positive = coefficients > 0.0
        negative = coefficients < 0.0
        if np.any(positive):
            coordinate_lower[coordinate] = max(
                coordinate_lower[coordinate],
                float(np.max(negative_margin[positive] / coefficients[positive])),
            )
            coordinate_upper[coordinate] = min(
                coordinate_upper[coordinate],
                float(np.min(positive_margin[positive] / coefficients[positive])),
            )
        if np.any(negative):
            coordinate_lower[coordinate] = max(
                coordinate_lower[coordinate],
                float(np.max(positive_margin[negative] / coefficients[negative])),
            )
            coordinate_upper[coordinate] = min(
                coordinate_upper[coordinate],
                float(np.min(negative_margin[negative] / coefficients[negative])),
            )
    if np.any(coordinate_lower >= coordinate_upper):
        raise ValueError(
            "The physical bounds contain no non-zero conservative box in the "
            "selected coordinate basis. Move the reference into the bounds or "
            "supply wider physical bounds."
        )
    return coordinate_lower, coordinate_upper


def _absolute_fisher_covariance(
    jacobian: np.ndarray,
    *,
    rank_rtol: float,
) -> tuple[np.ndarray, JacobianDiagnostics, np.ndarray]:
    """Return ``pinv(J.T@J)`` without estimating an extra noise scale."""

    diagnostics = jacobian_svd(jacobian, rtol=rank_rtol)
    rank = diagnostics.rank
    vectors = diagnostics.right_singular_vectors[:rank, :].T
    covariance = (
        vectors @ np.diag(1.0 / diagnostics.singular_values[:rank] ** 2) @ vectors.T
        if rank
        else np.zeros((jacobian.shape[1], jacobian.shape[1]), dtype=float)
    )
    null_basis = diagnostics.right_singular_vectors[rank:, :].T
    return covariance, diagnostics, null_basis


@dataclass(frozen=True)
class IdentifiableSubspaceFit:
    """Bounded fit performed only in experimentally resolved coordinates."""

    subspace: IdentifiableParameterSubspace
    active_fit: InversionResult
    parameters: np.ndarray
    active_coordinates: np.ndarray
    active_coordinate_bounds: tuple[np.ndarray, np.ndarray]
    active_fisher_covariance: np.ndarray
    identifiable_covariance: np.ndarray
    projected_parameter_standard_errors: np.ndarray
    fit_singular_values: np.ndarray
    fit_rank: int
    fit_threshold: float
    fit_null_basis_active: np.ndarray

    @property
    def success(self) -> bool:
        """Whether optimization converged and retained the declared active rank."""

        return bool(self.active_fit.success and self.fit_rank == self.subspace.rank)

    @property
    def cost(self) -> float:
        return self.active_fit.cost

    @property
    def residuals(self) -> np.ndarray:
        return self.active_fit.residuals

    @property
    def weighted_rmse(self) -> float:
        return self.active_fit.weighted_rmse


def fit_identifiable_subspace(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    reference_parameters: ArrayLike,
    parameter_scales: ArrayLike,
    *,
    rank_rtol: float,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    parameter_names: Sequence[str] = (),
    active_x0: ArrayLike | None = None,
    jacobian_mode: str = "forward",
    **fit_options: Any,
) -> IdentifiableSubspaceFit:
    """Fit only the practical-rank SVD coordinates of a whitened residual.

    The local subspace is formed at ``reference_parameters`` from the Jacobian
    with columns scaled by ``parameter_scales``.  Optimization then uses
    ``p = p0 + diag(scales) @ V_active @ eta``.  Physical box bounds are mapped
    to a conservative active-coordinate box, ensuring that no optimizer
    evaluation can move a lens outside the caller's declared limits.

    The returned covariance is the absolute-noise Fisher covariance.  Because
    the residual must already be whitened by known experimental covariance, it
    is *not* multiplied by the observed residual variance.
    """

    reference = np.asarray(reference_parameters, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if reference.ndim != 1 or reference.size == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("reference_parameters must be a non-empty finite vector.")
    if scales.shape != reference.shape or not np.all(np.isfinite(scales)) or np.any(
        scales <= 0.0
    ):
        raise ValueError("parameter_scales must be finite, positive, and match parameters.")
    if jacobian_mode not in {"forward", "reverse"}:
        raise ValueError("jacobian_mode must be 'forward' or 'reverse'.")
    derivative = jax.jacfwd if jacobian_mode == "forward" else jax.jacrev
    reference_jacobian = np.asarray(
        derivative(lambda value: jnp.ravel(residual_function(value)))(jnp.asarray(reference)),
        dtype=float,
    )
    subspace = scaled_identifiable_subspace(
        reference_jacobian,
        reference,
        scales,
        parameter_names=parameter_names,
        rank_rtol=rank_rtol,
    )
    if subspace.rank == 0:
        raise ValueError("The residual Jacobian has no direction above rank_rtol.")

    full_lower, full_upper = _normalize_bounds(bounds, reference.size)
    active_bounds = _conservative_linear_coordinate_bounds(
        reference, subspace.active_basis_physical, (full_lower, full_upper)
    )
    initial = (
        np.zeros(subspace.rank, dtype=float)
        if active_x0 is None
        else np.asarray(active_x0, dtype=float)
    )
    if initial.shape != (subspace.rank,) or not np.all(np.isfinite(initial)):
        raise ValueError(f"active_x0 must be finite with shape ({subspace.rank},).")
    if np.any(initial < active_bounds[0]) or np.any(initial > active_bounds[1]):
        raise ValueError("active_x0 lies outside the conservative physical bounds.")

    def active_residual(coordinates: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(residual_function(subspace.parameters_from_active(coordinates)))

    options = dict(fit_options)
    if "bounds" in options or "parameter_names" in options or "jacobian_mode" in options:
        raise TypeError(
            "bounds, parameter_names, and jacobian_mode are explicit "
            "fit_identifiable_subspace arguments."
        )
    # A whitened physical likelihood has an absolute noise scale.  Robust loss
    # remains opt-in, but the statistically matched default is ordinary linear
    # least squares.
    options.setdefault("loss", "linear")
    active_fit = fit_least_squares(
        active_residual,
        initial,
        bounds=active_bounds,
        parameter_names=tuple(f"active_svd_{index}" for index in range(subspace.rank)),
        jacobian_mode=jacobian_mode,
        **options,
    )
    parameters = np.asarray(
        subspace.parameters_from_active(jnp.asarray(active_fit.parameters)), dtype=float
    )
    tolerance = 128.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(parameters))
    if np.any(parameters < full_lower - tolerance) or np.any(parameters > full_upper + tolerance):
        raise RuntimeError("Embedded optimizer result escaped the declared physical bounds.")

    fisher_covariance, fit_diagnostics, fit_null = _absolute_fisher_covariance(
        np.asarray(active_fit.jacobian, dtype=float), rank_rtol=float(rank_rtol)
    )
    active_covariance = fisher_covariance.copy()
    active_standard_errors = np.sqrt(
        np.maximum(np.diag(fisher_covariance), 0.0)
    )
    if fit_null.size:
        affected = np.sum(fit_null**2, axis=1) > 100.0 * np.finfo(float).eps
        active_covariance[affected, :] = np.nan
        active_covariance[:, affected] = np.nan
        active_standard_errors[affected] = np.inf
    active_fit = replace(
        active_fit,
        singular_values=fit_diagnostics.singular_values,
        covariance=active_covariance,
        identifiable_covariance=fisher_covariance,
        parameter_standard_errors=active_standard_errors,
        null_space_basis=fit_null,
        jacobian_rank=fit_diagnostics.rank,
        degrees_of_freedom=max(
            active_fit.residuals.size - fit_diagnostics.rank, 1
        ),
    )
    physical_basis = subspace.active_basis_physical
    identifiable_covariance = physical_basis @ fisher_covariance @ physical_basis.T
    projected_errors = np.sqrt(np.maximum(np.diag(identifiable_covariance), 0.0))
    return IdentifiableSubspaceFit(
        subspace=subspace,
        active_fit=active_fit,
        parameters=parameters,
        active_coordinates=np.asarray(active_fit.parameters, dtype=float),
        active_coordinate_bounds=(active_bounds[0].copy(), active_bounds[1].copy()),
        active_fisher_covariance=fisher_covariance,
        identifiable_covariance=identifiable_covariance,
        projected_parameter_standard_errors=projected_errors,
        fit_singular_values=fit_diagnostics.singular_values,
        fit_rank=fit_diagnostics.rank,
        fit_threshold=fit_diagnostics.threshold,
        fit_null_basis_active=fit_null,
    )


@dataclass(frozen=True)
class DirectionProfileLikelihood:
    """Conditional costs along one normalized direction in scaled space."""

    direction_name: str
    direction_scaled: np.ndarray
    direction_physical: np.ndarray
    grid: np.ndarray
    costs: np.ndarray
    conditional_parameters: np.ndarray
    success: np.ndarray
    orthogonal_active_basis_scaled: np.ndarray

    @property
    def delta_cost(self) -> np.ndarray:
        return self.costs - np.nanmin(self.costs)


def _least_squares_cost(residuals: np.ndarray, loss: str, f_scale: float) -> float:
    residuals = np.asarray(residuals, dtype=float)
    scale = float(f_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("f_scale must be finite and positive.")
    z = (residuals / scale) ** 2
    if loss == "linear":
        rho = z
    elif loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss == "cauchy":
        rho = np.log1p(z)
    elif loss == "arctan":
        rho = np.arctan(z)
    else:
        raise ValueError(f"Unsupported profile loss {loss!r}.")
    return float(0.5 * scale**2 * np.sum(rho))


def profile_likelihood_direction(
    residual_function: Callable[[jnp.ndarray], ArrayLike],
    fit: IdentifiableSubspaceFit,
    scaled_direction: ArrayLike,
    grid: ArrayLike,
    *,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    direction_name: str = "scaled_direction",
    loss: str = "linear",
    f_scale: float = 1.0,
    jacobian_mode: str = "forward",
    max_nfev: int | None = None,
) -> DirectionProfileLikelihood:
    """Profile a scaled physical direction while refitting active orthogonal modes.

    ``grid`` gives signed displacement along the normalized direction in scaled
    parameter units.  A direction inside the active subspace leaves ``rank-1``
    active modes to refit; a local null direction leaves all active modes free.
    Every fixed point and every conditional optimizer evaluation is checked
    against the original physical parameter bounds.
    """

    if not isinstance(fit, IdentifiableSubspaceFit):
        raise TypeError("fit must be an IdentifiableSubspaceFit.")
    direction = np.asarray(scaled_direction, dtype=float)
    if direction.shape != (fit.subspace.n_parameters,) or not np.all(
        np.isfinite(direction)
    ):
        raise ValueError("scaled_direction must be finite and match full parameters.")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("scaled_direction must be non-zero.")
    direction = direction / norm
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("grid must be a finite vector containing at least two points.")
    full_lower, full_upper = _normalize_bounds(bounds, fit.subspace.n_parameters)

    active = fit.subspace.active_basis_scaled
    active_component = active.T @ direction
    component_norm = float(np.linalg.norm(active_component))
    if component_norm <= 256.0 * np.finfo(float).eps:
        coefficient_basis = np.eye(fit.subspace.rank)
    else:
        _left, _singular, right_t = np.linalg.svd(
            active_component[None, :], full_matrices=True
        )
        coefficient_basis = right_t[1:, :].T
    orthogonal_active = active @ coefficient_basis
    physical_orthogonal = fit.subspace.parameter_scales[:, None] * orthogonal_active
    physical_direction = fit.subspace.parameter_scales * direction

    costs: list[float] = []
    conditional_parameters: list[np.ndarray] = []
    successes: list[bool] = []
    previous: np.ndarray | None = None
    for coordinate in values:
        base = np.asarray(fit.parameters + physical_direction * coordinate, dtype=float)
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(base))
        if np.any(base < full_lower - tolerance) or np.any(base > full_upper + tolerance):
            raise ValueError(
                f"Profile grid coordinate {coordinate:g} places the fixed direction "
                "outside the physical bounds."
            )
        if orthogonal_active.shape[1] == 0:
            residual = np.asarray(residual_function(jnp.asarray(base)), dtype=float).ravel()
            conditional_parameters.append(base)
            costs.append(_least_squares_cost(residual, loss, f_scale))
            successes.append(True)
            continue

        free_bounds = _conservative_linear_coordinate_bounds(
            base, physical_orthogonal, (full_lower, full_upper)
        )
        initial = np.zeros(orthogonal_active.shape[1], dtype=float)
        if previous is not None and np.all(previous >= free_bounds[0]) and np.all(
            previous <= free_bounds[1]
        ):
            initial = previous

        def conditional_residual(free_coordinates: jnp.ndarray) -> jnp.ndarray:
            parameters = jnp.asarray(base) + jnp.asarray(physical_orthogonal) @ free_coordinates
            return jnp.ravel(residual_function(parameters))

        conditional_fit = fit_least_squares(
            conditional_residual,
            initial,
            bounds=free_bounds,
            parameter_names=tuple(
                f"orthogonal_active_{index}"
                for index in range(orthogonal_active.shape[1])
            ),
            loss=loss,
            f_scale=f_scale,
            multistart=1,
            jacobian_mode=jacobian_mode,
            max_nfev=max_nfev,
        )
        previous = np.asarray(conditional_fit.parameters, dtype=float)
        parameters = base + physical_orthogonal @ previous
        conditional_parameters.append(parameters)
        costs.append(conditional_fit.cost)
        successes.append(conditional_fit.success)

    return DirectionProfileLikelihood(
        direction_name=str(direction_name),
        direction_scaled=direction,
        direction_physical=physical_direction,
        grid=values.copy(),
        costs=np.asarray(costs, dtype=float),
        conditional_parameters=np.asarray(conditional_parameters, dtype=float),
        success=np.asarray(successes, dtype=bool),
        orthogonal_active_basis_scaled=orthogonal_active,
    )


def fit_dataset(
    model_function: Callable[[jnp.ndarray], ArrayLike],
    dataset: InversionDataset,
    x0: ArrayLike,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    parameter_names: Sequence[str] = (),
    include_held_out: bool = False,
    **fit_options: Any,
) -> InversionResult:
    """Fit model predictions to a weighted :class:`InversionDataset`."""

    observed = jnp.asarray(dataset.values)
    deviations = jnp.asarray(dataset.standard_deviations)
    mask = (
        np.ones(dataset.plan.n_states, dtype=bool)
        if include_held_out
        else dataset.plan.training_mask
    )
    indices = jnp.asarray(np.flatnonzero(mask))

    def residual(parameters: jnp.ndarray) -> jnp.ndarray:
        predicted = jnp.asarray(model_function(parameters))
        if predicted.shape != observed.shape:
            raise ValueError(
                "model_function output must have the same shape as dataset.values."
            )
        return ((predicted[indices] - observed[indices]) / deviations[indices]).reshape(-1)

    return fit_least_squares(
        residual,
        x0,
        bounds=bounds,
        parameter_names=parameter_names,
        **fit_options,
    )


def fit_heterogeneous_dataset(
    model_function: Callable[[jnp.ndarray], Sequence[ArrayLike]],
    dataset: HeterogeneousInversionDataset,
    x0: ArrayLike,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    parameter_names: Sequence[str] = (),
    include_held_out: bool = False,
    **fit_options: Any,
) -> InversionResult:
    """Fit frame-specific observable blocks with their full covariances."""

    def residual(parameters: jnp.ndarray) -> jnp.ndarray:
        return dataset.jax_whitened_residuals(
            model_function(parameters), include_held_out=include_held_out
        )

    return fit_least_squares(
        residual,
        x0,
        bounds=bounds,
        parameter_names=parameter_names,
        **fit_options,
    )


__all__ = [
    "AcquisitionPlan",
    "AcquisitionState",
    "CalibratedSourceState",
    "DACHardEdgeLens",
    "DACGlaserLens",
    "DACPolynomial",
    "DACRoundSymplecticSection",
    "DACThinLens",
    "DEFAULT_LENS_CHANNELS",
    "DOptimalSelection",
    "FULL_RESPONSE_OBSERVABLE_NAMES",
    "FrameObservationBlock",
    "HeterogeneousInversionDataset",
    "ILLUMINATION_IMAGE_MOMENT_NAMES",
    "IlluminationMomentExperiment",
    "IlluminationMomentFrame",
    "IlluminationProjection2D",
    "InversionDataset",
    "InversionResult",
    "IdentifiableParameterSubspace",
    "IdentifiableSubspaceFit",
    "EquivalentSolutionFamily",
    "EquivalentSolutions",
    "MultistartBasin",
    "MultistartBasinDiagnostics",
    "ProfileLikelihood",
    "DirectionProfileLikelihood",
    "JacobianDiagnostics",
    "POSITION_TILT_OBSERVABLE_NAMES",
    "PHASE_SPACE_COVARIANCE_NAMES",
    "PositionTiltResponse",
    "STATE_ORDER",
    "SamplePlaneState",
    "canonical_symplectic_form",
    "calibrated_source_basis",
    "compact_30_state_plan",
    "compact_illumination_moment_experiment",
    "compose_matrices",
    "d_optimal_logdet",
    "dac_hard_edge_solenoid_matrix",
    "dac_thin_lens_matrix",
    "diagnose_multistart_basins",
    "drift_matrix",
    "evaluate_dac_polynomial",
    "equivalent_solution_family",
    "equivalent_solutions",
    "extract_position_tilt_response",
    "fisher_information",
    "fit_dataset",
    "fit_heterogeneous_dataset",
    "fit_identifiable_subspace",
    "fit_least_squares",
    "generic_round_symplectic_matrix",
    "glaser_solenoid_matrix",
    "greedy_d_optimal_selection",
    "hard_edge_solenoid_matrix",
    "illumination_covariance_design_matrix",
    "jax_jacobian",
    "jacobian_svd",
    "larmor_rotation_4x4",
    "mathematical_sinc",
    "normalized_profile_solenoid_matrix",
    "predict_acquisition_plan",
    "predict_illumination_experiment",
    "predict_illumination_frame",
    "profile_likelihood",
    "profile_likelihood_direction",
    "project_illumination_moments",
    "propagate_phase_space_moments",
    "propagate_sample_state",
    "response_observable_vector",
    "smooth_basis_solenoid_matrix",
    "smooth_field_solenoid_matrix",
    "scaled_identifiable_subspace",
    "standard_illumination_projections",
    "symplectic_defect",
    "thin_lens_matrix",
]
