"""Independent scaled grid-wave references for round electron-optical maps.

This module contains no Gaussian-beamlet construction, re-expansion, or
evaluation code.  It evaluates the scalar Collins diffraction integral from
sampled complex fields, using Simpson quadrature and SciPy's chirp-z
transform for arbitrary uniform input/output sampling.  A fail-closed
dispatcher handles the singular specimen-conjugate limit, a stable scaled
Fresnel/FFT factorization for high-magnification near-conjugate states, and
the general CZT route while accounting explicitly for cumulative Larmor
rotation and detector affine calibration.

Coordinates are in metres, wavelength is in metres, and fields are complex
amplitudes whose squared modulus is power density per square metre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage, signal


class GridWaveSamplingError(RuntimeError):
    """The requested independent reference is not numerically certified."""

    def __init__(self, diagnostics: "GridWaveDiagnostics"):
        self.diagnostics = diagnostics
        reasons = ", ".join(diagnostics.failure_reasons) or "unknown sampling failure"
        super().__init__(
            f"Grid-wave reference route {diagnostics.route!r} failed: {reasons}."
        )


def _uniform_coords(name: str, coords: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(coords, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional array of length >= 2.")
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    step = float(np.mean(differences))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(step))
    if not np.allclose(differences, step, rtol=2e-11, atol=tolerance):
        raise ValueError(f"{name} must be uniformly sampled.")
    return values, step


def simpson_uniform_weights(n: int, step: float) -> np.ndarray:
    """Return the current SciPy Simpson weights without constructing an identity.

    For an even number of samples this includes SciPy's Cartwright correction
    on the final interval.  The result therefore matches
    ``scipy.integrate.simpson(..., dx=step)`` while requiring O(n), not O(n²),
    storage.
    """

    if isinstance(n, (bool, np.bool_)) or int(n) != n or int(n) < 2:
        raise ValueError("n must be an integer >= 2.")
    spacing = float(step)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("step must be finite and positive.")
    n = int(n)
    weights = np.zeros(n, dtype=float)
    if n == 2:
        weights[:] = 0.5 * spacing
        return weights
    simpson_n = n if n % 2 else n - 1
    weights[0] = weights[simpson_n - 1] = spacing / 3.0
    weights[1 : simpson_n - 1 : 2] = 4.0 * spacing / 3.0
    weights[2 : simpson_n - 1 : 2] = 2.0 * spacing / 3.0
    if n % 2 == 0:
        # Uniform-spacing specialization of SciPy's Cartwright correction.
        weights[-3] -= spacing / 12.0
        weights[-2] += 2.0 * spacing / 3.0
        weights[-1] += 5.0 * spacing / 12.0
    return weights


def _abcd(name: str, values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.shape == (2, 2):
        array = array.ravel()
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite (A, B, C, D) values.")
    return tuple(float(value) for value in array)  # type: ignore[return-value]


@dataclass(frozen=True)
class RadialSymplecticCertificate:
    """Floating-point certificate for a scalar first-order ray map.

    A highly magnifying symplectic map can have ``A*D`` and ``B*C`` of order
    ``1e8`` even though their difference is exactly one.  In that regime an
    absolute test of the *evaluated* determinant confuses cancellation at the
    last one or two representable floats with a physical violation.  This
    record keeps the raw residual while separately reporting the standard
    floating-point evaluation bound used for certification.  No coefficient
    is projected or otherwise changed.
    """

    radial_abcd: tuple[float, float, float, float]
    determinant: float
    determinant_error: float
    determinant_evaluation_roundoff_bound: float
    normalized_backward_error: float
    cancellation_scale: float
    absolute_tolerance: float
    certification_basis: str
    passed: bool


def certify_radial_symplectic_map(
    radial_abcd: Sequence[float],
    *,
    absolute_tolerance: float = 1e-8,
) -> RadialSymplecticCertificate:
    """Certify ``A*D - B*C = 1`` without rejecting roundoff cancellation.

    The determinant expression contains two products and two subtractions.
    With IEEE binary64 unit roundoff ``u``, its forward evaluation error is
    bounded to first order by ``gamma_4 * (|A D| + |B C| + 1)``, where
    ``gamma_4 = 4u / (1 - 4u)``.  A map passes when its observed residual is
    within the requested physical tolerance plus that numerical bound.

    This is a backward-error certificate, not a relaxed relative determinant
    test: a well-scaled map receives essentially the original absolute gate,
    while an ill-conditioned map is admitted only when its residual is no
    larger than the loss of information inherent in evaluating its supplied
    binary64 coefficients.  Non-finite products always fail closed.
    """

    matrix = _abcd("radial_abcd", radial_abcd)
    tolerance = float(absolute_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be finite and positive.")
    A, B, C, D = matrix
    with np.errstate(over="ignore", invalid="ignore"):
        ad = float(A * D)
        bc = float(B * C)
        determinant = float(ad - bc)
        error = float(abs(determinant - 1.0))
        cancellation_scale = float(abs(ad) + abs(bc) + 1.0)
    unit_roundoff = 0.5 * np.finfo(float).eps
    gamma_four = 4.0 * unit_roundoff / (1.0 - 4.0 * unit_roundoff)
    roundoff_bound = float(gamma_four * cancellation_scale)
    finite = bool(
        np.isfinite(determinant)
        and np.isfinite(error)
        and np.isfinite(cancellation_scale)
        and np.isfinite(roundoff_bound)
    )
    normalized = (
        float(error / cancellation_scale)
        if finite and cancellation_scale > 0.0
        else float("inf")
    )
    passed = bool(finite and error <= tolerance + roundoff_bound)
    if not finite:
        basis = "failed_nonfinite_determinant_evaluation"
    elif error <= tolerance:
        basis = "absolute_determinant_tolerance"
    elif passed:
        basis = "floating_point_backward_error"
    else:
        basis = "failed_determinant_residual"
    return RadialSymplecticCertificate(
        radial_abcd=matrix,
        determinant=determinant,
        determinant_error=error,
        determinant_evaluation_roundoff_bound=roundoff_bound,
        normalized_backward_error=normalized,
        cancellation_scale=cancellation_scale,
        absolute_tolerance=tolerance,
        certification_basis=basis,
        passed=passed,
    )


def _scaled_czt_axis(
    values: np.ndarray,
    coords_in: np.ndarray,
    coords_out: np.ndarray,
    *,
    abcd: tuple[float, float, float, float],
    wavelength_m: float,
    axis: int,
    include_output_chirp: bool = True,
) -> np.ndarray:
    """Apply one absolute-prefactor Collins transform along ``axis``.

    ``include_output_chirp=False`` returns the field with the exactly known
    ``exp(i*pi*D*X**2/(lambda*B))`` carrier removed.  This is useful when the
    result must subsequently be interpolated: the carrier may be arbitrarily
    fast even though the physically relevant complex envelope is adequately
    sampled.  It is restored analytically at the final query coordinates.
    """

    A, B, _C, D = abcd
    if abs(B) <= np.finfo(float).tiny:
        raise ValueError("The scaled Collins CZT requires non-zero B.")
    coordinates_in, step_in = _uniform_coords("coords_in", coords_in)
    coordinates_out, step_out = _uniform_coords("coords_out", coords_out)
    wavelength = float(wavelength_m)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength_m must be finite and positive.")
    data = np.asarray(values, dtype=np.complex128)
    if data.shape[axis] != coordinates_in.size:
        raise ValueError("The transformed field axis does not match coords_in.")

    weights = simpson_uniform_weights(coordinates_in.size, step_in)
    shape = [1] * data.ndim
    shape[axis] = coordinates_in.size
    weighted_chirped = data * (
        weights
        * np.exp(1j * np.pi * A * coordinates_in**2 / (wavelength * B))
    ).reshape(shape)

    # x_n=x_0+n*dx and X_m=X_0+m*dX.  The two linear phase factors
    # leave a geometric exp(-i*2π*dx*dX*n*m/(lambda*B)) sequence, which is
    # exactly the transform evaluated by scipy.signal.CZT.
    n = np.arange(coordinates_in.size, dtype=float)
    input_linear = np.exp(
        -2j
        * np.pi
        * step_in
        * coordinates_out[0]
        * n
        / (wavelength * B)
    )
    weighted_chirped *= input_linear.reshape(shape)
    transform = signal.CZT(
        n=coordinates_in.size,
        m=coordinates_out.size,
        w=np.exp(-2j * np.pi * step_in * step_out / (wavelength * B)),
        a=1.0 + 0.0j,
    )
    transformed = transform(weighted_chirped, axis=axis)
    output_factor = np.exp(
        -2j
        * np.pi
        * coordinates_in[0]
        * coordinates_out
        / (wavelength * B)
    )
    if include_output_chirp:
        output_factor *= np.exp(
            1j * np.pi * D * coordinates_out**2 / (wavelength * B)
        )
    output_factor /= np.sqrt(1j * wavelength * B + 0j)
    output_shape = [1] * data.ndim
    output_shape[axis] = coordinates_out.size
    return transformed * output_factor.reshape(output_shape)


def scaled_collins_czt_1d(
    field_x: np.ndarray,
    x_in_m: np.ndarray,
    x_out_m: np.ndarray,
    *,
    abcd: Sequence[float],
    wavelength_m: float,
) -> np.ndarray:
    """Evaluate a one-dimensional Collins integral on arbitrary uniform grids."""

    field = np.asarray(field_x, dtype=np.complex128)
    if field.ndim != 1 or not np.all(np.isfinite(field)):
        raise ValueError("field_x must be a finite one-dimensional complex array.")
    return _scaled_czt_axis(
        field,
        np.asarray(x_in_m),
        np.asarray(x_out_m),
        abcd=_abcd("abcd", abcd),
        wavelength_m=wavelength_m,
        axis=0,
    )


def scaled_collins_czt_2d(
    field_yx: np.ndarray,
    x_in_m: np.ndarray,
    y_in_m: np.ndarray,
    x_out_m: np.ndarray,
    y_out_m: np.ndarray,
    *,
    abcd_x: Sequence[float],
    abcd_y: Sequence[float] | None = None,
    wavelength_m: float,
) -> np.ndarray:
    """Evaluate the separable 2D Collins integral with absolute prefactors."""

    field = np.asarray(field_yx, dtype=np.complex128)
    x_in, _ = _uniform_coords("x_in_m", x_in_m)
    y_in, _ = _uniform_coords("y_in_m", y_in_m)
    x_out, _ = _uniform_coords("x_out_m", x_out_m)
    y_out, _ = _uniform_coords("y_out_m", y_out_m)
    if field.shape != (y_in.size, x_in.size) or not np.all(np.isfinite(field)):
        raise ValueError("field_yx must be finite with shape (len(y_in_m), len(x_in_m)).")
    matrix_x = _abcd("abcd_x", abcd_x)
    matrix_y = matrix_x if abcd_y is None else _abcd("abcd_y", abcd_y)
    after_x = _scaled_czt_axis(
        field,
        x_in,
        x_out,
        abcd=matrix_x,
        wavelength_m=wavelength_m,
        axis=1,
    )
    return _scaled_czt_axis(
        after_x,
        y_in,
        y_out,
        abcd=matrix_y,
        wavelength_m=wavelength_m,
        axis=0,
    )


def _scaled_collins_czt_2d_demodulated(
    field_yx: np.ndarray,
    x_in_m: np.ndarray,
    y_in_m: np.ndarray,
    x_out_m: np.ndarray,
    y_out_m: np.ndarray,
    *,
    abcd: tuple[float, float, float, float],
    wavelength_m: float,
) -> np.ndarray:
    """Return a 2-D Collins field without its known output lens chirp.

    The omitted quadratic phase has unit modulus.  It is therefore safe for
    power integration and, unlike the full Collins field, has an output
    bandwidth bounded solely by the sampled input support.  The dispatcher
    restores the omitted phase at exact detector query positions.
    """

    field = np.asarray(field_yx, dtype=np.complex128)
    x_in, _ = _uniform_coords("x_in_m", x_in_m)
    y_in, _ = _uniform_coords("y_in_m", y_in_m)
    x_out, _ = _uniform_coords("x_out_m", x_out_m)
    y_out, _ = _uniform_coords("y_out_m", y_out_m)
    if field.shape != (y_in.size, x_in.size) or not np.all(np.isfinite(field)):
        raise ValueError("field_yx must be finite with shape (len(y_in_m), len(x_in_m)).")
    after_x = _scaled_czt_axis(
        field,
        x_in,
        x_out,
        abcd=abcd,
        wavelength_m=wavelength_m,
        axis=1,
        include_output_chirp=False,
    )
    return _scaled_czt_axis(
        after_x,
        y_in,
        y_out,
        abcd=abcd,
        wavelength_m=wavelength_m,
        axis=0,
        include_output_chirp=False,
    )


def _rotation(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=float)


def _complex_map_coordinates(
    field_yx: np.ndarray,
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    query_xy_m: np.ndarray,
    *,
    order: int = 3,
) -> np.ndarray:
    x, dx = _uniform_coords("x_coords_m", x_coords_m)
    y, dy = _uniform_coords("y_coords_m", y_coords_m)
    query = np.asarray(query_xy_m, dtype=float)
    if query.shape[-1:] != (2,) or not np.all(np.isfinite(query)):
        raise ValueError("query_xy_m must be finite and end in dimension two.")
    coordinates = np.stack(
        ((query[..., 1] - y[0]) / dy, (query[..., 0] - x[0]) / dx), axis=0
    )
    real = ndimage.map_coordinates(
        np.asarray(field_yx).real,
        coordinates,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )
    imag = ndimage.map_coordinates(
        np.asarray(field_yx).imag,
        coordinates,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )
    return real + 1j * imag


def _power(field_yx: np.ndarray, x_coords_m: np.ndarray, y_coords_m: np.ndarray) -> float:
    x, dx = _uniform_coords("x_coords_m", x_coords_m)
    y, dy = _uniform_coords("y_coords_m", y_coords_m)
    wx = simpson_uniform_weights(x.size, dx)
    wy = simpson_uniform_weights(y.size, dy)
    return float(wy @ (np.abs(field_yx) ** 2) @ wx)


def _edge_power_fraction(field_yx: np.ndarray) -> float:
    power = np.abs(field_yx) ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        return float("inf")
    mask = np.zeros(power.shape, dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return float(np.sum(power[mask]) / total)


def _kernel_phase_steps(
    x_in: np.ndarray,
    y_in: np.ndarray,
    x_out: np.ndarray,
    y_out: np.ndarray,
    abcd: tuple[float, float, float, float],
    wavelength: float,
    *,
    output_chirp_removed: bool = False,
) -> tuple[float, float]:
    A, B, _C, D = abcd
    if B == 0.0:
        return 0.0, 0.0

    def axis_steps(values_in: np.ndarray, values_out: np.ndarray) -> tuple[float, float]:
        step_in = float(np.diff(values_in).mean())
        step_out = float(np.diff(values_out).mean())
        in_mid = 0.5 * (values_in[:-1] + values_in[1:])
        out_mid = 0.5 * (values_out[:-1] + values_out[1:])
        input_step = (
            2.0
            * np.pi
            * step_in
            / (wavelength * abs(B))
            * max(
                abs(A * in_mid[0] - values_out[0]),
                abs(A * in_mid[0] - values_out[-1]),
                abs(A * in_mid[-1] - values_out[0]),
                abs(A * in_mid[-1] - values_out[-1]),
            )
        )
        if output_chirp_removed:
            # The dispatcher evaluates and interpolates the chirp-demodulated
            # CZT envelope.  Its output phase derivative is due only to the
            # bilinear -2*x*X term; D*X is an analytic carrier restored after
            # interpolation and must not be mistaken for sampled bandwidth.
            output_support = max(abs(values_in[0]), abs(values_in[-1]))
        else:
            output_support = max(
                abs(D * out_mid[0] - values_in[0]),
                abs(D * out_mid[0] - values_in[-1]),
                abs(D * out_mid[-1] - values_in[0]),
                abs(D * out_mid[-1] - values_in[-1]),
            )
        output_step = (
            2.0
            * np.pi
            * step_out
            / (wavelength * abs(B))
            * output_support
        )
        return float(input_step), float(output_step)

    x_steps = axis_steps(x_in, x_out)
    y_steps = axis_steps(y_in, y_out)
    return max(x_steps[0], y_steps[0]), max(x_steps[1], y_steps[1])


@dataclass(frozen=True)
class GridWaveOptions:
    """Fail-closed numerical tolerances for the independent reference."""

    conjugate_maximum_b_displacement_px: float = 1e-3
    czt_maximum_kernel_phase_step_rad: float = 0.8 * np.pi
    maximum_input_edge_power_fraction: float = 1e-3
    scaled_fresnel_fft_enabled: bool = True
    fft_padding_factor: float = 3.0
    fft_maximum_nyquist_shift_fraction_of_padding: float = 0.8
    fft_maximum_paraxial_angle_rad: float = 0.1
    fft_minimum_samples_per_detector_pixel: float = 1.0
    output_oversampling: float = 2.0
    interpolation_halo_pixels: int = 3
    maximum_intermediate_points: int = 16_000_000
    symplectic_determinant_tolerance: float = 1e-8
    require_power_conservation: bool = False
    maximum_power_relative_error: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.scaled_fresnel_fft_enabled, (bool, np.bool_)):
            raise TypeError("scaled_fresnel_fft_enabled must be Boolean.")
        object.__setattr__(
            self, "scaled_fresnel_fft_enabled", bool(self.scaled_fresnel_fft_enabled)
        )
        positive = (
            "conjugate_maximum_b_displacement_px",
            "czt_maximum_kernel_phase_step_rad",
            "maximum_input_edge_power_fraction",
            "fft_padding_factor",
            "fft_maximum_nyquist_shift_fraction_of_padding",
            "fft_maximum_paraxial_angle_rad",
            "fft_minimum_samples_per_detector_pixel",
            "output_oversampling",
            "symplectic_determinant_tolerance",
            "maximum_power_relative_error",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        for name in ("interpolation_halo_pixels", "maximum_intermediate_points"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
            object.__setattr__(self, name, int(value))
        if self.maximum_intermediate_points < 4:
            raise ValueError("maximum_intermediate_points must be at least four.")
        if self.fft_padding_factor < 1.0:
            raise ValueError("fft_padding_factor must be at least one.")


@dataclass(frozen=True)
class GridWaveDiagnostics:
    route: str
    radial_abcd: tuple[float, float, float, float]
    cumulative_rotation_rad: float
    symplectic_determinant_error: float
    b_displacement_px: float
    angular_bandlimit_rad: float
    effective_defocus_m: float | None
    fft_padded_shape_yx: tuple[int, int] | None
    fft_nyquist_shift_xy_m: tuple[float, float] | None
    fft_padding_margin_xy_m: tuple[float, float] | None
    fft_samples_per_detector_pixel: float | None
    maximum_output_lens_phase_step_rad: float | None
    maximum_input_kernel_phase_step_rad: float | None
    maximum_output_kernel_phase_step_rad: float | None
    input_edge_power_fraction: float
    input_power: float
    output_power: float | None
    power_relative_error: float | None
    intermediate_shape_yx: tuple[int, int]
    failure_reasons: tuple[str, ...]
    passed: bool
    alternative_route_rejections: tuple[str, ...] = ()
    symplectic_determinant_roundoff_bound: float = 0.0
    symplectic_normalized_backward_error: float = 0.0
    symplectic_certification_basis: str = "absolute_determinant_tolerance"


@dataclass(frozen=True)
class GridWaveReferenceResult:
    field_yx: np.ndarray
    intensity_yx: np.ndarray
    intermediate_field_yx: np.ndarray
    intermediate_x_coords_m: np.ndarray
    intermediate_y_coords_m: np.ndarray
    diagnostics: GridWaveDiagnostics


@dataclass(frozen=True)
class GridWaveConvergenceDiagnostics:
    """Noise-relative convergence of two absolute-intensity references.

    The arrays must already have the same physical units, normally detector
    electrons.  No flux or shape normalisation is performed, so a
    refinement-dependent change in total intensity remains visible.
    """

    detector_rms_in_noise_sigma: float
    observable_residuals_in_sigma: tuple[tuple[str, float], ...]
    maximum_detector_rms_in_noise_sigma: float
    maximum_observable_residual_in_sigma: float
    failure_reasons: tuple[str, ...]
    passed: bool

    @property
    def observable_residual_map(self) -> dict[str, float]:
        return dict(self.observable_residuals_in_sigma)


def poisson_read_total_standard_error(
    expected_counts_yx: np.ndarray,
    *,
    read_noise_e: float = 0.0,
) -> float:
    """Return the exact standard error of an unnormalised detector sum.

    Pixels are independent Poisson variables with means ``expected_counts_yx``
    followed by independent additive read noise.  This analytic variance is
    preferable to a small bootstrap for the linear total-count observable;
    nonlinear image geometry can still use bootstrap covariance.
    """

    expected = np.asarray(expected_counts_yx, dtype=float)
    read_noise = float(read_noise_e)
    if expected.ndim != 2 or not np.all(np.isfinite(expected)):
        raise ValueError("expected_counts_yx must be a finite two-dimensional array.")
    if np.any(expected < 0.0):
        raise ValueError("expected_counts_yx must be non-negative.")
    if not np.isfinite(read_noise) or read_noise < 0.0:
        raise ValueError("read_noise_e must be finite and non-negative.")
    variance = float(np.sum(expected) + expected.size * read_noise**2)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("The total-count variance must be finite and positive.")
    return float(np.sqrt(variance))


def assess_grid_reference_convergence(
    previous_intensity_yx: np.ndarray,
    refined_intensity_yx: np.ndarray,
    detector_noise_sigma_yx: np.ndarray,
    *,
    observable_residuals_in_sigma: Mapping[str, float] | None = None,
    maximum_detector_rms_in_noise_sigma: float = 0.01,
    maximum_observable_residual_in_sigma: float = 0.01,
) -> GridWaveConvergenceDiagnostics:
    """Certify successive references while preserving absolute counts.

    Observable residuals are supplied by the caller because their extraction
    is experiment-specific. They must be standardized magnitudes computed
    from the same pair of absolute-intensity images.
    """

    previous = np.asarray(previous_intensity_yx, dtype=float)
    refined = np.asarray(refined_intensity_yx, dtype=float)
    noise = np.asarray(detector_noise_sigma_yx, dtype=float)
    if previous.ndim != 2 or refined.shape != previous.shape:
        raise ValueError(
            "previous_intensity_yx and refined_intensity_yx must be matching "
            "two-dimensional arrays."
        )
    if noise.shape != previous.shape:
        raise ValueError("detector_noise_sigma_yx must match the intensity shape.")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(refined)):
        raise ValueError("Grid-reference intensities must be finite.")
    if not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
        raise ValueError("detector_noise_sigma_yx must be finite and strictly positive.")

    detector_limit = float(maximum_detector_rms_in_noise_sigma)
    observable_limit = float(maximum_observable_residual_in_sigma)
    if not np.isfinite(detector_limit) or detector_limit <= 0.0:
        raise ValueError(
            "maximum_detector_rms_in_noise_sigma must be finite and positive."
        )
    if not np.isfinite(observable_limit) or observable_limit <= 0.0:
        raise ValueError(
            "maximum_observable_residual_in_sigma must be finite and positive."
        )

    residual_items: list[tuple[str, float]] = []
    for raw_name, raw_value in (observable_residuals_in_sigma or {}).items():
        name = str(raw_name)
        if not name:
            raise ValueError("Observable residual names must be non-empty.")
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "Observable residuals in sigma must be finite and non-negative."
            )
        residual_items.append((name, value))
    residual_items.sort(key=lambda item: item[0])

    detector_rms = float(np.sqrt(np.mean(((previous - refined) / noise) ** 2)))
    failures: list[str] = []
    if detector_rms > detector_limit:
        failures.append("detector_rms")
    failures.extend(
        f"observable:{name}"
        for name, value in residual_items
        if value > observable_limit
    )
    return GridWaveConvergenceDiagnostics(
        detector_rms_in_noise_sigma=detector_rms,
        observable_residuals_in_sigma=tuple(residual_items),
        maximum_detector_rms_in_noise_sigma=detector_limit,
        maximum_observable_residual_in_sigma=observable_limit,
        failure_reasons=tuple(failures),
        passed=not failures,
    )


def _detector_query_coordinates(
    detector_x_coords_m: np.ndarray,
    detector_y_coords_m: np.ndarray,
    detector_affine_xy: np.ndarray,
    detector_centre_xy_m: Sequence[float],
    cumulative_rotation_rad: float,
) -> np.ndarray:
    x_detector, _ = _uniform_coords("detector_x_coords_m", detector_x_coords_m)
    y_detector, _ = _uniform_coords("detector_y_coords_m", detector_y_coords_m)
    affine = np.asarray(detector_affine_xy, dtype=float)
    if affine.shape != (2, 2) or not np.all(np.isfinite(affine)):
        raise ValueError("detector_affine_xy must be a finite 2x2 matrix.")
    if abs(float(np.linalg.det(affine))) <= np.finfo(float).eps:
        raise ValueError("detector_affine_xy must be invertible.")
    centre = np.asarray(detector_centre_xy_m, dtype=float)
    if centre.shape != (2,) or not np.all(np.isfinite(centre)):
        raise ValueError("detector_centre_xy_m must contain two finite values.")
    xx, yy = np.meshgrid(x_detector, y_detector, indexing="xy")
    detector_xy = np.stack((xx, yy), axis=-1)
    column_xy = centre + np.einsum(
        "ij,...j->...i", np.linalg.inv(affine), detector_xy - centre
    )
    return np.einsum(
        "ij,...j->...i", _rotation(-float(cumulative_rotation_rad)), column_xy
    )


def _intermediate_grid(
    query_xy: np.ndarray,
    detector_x: np.ndarray,
    detector_y: np.ndarray,
    affine: np.ndarray,
    options: GridWaveOptions,
    *,
    maximum_step_xy_m: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    detector_step = np.asarray(
        (float(np.diff(detector_x).mean()), float(np.diff(detector_y).mean()))
    )
    optical_step_map = np.linalg.inv(affine) @ np.diag(detector_step)
    base_step = (
        float(np.min(np.linalg.svd(optical_step_map, compute_uv=False)))
        / options.output_oversampling
    )
    if maximum_step_xy_m is None:
        steps = np.asarray((base_step, base_step), dtype=float)
    else:
        maximum_steps = np.asarray(maximum_step_xy_m, dtype=float)
        if (
            maximum_steps.shape != (2,)
            or not np.all(np.isfinite(maximum_steps))
            or np.any(maximum_steps <= 0.0)
        ):
            raise ValueError("maximum_step_xy_m must contain two finite positive values.")
        steps = np.minimum(base_step, maximum_steps)

    def axis(values: np.ndarray, step: float) -> np.ndarray:
        halo = options.interpolation_halo_pixels * step
        lower = float(np.min(values)) - halo
        upper = float(np.max(values)) + halo
        count = max(2, int(np.ceil((upper - lower) / step)) + 1)
        return lower + step * np.arange(count, dtype=float)

    x_out = axis(query_xy[..., 0], float(steps[0]))
    y_out = axis(query_xy[..., 1], float(steps[1]))
    if x_out.size * y_out.size > options.maximum_intermediate_points:
        raise ValueError(
            "The rotated/affine intermediate grid requires "
            f"{x_out.size * y_out.size:,} points, exceeding "
            f"maximum_intermediate_points={options.maximum_intermediate_points:,}."
        )
    return x_out, y_out


def _fft_padding_geometry(
    shape_yx: tuple[int, int],
    dx: float,
    dy: float,
    options: GridWaveOptions,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int], tuple[float, float]]:
    target_y = max(shape_yx[0], int(np.ceil(options.fft_padding_factor * shape_yx[0])))
    target_x = max(shape_yx[1], int(np.ceil(options.fft_padding_factor * shape_yx[1])))
    extra_y = target_y - shape_yx[0]
    extra_x = target_x - shape_yx[1]
    pads = (
        (extra_y // 2, extra_y - extra_y // 2),
        (extra_x // 2, extra_x - extra_x // 2),
    )
    margins = (
        min(pads[1]) * dx,
        min(pads[0]) * dy,
    )
    return pads, (target_y, target_x), margins


def _fft_fresnel_plane(
    field_yx: np.ndarray,
    x_in: np.ndarray,
    y_in: np.ndarray,
    *,
    wavelength: float,
    effective_defocus: float,
    pads: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paraxial free-space propagation on a zero-padded native grid."""

    dx = float(np.diff(x_in).mean())
    dy = float(np.diff(y_in).mean())
    padded = np.pad(field_yx, pads, mode="constant")
    x_padded = x_in[0] - pads[1][0] * dx + dx * np.arange(padded.shape[1])
    y_padded = y_in[0] - pads[0][0] * dy + dy * np.arange(padded.shape[0])
    fx = np.fft.fftfreq(padded.shape[1], d=dx)
    fy = np.fft.fftfreq(padded.shape[0], d=dy)
    transfer = np.exp(
        -1j
        * np.pi
        * wavelength
        * effective_defocus
        * (fy[:, None] ** 2 + fx[None, :] ** 2)
    )
    propagated = np.fft.ifft2(np.fft.fft2(padded) * transfer)
    return propagated, x_padded, y_padded


def _maximum_quadratic_phase_step(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    coefficient_rad_per_m2: float,
) -> float:
    def axis_max(coords: np.ndarray) -> float:
        if coords.size < 2:
            return 0.0
        return float(
            np.max(
                np.abs(
                    coefficient_rad_per_m2
                    * (coords[1:] ** 2 - coords[:-1] ** 2)
                )
            )
        )

    return max(axis_max(x_coords), axis_max(y_coords))


def round_solenoid_grid_reference(
    field_yx: np.ndarray,
    x_in_m: np.ndarray,
    y_in_m: np.ndarray,
    detector_x_coords_m: np.ndarray,
    detector_y_coords_m: np.ndarray,
    *,
    radial_abcd: Sequence[float],
    cumulative_rotation_rad: float,
    detector_affine_xy: np.ndarray,
    detector_centre_xy_m: Sequence[float] = (0.0, 0.0),
    wavelength_m: float,
    detector_mtf_sigma_px: float = 0.0,
    route: str = "auto",
    options: GridWaveOptions = GridWaveOptions(),
) -> GridWaveReferenceResult:
    """Dispatch a round map through conjugate, scaled-Fresnel, or CZT routes.

    ``radial_abcd`` is the co-rotating scalar map.  The output of that map is
    actively rotated by ``cumulative_rotation_rad`` into the column frame and
    then sampled through the detector affine.  ``route='auto'`` uses the
    inverse-map limit only when the Nyquist-band B displacement is certified.
    The next branch uses the exact first-order factorization
    ``M = L_(C/A) S_A F_(B/A)`` and a padded FFT; this is the stable route for
    high-magnification states having tiny effective defocus ``B/A`` but a
    detector-resolved B displacement.  Remaining nonzero-B states use CZT.
    Explicitly requesting an uncertified branch fails rather than silently
    changing the optical state.  The CZT branch removes the analytic
    ``D*X**2`` output chirp before interpolation, adaptively refines the
    remaining finite-support Fourier envelope to
    ``czt_maximum_kernel_phase_step_rad / output_oversampling``, and restores
    the chirp only at the exact detector coordinates.  Consequently its
    reported output-kernel phase step is the sampled envelope step; the
    separately reported output lens-phase step may be much larger without
    representing aliasing.
    """

    if route == "fft":
        route = "scaled_fresnel_fft"
    if route not in {"auto", "conjugate", "scaled_fresnel_fft", "czt"}:
        raise ValueError(
            "route must be 'auto', 'conjugate', 'scaled_fresnel_fft', or 'czt'."
        )
    if not isinstance(options, GridWaveOptions):
        raise TypeError("options must be GridWaveOptions.")
    matrix = _abcd("radial_abcd", radial_abcd)
    A, B, C, D = matrix
    wavelength = float(wavelength_m)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength_m must be finite and positive.")
    rotation = float(cumulative_rotation_rad)
    if not np.isfinite(rotation):
        raise ValueError("cumulative_rotation_rad must be finite.")
    mtf = float(detector_mtf_sigma_px)
    if not np.isfinite(mtf) or mtf < 0.0:
        raise ValueError("detector_mtf_sigma_px must be finite and non-negative.")
    x_in, dx = _uniform_coords("x_in_m", x_in_m)
    y_in, dy = _uniform_coords("y_in_m", y_in_m)
    detector_x, _ = _uniform_coords("detector_x_coords_m", detector_x_coords_m)
    detector_y, _ = _uniform_coords("detector_y_coords_m", detector_y_coords_m)
    field = np.asarray(field_yx, dtype=np.complex128)
    if field.shape != (y_in.size, x_in.size) or not np.all(np.isfinite(field)):
        raise ValueError("field_yx must be finite with shape (len(y_in_m), len(x_in_m)).")
    if float(np.sum(np.abs(field) ** 2)) <= 0.0:
        raise ValueError("field_yx must contain positive power.")

    affine = np.asarray(detector_affine_xy, dtype=float)
    query_radial = _detector_query_coordinates(
        detector_x,
        detector_y,
        affine,
        detector_centre_xy_m,
        rotation,
    )
    x_out, y_out = _intermediate_grid(
        query_radial, detector_x, detector_y, affine, options
    )
    angular_bandlimit = wavelength / (2.0 * min(dx, dy))
    detector_step_map = np.linalg.inv(affine) @ np.diag(
        (float(np.diff(detector_x).mean()), float(np.diff(detector_y).mean()))
    )
    smallest_optical_pixel = float(
        np.min(np.linalg.svd(detector_step_map, compute_uv=False))
    )
    b_displacement = (
        abs(B) * np.sqrt(2.0) * angular_bandlimit / smallest_optical_pixel
    )
    symplectic_certificate = certify_radial_symplectic_map(
        matrix,
        absolute_tolerance=options.symplectic_determinant_tolerance,
    )
    symplectic_error = symplectic_certificate.determinant_error
    edge_fraction = _edge_power_fraction(field)
    input_power = _power(field, x_in, y_in)
    conjugate_admissible = (
        b_displacement <= options.conjugate_maximum_b_displacement_px
    )
    effective_defocus = None if abs(A) <= np.finfo(float).eps else B / A
    fft_pads, fft_shape, fft_margins = _fft_padding_geometry(
        field.shape, dx, dy, options
    )
    fft_shifts = (
        None
        if effective_defocus is None
        else (
            abs(effective_defocus) * wavelength / (2.0 * dx),
            abs(effective_defocus) * wavelength / (2.0 * dy),
        )
    )
    fft_samples_per_detector_pixel = (
        None
        if effective_defocus is None
        else smallest_optical_pixel / (abs(A) * max(dx, dy))
    )
    fft_query_inside = False
    if effective_defocus is not None:
        scaled_query = query_radial / A
        x_min = x_in[0] - fft_pads[1][0] * dx
        x_max = x_in[-1] + fft_pads[1][1] * dx
        y_min = y_in[0] - fft_pads[0][0] * dy
        y_max = y_in[-1] + fft_pads[0][1] * dy
        fft_query_inside = bool(
            np.min(scaled_query[..., 0]) >= x_min + 2.0 * dx
            and np.max(scaled_query[..., 0]) <= x_max - 2.0 * dx
            and np.min(scaled_query[..., 1]) >= y_min + 2.0 * dy
            and np.max(scaled_query[..., 1]) <= y_max - 2.0 * dy
        )
    fft_point_count_admissible = (
        fft_shape[0] * fft_shape[1] <= options.maximum_intermediate_points
    )
    fft_shift_admissible = bool(
        fft_shifts is not None
        and fft_shifts[0]
        <= options.fft_maximum_nyquist_shift_fraction_of_padding * fft_margins[0]
        and fft_shifts[1]
        <= options.fft_maximum_nyquist_shift_fraction_of_padding * fft_margins[1]
    )
    fft_paraxial_admissible = bool(
        np.sqrt(2.0) * angular_bandlimit
        <= options.fft_maximum_paraxial_angle_rad
    )
    fft_detector_sampling_admissible = bool(
        fft_samples_per_detector_pixel is not None
        and fft_samples_per_detector_pixel
        >= options.fft_minimum_samples_per_detector_pixel
    )
    fft_admissible = bool(
        options.scaled_fresnel_fft_enabled
        and effective_defocus is not None
        and B != 0.0
        and fft_point_count_admissible
        and fft_shift_admissible
        and fft_paraxial_admissible
        and fft_detector_sampling_admissible
        and fft_query_inside
    )
    auto_rejections: list[str] = []
    if not conjugate_admissible:
        auto_rejections.append("conjugate:b_displacement")
    if not options.scaled_fresnel_fft_enabled:
        auto_rejections.append("scaled_fresnel_fft:disabled")
    if effective_defocus is None or B == 0.0:
        auto_rejections.append("scaled_fresnel_fft:factorization_singular")
    if not fft_point_count_admissible:
        auto_rejections.append("scaled_fresnel_fft:memory")
    if not fft_shift_admissible:
        auto_rejections.append("scaled_fresnel_fft:wraparound")
    if not fft_paraxial_admissible:
        auto_rejections.append("scaled_fresnel_fft:nonparaxial_bandlimit")
    if not fft_detector_sampling_admissible:
        auto_rejections.append("scaled_fresnel_fft:detector_undersampling")
    if not fft_query_inside:
        auto_rejections.append("scaled_fresnel_fft:output_outside_padding")
    if route == "auto":
        if conjugate_admissible:
            selected = "conjugate"
        elif fft_admissible:
            selected = "scaled_fresnel_fft"
        else:
            selected = "czt"
    else:
        selected = route
    czt_preliminary_input_phase_step: float | None = None
    if selected == "czt" and B != 0.0:
        czt_preliminary_input_phase_step, _ = _kernel_phase_steps(
            x_in,
            y_in,
            x_out,
            y_out,
            matrix,
            wavelength,
            output_chirp_removed=True,
        )
    if (
        selected == "czt"
        and B != 0.0
        and czt_preliminary_input_phase_step is not None
        and czt_preliminary_input_phase_step
        <= options.czt_maximum_kernel_phase_step_rad
        and symplectic_certificate.passed
        and edge_fraction <= options.maximum_input_edge_power_fraction
    ):
        # The full Collins output contains a known D*X^2 chirp which is
        # demodulated before interpolation.  Refine the intermediate grid only
        # enough to sample the remaining Fourier envelope at the same strict
        # phase-step limit used by the fail-closed CZT gate, with the requested
        # output oversampling applied to that limit as well as detector pixels.
        # This is essential in diffraction states: detector oversampling alone
        # need not resolve the finite-input-support bandwidth.
        maximum_input_abs_xy = np.asarray(
            (
                max(abs(x_in[0]), abs(x_in[-1])),
                max(abs(y_in[0]), abs(y_in[-1])),
            ),
            dtype=float,
        )
        czt_maximum_step_xy = (
            options.czt_maximum_kernel_phase_step_rad
            * wavelength
            * abs(B)
            / (
                2.0
                * np.pi
                * maximum_input_abs_xy
                * options.output_oversampling
            )
        ) * (1.0 - 64.0 * np.finfo(float).eps)
        x_out, y_out = _intermediate_grid(
            query_radial,
            detector_x,
            detector_y,
            affine,
            options,
            maximum_step_xy_m=czt_maximum_step_xy,
        )
    failures: list[str] = []
    if not symplectic_certificate.passed:
        failures.append("radial_abcd_not_symplectic")
    if edge_fraction > options.maximum_input_edge_power_fraction:
        failures.append("input_window_not_tapered")

    if B == 0.0:
        input_phase_step: float | None = None
        output_phase_step: float | None = None
    else:
        input_phase_step, output_phase_step = _kernel_phase_steps(
            x_in,
            y_in,
            x_out,
            y_out,
            matrix,
            wavelength,
            output_chirp_removed=selected == "czt",
        )
    output_lens_phase_step: float | None = None
    padded_fresnel: np.ndarray | None = None
    padded_x: np.ndarray | None = None
    padded_y: np.ndarray | None = None
    czt_demodulated: np.ndarray | None = None
    if selected == "conjugate":
        if not conjugate_admissible:
            failures.append("conjugate_b_displacement")
        if abs(A) <= np.finfo(float).eps:
            failures.append("conjugate_a_singular")
        if failures:
            diagnostics = GridWaveDiagnostics(
                route=selected,
                radial_abcd=matrix,
                cumulative_rotation_rad=rotation,
                symplectic_determinant_error=symplectic_error,
                b_displacement_px=float(b_displacement),
                angular_bandlimit_rad=float(angular_bandlimit),
                effective_defocus_m=effective_defocus,
                fft_padded_shape_yx=fft_shape,
                fft_nyquist_shift_xy_m=fft_shifts,
                fft_padding_margin_xy_m=fft_margins,
                fft_samples_per_detector_pixel=fft_samples_per_detector_pixel,
                maximum_output_lens_phase_step_rad=None,
                maximum_input_kernel_phase_step_rad=input_phase_step,
                maximum_output_kernel_phase_step_rad=output_phase_step,
                input_edge_power_fraction=edge_fraction,
                input_power=input_power,
                output_power=None,
                power_relative_error=None,
                intermediate_shape_yx=(y_out.size, x_out.size),
                failure_reasons=tuple(failures),
                passed=False,
                alternative_route_rejections=tuple(auto_rejections),
                symplectic_determinant_roundoff_bound=(
                    symplectic_certificate.determinant_evaluation_roundoff_bound
                ),
                symplectic_normalized_backward_error=(
                    symplectic_certificate.normalized_backward_error
                ),
                symplectic_certification_basis=(
                    symplectic_certificate.certification_basis
                ),
            )
            raise GridWaveSamplingError(diagnostics)
        xx_out, yy_out = np.meshgrid(x_out, y_out, indexing="xy")
        radial_xy = np.stack((xx_out, yy_out), axis=-1)
        intermediate = _complex_map_coordinates(
            field,
            x_in,
            y_in,
            radial_xy / A,
        )
        intermediate *= np.exp(
            1j * np.pi * C * (xx_out**2 + yy_out**2) / (wavelength * A)
        ) / abs(A)
        output_lens_phase_step = _maximum_quadratic_phase_step(
            x_out, y_out, np.pi * C / (wavelength * A)
        )
    elif selected == "scaled_fresnel_fft":
        if not options.scaled_fresnel_fft_enabled:
            failures.append("scaled_fresnel_fft_disabled")
        if effective_defocus is None or B == 0.0:
            failures.append("scaled_fresnel_fft_factorization_singular")
        if not fft_point_count_admissible:
            failures.append("scaled_fresnel_fft_memory")
        if not fft_shift_admissible:
            failures.append("scaled_fresnel_fft_wraparound")
        if not fft_paraxial_admissible:
            failures.append("scaled_fresnel_fft_nonparaxial_bandlimit")
        if not fft_detector_sampling_admissible:
            failures.append("scaled_fresnel_fft_detector_undersampling")
        if not fft_query_inside:
            failures.append("scaled_fresnel_fft_output_outside_padding")
        if failures:
            diagnostics = GridWaveDiagnostics(
                route=selected,
                radial_abcd=matrix,
                cumulative_rotation_rad=rotation,
                symplectic_determinant_error=symplectic_error,
                b_displacement_px=float(b_displacement),
                angular_bandlimit_rad=float(angular_bandlimit),
                effective_defocus_m=effective_defocus,
                fft_padded_shape_yx=fft_shape,
                fft_nyquist_shift_xy_m=fft_shifts,
                fft_padding_margin_xy_m=fft_margins,
                fft_samples_per_detector_pixel=fft_samples_per_detector_pixel,
                maximum_output_lens_phase_step_rad=None,
                maximum_input_kernel_phase_step_rad=input_phase_step,
                maximum_output_kernel_phase_step_rad=output_phase_step,
                input_edge_power_fraction=edge_fraction,
                input_power=input_power,
                output_power=None,
                power_relative_error=None,
                intermediate_shape_yx=(y_out.size, x_out.size),
                failure_reasons=tuple(failures),
                passed=False,
                alternative_route_rejections=tuple(auto_rejections),
                symplectic_determinant_roundoff_bound=(
                    symplectic_certificate.determinant_evaluation_roundoff_bound
                ),
                symplectic_normalized_backward_error=(
                    symplectic_certificate.normalized_backward_error
                ),
                symplectic_certification_basis=(
                    symplectic_certificate.certification_basis
                ),
            )
            raise GridWaveSamplingError(diagnostics)
        assert effective_defocus is not None
        padded_fresnel, padded_x, padded_y = _fft_fresnel_plane(
            field,
            x_in,
            y_in,
            wavelength=wavelength,
            effective_defocus=effective_defocus,
            pads=fft_pads,
        )
        xx_out, yy_out = np.meshgrid(x_out, y_out, indexing="xy")
        radial_xy = np.stack((xx_out, yy_out), axis=-1)
        intermediate = _complex_map_coordinates(
            padded_fresnel,
            padded_x,
            padded_y,
            radial_xy / A,
        )
        output_coefficient = np.pi * C / (wavelength * A)
        intermediate *= np.exp(
            1j * output_coefficient * (xx_out**2 + yy_out**2)
        ) / abs(A)
        output_lens_phase_step = _maximum_quadratic_phase_step(
            x_out, y_out, output_coefficient
        )
    else:
        if B == 0.0:
            failures.append("czt_requires_nonzero_b")
        else:
            assert input_phase_step is not None and output_phase_step is not None
            if input_phase_step > options.czt_maximum_kernel_phase_step_rad:
                failures.append("czt_input_kernel_aliasing")
            if output_phase_step > options.czt_maximum_kernel_phase_step_rad:
                failures.append("czt_output_kernel_aliasing")
        if failures:
            diagnostics = GridWaveDiagnostics(
                route=selected,
                radial_abcd=matrix,
                cumulative_rotation_rad=rotation,
                symplectic_determinant_error=symplectic_error,
                b_displacement_px=float(b_displacement),
                angular_bandlimit_rad=float(angular_bandlimit),
                effective_defocus_m=effective_defocus,
                fft_padded_shape_yx=fft_shape,
                fft_nyquist_shift_xy_m=fft_shifts,
                fft_padding_margin_xy_m=fft_margins,
                fft_samples_per_detector_pixel=fft_samples_per_detector_pixel,
                maximum_output_lens_phase_step_rad=None,
                maximum_input_kernel_phase_step_rad=input_phase_step,
                maximum_output_kernel_phase_step_rad=output_phase_step,
                input_edge_power_fraction=edge_fraction,
                input_power=input_power,
                output_power=None,
                power_relative_error=None,
                intermediate_shape_yx=(y_out.size, x_out.size),
                failure_reasons=tuple(failures),
                passed=False,
                alternative_route_rejections=tuple(auto_rejections),
                symplectic_determinant_roundoff_bound=(
                    symplectic_certificate.determinant_evaluation_roundoff_bound
                ),
                symplectic_normalized_backward_error=(
                    symplectic_certificate.normalized_backward_error
                ),
                symplectic_certification_basis=(
                    symplectic_certificate.certification_basis
                ),
            )
            raise GridWaveSamplingError(diagnostics)
        czt_demodulated = _scaled_collins_czt_2d_demodulated(
            field,
            x_in,
            y_in,
            x_out,
            y_out,
            abcd=matrix,
            wavelength_m=wavelength,
        )
        xx_out, yy_out = np.meshgrid(x_out, y_out, indexing="xy")
        czt_output_coefficient = np.pi * D / (wavelength * B)
        intermediate = czt_demodulated * np.exp(
            1j * czt_output_coefficient * (xx_out**2 + yy_out**2)
        )
        output_lens_phase_step = _maximum_quadratic_phase_step(
            x_out, y_out, np.pi * D / (wavelength * B)
        )

    output_power = _power(intermediate, x_out, y_out)
    power_error = abs(output_power / input_power - 1.0)
    if options.require_power_conservation and power_error > options.maximum_power_relative_error:
        failures.append("power_conservation")
    query_radius_squared = query_radial[..., 0] ** 2 + query_radial[..., 1] ** 2
    if selected == "conjugate":
        sampled_field = _complex_map_coordinates(
            field, x_in, y_in, query_radial / A
        )
        sampled_field *= np.exp(
            1j * np.pi * C * query_radius_squared / (wavelength * A)
        ) / abs(A)
    elif selected == "scaled_fresnel_fft":
        assert padded_fresnel is not None and padded_x is not None and padded_y is not None
        sampled_field = _complex_map_coordinates(
            padded_fresnel, padded_x, padded_y, query_radial / A
        )
        sampled_field *= np.exp(
            1j * np.pi * C * query_radius_squared / (wavelength * A)
        ) / abs(A)
    else:
        # Remove the known rapid Collins output chirp before cubic sampling,
        # then restore it analytically at the exact rotated/affine detector
        # locations.  This prevents interpolation from attenuating intensity
        # merely because the detector field has a fast but known phase carrier.
        assert czt_demodulated is not None
        czt_output_coefficient = np.pi * D / (wavelength * B)
        sampled_field = _complex_map_coordinates(
            czt_demodulated, x_out, y_out, query_radial
        ) * np.exp(1j * czt_output_coefficient * query_radius_squared)
    intensity = np.abs(sampled_field) ** 2
    if mtf > 0.0:
        intensity = ndimage.gaussian_filter(intensity, mtf, mode="nearest")
    if not np.all(np.isfinite(sampled_field)) or not np.all(np.isfinite(intensity)):
        failures.append("nonfinite_output")
    diagnostics = GridWaveDiagnostics(
        route=selected,
        radial_abcd=matrix,
        cumulative_rotation_rad=rotation,
        symplectic_determinant_error=symplectic_error,
        b_displacement_px=float(b_displacement),
        angular_bandlimit_rad=float(angular_bandlimit),
        effective_defocus_m=effective_defocus,
        fft_padded_shape_yx=fft_shape,
        fft_nyquist_shift_xy_m=fft_shifts,
        fft_padding_margin_xy_m=fft_margins,
        fft_samples_per_detector_pixel=fft_samples_per_detector_pixel,
        maximum_output_lens_phase_step_rad=output_lens_phase_step,
        maximum_input_kernel_phase_step_rad=input_phase_step,
        maximum_output_kernel_phase_step_rad=output_phase_step,
        input_edge_power_fraction=edge_fraction,
        input_power=input_power,
        output_power=output_power,
        power_relative_error=power_error,
        intermediate_shape_yx=(y_out.size, x_out.size),
        failure_reasons=tuple(failures),
        passed=not failures,
        alternative_route_rejections=tuple(auto_rejections),
        symplectic_determinant_roundoff_bound=(
            symplectic_certificate.determinant_evaluation_roundoff_bound
        ),
        symplectic_normalized_backward_error=(
            symplectic_certificate.normalized_backward_error
        ),
        symplectic_certification_basis=symplectic_certificate.certification_basis,
    )
    if failures:
        raise GridWaveSamplingError(diagnostics)
    for array in (sampled_field, intensity, intermediate, x_out, y_out):
        array.setflags(write=False)
    return GridWaveReferenceResult(
        field_yx=sampled_field,
        intensity_yx=intensity,
        intermediate_field_yx=intermediate,
        intermediate_x_coords_m=x_out,
        intermediate_y_coords_m=y_out,
        diagnostics=diagnostics,
    )


__all__ = [
    "GridWaveConvergenceDiagnostics",
    "GridWaveSamplingError",
    "GridWaveOptions",
    "GridWaveDiagnostics",
    "GridWaveReferenceResult",
    "RadialSymplecticCertificate",
    "assess_grid_reference_convergence",
    "certify_radial_symplectic_map",
    "poisson_read_total_standard_error",
    "simpson_uniform_weights",
    "scaled_collins_czt_1d",
    "scaled_collins_czt_2d",
    "round_solenoid_grid_reference",
]
