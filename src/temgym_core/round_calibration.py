"""Shared physical-scale calibration for round magnetic response tomography.

These errors describe an existing sample-state/detector calibration and its
covariance. They are not uncalibrated beam-command gains inferred from images.
The same calibration realization applies to every affected exposure.
"""

import jax.numpy as jnp


GAIN_NAMES = ("log_sample_position_gain", "log_sample_tilt_gain", "log_detector_pitch_gain")


def calibrated_position_response(response, log_gains):
    """Apply shared actual/nominal position, tilt, and isotropic pitch gains.

    Input columns are grouped (x, y, alpha_x, alpha_y); output is in detector
    pixels. Nominal response covariance remains statistical detector noise.
    Shared calibration covariance belongs to the latent gains, once globally.
    """
    response, gains = jnp.asarray(response), jnp.asarray(log_gains)
    if response.shape[-2:] != (2, 4) or gains.shape != (3,):
        raise ValueError("Expected (...,2,4) position responses and three shared log gains.")
    factors = jnp.exp(gains[jnp.array([0,0,1,1])]-gains[2])
    return response*factors


def round_response_gain_family(stages, prefix, log_gains):
    """Re-express distorted responses as nominally calibrated round optics.

    Each stage is [log_length, log_following_gap, q_bias, q_slope]. The prefix
    uses [log_row_norm, row_angle, row_shear, rotation, kappa*balance_length],
    where its balanced radial matrix has first row exp(rho)*(cos(w),sin(w))
    and second row h*first_row + exp(-rho)*(-sin(w),cos(w)).

    The transformed nominal response equals the original response with gains
    applied. Thus *unmodelled* gains bias every length by sp*st/pp². Conversely,
    for fixed observations and unknown actual gains, apply this function with
    NEGATIVE log gains to obtain the exactly compensating physical optics.
    Excitation phases and gap/length ratios are unchanged. This is an exact
    calibration family, not an additional excitation-branch ambiguity.
    """
    stages, prefix, gains = map(jnp.asarray,(stages,prefix,log_gains))
    if stages.ndim != 2 or stages.shape[1] != 4 or stages.shape[0] < 1:
        raise ValueError("At least one four-coordinate physical stage is required.")
    if prefix.shape != (5,) or gains.shape != (3,):
        raise ValueError("Expected five prefix coordinates and three log gains.")
    fp, ft = jnp.exp(gains[:2]-gains[2])
    rho, angle, shear, rotation, kappa = prefix
    r, c, s = jnp.exp(rho), jnp.cos(angle), jnp.sin(angle)
    original = jnp.array([[r*c,r*s],[shear*r*c-s/r,shear*r*s+c/r]])
    transformed = original*jnp.array([[fp,ft],[1/ft,1/fp]])
    norm = jnp.linalg.norm(transformed[0])
    output_prefix = jnp.array([jnp.log(norm),jnp.arctan2(transformed[0,1],transformed[0,0]),
        jnp.dot(transformed[0],transformed[1])/norm**2,rotation,kappa*fp/ft])
    output_stages = stages.at[:,:2].add(gains[0]+gains[1]-2*gains[2])
    return output_stages,output_prefix
