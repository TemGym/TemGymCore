"""Calibrate a microscope from recorded Gaussian images and aperture currents.

The public functions follow the notebook: acquire, measure, fit, check.
Lens propagation and differentiation use the native TemGym components.
"""

from .checks import show_result
from .workflow import (
    check_full_microscope,
    copy_recorded_experiment,
    fit_microscope,
    fit_projector,
    measure_condenser,
    measure_projector,
    show_images,
    simulate_experiment,
)

__all__ = [
    "simulate_experiment",
    "measure_projector",
    "fit_projector",
    "measure_condenser",
    "fit_microscope",
    "check_full_microscope",
    "show_images",
    "show_result",
    "copy_recorded_experiment",
]
