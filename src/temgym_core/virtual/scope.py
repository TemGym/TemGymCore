"""The virtual microscope: control words in, camera frames out, truth hidden.

:class:`VirtualMicroscope` keeps the state of a JEOL-like TEM column in the
units its control interface uses (16-bit words, volts, nanometres, indices). A
frame is rendered only when the optical model can justify the state: anything
unmodelled (another function mode, a moved unmodelled lens or deflector, an
untabulated high tension, an open condenser aperture, ...) raises
:class:`UnmodelledStateError` listing every reason, instead of returning an
image the model cannot produce.

The optical truth is the calibration model itself (finite solenoid lenses, a
Gaussian source per high tension, a sigmoid condenser aperture, a post-aperture
double deflector); images come from native ray tracing through
:func:`temgym_core.calibration.acquisition.setting_column` and
:func:`~temgym_core.calibration.acquisition.render_rays`.
"""

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge

from ..calibration.acquisition import render_rays, setting_column
from ..config import require_x64
from .camera import CameraModel
from .electronics import NEUTRAL_WORD, Electronics

#: PyJEM free-lens-control identifiers.
LENS_IDS = {"CL1": 0, "CL2": 1, "CL3": 2, "CM": 3, "OLc": 6, "OLf": 7, "OM": 8, "OM2": 9,
            "IL1": 10, "IL2": 11, "IL3": 12, "IL4": 13, "PL1": 14, "PL2": 15, "PL3": 16}
PROJECTOR_LENSES = ("IL1", "IL2", "IL3", "PL1")
CONDENSER_LENSES = ("CL1", "CL3", "CM")
OBJECTIVE_LENSES = ("OLc", "OLf")
UNMODELLED_LENSES = ("CL2", "OM", "OM2", "IL4", "PL2", "PL3")
#: PyJEM Def3 deflector pairs; only CLA1 (beam shift) and CLA2 (beam tilt) are modelled.
DEFLECTORS = ("GunA1", "GunA2", "CLA1", "CLA2", "CLs", "IS1", "IS2", "PLA", "OLs", "ILs",
              "SpotA", "ShifBal", "TiltBal", "AngBal")
TEM_FUNCTION_MODES = {0: "MAG", 1: "MAG2", 2: "LowMAG", 3: "SAMAG", 4: "DIFF"}
STEM_FUNCTION_MODES = {0: "Align", 1: "SM-LMAG", 2: "SM-MAG", 3: "AMAG", 4: "uuDIFF", 5: "Rocking"}
#: Apt3 kinds; kind 1 is the condenser aperture, the only one modelled.
APERTURE_KINDS = {1: "CLA", 2: "OLA", 3: "HCA", 4: "SAA", 5: "ENTA", 6: "EDS"}
CONDENSER_APERTURE = 1
APERTURE_HOME = (2048, 2048)


class UnmodelledStateError(RuntimeError):
    """The microscope state lies outside what the optical model can render.

    Parameters
    ----------
    reasons : list of str
        Every unmodelled aspect of the state.
    """

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("The virtual microscope cannot render this state: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class OpticalTruth:
    """Hidden optics: calibration-model parameters and the modelled operating points.

    Parameters
    ----------
    condenser_parameters, projector_parameters : tuple of float
        Physical calibration-model parameters (31 and 16 values).
    voltages_v : tuple of float, optional
        Modelled high tensions; the first is the parameters' reference voltage and
        each has its own source (index in this tuple).
    aperture_diameters_um : tuple of float, optional
        Diameters of the two condenser-aperture holes with fitted centres
        (aperture index 0 and 1).
    aperture_edge_width_m : float, optional
        Sigmoid aperture edge width.
    gap_fractions : tuple of float, optional
        Deflector positions in the aperture-to-mini gap (illustrative, hidden).
    """

    condenser_parameters: tuple
    projector_parameters: tuple
    voltages_v: tuple = (200e3, 195e3)
    aperture_diameters_um: tuple = (20.0, 50.0)
    aperture_edge_width_m: float = 150e-9
    gap_fractions: tuple = (0.3, 0.7)

    def __post_init__(self):
        if len(self.condenser_parameters) != 31 or len(self.projector_parameters) != 16:
            raise ValueError("Supply 31 condenser and 16 projector parameters.")
        for name in ("condenser_parameters", "projector_parameters", "voltages_v",
                     "aperture_diameters_um", "gap_fractions"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))

    @classmethod
    def from_files(cls, experiment_path, scoring_truth_path):
        """Load a saved synthetic acquisition's truth.

        Parameters
        ----------
        experiment_path : str or Path
            ``experiment.json`` with ``voltages_v``, ``aperture_diameters_um`` and
            ``aperture_edge_width_m`` (and optionally ``deflector_gap_fractions``).
        scoring_truth_path : str or Path
            ``scoring_truth.json`` with the 31 + 16 physical parameters.

        Returns
        -------
        OpticalTruth
        """
        spec = json.loads(Path(experiment_path).read_text())
        truth = json.loads(Path(scoring_truth_path).read_text())
        return cls(truth["condenser_parameters"], truth["projector_parameters"],
                   voltages_v=spec["voltages_v"], aperture_diameters_um=spec["aperture_diameters_um"],
                   aperture_edge_width_m=spec["aperture_edge_width_m"],
                   gap_fractions=spec.get("deflector_gap_fractions", (0.3, 0.7)))

    def scoring_truth(self):
        """The truth in ``scoring_truth.json`` form, for scoring a fit only.

        Returns
        -------
        dict
        """
        return {"condenser_parameters": list(self.condenser_parameters),
                "projector_parameters": list(self.projector_parameters)}


@dataclass(frozen=True)
class PhysicalSetting:
    """Physical controls of one frame, in calibration-model units.

    Parameters
    ----------
    voltage_v : float
        High tension.
    source_index : int
        Index of the high tension in :attr:`OpticalTruth.voltages_v`.
    condenser_controls : tuple of float
        Relative CL1, CL3 and mini-lens controls.
    projector_controls : tuple of float
        Relative Objective, IL1, IL2, IL3 and PL1 controls.
    specimen_kick : tuple of float
        Specimen displacement (m) and angle (rad).
    aperture_index : int
        Inserted condenser-aperture hole (0 or 1).
    """

    voltage_v: float
    source_index: int
    condenser_controls: tuple
    projector_controls: tuple
    specimen_kick: tuple
    aperture_index: int


class VirtualMicroscope:
    """A JEOL-like TEM column driven by control words, with hidden optics.

    Parameters
    ----------
    truth : OpticalTruth
        Hidden optics.
    electronics : Electronics
        Hidden word-to-control laws.
    magnification_table : dict
        Nominal magnification -> words for ``IL1``, ``IL2``, ``IL3`` and ``PL1``
        (the TEM MAG selector, in increasing magnification).
    spot_table : dict
        PyJEM spot index (0-7) -> words for ``CL1``, ``CL3`` and ``CM``.
    diffraction_table : dict, optional
        Camera length (m, at this camera) -> words for ``IL1``, ``IL2``, ``IL3`` and
        ``PL1`` (the TEM DIFF selector, in increasing camera length). Without it DIFF
        mode is refused as unmodelled.
    camera : CameraModel, optional
        The camera; ideal by default.
    aperture_holes : dict, optional
        Condenser-aperture hole number (``Apt3.SetSize``) -> diameter in µm, each
        one of ``truth.aperture_diameters_um``. Default: hole 1 is the larger.
    beam_current_a : float, optional
        Current leaving the virtual source; with the exposure it sets the dose.
    ray_sample_power : int, optional
        Trace ``2**ray_sample_power`` rays per image.
    detector_samples_per_pixel : int, optional
        Internal detector sub-samples per output pixel axis.
    seed : int, optional
        Seed of the camera noise.
    """

    def __init__(self, truth, electronics, *, magnification_table, spot_table, diffraction_table=None,
                 camera=None, aperture_holes=None, beam_current_a=1.602176634e-9, ray_sample_power=18,
                 detector_samples_per_pixel=1, seed=0):
        require_x64("the virtual microscope")
        if not isinstance(electronics, Electronics):
            raise TypeError("electronics must be an Electronics instance.")
        self.truth, self.electronics = truth, electronics
        self.magnification_table = tuple(sorted(
            (float(m), {k: int(w) for k, w in words.items()}) for m, words in magnification_table.items()))
        self.diffraction_table = tuple(sorted(
            (float(m), {k: int(w) for k, w in words.items()}) for m, words in (diffraction_table or {}).items()))
        self.spot_table = {int(k): {n: int(w) for n, w in v.items()} for k, v in spot_table.items()}
        if not self.magnification_table or not self.spot_table:
            raise ValueError("Supply at least one magnification row and one spot row.")
        large, small = sorted(truth.aperture_diameters_um, reverse=True)
        self.aperture_holes = dict(aperture_holes or {1: large, 2: small})
        if not set(self.aperture_holes.values()) <= set(truth.aperture_diameters_um):
            raise ValueError("Every aperture hole must be one of the truth's aperture diameters.")
        self.camera = camera or CameraModel()
        self.beam_current_a = float(beam_current_a)
        self.ray_sample_power = int(ray_sample_power)
        self.detector_samples_per_pixel = int(detector_samples_per_pixel)
        self.rng = np.random.default_rng(seed)
        self._images = OrderedDict()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Return every control to its power-on value (first table rows, neutral words)."""
        self.ht_v = self.truth.voltages_v[0]
        self.tem_stem = 0
        self.function_mode = 0
        self.probe_mode = 0
        self.alpha = 0
        self.selector = 0
        self.camera_length_selector = 0
        self.spot = min(self.spot_table)
        self.direct = {}
        self.flc_switch = {name: 0 for name in LENS_IDS}
        self.flc_value = {name: None for name in LENS_IDS}
        self.deflectors = {name: [NEUTRAL_WORD, NEUTRAL_WORD] for name in DEFLECTORS}
        self.beam_blank = 0
        self.aperture_kind = CONDENSER_APERTURE
        self.exp_kind = 0
        self.apertures = {kind: {"size": 0, "position": list(APERTURE_HOME)} for kind in APERTURE_KINDS}
        self.apertures[CONDENSER_APERTURE]["size"] = min(self.aperture_holes)
        self.stage = [0.0, 0.0, 0.0, 0.0, 0.0]
        binning = max(self.camera.binnings)
        shape, _, _ = self.camera.geometry(binning)
        self.detector = {"inserted": True, "binning": binning, "exposure_us": 10_000.0, "frames": 1,
                         "area": [0, 0, shape[1], shape[0]], "gain_index": 0, "offset_index": 0}

    @property
    def magnification(self):
        """Nominal magnification of the selected TEM MAG row."""
        return self.magnification_table[self.selector][0]

    @property
    def diffraction(self):
        """True in TEM DIFF mode."""
        return self.tem_stem == 0 and self.function_mode == 4

    @property
    def camera_length(self):
        """Camera length (m) of the selected TEM DIFF row (``None`` without a diffraction table)."""
        if not self.diffraction_table:
            return None
        return self.diffraction_table[self.camera_length_selector][0]

    def table_word(self, name):
        """Word a lens takes from the selector, spot or power-on tables (``None`` if absent).

        Parameters
        ----------
        name : str
            Lens name (a key of :data:`LENS_IDS`).

        Returns
        -------
        int or None
        """
        if name in PROJECTOR_LENSES:
            if self.diffraction:
                if not self.diffraction_table:
                    return None
                return self.diffraction_table[self.camera_length_selector][1].get(name)
            return self.magnification_table[self.selector][1].get(name)
        if name in CONDENSER_LENSES:
            return self.spot_table.get(self.spot, {}).get(name)
        if name in OBJECTIVE_LENSES:
            return self.electronics.objective_words[OBJECTIVE_LENSES.index(name)]
        return NEUTRAL_WORD

    def lens_word(self, name):
        """Effective word: free lens control, else a direct setting, else the tables.

        Parameters
        ----------
        name : str
            Lens name.

        Returns
        -------
        int or None
            ``None`` when no table provides the lens (an unmodelled spot).
        """
        if self.flc_switch[name]:
            return self.flc_value[name]
        if name in self.direct:
            return self.direct[name]
        return self.table_word(name)

    def reload_tables(self):
        """A selector, function-mode or spot change reloads table values and clears direct settings."""
        self.direct = {}

    # ---------------------------------------------------------------- physics
    def physical_setting(self):
        """Translate the current words into calibration-model controls.

        Returns
        -------
        PhysicalSetting

        Raises
        ------
        UnmodelledStateError
            Listing every aspect of the state the optics cannot render.
        """
        reasons = []
        if self.ht_v not in self.truth.voltages_v:
            reasons.append(f"high tension {self.ht_v:g} V is not one of the modelled {self.truth.voltages_v}")
        if self.tem_stem != 0:
            reasons.append("STEM mode is not modelled")
        elif self.diffraction and not self.diffraction_table:
            reasons.append("TEM function mode DIFF is not modelled (no diffraction table)")
        elif self.function_mode not in (0, 4):
            reasons.append(f"TEM function mode {TEM_FUNCTION_MODES.get(self.function_mode)} is not modelled "
                           "(MAG and DIFF only)")
        if self.probe_mode != 0:
            reasons.append("only the TEM probe mode is modelled")
        if self.alpha != 0:
            reasons.append("only alpha selector 0 is modelled")
        for name in UNMODELLED_LENSES:
            if self.lens_word(name) != NEUTRAL_WORD:
                reasons.append(f"{name} is not modelled and must stay at {NEUTRAL_WORD}")
        objective = tuple(self.lens_word(name) for name in OBJECTIVE_LENSES)
        if objective != self.electronics.objective_words:
            reasons.append(f"objective (OLc, OLf) = {objective} differs from the modelled "
                           f"{self.electronics.objective_words}")
        for name in CONDENSER_LENSES + PROJECTOR_LENSES:
            if self.lens_word(name) is None and name in CONDENSER_LENSES:     # (DIFF without a table: above)
                reasons.append(f"{name} has no word: spot {self.spot} is not tabulated and free lens control is off")
        for name in DEFLECTORS:
            if name not in ("CLA1", "CLA2") and self.deflectors[name] != [NEUTRAL_WORD, NEUTRAL_WORD]:
                reasons.append(f"deflector {name} is not modelled and must stay neutral")
        condenser = self.apertures[CONDENSER_APERTURE]
        if condenser["size"] not in self.aperture_holes:
            reasons.append(f"condenser aperture hole {condenser['size']} is not modelled "
                           f"(modelled holes: {sorted(self.aperture_holes)})")
        if condenser["position"] != list(APERTURE_HOME):
            reasons.append("a moved condenser aperture is not modelled")
        for kind, state in self.apertures.items():
            if kind != CONDENSER_APERTURE and state["size"] != 0:
                reasons.append(f"aperture {APERTURE_KINDS[kind]} must stay open")
        if reasons:
            raise UnmodelledStateError(reasons)
        drives = self.electronics.lenses
        condenser_controls = tuple(drives[n].control(self.lens_word(n)) for n in CONDENSER_LENSES)
        projector_controls = (1.0,) + tuple(drives[n].control(self.lens_word(n)) for n in PROJECTOR_LENSES)
        words = (*self.deflectors["CLA1"], *self.deflectors["CLA2"])
        kick = tuple(float(k) for k in self.electronics.deflector.kick(words, self.ht_v))
        diameter = self.aperture_holes[condenser["size"]]
        return PhysicalSetting(
            voltage_v=float(self.ht_v), source_index=self.truth.voltages_v.index(self.ht_v),
            condenser_controls=condenser_controls, projector_controls=projector_controls,
            specimen_kick=kick, aperture_index=self.truth.aperture_diameters_um.index(diameter))

    def expected_image(self, setting, binning):
        """Mean transmitted fraction per output pixel for a setting (cached).

        Parameters
        ----------
        setting : PhysicalSetting
            Physical controls.
        binning : int
            Camera binning.

        Returns
        -------
        numpy.ndarray
            Read-only image per incident electron, full binned frame.
        """
        key = (setting, binning, self.ray_sample_power, self.detector_samples_per_pixel)
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        t = self.truth
        column, mean, covariance, source_z = setting_column(
            np.asarray(t.condenser_parameters), np.asarray(t.projector_parameters),
            voltage_v=setting.voltage_v, reference_voltage_v=t.voltages_v[0],
            source_index=setting.source_index, condenser_controls=np.asarray(setting.condenser_controls),
            projector_controls=np.asarray(setting.projector_controls),
            specimen_kick=np.asarray(setting.specimen_kick), aperture_index=setting.aperture_index,
            aperture_diameter_um=t.aperture_diameters_um[setting.aperture_index],
            aperture_edge_width_m=t.aperture_edge_width_m, gap_fractions=t.gap_fractions)
        shape, pitch, psf = self.camera.geometry(binning)
        image = render_rays(column, mean, covariance, source_z, setting.voltage_v, shape_yx=shape,
                            pixel_pitch_m=pitch, psf_sigma_pixels=psf,
                            ray_sample_power=self.ray_sample_power,
                            detector_samples_per_pixel=self.detector_samples_per_pixel)
        image.setflags(write=False)
        self._images[key] = image
        if len(self._images) > 64:
            self._images.popitem(last=False)
        return image

    def acquire(self):
        """Record one camera exposure with the current state and detector settings.

        Returns
        -------
        numpy.ndarray
            Integer ADU frame (rows, columns) of the imaging area, in the camera's
            output type.

        Raises
        ------
        RuntimeError
            If the camera is retracted.
        UnmodelledStateError
            If the beam is on and the state is not modelled.
        """
        detector = self.detector
        if not detector["inserted"]:
            raise RuntimeError(f"The {self.camera.name} detector is retracted.")
        binning = detector["binning"]
        shape, _, _ = self.camera.geometry(binning)
        x, y, width, height = detector["area"]
        if self.beam_blank:
            expected = np.zeros(shape)
        else:
            incident = self.beam_current_a * detector["exposure_us"] * 1e-6 / elementary_charge
            expected = self.expected_image(self.physical_setting(), binning) * incident
        return self.camera.expose(expected[y:y + height, x:x + width], self.rng, detector["frames"])
