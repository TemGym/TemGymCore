"""Acquire the calibration images through a PyJEM-style microscope interface.

The protocol uses only what a real JEOL microscope offers: the ``TEM3`` and
``detector`` interfaces of PyJEM 1.4 (passed in, so the real PyJEM, the
virtual microscope in :mod:`temgym_core.virtual` or a test double can drive it)
and a :class:`VendorCalibration` of the CLA deflectors and the camera. It never
reads lens positions or other generating parameters.

Lens settings are planned as relative controls (current divided by a reference
current, the convention of :func:`~temgym_core.calibration.recorded.import_images`).
The microscope is put in a reference state (a magnification-selector row and a
spot size); its lens words there define the gauge::

    word = round(control * reference_word / reference_control)

for IL1, IL2, IL3 and PL1 (free lens control) and CL1, CL3 and the condenser
mini lens CM. This assumes the lens supplies are linear through zero; the
recorded controls are recomputed from the words actually set. Every word is
checked against 0-65535 before the microscope is touched.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .condenser_model import magnetic_strength_ratio
from .recorded import import_images

#: PyJEM free-lens-control identifiers of the lenses the calibration drives.
PROJECTOR_LENS_IDS = {"IL1": 10, "IL2": 11, "IL3": 12, "PL1": 14}
CONDENSER_LENS_IDS = {"CL1": 0, "CL3": 2, "CM": 3}
CONDENSER_APERTURE_KIND = 1
WORD_MAX = 65535


class PlanError(ValueError):
    """A planned setting needs a control word outside 0-65535."""


@dataclass(frozen=True)
class VendorCalibration:
    """What a user is given about the deflectors and the camera.

    Parameters
    ----------
    kick_per_word : tuple
        4x4 nested tuple: specimen ``[x, y, theta_x, theta_y]`` (m, rad) per word
        offset of ``[CLA1 x, CLA1 y, CLA2 x, CLA2 y]`` at ``reference_voltage_v``.
    zero_words : tuple of int, optional
        CLA words giving no kick.
    reference_voltage_v : float, optional
        High tension of ``kick_per_word``; a magnetic deflector's kick scales as
        ``p(reference) / p(voltage)``.
    camera_gain_adu_per_electron, camera_offset_adu : float, optional
        Per frame: ``ADU = gain * electrons + offset``.
    camera_read_noise_electrons, camera_background_electrons : float, optional
        Per output pixel and frame.
    sensor_pixel_m : float, optional
        Unbinned camera pixel pitch.
    psf_sigma_sensor_pixels : float, optional
        Camera point-spread function in unbinned pixels.
    """

    kick_per_word: tuple
    zero_words: tuple = (32768, 32768, 32768, 32768)
    reference_voltage_v: float = 200e3
    camera_gain_adu_per_electron: float = 1.0
    camera_offset_adu: float = 0.0
    camera_read_noise_electrons: float = 2.0
    camera_background_electrons: float = 3.0
    sensor_pixel_m: float = 13.75e-6
    psf_sigma_sensor_pixels: float = 8.0

    def __post_init__(self):
        matrix = np.asarray(self.kick_per_word, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("kick_per_word must be a finite 4x4 matrix.")
        object.__setattr__(self, "kick_per_word", tuple(tuple(map(float, r)) for r in matrix))
        object.__setattr__(self, "zero_words", tuple(int(w) for w in self.zero_words))

    def cla_words(self, kick, voltage_v):
        """CLA1/CLA2 words for a specimen kick at a high tension.

        Parameters
        ----------
        kick : array_like
            ``[x, y, theta_x, theta_y]`` in metres and radians.
        voltage_v : float
            High tension.

        Returns
        -------
        tuple of int
            ``(CLA1 x, CLA1 y, CLA2 x, CLA2 y)``.

        Raises
        ------
        PlanError
            If a word falls outside 0-65535.
        """
        ratio = float(magnetic_strength_ratio(voltage_v, self.reference_voltage_v))
        offsets = np.linalg.solve(np.asarray(self.kick_per_word) * ratio, np.asarray(kick, dtype=float))
        words = tuple(int(np.rint(o)) + z for o, z in zip(offsets, self.zero_words))
        if not all(0 <= w <= WORD_MAX for w in words):
            raise PlanError(f"Kick {list(kick)} needs CLA words {words} outside 0-{WORD_MAX}.")
        return words

    def to_json(self, path):
        """Write the calibration as JSON.

        Parameters
        ----------
        path : str or Path
            Destination file.
        """
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path):
        """Read a calibration written by :meth:`to_json`.

        Parameters
        ----------
        path : str or Path
            JSON file.

        Returns
        -------
        VendorCalibration
        """
        return cls(**json.loads(Path(path).read_text()))


def plan_words(plan, *, reference_words, projector_reference_ratios, condenser_reference_ratios):
    """Lens words for every planned setting, in the reference-state gauge.

    Parameters
    ----------
    plan : dict
        Calibration settings as in ``settings.npz``: ``projector_controls``
        (Objective, IL1, IL2, IL3, PL1) and ``condenser_controls`` (CL1, CL3, mini)
        per row.
    reference_words : dict
        Words of ``IL1``, ``IL2``, ``IL3``, ``PL1``, ``CL1``, ``CL3`` and ``CM`` read in
        the reference state.
    projector_reference_ratios, condenser_reference_ratios : array_like
        Relative controls of IL1-PL1 and CL1, CL3, mini in that reference state.

    Returns
    -------
    words : dict
        Integer word array per lens name.
    infeasible : list of str
        One entry per lens and voltage whose words leave 0-65535 (empty if feasible).
    """
    ratios = {**dict(zip(PROJECTOR_LENS_IDS, np.asarray(projector_reference_ratios, dtype=float))),
              **dict(zip(CONDENSER_LENS_IDS, np.asarray(condenser_reference_ratios, dtype=float)))}
    controls = {**dict(zip(PROJECTOR_LENS_IDS, np.asarray(plan["projector_controls"], dtype=float)[:, 1:].T)),
                **dict(zip(CONDENSER_LENS_IDS, np.asarray(plan["condenser_controls"], dtype=float).T))}
    words, infeasible = {}, []
    for name in (*PROJECTOR_LENS_IDS, *CONDENSER_LENS_IDS):
        words[name] = np.rint(controls[name] * reference_words[name] / ratios[name]).astype(np.int64)
        bad = (words[name] < 0) | (words[name] > WORD_MAX)
        for vi in np.unique(np.asarray(plan["voltage_index"])[bad]):
            rows = bad & (np.asarray(plan["voltage_index"]) == vi)
            infeasible.append(f"{name}: {int(rows.sum())} rows at voltage index {int(vi)} need words up to "
                              f"{int(words[name][rows].max())} (limit {WORD_MAX})")
    return words, infeasible


def _select_magnification(eos, magnification):
    """Step the TEM MAG selector to a nominal magnification; raise if absent."""
    previous = None
    while eos.GetCurrentMagSelectorID()[0] != previous:
        previous = eos.GetCurrentMagSelectorID()[0]
        eos.DownSelector()
    previous = None
    while True:
        index, value = eos.GetCurrentMagSelectorID()
        if value == magnification:
            return index
        if index == previous:
            raise PlanError(f"The magnification selector has no {magnification:g}x row.")
        previous = index
        eos.UpSelector()


def _decode(raw, setting):
    """Raw detector bytes to a (rows, columns) array; signed little-endian words."""
    bits = setting["OutputImageInformation"]["DataBits"]
    dtype = {16: "<i2", 32: "<i4"}.get(bits)
    if dtype is None:
        raise ValueError(f"Unsupported detector DataBits {bits}.")
    area = setting["ImagingArea"]
    return np.frombuffer(raw, dtype=dtype).reshape(area["Height"], area["Width"]).astype(float)


def acquire_calibration(
    tem3,
    detector,
    *,
    plan,
    vendor,
    output_directory,
    voltages_v,
    aperture_diameters_um,
    aperture_holes,
    reference_magnification,
    reference_spot,
    projector_reference_ratios,
    condenser_reference_ratios,
    exposure_s,
    binning,
    model_assumptions,
    frames=1,
    projector_aperture_um=None,
    camera="camera",
    log=print,
):
    """Record a calibration plan through PyJEM calls and import it for fitting.

    Parameters
    ----------
    tem3, detector : module
        ``PyJEM.TEM3`` and ``PyJEM.detector`` (real or virtual).
    plan : dict
        Settings as in ``settings.npz`` (``voltage_index``, ``purpose``,
        ``projector_controls``, ``condenser_controls``, ``specimen_kick``,
        ``response_group``; optional ``nominal_magnification``).
    vendor : VendorCalibration
        Deflector and camera calibration.
    output_directory : str or Path
        Empty folder for :func:`~temgym_core.calibration.recorded.import_images`.
    voltages_v : sequence of float
        High tension per ``voltage_index``; the first is the reference voltage.
    aperture_diameters_um : sequence of float
        Condenser-aperture diameters, in the order the model indexes them.
    aperture_holes : dict
        Diameter in µm -> ``Apt3.SetSize`` hole number.
    reference_magnification : float
        Selector row whose lens words define the projector gauge.
    reference_spot : int
        Spot-size index whose condenser words define the condenser gauge.
    projector_reference_ratios, condenser_reference_ratios : array_like
        Relative controls assigned to the reference words (IL1-PL1; CL1, CL3, mini).
    exposure_s : float
        Exposure per image.
    binning : int
        Camera binning.
    model_assumptions : dict
        Fit assumptions for the specification: ``aperture_edge_width_m``,
        ``aperture_diffraction`` and optionally ``crossover_fraction_bounds`` and
        ``moment_margin_pixels``.
    frames : int, optional
        Frames integrated per image (``frameIntegration``).
    projector_aperture_um : float, optional
        Record projector settings through this aperture only; ``None`` records both.
    camera : str, optional
        Detector name.
    log : callable, optional
        Progress messages.

    Returns
    -------
    Path
        The imported experiment directory.

    Raises
    ------
    PlanError
        If any planned word lies outside 0-65535 (nothing is changed then).
    """
    lens, deflector, eos, ht, aperture = tem3.Lens3(), tem3.Def3(), tem3.EOS3(), tem3.HT3(), tem3.Apt3()
    cam = detector.Detector(camera)
    driven = {**PROJECTOR_LENS_IDS, **CONDENSER_LENS_IDS}
    initial = dict(
        ht=ht.GetHtValue(), mode=eos.GetFunctionMode()[0], selector=eos.GetCurrentMagSelectorID()[0],
        spot=eos.GetSpotSize(), flc={name: lens.GetFLCInfo(i) for name, i in driven.items()},
        cla1=deflector.GetCLA1(), cla2=deflector.GetCLA2(), blank=deflector.GetBeamBlank(),
        kind=aperture.GetKind(), hole=aperture.GetSize(CONDENSER_APERTURE_KIND),
        camera=cam.get_detectorsetting())
    images, records = [], []
    try:
        ht.SetHtValue(float(voltages_v[0]))
        eos.SelectFunctionMode(0)
        _select_magnification(eos, reference_magnification)
        eos.SelectSpotSize(int(reference_spot))
        reference = {name: int(getattr(lens, f"Get{name}")()) for name in driven}
        words, infeasible = plan_words(plan, reference_words=reference,
                                       projector_reference_ratios=projector_reference_ratios,
                                       condenser_reference_ratios=condenser_reference_ratios)
        if infeasible:
            raise PlanError("The plan cannot be set on this microscope: " + "; ".join(infeasible))
        cam.set_detectorsetting({"BinningSize": int(binning), "ExposureTimeValue": float(exposure_s) * 1e6,
                                 "frameIntegration": int(frames)})
        setting = cam.get_detectorsetting()
        for i in driven.values():
            lens.SetFLCSw(i, 1)
        deflector.SetBeamBlank(1)
        dark = _decode(cam.snapshot_rawdata(), setting)
        deflector.SetBeamBlank(0)
        expected_dark = frames * (vendor.camera_offset_adu + vendor.camera_gain_adu_per_electron
                                  * vendor.camera_background_electrons)
        log(f"Dark frame mean {dark.mean():.2f} ADU (vendor calibration predicts {expected_dark:.2f}).")
        ratio = {name: (r, reference[name]) for name, r in zip(
            (*PROJECTOR_LENS_IDS, *CONDENSER_LENS_IDS),
            (*np.asarray(projector_reference_ratios, float), *np.asarray(condenser_reference_ratios, float)))}
        voltage_index = np.asarray(plan["voltage_index"])
        for vi, voltage in enumerate(voltages_v):
            rows = np.flatnonzero(voltage_index == vi)
            if not len(rows):
                continue
            ht.SetHtValue(float(voltage))
            for count, row in enumerate(rows):
                for name, i in driven.items():
                    lens.SetFLCAbs(i, int(words[name][row]))
                kick = np.asarray(plan["specimen_kick"][row], dtype=float)
                cla = vendor.cla_words(kick, voltage)
                deflector.SetCLA1(cla[0], cla[1])
                deflector.SetCLA2(cla[2], cla[3])
                purpose = str(plan["purpose"][row])
                diameters = ([projector_aperture_um] if purpose == "projector" and projector_aperture_um
                             else list(aperture_diameters_um))
                applied = {name: int(words[name][row]) * r / w for name, (r, w) in ratio.items()}
                for diameter in diameters:
                    aperture.SelectKind(CONDENSER_APERTURE_KIND)
                    aperture.SetSize(int(aperture_holes[diameter]))
                    adu = _decode(cam.snapshot_rawdata(), setting)
                    images.append((adu - frames * vendor.camera_offset_adu) / vendor.camera_gain_adu_per_electron)
                    record = dict(
                        measurement_id=f"{int(row):05d}", aperture_um=float(diameter), voltage_v=float(voltage),
                        exposure_s=float(exposure_s), purpose=purpose,
                        response_group=int(plan["response_group"][row]),
                        projector_controls=[1.0] + [applied[n] for n in PROJECTOR_LENS_IDS],
                        condenser_controls=[applied[n] for n in CONDENSER_LENS_IDS],
                        specimen_kick=kick.tolist(),
                        lens_words={name: int(words[name][row]) for name in driven}, cla_words=list(cla))
                    if "nominal_magnification" in plan:
                        record["nominal_magnification"] = float(plan["nominal_magnification"][row])
                    records.append(record)
                if count % 25 == 0:
                    log(f"{voltage / 1e3:g} kV: setting {count + 1}/{len(rows)}")
    finally:
        for name, i in driven.items():
            sw, value = initial["flc"][name]
            lens.SetFLCAbs(i, int(value))
            lens.SetFLCSw(i, int(sw))
        deflector.SetCLA1(*initial["cla1"])
        deflector.SetCLA2(*initial["cla2"])
        deflector.SetBeamBlank(int(initial["blank"]))
        aperture.SelectKind(CONDENSER_APERTURE_KIND)
        aperture.SetSize(int(initial["hole"]))
        aperture.SelectKind(int(initial["kind"]))
        eos.SetSelector(int(initial["selector"]))
        eos.SelectSpotSize(int(initial["spot"]))
        eos.SelectFunctionMode(int(initial["mode"]))
        ht.SetHtValue(float(initial["ht"]))
        cam.set_detectorsetting({key: initial["camera"][key] for key in
                                 ("BinningSize", "ImagingArea", "ExposureTimeValue", "frameIntegration")})
    specification = dict(
        model_assumptions,
        voltages_v=[float(v) for v in voltages_v],
        aperture_diameters_um=[float(d) for d in aperture_diameters_um],
        pixel_pitch_m=vendor.sensor_pixel_m * binning,
        psf_sigma_pixels=vendor.psf_sigma_sensor_pixels / binning,
        background_counts=frames * vendor.camera_background_electrons,
        read_noise_counts=float(np.sqrt(frames)) * vendor.camera_read_noise_electrons,
        image_units="electrons", image_axes="row_down_column_right",
        control_units="current/reference_current",
        acquisition="PyJEM protocol: free lens control in the reference gauge; vendor CLA and camera calibration",
    )
    return import_images(images=np.asarray(images), records=records, specification=specification,
                         output_directory=output_directory)
