"""Import calibrated image arrays without a generating microscope or truth file."""

import json
from pathlib import Path

import numpy as np


def import_images(*, images, records, specification, output_directory):
    """Pack recorded frames for the existing measurement and fitting functions.

    Images are finite electron-count arrays (frame, row, column), with rows down
    and columns right. Each record supplies a measurement_id, aperture_um,
    voltage_v, exposure_s, purpose, projector_controls (5 current ratios),
    condenser_controls (3), specimen_kick (x,y,theta_x,theta_y in m/rad), and
    response_group (-1 except for a signed steering pair). The two aperture
    images share a measurement_id; projector settings may have just one image.

    Detector calibration and explicit model assumptions are supplied in
    specification. No TIFF/ADU conversion, steering calibration or current-to-DAC
    conversion is inferred. The destination must be empty to avoid stale fits.
    """
    images = np.asarray(images, dtype=float)
    records = list(records)
    spec = dict(specification)
    required = ["voltages_v", "aperture_diameters_um", "pixel_pitch_m",
                "psf_sigma_pixels", "background_counts", "read_noise_counts",
                "aperture_edge_width_m", "aperture_diffraction"]
    for name in required:
        if name not in spec:
            raise ValueError(f"Supply specification[{name!r}] explicitly.")
    conventions = dict(image_units="electrons", image_axes="row_down_column_right",
                       control_units="current/reference_current")
    for name, value in conventions.items():
        if spec.get(name) != value:
            raise ValueError(f"Supply {name}={value!r}; conversion is not inferred.")
    if images.ndim != 3 or len(images) != len(records) or not len(images) or not np.isfinite(images).all():
        raise ValueError("Supply one finite 2D electron-count image per record.")
    for name in ["voltages_v", "aperture_diameters_um"]:
        a = np.asarray(spec[name], dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all() or min(a) <= 0 or a[0] == a[1]:
            raise ValueError(f"{name} must contain two distinct positive values.")
        spec[name] = a.tolist()
    for name in ["pixel_pitch_m", "aperture_edge_width_m", "psf_sigma_pixels",
                 "background_counts", "read_noise_counts"]:
        value = float(spec[name])
        if not np.isfinite(value) or value < 0 or (name in ["pixel_pitch_m", "aperture_edge_width_m"] and value == 0):
            raise ValueError(f"Invalid {name}.")
    if not isinstance(spec["aperture_diffraction"], bool):
        raise ValueError("Choose aperture_diffraction=True or False explicitly.")
    if "saturation_counts" in spec and np.any(images >= spec["saturation_counts"]):
        raise ValueError("A frame reaches the supplied saturation level.")

    ids = list(dict.fromkeys(str(r["measurement_id"]) for r in records))
    slots = {name: i for i, name in enumerate(ids)}
    packed = np.full((len(ids), 2, *images.shape[1:]), np.nan)
    rows, exposures = [None] * len(ids), [None] * len(ids)
    for frame, record in zip(images, records):
        i = slots[str(record["measurement_id"])]
        ai = spec["aperture_diameters_um"].index(record["aperture_um"])
        vi = spec["voltages_v"].index(record["voltage_v"])
        row = dict(voltage_index=vi, purpose=record["purpose"],
                   response_group=int(record["response_group"]))
        if row["response_group"] != record["response_group"] or row["response_group"] < -1:
            raise ValueError("response_group must be -1 or a nonnegative integer.")
        for name, size in [("projector_controls", 5), ("condenser_controls", 3), ("specimen_kick", 4)]:
            a = np.asarray(record[name], dtype=float)
            if a.shape != (size,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {name} for measurement {ids[i]}.")
            if name != "specimen_kick" and np.any(a <= 0):
                raise ValueError("This calibration assumes positive lens current ratios.")
            row[name] = a
        if row["purpose"] not in ["projector", "condenser", "check"]:
            raise ValueError("purpose must be projector, condenser or check.")
        exposure = float(record["exposure_s"])
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("Record a positive exposure_s for every frame.")
        if np.isfinite(packed[i, ai]).any():
            raise ValueError(f"Duplicate aperture image for measurement {ids[i]}.")
        if rows[i] is not None:
            if any(not np.array_equal(row[k], rows[i][k]) for k in row) or exposure != exposures[i]:
                raise ValueError("Aperture pairs must share controls, steering and exposure.")
        rows[i], exposures[i], packed[i, ai] = row, exposure, frame
    recorded = np.isfinite(packed).all(axis=(-2, -1))
    settings = {name: np.array([r[name] for r in rows]) for name in rows[0]}
    settings["nominal_magnification"] = np.array([
        next(r.get("nominal_magnification", np.nan) for r in records if str(r["measurement_id"]) == name)
        for name in ids
    ])
    settings["lens_change"] = np.full(len(ids), "Recorded setting")
    if not recorded[settings["purpose"] != "projector"].all():
        raise ValueError("Condenser and check settings require both aperture images.")
    if not np.allclose(settings["projector_controls"][:, 0], settings["projector_controls"][0, 0], rtol=0, atol=1e-12):
        raise ValueError("The calibrated workflow keeps objective current fixed.")
    for vi in range(2):
        mask = (settings["purpose"] == "projector") & (settings["voltage_index"] == vi)
        if len(np.unique(settings["condenser_controls"][mask], axis=0)) > 1:
            raise ValueError("Projector calibration needs one illumination per voltage.")
    for group in np.unique(settings["response_group"]):
        if group < 0:
            continue
        pair = np.flatnonzero(settings["response_group"] == group)
        if len(pair) != 2:
            raise ValueError("Each response_group must contain two opposite steering settings.")
        a, b = pair
        if (not np.array_equal(settings["specimen_kick"][a], -settings["specimen_kick"][b])
                or np.count_nonzero(settings["specimen_kick"][a]) != 1
                or any(not np.array_equal(settings[k][a], settings[k][b]) for k in
                       ["projector_controls", "condenser_controls", "voltage_index", "purpose"])
                or not np.array_equal(recorded[a], recorded[b])):
            raise ValueError("Signed responses need matching optics/apertures and opposite single-axis kicks.")
    if any(r["purpose"] == "projector" and np.any(r["specimen_kick"]) and r["response_group"] < 0 for r in rows):
        raise ValueError("Every nonzero projector steering command needs a response_group.")

    root = Path(output_directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Import into an empty directory; existing data are not overwritten.")
    spec.update(shape_yx=list(images.shape[1:]), data_source="recorded electron-count arrays",
                measurement_noise_added=True, aperture_inserted_for_every_image=True,
                objective_current=float(settings["projector_controls"][0, 0]))
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "images.npz", images=packed)
    np.savez_compressed(root / "settings.npz", **settings)
    (root / "experiment.json").write_text(json.dumps(spec, indent=2) + "\n")
    (root / "records.json").write_text(json.dumps(records, indent=2, default=lambda x: np.asarray(x).tolist()) + "\n")
    return root
