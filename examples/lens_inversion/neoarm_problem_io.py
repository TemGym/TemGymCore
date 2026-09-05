"""Portable, checksummed numeric snapshot of the archived inverse problem.

No Python functions or pickles are serialized. Only explicitly allowed TemGym
dataclasses, numeric arrays, tuples and JSON values are reconstructed.
"""

from dataclasses import fields, is_dataclass
import hashlib
import importlib
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np


MODULES = (
    "temgym_core.microscope_inversion",
    "temgym_core.neoarm_calibration",
    "temgym_core.physical_observables",
    "temgym_core.layer_metrology",
    "temgym_core.physical_nuisance",
    "temgym_core.condenser_astigmatic_likelihood",
)


def _encode(value):
    if is_dataclass(value):
        name = type(value).__module__ + ":" + type(value).__name__
        if type(value).__module__ not in MODULES:
            raise TypeError(f"Unsupported snapshot class {name}")
        return {"class": name, "fields": {
            field.name: _encode(getattr(value, field.name))
            for field in fields(value) if field.init}}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError("Snapshot arrays must be finite real numeric arrays.")
        return {"array": value.tolist(), "dtype": value.dtype.str}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"tuple": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported snapshot value {type(value)}")


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"class", "fields"}:
        module, name = value["class"].split(":")
        if module not in MODULES or name.startswith("_"):
            raise ValueError("Snapshot class is not permitted.")
        cls = getattr(importlib.import_module(module), name)
        if not isinstance(cls, type) or not is_dataclass(cls):
            raise ValueError("Snapshot class must be a public dataclass.")
        return cls(**{key: _decode(item) for key, item in value["fields"].items()})
    if set(value) == {"array", "dtype"}:
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Snapshot dtype must be real numeric.")
        array = np.asarray(value["array"], dtype=dtype)
        if not np.all(np.isfinite(array)):
            raise ValueError("Snapshot arrays must be finite.")
        return array
    if set(value) == {"tuple"}:
        return tuple(_decode(item) for item in value["tuple"])
    return {key: _decode(item) for key, item in value.items()}


def save_problem(path, **values):
    payload = _encode(values)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document = {"schema": "neoarm-recovery-problem-v1", "payload": payload,
                "sha256": hashlib.sha256(encoded.encode()).hexdigest()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n")


def load_problem(path):
    document = json.loads(Path(path).read_text())
    if document["schema"] != "neoarm-recovery-problem-v1":
        raise ValueError("Unsupported problem schema.")
    encoded = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(encoded.encode()).hexdigest() != document["sha256"]:
        raise ValueError("Problem snapshot checksum mismatch.")
    return _decode(document["payload"])
