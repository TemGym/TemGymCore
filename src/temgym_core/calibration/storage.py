"""Content fingerprints for recorded arrays and calibration stage reuse."""

import hashlib
import json
from pathlib import Path

import numpy as np


def input_signature(*, paths, settings):
    """Fingerprint array contents, avoiding irrelevant NPZ zip timestamps."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for filename in paths:
        path = Path(filename)
        if path.suffix == ".npz":
            with np.load(path) as data:
                for key in sorted(data.files):
                    value = np.asarray(data[key])
                    digest.update(key.encode())
                    digest.update(str((value.shape, value.dtype)).encode())
                    digest.update(value.tobytes())
        else:
            digest.update(path.read_bytes())
    return digest.hexdigest()
