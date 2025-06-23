from numba import njit
import numpy as np
from libertem.udf import UDF

from .model import ModelParameters, create_stem_model
from .stemoverfocus import project_frame_backward


@njit
def mask_via_for(px_y, px_x, values, buffer):
    ny, nx = buffer.shape
    n = px_y.shape[0]
    for i in range(n):
        py = px_y[i]
        px = px_x[i]
        if 0 <= py < ny and 0 <= px < nx:
            buffer[py, px] += values[i]


class ShiftedSumUDF(UDF):
    def get_task_data(self):
        # Ran once per-partition and re-used
        params_dict = ModelParameters(**self.params.model_parameters)
        model = create_stem_model(params_dict)
        scan_coords = model.scan_grid.coords
        detector_coords = model.detector.coords
        return {
            "model": model,
            "scan_coords": scan_coords,
            "detector_coords": detector_coords,
        }

    def get_result_buffers(self):
        dtype = np.result_type(
            self.meta.input_dtype,
            np.float32,
        )
        return {
            "shifted_sum": self.buffer(
                kind="single",
                dtype=dtype,
                extra_shape=self.meta.dataset_shape.nav,
            ),
        }

    def process_frame(self, frame: np.ndarray):
        if self.params.get("shifts", None) is not None:
            shifts = self.params.shifts
            frame = np.roll(frame, -1 * shifts, axis=(0, 1))
        scan_pos_flat = np.ravel_multi_index(
            self.meta.coordinates.ravel(),
            self.meta.dataset_shape.nav,
        )
        det_coords = self.task_data.detector_coords
        scan_pos = self.task_data.scan_coords[scan_pos_flat]
        model = self.task_data.model
        # if self.params.get('shifts') is not None:
        #     # correct descan error in the pixel coordinate system
        #     frame = np.roll(frame, self.params.get('shifts'), axis=(0, 1))
        px_y, px_x, values = project_frame_backward(model, det_coords, frame, scan_pos)
        mask_via_for(
            np.array(px_y), np.array(px_x), np.array(values), self.results.shifted_sum
        )

    def merge(self, dest, src):
        dest.shifted_sum += src.shifted_sum
