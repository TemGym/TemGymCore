import numpy as np
from libertem.udf import UDF
from libertem.common.buffers import AuxBufferWrapper

from .model import ModelParameters, create_stem_model
from .stemoverfocus import project_coordinates_backward, inplace_sum, solve_model_fourdstem_wrapper


class ShiftedSumUDF(UDF):
    def __init__(
        self,
        model_parameters: ModelParameters,
        shifts: AuxBufferWrapper | None = None,
    ):
        super().__init__(model_parameters=model_parameters, shifts=shifts)

    def get_task_data(self):
        # Ran once per-partition and re-used
        params_dict = ModelParameters(**self.params.model_parameters)
        model = create_stem_model(params_dict)
        scan_coords = model.scan_grid.coords
        detector_coords = model.detector.coords

        total_tm_and_grad, scangrid_tm_and_grad = solve_model_fourdstem_wrapper(model)

        return {
            "model": model,
            "total_tm_and_grad": total_tm_and_grad,
            "scangrid_tm_and_grad": scangrid_tm_and_grad,
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
        model = self.task_data.model
        total_tm_and_grad = self.task_data.total_tm_and_grad
        scangrid_tm_and_grad = self.task_data.scangrid_tm_and_grad
        det_coords = self.task_data.detector_coords
        scan_pos = self.task_data.scan_coords[scan_pos_flat]

        px_y, px_x, mask = project_coordinates_backward(
            model,
            total_tm_and_grad,
            scangrid_tm_and_grad,
            det_coords,
            scan_pos,
        )
        inplace_sum(
            np.array(px_y),
            np.array(px_x),
            np.array(mask),
            frame.ravel(),
            self.results.shifted_sum,
        )

    def merge(self, dest, src):
        dest.shifted_sum += src.shifted_sum
