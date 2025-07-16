from typing import Optional
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

@dataclass
class Rays:
    xy_coords: NDArray
    z_coords: NDArray
    path_length: NDArray
    wavelength: Optional[float] = None
    mask: Optional[NDArray] = None
    blocked: Optional[NDArray] = None

    def __eq__(self, other: 'Rays') -> bool:
        return self.num == other.num and (self.xy_coords == other.xy_coords).all()

    @property
    def num(self):
        return self.xy_coords.shape[0]

    @property
    def num_display(self):
        return self.num

    @property
    def x(self):
        return self.xy_coords[:, :, 0]

    @x.setter
    def x(self, xpos):
        self.xy_coords[:, :, 0] = xpos

    @property
    def y(self):
        return self.xy_coords[:, :, 1]

    @y.setter
    def y(self, ypos):
        self.xy_coords[:, :, 1] = ypos

    @property
    def yx(self):
        return self.xy_coords[:, :, [1, 0]]

    @property
    def dx(self):
        return self.xy_coords[:, :, 2]

    @dx.setter
    def dx(self, xslope):
        self.xy_coords[:, :, 2] = xslope

    @property
    def dy(self):
        return self.xy_coords[:, :, 3]

    @dy.setter
    def dy(self, yslope):
        self.xy_coords[:, :, 3] = yslope

    @property
    def num_display(self):
        return self.num

    @property
    def x_central(self):
        return self.x

    @property
    def y_central(self):
        return self.y
    
    @property
    def dx_central(self):
        return self.dx

    @property
    def dy_central(self):
        return self.dy

    @property
    def mask_display(self):
        return self.mask
    
    @property
    def num_matrices(self):
        return self.xy_coords.shape[1]
    
    @property
    def num_components(self):
        return np.ceil(self.xy_coords.shape[1] / 2)
    
    @property
    def xyz_coords(self):
        return np.concatenate((self.xy_coords[:, :, [0, 1]], self.z_coords[:, :, None]), axis=-1)
