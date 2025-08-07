from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

from PySide6.QtGui import QVector3D
from PySide6.QtCore import (
    Slot,
)
from PySide6.QtWidgets import (
    QMainWindow,
    QTreeView,
)
from PySide6.QtGui import (
    QStandardItemModel,
    QStandardItem,
)
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from dataclasses import asdict

import numpy as np

from . import shapes as comp_geom
from .utils import P2R, R2P, as_gl_lines
from .widgets import GLImageItem, MyDockLabel
from microscope_calibration.stemoverfocus import inplace_sum

from jaxgym.ray import Ray
from jaxgym.run import solve_model, run_iter
from jaxgym.transfer import transfer_rays

if TYPE_CHECKING:
    from jaxgym import Radians


LABEL_RADIUS = 0.3
Z_ORIENT = -1
RAY_COLOR = (0.0, 0.8, 0.0)
XYZ_SCALING = np.asarray((1, 1, 1.0))
LENGTHSCALING = 1
MRAD = 1e-3
UPDATE_RATE = 100
BKG_COLOR_3D = (150, 150, 150, 255)


class GUIWrapper:
    @staticmethod
    def geometry(component):
        vertices = comp_geom.lens(
            0.2,
            Z_ORIENT * component.z,
            64,
        )
        return (gl.GLLinePlotItem(pos=vertices.T, color="white", width=5),)

    @staticmethod
    def label(component) -> gl.GLTextItem:
        return gl.GLTextItem(
            pos=np.array([-LABEL_RADIUS, LABEL_RADIUS, Z_ORIENT * component.z]),
            text=type(component).__name__,
            color="w",
        )


class GridGeomParams(NamedTuple):
    w: float
    h: float
    cx: float
    cy: float
    rotation: "Radians"
    z: float
    shape: Optional[Tuple[int, int]]


class GridGeomMixin:
    def _get_extents(self) -> GridGeomParams:
        # (cx, cy, w, h, rotation, z)
        raise NotImplementedError()

    def _get_image(self):
        return np.asarray(
            (((255, 128, 128, 255),),),
            dtype=np.uint8,
        )

    def _get_grid_verts(self):
        geom = self._get_extents()
        rotation = geom.rotation
        shape = geom.shape
        if shape is None or any(s <= 0 for s in shape):
            return None
        vertices = self._get_mesh(rotation=0.0)
        min_x, min_y, z = vertices.min(axis=0)
        max_x, max_y, _ = vertices.max(axis=0)
        ny, nx = shape
        # this can be done with clever striding / reshaping
        xvals = np.linspace(min_x, max_x, num=nx + 1, endpoint=True)[1:-1]
        xfill = np.asarray((min_y, max_y))
        xfill = np.tile(xfill, xvals.size)
        xvals = np.repeat(xvals, 2)
        yvals = np.linspace(min_y, max_y, num=ny + 1, endpoint=True)[1:-1]
        yfill = np.asarray((min_x, max_x))
        yfill = np.tile(yfill, yvals.size)
        yvals = np.repeat(yvals, 2)

        if rotation != 0.0:
            mag, ang = R2P(xvals + xfill * 1j)
            xcplx = P2R(mag, ang + rotation)
            xvals, xfill = xcplx.real, xcplx.imag
            mag, ang = R2P(yfill + yvals * 1j)
            ycplx = P2R(mag, ang + rotation)
            yfill, yvals = ycplx.real, ycplx.imag

        xlines = np.stack((xvals, xfill, np.full_like(xvals, z)), axis=1)
        ylines = np.stack((yfill, yvals, np.full_like(yvals, z)), axis=1)
        return np.concatenate((xlines, ylines), axis=0)

    def get_geom(self):
        self._geom_state = self._get_extents()
        vertices = self._get_mesh()
        vertices *= XYZ_SCALING
        self.geom_border = gl.GLLinePlotItem(
            pos=np.concatenate((vertices, vertices[:1, :]), axis=0),
            color=(0.0, 0.0, 0.0, 8.0),
            antialias=True,
            mode="line_strip",
        )
        grid_verts = self._get_grid_verts()
        if grid_verts is not None:
            grid_verts *= XYZ_SCALING
            self.geom_grid = gl.GLLinePlotItem(
                pos=grid_verts,
                color=(0.0, 0.0, 0.0, 0.2),
                antialias=True,
                mode="lines",
            )
        self.geom_image = GLImageItem(
            vertices,
            self._get_image(),
        )
        return [self.geom_image, self.geom_grid, self.geom_border]

    def update_geometry(self):
        geom_state = self._get_extents()
        if geom_state == self._geom_state:
            self._geom_state = geom_state
        vertices = self._get_mesh()
        vertices *= XYZ_SCALING
        self.geom_image.setVertices(
            vertices,
        )
        grid_verts = self._get_grid_verts()
        if grid_verts is not None:
            grid_verts *= XYZ_SCALING
            self.geom_grid.setData(
                pos=grid_verts,
                color=(0.0, 0.0, 0.0, 0.3),
                antialias=True,
            )
        self.geom_border.setData(
            pos=np.concatenate((vertices, vertices[:1, :]), axis=0),
            color=(0.0, 0.0, 0.0, 1.0),
            antialias=True,
        )

    def _get_mesh(self, rotation=None):
        geom = self._get_extents()
        if rotation is None:
            rotation = geom.rotation
        vertices, _ = comp_geom.rectangle(
            w=geom.w,
            h=geom.h,
            x=geom.cx,
            y=geom.cy,
            z=Z_ORIENT * geom.z,
            rotation=rotation,
        )
        return vertices


class TemGymWindow(QMainWindow):
    """
    Create the UI Window
    """

    def __init__(self, *args, num_rays: int = 64, **kwargs):
        """Init important parameters

        Parameters
        ----------
        model : class
            Microscope model
        """
        super().__init__(*args, **kwargs)
        self.num_rays = num_rays
        self._model = None

        # Set some main window's properties
        self.setWindowTitle("TemGym")
        self.resize(600, 400)

        # Create Docks
        label = MyDockLabel("3D View")
        self.tem_dock = Dock(label.text(), label=label)
        label = MyDockLabel("Detector")
        self.detector_dock = Dock(label.text(), label=label)
        label = MyDockLabel("Controls")
        self.gui_dock = Dock(label.text(), label=label)

        self.params_tree = QTreeView()
        self.params_tree.setAlternatingRowColors(True)
        self.params_tree.setSortingEnabled(False)
        self.params_tree.setHeaderHidden(False)
        self.params_tree.setIndentation(10)
        self.gui_dock.addWidget(self.params_tree)

        self.centralWidget = DockArea()
        self.setCentralWidget(self.centralWidget)
        self.centralWidget.addDock(self.tem_dock, "left")
        self.centralWidget.addDock(self.gui_dock, "right")
        self.centralWidget.addDock(self.detector_dock, "bottom", self.gui_dock)

        # Create the display and the buttons
        self.create3DDisplay()
        self.createDetectorDisplay()

    def set_model(
        self, model, tree: bool = True, geometry: bool = True, camera: bool = True
    ):
        self._model = model
        if geometry:
            self.add_geometry(model)
        if camera:
            self.update_camera(model)
        if tree:
            self.update_tree(model)
        self.update_rays(model, self.num_rays)

    def add_geometry(self, model):
        self.tem_window.clear()
        # Loop through all of the model model
        # and add their geometry to the TEM window.
        # FIXME Add in reverse to simulate better depth stacking
        wrappers = []
        for component in model:
            try:
                wrapper = component.gui()
            except AttributeError:
                wrapper = GUIWrapper
            wrappers.append(wrapper)
        for wrapper, component in zip(wrappers, model):
            for geometry in wrapper.geometry(component):
                self.tem_window.addItem(geometry)
        # Add labels next so they appear above geometry
        for wrapper, component in zip(wrappers, model):
            self.tem_window.addItem(wrapper.label(component))
        # Add the ray geometry last so it is always on top
        self.tem_window.addItem(self.ray_geometry)

    def update_camera(self, components):
        z_vals = tuple(c.z for c in components)
        mid_z = (min(z_vals) + max(z_vals)) / 2.0
        mid_z *= Z_ORIENT
        xyoffset = (0.2 * mid_z, -0.2 * mid_z)
        self.tem_window.setCameraParams(center=QVector3D(*xyoffset, mid_z))

    def update_tree(self, components: list):
        params_model = QStandardItemModel()
        params_model.setHorizontalHeaderLabels(["Parameter", "Value"])
        params_model.itemChanged.connect(self.handleChanged)
        for component in components:
            comp_row = QStandardItem(type(component).__name__)
            comp_row.setEditable(False)
            for name, val in asdict(component).items():
                if hasattr(val, "size"):
                    val = np.asarray(val).tolist()
                key = QStandardItem(name)
                key.setCheckable(False)
                key.setEditable(False)
                comp_row.appendRow(
                    [
                        key,
                        (param := QStandardItem(f"{val}")),
                    ]
                )
                param.setCheckable(False)
        self.params_tree.setModel(params_model)
        self.params_tree.expandAll()

    @Slot()
    def update_rays(self, model, num_rays: int):
        optical_axis_ray = Ray(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        transfer_matrices = solve_model(optical_axis_ray, model)
        z_vals = np.asarray(
            tuple(ray.z for _, ray in run_iter(optical_axis_ray, model))
        )
        input_rays = model[0].generate_array(num_rays, random=False)

        xy_coords = transfer_rays(input_rays, transfer_matrices)

        vertices = as_gl_lines(xy_coords, z_vals, z_mult=Z_ORIENT)
        self.ray_geometry.setData(
            pos=vertices * XYZ_SCALING,
            color=RAY_COLOR + (0.05,),
        )

        y_px, x_px = model[-1].metres_to_pixels(
            xy_coords[:, -1, :2].T,
        )
        image = np.zeros(model[-1].det_shape, dtype=np.float32)
        inplace_sum(
            np.asarray(y_px),
            np.asarray(x_px),
            np.ones(y_px.shape, dtype=bool),
            np.ones(y_px.shape, dtype=np.float32),
            image,
        )
        self.spot_img.setImage(image)

    @staticmethod
    def qtmodel_to_model(qt_model: QStandardItemModel, model):
        new_model = []
        for i in range(qt_model.rowCount()):
            params = {}
            component = qt_model.item(i)
            temgym_comp = model[i]
            for j in range(component.rowCount()):
                key = component.child(j, 0).text()
                value = component.child(j, 1)
                text = value.text()
                param_type = getattr(temgym_comp, key)
                try:
                    py_val = eval(text, {})
                except (NameError, SyntaxError):
                    py_val = text
                if hasattr(param_type, "size"):
                    py_val = np.asarray(py_val)
                else:
                    py_val = type(param_type)(py_val)
                params[key] = py_val
            new_model.append(type(temgym_comp)(**params))
        return tuple(new_model)

    @Slot()
    def handleChanged(self, item: QStandardItem):
        self.set_model(
            self.qtmodel_to_model(item.model(), self._model)
        )

    def create3DDisplay(self):
        """Create the 3D Display"""
        # Create the 3D TEM Widnow, and plot the components in 3D
        self.tem_window = gl.GLViewWidget()
        self.tem_window.setBackgroundColor(BKG_COLOR_3D)

        # Get the model mean height to centre the camera origin
        mean_z = 0.0
        mean_z *= Z_ORIENT

        xyoffset = (0.2 * mean_z, -0.2 * mean_z)
        # Define Camera Parameters
        initial_camera_params = {
            "center": QVector3D(*xyoffset, mean_z),
            "fov": 35,
            "azimuth": 45.0,
            "distance": 3.5 * abs(mean_z),
            "elevation": 25.0,
        }
        self.tem_window.setCameraParams(**initial_camera_params)

        self.ray_geometry = gl.GLLinePlotItem(mode="lines", width=2)

        # Add the window to the dock
        self.tem_dock.addWidget(self.tem_window)

    def createDetectorDisplay(self):
        """Create the detector display"""
        # Create the detector window, which shows where rays land at the bottom
        self.spot_img = pg.ImageView(
            parent=self.detector_dock,
        )
        self.spot_img.setImage(
            np.random.uniform(size=(128, 128)).astype(np.float32)
        )
        self.spot_img.adjustSize()
        self.detector_dock.addWidget(self.spot_img)
