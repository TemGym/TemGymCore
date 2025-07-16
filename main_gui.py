import sys

from PySide6.QtWidgets import QApplication
from temgym_ui.window import TemGymWindow
import microscope_calibration.components as comp


model = (
    comp.PointSource(0., 0.01),
    comp.Detector(0.5, (0.01,) * 2, (128, 128)),
    comp.Detector(1., (0.001,) * 2, (128, 128)),
)

AppWindow = QApplication(sys.argv)
viewer = TemGymWindow()
viewer.set_model(model)
viewer.show()
AppWindow.exec()
