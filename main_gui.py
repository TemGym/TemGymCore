import sys

from PySide6.QtWidgets import QApplication
from temgym_ui.window import TemGymWindow
import jaxgym.components as comp


model = (
    comp.ParallelBeam(0., 0.01),
    comp.Lens(0.5, 0.05),
    comp.Detector(1., (0.001,) * 2, (128, 128)),
)

AppWindow = QApplication(sys.argv)
viewer = TemGymWindow()
viewer.set_model(model)
viewer.show()
AppWindow.exec()
