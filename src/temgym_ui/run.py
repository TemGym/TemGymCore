import sys

from PySide6.QtWidgets import QApplication
from .window import TemGymWindow


def temgym(model):
    AppWindow = QApplication(sys.argv)
    viewer = TemGymWindow()
    viewer.set_model(model)
    viewer.show()
    AppWindow.exec()
