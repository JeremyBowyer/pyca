import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtWidgets
import PyQt5.QtWidgets
import PyQt5.QtGui
from PyQt5.QtCore import QSize 

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ColumnList(QtWidgets.QListWidget):
    def __init__(self, parent, controller):
        QtWidgets.QListWidget.__init__(self, parent)
        self.controller = controller
        self.controller.columnLists.append(self)
        if self.controller.is_df_loaded:
            self.addItems(self.controller.df.columns)
        self.setSelectionMode(QtWidgets.QListWidget.MultiSelection)

        
class ColumnComboBox(QtWidgets.QComboBox):
    def __init__(self, parent, controller):
        QtWidgets.QComboBox.__init__(self, parent)
        self.controller = controller
        self.controller.columnLists.append(self)
        if self.controller.is_df_loaded:
            self.addItems(self.controller.df.columns)

class ScatterCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.ax = self.figure.add_subplot(111)
        self.setParent(parent)
    
    def clear(self):
        self.ax.cla()
        
    def scatter(self, df, x, y, color=None, size=None, alpha=0.5):
        self.clear()
        sizecol = size if size is not None else x
        df.plot.scatter(x=x,y=y,s=df[sizecol].values, alpha=alpha,ax=self.ax)
        self.draw()