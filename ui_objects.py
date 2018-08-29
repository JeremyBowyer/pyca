import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import style

style.use("ggplot")

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
    def __init__(self, parent=None, width=5, height=5, dpi=100, controller=None):
        self.controller = controller
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.ax = self.figure.add_subplot(111)
        self.setParent(parent)
    
    def style_plot(self, fig=None, ax=None, xlab="", ylab=""):
        fig.patch.set_facecolor(self.controller.BACKGROUND_SECONDARY)
        ax.set_xlabel(xlab, fontsize=14, color=self.controller.FONT_COLOR)
        ax.set_ylabel(ylab, fontsize=14, color=self.controller.FONT_COLOR)
        ax.tick_params(labelcolor=self.controller.FONT_COLOR)
        fig.tight_layout()
        
    def clear(self):
        self.ax.cla()
        
    def scatter(self, df, x, y, color=None, size=None, alpha=0.5):
        self.clear()
        sizecol = size if size is not None else x
        df.plot.scatter(x=x,y=y,s=df[sizecol].values, alpha=alpha,ax=self.ax)
        self.style_plot(self.figure, self.ax, xlab=x, ylab=y)
        self.draw()