import numpy as np
import pandas as pd
import transformations
import filters

from IPython import embed;

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QSortFilterProxyModel, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QCompleter, QComboBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
import pyqtgraph as pg


class ExtendedComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(ExtendedComboBox, self).__init__(parent)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setEditable(True)

        # add a filter model to filter matching items
        self.pFilterModel = QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())

        # add a completer, which uses the filter model
        self.completer = QCompleter(self.pFilterModel, self)
        # always show all (filtered) completions
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)

        # connect signals
        self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)


    # on selection of an item from the completer, select the corresponding item from combobox 
    def on_completer_activated(self, text):
        if text:
            index = self.findText(text)
            self.setCurrentIndex(index)
            self.activated[str].emit(self.itemText(index))


    # on model change, update the models of the filter and completer as well 
    def setModel(self, model):
        super(ExtendedComboBox, self).setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)


    # on model column change, update the model column of the filter and completer as well
    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super(ExtendedComboBox, self).setModelColumn(column) 


class ColumnComboBox(ExtendedComboBox):
    def __init__(self, parent, controller):
        super(ColumnComboBox, self).__init__(parent)
        self.controller = controller
        self.controller.columnLists.append(self)
        if self.controller.is_df_loaded:
            self.addItems(self.controller.all_cols)

    def update_cols(self, all_cols, x_cols):
        new_cols = [""] + all_cols
        try:
            idx = new_cols.index(self.currentText())
        except ValueError:
            idx = 0
        self.clear()
        self.addItems(new_cols)
        self.setCurrentIndex(idx)

    def deleteLater(self):
        self.controller.columnLists.remove(self)
        super(ColumnComboBox, self).deleteLater()


class XColumnComboBox(ExtendedComboBox):
    def __init__(self, parent, controller):
        super(XColumnComboBox, self).__init__(parent)
        self.controller = controller
        self.controller.columnLists.append(self)
        if self.controller.is_df_loaded:
            self.addItems(self.controller.x_cols)

    def update_cols(self, all_cols, x_cols):
        new_cols = [""] + x_cols
        try:
            idx = new_cols.index(self.currentText())
        except ValueError:
            idx = 0
        self.clear()
        self.addItems(new_cols)
        self.setCurrentIndex(idx)

    def deleteLater(self):
        self.controller.columnLists.remove(self)
        super(XColumnComboBox, self).deleteLater()


class ColumnList(QtWidgets.QListWidget):
    def __init__(self, parent, controller):
        QtWidgets.QListWidget.__init__(self, parent)
        self.controller = controller
        self.controller.columnLists.append(self)
        if self.controller.is_df_loaded:
            self.addItems(self.controller.all_cols)
        self.setSelectionMode(QtWidgets.QListWidget.MultiSelection)

    def update_cols(self, all_cols, x_cols):
        new_cols = [""] + all_cols
        current_selection = [x.text() for x in self.selectedItems()]
        self.clear()
        for col in new_cols:
            item = QtWidgets.QListWidgetItem(col)
            self.addItem(item)
            if col in current_selection:
                item.setSelected(True)

    def deleteLater(self):
        self.controller.columnLists.remove(self)
        super(ColumnList, self).deleteLater()


class XColumnList(QtWidgets.QListWidget):
    def __init__(self, parent, controller):
        QtWidgets.QListWidget.__init__(self, parent)
        self.controller = controller
        self.controller.columnLists.append(self)
        if self.controller.is_df_loaded:
            self.addItems(self.controller.x_cols)
        self.setSelectionMode(QtWidgets.QListWidget.MultiSelection)

    def update_cols(self, all_cols, x_cols):
        new_cols = [""] + x_cols
        current_selection = [x.text() for x in self.selectedItems()]
        self.clear()
        for col in new_cols:
            item = QtWidgets.QListWidgetItem(col)
            self.addItem(item)
            if col in current_selection:
                item.setSelected(True)

    def deleteLater(self):
        self.controller.columnLists.remove(self)
        super(XColumnList, self).deleteLater()

class FlowLayout(QtGui.QLayout):
    # https://stackoverflow.com/questions/41621354/pyqt-wrap-around-layout-of-widgets-inside-a-qscrollarea
    def __init__(self, parent=None, margin=-1, hspacing=-1, vspacing=-1):
        super(FlowLayout, self).__init__(parent)
        self._hspacing = hspacing
        self._vspacing = vspacing
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        del self._items[:]

    def addItem(self, item):
        self._items.append(item)

    def horizontalSpacing(self):
        if self._hspacing >= 0:
            return self._hspacing
        else:
            return self.smartSpacing(
                QtGui.QStyle.PM_LayoutHorizontalSpacing)

    def verticalSpacing(self):
        if self._vspacing >= 0:
            return self._vspacing
        else:
            return self.smartSpacing(
                QtGui.QStyle.PM_LayoutVerticalSpacing)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)

    def expandingDirections(self):
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.doLayout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        size += QtCore.QSize(left + right, top + bottom)
        return size

    def doLayout(self, rect, testonly):
        left, top, right, bottom = self.getContentsMargins()
        effective = rect.adjusted(+left, +top, -right, -bottom)
        x = effective.x()
        y = effective.y()
        lineheight = 0
        for item in self._items:
            widget = item.widget()
            hspace = self.horizontalSpacing()
            if hspace == -1:
                hspace = widget.style().layoutSpacing(
                    QtGui.QSizePolicy.PushButton,
                    QtGui.QSizePolicy.PushButton, QtCore.Qt.Horizontal)
            vspace = self.verticalSpacing()
            if vspace == -1:
                vspace = widget.style().layoutSpacing(
                    QtGui.QSizePolicy.PushButton,
                    QtGui.QSizePolicy.PushButton, QtCore.Qt.Vertical)
            nextX = x + item.sizeHint().width() + hspace
            if nextX - hspace > effective.right() and lineheight > 0:
                x = effective.x()
                y = y + lineheight + vspace
                nextX = x + item.sizeHint().width() + hspace
                lineheight = 0
            if not testonly:
                item.setGeometry(
                    QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))
            x = nextX
            lineheight = max(lineheight, item.sizeHint().height())
        return y + lineheight - rect.y() + bottom

    def smartSpacing(self, pm):
        parent = self.parent()
        if parent is None:
            return -1
        elif parent.isWidgetType():
            return parent.style().pixelMetric(pm, None, parent)
        else:
            return parent.spacing()


class TransformationSummaryBox(QWidget):
    def __init__(self, parent, controller, layout):
        super(TransformationSummaryBox, self).__init__(parent)
        self.controller = controller
        self.parent = parent
        self.layout = layout
        self.build_layout()

    def build_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignLeft)

        # Top Layout
        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignCenter)
        self.tran_label = QLabel("Transformation", self)
        self.tran_label.setAlignment(Qt.AlignCenter)
        self.tran_label.setFont(self.controller.LARGE_FONT)
        self.top_layout.addWidget(self.tran_label)

        # Content Layout
        self.content_layout = QVBoxLayout()
        self.content_layout.setAlignment(Qt.AlignLeft)

        # Add to main layout
        self.main_layout.addLayout(self.top_layout, 1)
        self.main_layout.addLayout(self.content_layout, 10)

        # Create close button
        self.remove_btn = QPushButton("", self)
        self.remove_btn.setIcon(QtGui.QIcon("icons/close.png"))
        self.remove_btn.setIconSize(QtCore.QSize(28,28))
        self.remove_btn.resize(32, 32)
        self.remove_btn.clicked.connect(self.remove_widget)
        self.remove_btn.move(self.width() - self.remove_btn.width() - 1, 1)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(42, 42, 42))
        self.setPalette(p)
    
    def remove_widget(self):
        self.parent.remove_transformation_signal.emit(self.transformation)
        self.layout.removeWidget(self)
        self.deleteLater()
        self = None

    def resizeEvent(self, event):
        # move to top-right corner
        self.remove_btn.move(self.width() - self.remove_btn.width() - 5, 5)
        super(TransformationSummaryBox, self).resizeEvent(event)


class TransformationPopupBox(QWidget):

    def __init__(self, parent, controller):
        super(TransformationPopupBox, self).__init__()
        self.parent = parent
        self.controller = controller
        self.build_layout()        
        self.setMinimumHeight(500)
        self.setMinimumWidth(500)
        self.show()

    def build_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        
        # Dropdown
        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignCenter)
        
        self.dropdown = QtWidgets.QComboBox(self)
        self.dropdown.addItems([""] + list(transformations.transformation_types.keys()))
        self.dropdown.currentTextChanged.connect(self.load_transformation)
        self.top_layout.addWidget(self.dropdown)

        # Content
        self.content_layout = QtWidgets.QVBoxLayout()
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QtWidgets.QWidget(self)
        self.widget_layout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.content_layout.addWidget(self.scrollArea)
        
        # Confirm
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.btn_layout.setAlignment(Qt.AlignCenter)
        self.confirm_btn = QPushButton("Create Transformation", self)
        self.confirm_btn.clicked.connect(self.validate_inputs)
        self.confirm_btn.setFont(self.controller.NORMAL_BTN_FONT)
        self.confirm_btn.setFixedSize(225, 50)
        self.btn_layout.addWidget(self.confirm_btn)

        self.main_layout.addLayout(self.top_layout, 1)
        self.main_layout.addLayout(self.content_layout, 10)
        self.main_layout.addLayout(self.btn_layout, 1)

    def validate_inputs(self):
        if self.transformation.validate_inputs(self):
            self.parent.create_transformation_signal.emit(self.transformation)
            self.unload_transformation()
            self.close()

    def load_transformation(self, value):
        self.unload_transformation()
        try:
            self.transformation = transformations.transformation_types[self.dropdown.currentText()]()
        except KeyError:
            return
        self.transformation.add_widgets(self.widget_layout, self, self.controller)

    def unload_transformation(self):
        self.transformation = None
        for i in reversed(range(self.widget_layout.count())): 
            widgetToRemove = self.widget_layout.itemAt(i).widget()
            self.widget_layout.removeWidget(widgetToRemove)
            widgetToRemove.deleteLater()

    def closeEvent(self, event):
        self.unload_transformation()
        event.accept()


class FilterSummaryBox(QWidget):
    def __init__(self, parent, controller, layout):
        super(FilterSummaryBox, self).__init__(parent)
        self.controller = controller
        self.parent = parent
        self.layout = layout
        self.build_layout()

    def build_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignLeft)

        # Top Layout
        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignCenter)
        self.filter_label = QLabel("Filter", self)
        self.filter_label.setAlignment(Qt.AlignCenter)
        self.filter_label.setFont(self.controller.LARGE_FONT)
        self.top_layout.addWidget(self.filter_label)

        # Content Layout
        self.content_layout = QVBoxLayout()
        self.content_layout.setAlignment(Qt.AlignLeft)

        # Add to main layout
        self.main_layout.addLayout(self.top_layout, 1)
        self.main_layout.addLayout(self.content_layout, 10)

        # Create close button
        self.remove_btn = QPushButton("", self)
        self.remove_btn.setIcon(QtGui.QIcon("icons/close.png"))
        self.remove_btn.setIconSize(QtCore.QSize(28,28))
        self.remove_btn.resize(32, 32)
        self.remove_btn.clicked.connect(self.remove_widget)
        self.remove_btn.move(self.width() - self.remove_btn.width() - 1, 1)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(42, 42, 42))
        self.setPalette(p)
    
    def remove_widget(self):
        self.parent.remove_filter_signal.emit(self.filter)
        self.layout.removeWidget(self)
        self.deleteLater()
        self = None

    def resizeEvent(self, event):
        # move to top-right corner
        self.remove_btn.move(self.width() - self.remove_btn.width() - 5, 5)
        super(FilterSummaryBox, self).resizeEvent(event)


class FilterPopupBox(QWidget):

    def __init__(self, parent, controller):
        super(FilterPopupBox, self).__init__()
        self.parent = parent
        self.controller = controller
        self.build_layout()        
        self.setMinimumHeight(500)
        self.setMinimumWidth(500)
        self.show()

    def build_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        
        # Dropdown
        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignCenter)
        
        self.dropdown = QtWidgets.QComboBox(self)
        self.dropdown.addItems([""] + list(filters.filter_types.keys()))
        self.dropdown.currentTextChanged.connect(self.load_filter)
        self.top_layout.addWidget(self.dropdown)

        # Content
        self.content_layout = QtWidgets.QVBoxLayout()
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QtWidgets.QWidget(self)
        self.widget_layout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.content_layout.addWidget(self.scrollArea)
        
        # Confirm
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.btn_layout.setAlignment(Qt.AlignCenter)
        self.confirm_btn = QPushButton("Apply Filter", self)
        self.confirm_btn.clicked.connect(self.validate_inputs)
        self.confirm_btn.setFont(self.controller.NORMAL_BTN_FONT)
        self.confirm_btn.setFixedSize(225, 50)
        self.btn_layout.addWidget(self.confirm_btn)

        self.main_layout.addLayout(self.top_layout, 1)
        self.main_layout.addLayout(self.content_layout, 10)
        self.main_layout.addLayout(self.btn_layout, 1)

    def validate_inputs(self):
        if self.filter.validate_inputs(self):
            self.parent.apply_filter_signal.emit(self.filter)
            self.unload_filter()
            self.close()

    def load_filter(self, value):
        self.unload_filter()
        try:
            self.filter = filters.filter_types[self.dropdown.currentText()]()
        except KeyError:
            return
        self.filter.add_widgets(self.widget_layout, self, self.controller)

    def unload_filter(self):
        self.filter = None
        for i in reversed(range(self.widget_layout.count())): 
            widgetToRemove = self.widget_layout.itemAt(i).widget()
            self.widget_layout.removeWidget(widgetToRemove)
            widgetToRemove.deleteLater()

    def closeEvent(self, event):
        self.unload_filter()
        event.accept()



class ScatterPlot():
    def __init__(self, parent=None, controller=None):
        self.parent = parent
        self.controller = controller
        self.view = pg.GraphicsLayoutWidget()
        self.plot = self.view.addPlot()
        self.lastClicked = []
        self.lastHover = []

    def scatter(self, x, y, size=10, alpha=0.5):
        self.plot.clear()
        series = pg.ScatterPlotItem(size=size, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 255*alpha))
        series.setData(x, y)
        series.sigClicked.connect(self.on_click)
        self.plot.addItem(series)
        self.plot.autoRange()

    def on_click(self, plot, points):
        for p in self.lastClicked:
            p.resetPen()
        for p in points:
            p.setPen('b', width=2)
        self.lastClicked = points

class Histogram():
    def __init__(self, parent=None, controller=None):
        self.parent = parent
        self.controller = controller
        self.view = pg.GraphicsLayoutWidget()
        self.plot = self.view.addPlot()

    def histogram(self, vals):
        self.plot.clear()
        y,x = np.histogram(vals, bins=10)
        self.plot.plot(x,y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
