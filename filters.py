import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.stats import pearsonr
import time
import psutil
import csv
import datetime
import gc
import re
from abc import ABC, abstractmethod

import ui_objects
from qrangeslider import QRangeSlider

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel, QLineEdit, QMessageBox, QCheckBox, QListWidget, QComboBox
from PyQt5.QtGui import *

from IPython import embed;

class Filter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def validate_inputs(self, owner):
        pass

    @abstractmethod
    def apply_filter(self, df):
        pass

    @abstractmethod
    def add_widgets(self, layout, owner, controller):
        pass

    @abstractmethod
    def add_summary_widgets(self, box, controller):
        pass

class ValueFilter(Filter):
    def __init__(self):
        super(ValueFilter).__init__()
        self.col_ids = []
        self.name = "Value Filter"

    def add_widgets(self, layout, owner, controller):
        filter_col_label = QLabel("Column to filter on", owner)
        self.filter_col_box = ui_objects.ColumnComboBox(owner, controller)
        self.filter_col_box.setFixedWidth(450)
        min_label = QLabel("Minimum Value (inclusive)", owner)
        self.min_value_le = QLineEdit(owner)
        max_label = QLabel("Maximum Value (inclusive)", owner)
        self.max_value_le = QLineEdit(owner)
        self.remove_matched_box = QCheckBox("REMOVE matched rows?")

        layout.addWidget(filter_col_label)
        layout.addWidget(self.filter_col_box)
        layout.addWidget(min_label)
        layout.addWidget(self.min_value_le)
        layout.addWidget(max_label)
        layout.addWidget(self.max_value_le)
        layout.addWidget(self.remove_matched_box)

    def add_summary_widgets(self, box, controller):
        box.filter = self

        box.filter_col_label = QLabel("Column Filtered On:", box)
        box.filter_col_label.setFont(controller.NORMAL_FONT)
        box.filter_col = QLabel(self.filter_col, box)
        box.filter_col.setFont(controller.SMALL_FONT)

        box.filter_label.setText("Value Filter")
        box.min_label = QLabel("Minimum Value:", box)
        box.min_label.setFont(controller.NORMAL_FONT)
        box.min_value = QLabel(str(self.min_value), box)
        box.min_value.setFont(controller.SMALL_FONT)

        box.max_label = QLabel("Minimum Value:", box)
        box.max_label.setFont(controller.NORMAL_FONT)
        box.max_value = QLabel(str(self.max_value), box)
        box.max_value.setFont(controller.SMALL_FONT)

        box.filter_out_label = QLabel("REMOVE matched rows?", box)
        box.filter_out_label.setFont(controller.NORMAL_FONT)
        box.filter_out_value = QLabel(str(self.remove_matched), box)
        box.filter_out_value.setFont(controller.SMALL_FONT)

        box.content_layout.addWidget(box.filter_col_label)
        box.content_layout.addWidget(box.filter_col)
        box.content_layout.addWidget(box.min_label)
        box.content_layout.addWidget(box.min_value)
        box.content_layout.addWidget(box.max_label)
        box.content_layout.addWidget(box.max_value)
        box.content_layout.addWidget(box.filter_out_label)
        box.content_layout.addWidget(box.filter_out_value)

        box.setFixedHeight(500)
        box.setFixedWidth(500)

    def validate_inputs(self, owner):
        self.filter_col = self.filter_col_box.currentText()

        try:
            self.min_value = float(self.min_value_le.text())
        except ValueError:
            self.min_value = np.nan

        try:
            self.max_value = float(self.max_value_le.text())
        except ValueError:
            self.max_value = np.nan

        self.remove_matched = self.remove_matched_box.isChecked()

        if self.filter_col == "":
            QMessageBox.about(owner, "Warning", "Please provide a column to filter on.")
            return False
        if  np.isnan(self.min_value):
            QMessageBox.about(owner, "Warning", "Please provide a valid number for the minimum value.")
            return False
        if  np.isnan(self.max_value):
            QMessageBox.about(owner, "Warning", "Please provide a valid number for the maximum value.")
            return False
        return True

    def apply_filter(self, df):
        num_col = pd.to_numeric(df[self.filter_col], errors='coerce')
        bools = num_col.between(self.min_value, self.max_value, inclusive=True) 
        if self.remove_matched:
            return ~bools
        else:
            return bools


filter_types = {
        "Value Filter": ValueFilter
    }