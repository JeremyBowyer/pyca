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
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel, QLineEdit, QMessageBox, QCheckBox, QListWidget
from PyQt5.QtGui import *

from IPython import embed;

class Transformation(ABC):
    def __init__(self):
        pass

    def generate_column_name(self, df, col):
        cnt = list(df.columns).count(col)
        if cnt > 0:
            return col+"_"+str(cnt)
        return col

    @abstractmethod
    def validate_inputs(self, owner):
        pass

    @abstractmethod
    def create_transformation(self, df):
        pass

    @abstractmethod
    def add_widgets(self, layout, owner, controller):
        pass

    @abstractmethod
    def add_summary_widgets(self, box, controller):
        pass

class BinaryStringTransformation(Transformation):
    def __init__(self):
        super(BinaryStringTransformation).__init__()
        self.col_ids = []
        self.name = "Binary (String)"

    def add_widgets(self, layout, owner, controller):
        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        transform_cols_label = QLabel("Columns to transform", owner)
        self.transformation_cols_box = ui_objects.ColumnList(owner, controller)
        string_label = QLabel("Value to be flagged as 1", owner)
        self.string_box = QLineEdit(owner)
        self.checkbox = QCheckBox("Treat NAs as 0s? (Checking this box means any missing data will result in a 0)")

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(transform_cols_label)
        layout.addWidget(self.transformation_cols_box)
        layout.addWidget(string_label)
        layout.addWidget(self.string_box)
        layout.addWidget(self.checkbox)

    def add_summary_widgets(self, box, controller):
        box.transformation = self

        box.tran_label.setText("Binary (String)")
        box.string_label = QLabel("String Matched:", box)
        box.string_label.setFont(controller.NORMAL_FONT)
        box.string = QLabel(self.string, box)
        box.string.setFont(controller.SMALL_FONT)
        box.transformation_cols_label = QLabel("Columns transformed:", box)
        box.transformation_cols_label.setFont(controller.NORMAL_FONT)
        box.transformation_cols = QListWidget(box)
        box.transformation_cols.addItems(self.col_ids)
        box.treat_nas_as_zero_label = QLabel("Treat NAs as 0?", box)
        box.treat_nas_as_zero_label.setFont(controller.NORMAL_FONT)
        box.treat_nas_as_zero_value = QLabel(str(self.treat_nas_as_zero), box)
        box.treat_nas_as_zero_value.setFont(controller.SMALL_FONT)

        box.content_layout.addWidget(box.string_label)
        box.content_layout.addWidget(box.string)
        box.content_layout.addWidget(box.transformation_cols_label)
        box.content_layout.addWidget(box.transformation_cols)
        box.content_layout.addWidget(box.treat_nas_as_zero_label)
        box.content_layout.addWidget(box.treat_nas_as_zero_value)

        box.setFixedHeight(500)
        box.setFixedWidth(500)

    def validate_inputs(self, owner):
        self.suffix = self.suffix_box.text()
        self.transform_cols = [x.text() for x in self.transformation_cols_box.selectedItems()]
        self.string = self.string_box.text()
        self.treat_nas_as_zero = self.checkbox.isChecked()
        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if  len(self.transform_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to transform.")
            return False
        if self.string == "":
            QMessageBox.about(owner, "Warning", "Please provide a string to be searched for.")
            return False
        return True

    def create_transformation(self, df):
        for col in self.transform_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            old_col = df[col].apply(str)
            new_col = np.where(old_col.str.match(self.string), 1, 0)
            if not self.treat_nas_as_zero:
                new_col = np.where(np.logical_or(old_col.isnull(),old_col == ''), np.nan, new_col)
            df[col_name] = new_col

        return df


class BinaryPercentileTransformation(Transformation):
    def __init__(self):
        super(BinaryPercentileTransformation).__init__()
        self.col_ids = []
        self.name = "Binary (Percentile)"

    def add_widgets(self, layout, owner, controller):
        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        transform_cols_label = QLabel("Columns to transform", owner)
        self.transformation_cols_box = ui_objects.ColumnList(owner, controller)
        perc_label = QLabel("Select quantile range to flag (inclusive)", owner)
        self.perc_slider = QRangeSlider(owner)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(transform_cols_label)
        layout.addWidget(self.transformation_cols_box)
        layout.addWidget(perc_label)
        layout.addWidget(self.perc_slider)

    def add_summary_widgets(self, box, controller):
        box.transformation = self

        box.tran_label.setText("Binary (Percentile)")

        box.lower_perc_label = QLabel("Lower Percentile:", box)
        box.lower_perc_label.setFont(controller.NORMAL_FONT)

        box.lower_perc = QLabel(str(self.lower_perc), box)
        box.lower_perc.setFont(controller.SMALL_FONT)

        box.upper_perc_label = QLabel("Upper Percentile:", box)
        box.upper_perc_label.setFont(controller.NORMAL_FONT)

        box.upper_perc = QLabel(str(self.upper_perc), box)
        box.upper_perc.setFont(controller.SMALL_FONT)

        box.transformation_cols_label = QLabel("Columns transformed:", box)
        box.transformation_cols_label.setFont(controller.NORMAL_FONT)
        box.transformation_cols = QListWidget(box)
        box.transformation_cols.addItems(self.col_ids)

        box.content_layout.addWidget(box.lower_perc_label)
        box.content_layout.addWidget(box.lower_perc)
        box.content_layout.addWidget(box.upper_perc_label)
        box.content_layout.addWidget(box.upper_perc)
        box.content_layout.addWidget(box.transformation_cols_label)
        box.content_layout.addWidget(box.transformation_cols)

        box.setFixedHeight(500)
        box.setFixedWidth(500)

    def validate_inputs(self, owner):
        self.suffix = self.suffix_box.text()
        self.transform_cols = [x.text() for x in self.transformation_cols_box.selectedItems()]
        self.lower_perc = self.perc_slider.getRange()[0]
        self.upper_perc = self.perc_slider.getRange()[1]
        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if  len(self.transform_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to transform.")
            return False
        return True

    def create_transformation(self, df):
        for col in self.transform_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            numeric_col = pd.to_numeric(df[col], errors="coerce")

            lower_q = np.nanquantile(numeric_col, self.lower_perc/100)
            upper_q = np.nanquantile(numeric_col, self.upper_perc/100)


            new_col = np.where(np.logical_and(numeric_col >= lower_q, numeric_col <= upper_q), 1, 0)
            new_col = np.where(new_col == np.nan, np.nan, new_col)
            df[col_name] = new_col

        return df  


transformation_types = {
        "Binary (String)": BinaryStringTransformation,
        "Binary (Percentile)": BinaryPercentileTransformation
    }