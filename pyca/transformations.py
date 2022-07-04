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
import util
from abc import ABC, abstractmethod

import IPython

import widgets
from qrangeslider import QRangeSlider

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel, QLineEdit, QMessageBox, QCheckBox, QListWidget, QComboBox, QHBoxLayout
from PyQt5.QtGui import *

class Transformation(ABC):
    def __init__(self, controller):
        self.controller = controller
        self.col_ids    = []

    def generate_column_name(self, df, col):
        cnt = len([column for column in df.columns if column.startswith(col)])
        if cnt > 0:
            while(col+"_"+str(cnt) in df.columns):
                cnt = cnt+1
            return col+"_"+str(cnt)
        return col

    @abstractmethod
    def validate_inputs(self, owner):
        pass

    @abstractmethod
    def create_transformation(self, df):
        """Perform calculations germane to the particular transformation

        The transformation calculation takes a DataFrame and must output a dictionary, 
        whose keys represent the names of the new column(s), and whose values are a 
        corresponding pandas Series.

        IMPORTANT: The order of the data in the resulting Series must be the same order 
        as the DataFrame being worked on.

        Parameters
        ----------
        df : DataFrame
            The DataFrame that the transformation should be performed on.
        """
        pass

    @abstractmethod
    def add_widgets(self, layout, owner, controller):
        pass

    def validate_category_col(self, col, df):
        isEmpty     = col == ""
        isInColumns = col in df.columns
        return not isEmpty and isInColumns

    def add_summary_widgets(self, box, controller):
        box.transformation = self
        box.tran_label.setText(self.name)

        if len(getattr(self, 'col_ids', [])) > 0:
            box.x_cols_label = QLabel("New Column(s):", box)
            box.x_cols_label.setFont(self.controller.NORMAL_FONT)
            box.x_cols = QListWidget(box)
            box.x_cols.addItems(self.col_ids)
            box.content_layout.addWidget(box.x_cols_label)
            box.content_layout.addWidget(box.x_cols)

        if len(getattr(self, 'cols_included', [])) > 0:
            box.x_cols_label = QLabel("Columns Included:", box)
            box.x_cols_label.setFont(self.controller.NORMAL_FONT)
            box.x_cols = QListWidget(box)
            box.x_cols.addItems(self.cols_included)
            box.content_layout.addWidget(box.x_cols_label)
            box.content_layout.addWidget(box.x_cols)

        if getattr(self, 'resulting_column', "") != "":
            box.resulting_column_label = QLabel("Resulting Column Name:", box)
            box.resulting_column_label.setFont(self.controller.NORMAL_FONT)
            box.resulting_column = QLabel(self.resulting_column, box)
            box.resulting_column.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.resulting_column_label)
            box.content_layout.addWidget(box.resulting_column)

        if getattr(self, 'y_col', "") != "":
            box.y_col_label = QLabel("Y Column:", box)
            box.y_col_label.setFont(self.controller.NORMAL_FONT)
            box.y_col = QLabel(self.y_col, box)
            box.y_col.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.y_col_label)
            box.content_layout.addWidget(box.y_col)

        if getattr(self, 'string', "") != "":
            box.string_label = QLabel("String:", box)
            box.string_label.setFont(controller.NORMAL_FONT)
            box.string = QLabel(self.string, box)
            box.string.setFont(controller.SMALL_FONT)
            box.content_layout.addWidget(box.string_label)
            box.content_layout.addWidget(box.string)
            
        if hasattr(self, 'case_sensitive'):
            box.case_sensitive_label = QLabel("Case Sensitive?", box)
            box.case_sensitive_label.setFont(controller.NORMAL_FONT)
            box.case_sensitive = QLabel(str(self.case_sensitive), box)
            box.case_sensitive.setFont(controller.SMALL_FONT)
            box.content_layout.addWidget(box.case_sensitive_label)
            box.content_layout.addWidget(box.case_sensitive)

        if getattr(self, 'sort_col', "") != "":
            box.sort_col_label = QLabel("Sort Column:", box)
            box.sort_col_label.setFont(self.controller.NORMAL_FONT)
            box.sort_col = QLabel(self.sort_col, box)
            box.sort_col.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.sort_col_label)
            box.content_layout.addWidget(box.sort_col)

        if getattr(self, 'window', "") != "":
            box.window_label = QLabel("Window:", box)
            box.window_label.setFont(self.controller.NORMAL_FONT)
            box.window = QLabel(str(self.window), box)
            box.window.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.window_label)
            box.content_layout.addWidget(box.window)

        if getattr(self, 'lag', "") != "":
            box.lag_label = QLabel("Lag:", box)
            box.lag_label.setFont(self.controller.NORMAL_FONT)
            box.lag = QLabel(str(self.lag), box)
            box.lag.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.lag_label)
            box.content_layout.addWidget(box.lag)

        if getattr(self, 'cutoff', "") != "":
            box.cutoff_label = QLabel("Lag:", box)
            box.cutoff_label.setFont(self.controller.NORMAL_FONT)
            box.cutoff = QLabel(str(self.cutoff), box)
            box.cutoff.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.cutoff_label)
            box.content_layout.addWidget(box.cutoff)

        if getattr(self, 'upper_perc', "") != "":
            box.upper_perc_label = QLabel("Upper Percentile:", box)
            box.upper_perc_label.setFont(self.controller.NORMAL_FONT)
            box.upper_perc = QLabel(str(self.upper_perc), box)
            box.upper_perc.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.upper_perc_label)
            box.content_layout.addWidget(box.upper_perc)

        if getattr(self, 'lower_perc', "") != "":
            box.lower_perc_label = QLabel("Lower Percentile:", box)
            box.lower_perc_label.setFont(self.controller.NORMAL_FONT)
            box.lower_perc = QLabel(str(self.lower_perc), box)
            box.lower_perc.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.lower_perc_label)
            box.content_layout.addWidget(box.lower_perc)

        if getattr(self, 'min_periods', "") != "":
            box.min_periods_label = QLabel("Minimum Periods:", box)
            box.min_periods_label.setFont(self.controller.NORMAL_FONT)
            box.min_periods = QLabel(str(self.min_periods), box)
            box.min_periods.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.min_periods_label)
            box.content_layout.addWidget(box.min_periods)

        if getattr(self, 'category_col', "") != "":
            box.category_col_label = QLabel("Category Column:", box)
            box.category_col_label.setFont(self.controller.NORMAL_FONT)
            box.category_col = QLabel(self.category_col, box)
            box.category_col.setFont(self.controller.NORMAL_FONT)
            box.content_layout.addWidget(box.category_col_label)
            box.content_layout.addWidget(box.category_col)

        if hasattr(self, 'treat_nas_as_zero'):
            box.treat_nas_as_zero_label = QLabel("Treat NAs as 0?", box)
            box.treat_nas_as_zero_label.setFont(controller.NORMAL_FONT)
            box.treat_nas_as_zero_value = QLabel(str(self.treat_nas_as_zero), box)
            box.treat_nas_as_zero_value.setFont(controller.SMALL_FONT)
            box.content_layout.addWidget(box.treat_nas_as_zero_label)
            box.content_layout.addWidget(box.treat_nas_as_zero_value)

        if hasattr(self, 'treat_nas_as'):
            box.treat_nas_as_label = QLabel("Treat NAs as:", box)
            box.treat_nas_as_label.setFont(controller.NORMAL_FONT)
            box.treat_nas_as_value = QLabel(str(self.treat_nas_as), box)
            box.treat_nas_as_value.setFont(controller.SMALL_FONT)
            box.content_layout.addWidget(box.treat_nas_as_label)
            box.content_layout.addWidget(box.treat_nas_as_value)

        if hasattr(self, 'all_or_any'):
            box.all_or_any_label = QLabel("All? (as opposed to Any)", box)
            box.all_or_any_label.setFont(controller.NORMAL_FONT)
            box.all_or_any_value = QLabel(str(self.all_or_any), box)
            box.all_or_any_value.setFont(controller.SMALL_FONT)
            box.content_layout.addWidget(box.all_or_any_label)
            box.content_layout.addWidget(box.all_or_any_value)

        if hasattr(self, 'percentile'):
            box.percentile_label = QLabel("Percentile?", box)
            box.percentile_label.setFont(controller.NORMAL_FONT)
            box.percentile_value = QLabel(str(self.percentile), box)
            box.percentile_value.setFont(controller.SMALL_FONT)
            box.content_layout.addWidget(box.percentile_label)
            box.content_layout.addWidget(box.percentile_value)

        if hasattr(self, 'ascending'):
            box.ascending_label = QLabel("Ascending?", box)
            box.ascending_label.setFont(controller.NORMAL_FONT)
            box.ascending_value = QLabel(str(self.ascending), box)
            box.ascending_value.setFont(controller.SMALL_FONT)
            box.content_layout.addWidget(box.ascending_label)
            box.content_layout.addWidget(box.ascending_value)

        if hasattr(self, 'ties'):
            box.ties_label = QLabel("Ties:", box)
            box.ties_label.setFont(controller.NORMAL_FONT)
            box.ties_value = QLabel(str(self.ascending), box)
            box.ties_value.setFont(controller.SMALL_FONT)
            box.content_layout.addWidget(box.ties_label)
            box.content_layout.addWidget(box.ties_value)

        box.setMinimumHeight(box.sizeHint().height()+25)
        box.setMinimumWidth(box.sizeHint().width()+75)
        #box.setMinimumSize(box.sizeHint())


class BinaryStringTransformation(Transformation):
    def __init__(self, controller):
        super(BinaryStringTransformation, self).__init__(controller)
        self.name        = "Binary (String)"
        self.description = "Transform a column into a binary (0s and 1s), based on a given string. Values that match that string are represented as 1s."

    def add_widgets(self, layout, owner, controller):
        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.transformation_cols_box = widgets.SearchableListView(owner, controller, title="Columns to transform", track_columns=True)
        string_label = QLabel("Value to be flagged as 1", owner)
        self.string_box = QLineEdit(owner)
        self.case_checkbox = QCheckBox("Case sensitive?")
        self.nas_checkbox = QCheckBox("Treat NAs as 0s? (Checking this box means any missing data will result in a 0)")

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.transformation_cols_box)
        layout.addWidget(string_label)
        layout.addWidget(self.string_box)
        layout.addWidget(self.case_checkbox)
        layout.addWidget(self.nas_checkbox)

    def validate_inputs(self, owner):
        self.suffix            = self.suffix_box.text()
        self.transform_cols    = self.transformation_cols_box.selectedItems()
        self.string            = self.string_box.text()
        self.case_sensitive    = self.case_checkbox.isChecked()
        self.treat_nas_as_zero = self.nas_checkbox.isChecked()

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
        cols = {}
        for col in self.transform_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            old_col = df[col].apply(str)
            new_col = np.where(old_col.str.match(self.string, case=self.case_sensitive), 1, 0)
            if not self.treat_nas_as_zero:
                new_col = np.where(np.logical_or(old_col.isnull(),old_col == ''), np.nan, new_col)
            cols[col_name] = pd.Series(new_col)

        return cols


class BinaryPercentileTransformation(Transformation):
    def __init__(self, controller):
        super(BinaryPercentileTransformation, self).__init__(controller)
        self.category_col = ""
        self.name         = "Binary (Percentile)"
        self.description  = "Transform a column into a binary (0s and 1s), based on a given percentile range, inclusive. Values within that range will be represented as 1."

    def add_widgets(self, layout, owner, controller):
        self.suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.transformation_cols_box = widgets.SearchableListView(owner, controller, title="Columns to transform:", track_columns=True)
        self.values_holder = QHBoxLayout()
        self.startValue = widgets.IntInput(owner,starting_value=0)
        self.endValue = widgets.IntInput(owner, starting_value=100)
        self.values_holder.addWidget(self.startValue)
        self.values_holder.addWidget(self.endValue)
        self.perc_label = QLabel("Select quantile range to flag (inclusive)", owner)
        self.perc_slider = QRangeSlider(owner)
        self.category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        # Connect signals
        self.startValue.valid_int_changed.connect(lambda x: self.perc_slider.setStart(int(x)))
        self.endValue.valid_int_changed.connect(lambda x: self.perc_slider.setEnd(int(x)))
        self.perc_slider.startValueChanged.connect(lambda x: self.startValue.setText(str(x)))
        self.perc_slider.endValueChanged.connect(lambda x: self.endValue.setText(str(x)))

        layout.addWidget(self.suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.transformation_cols_box)
        layout.addLayout(self.values_holder)
        layout.addWidget(self.perc_label)
        layout.addWidget(self.perc_slider)
        layout.addWidget(self.category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix = self.suffix_box.text()
        self.transform_cols = self.transformation_cols_box.selectedItems()
        self.lower_perc = self.perc_slider.getRange()[0]
        self.upper_perc = self.perc_slider.getRange()[1]
        self.category_col = self.category_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if  len(self.transform_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to transform.")
            return False
        return True

    def create_transformation(self, df):

        def create_perc_binary(_df, _upper, _lower, _col, _new_col):
            numeric_col = pd.to_numeric(_df[_col], errors="coerce")

            lower_q = np.nanquantile(numeric_col, _lower/100)

            if _upper >= 100:
                clean_numeric_col = numeric_col.replace([np.inf, -np.inf], np.nan).dropna()
                try:
                    upper_q = max(clean_numeric_col)
                except:
                    upper_q = np.nan
            else:
                upper_q = np.nanquantile(numeric_col, _upper/100)
      

            new_col = np.where(np.logical_and(numeric_col >= lower_q, numeric_col <= upper_q), 1, 0)
            new_col = np.where(numeric_col.isnull(), np.nan, new_col)

            _df[_new_col] = new_col
            return _df

        cols = {}
        for col in self.transform_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_perc_binary(df, self.upper_perc, self.lower_perc, col, col_name)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_perc_binary(grp, self.upper_perc, self.lower_perc, col, col_name))
                

            cols[col_name] = df[col_name]

        return cols


class BinaryValueTransformation(Transformation):
    def __init__(self, controller):
        super(BinaryValueTransformation, self).__init__(controller)
        self.category_col = ""
        self.name         = "Binary (Value)"
        self.description  = "Transform a column into a binary (0s and 1s), based on a given value, inclusive. Values greater than or equal to this value will be represented as 1."

    def add_widgets(self, layout, owner, controller):
        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.transformation_cols_box = widgets.SearchableListView(owner, controller, title="Columns to transform:", track_columns=True)
        cutoff_label = QLabel("Cutoff (values greater than or equal to this will be 1", owner)
        self.cutoff_box = QLineEdit(owner)
        self.nas_checkbox = QCheckBox("Treat NAs as 0s? (Checking this box means any missing data will result in a 0)")

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.transformation_cols_box)
        layout.addWidget(cutoff_label)
        layout.addWidget(self.cutoff_box)
        layout.addWidget(self.nas_checkbox)

    def validate_inputs(self, owner):
        self.suffix = self.suffix_box.text()
        self.transform_cols = self.transformation_cols_box.selectedItems()
        self.cutoff = self.cutoff_box.text()
        self.treat_nas_as_zero = self.nas_checkbox.isChecked()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if  len(self.transform_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to transform.")
            return False
        try:
            self.cutoff = float(self.cutoff)
        except:
            QMessageBox.about(owner, "Warning", "Please provide a number as a cutoff.")
            return False
        return True

    def create_transformation(self, df):
        cols = {}
        for col in self.transform_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            n = pd.to_numeric(df[col], errors='coerce')

            b = np.where(n >= self.cutoff, 1, 0)
            if not self.treat_nas_as_zero:
                b = np.where(np.logical_or(n.isnull(),n == ''), np.nan, b)

            cols[col_name] = b

        return cols


class SingleLinearPredictedTransformation(Transformation):
    def __init__(self, controller):
        super(SingleLinearPredictedTransformation, self).__init__(controller)
        self.y_col        = ""
        self.category_col = ""
        self.name         = "Single Linear Predicted Values"
        self.description  = "Regress each individual given X against a given Y. Those regressions are then used to create predicted values for each relationship."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select X Columns:", track_columns=True)
        y_col_label = QLabel("Select Y Column", owner)
        self.y_col_box = widgets.ColumnComboBox(owner, controller)
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(y_col_label)
        layout.addWidget(self.y_col_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.y_col        = self.y_col_box.currentText()
        self.category_col = self.category_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be regressed against Y.")
            return False
        if self.y_col == "":
            QMessageBox.about(owner, "Warning", "Please select a Y column to be used in the linear regression.")
            return False
        return True

    def create_transformation(self, df):

        def create_predicted(_df, _x, _y, _new_col):
            clean_y = pd.to_numeric(_df[_y], errors='coerce')
            clean_x = pd.to_numeric(_df[_x], errors='coerce')
            col_df = pd.DataFrame({'x': clean_x, 'y': clean_y}).dropna()
            count = col_df.shape[0]

            if count < 3:
                _df[_new_col] = [np.nan] * _df.shape[0]
            else:
                x = col_df['x']
                y = col_df['y']
                slope, intercept, r_value, p_value, std_err = linregress(x,y)
                _df[_new_col] = clean_x*slope + intercept
            
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_predicted(df, col, self.y_col, col_name)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_predicted(grp, col, self.y_col, col_name))

            cols[col_name] = df[col_name]

        return cols


class SingleLinearResidualTransformation(Transformation):
    def __init__(self, controller):
        super(SingleLinearResidualTransformation, self).__init__(controller)
        self.y_col        = ""
        self.category_col = ""
        self.name         = "Single Linear Residual"
        self.description  = "Regress each individual given X against a given Y. That regression is then used to create a residual, defined as: y - predicted x."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select X Columns:", track_columns=True)
        y_col_label = QLabel("Select Y Column", owner)
        self.y_col_box = widgets.ColumnComboBox(owner, controller)
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(y_col_label)
        layout.addWidget(self.y_col_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.y_col        = self.y_col_box.currentText()
        self.category_col = self.category_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be regressed against Y.")
            return False
        if self.y_col == "":
            QMessageBox.about(owner, "Warning", "Please select a Y column to be used in the linear regression.")
            return False
        return True

    def create_transformation(self, df):

        def create_resid(_df, _x, _y, _new_col):
            clean_y = pd.to_numeric(_df[_y], errors='coerce')
            clean_x = pd.to_numeric(_df[_x], errors='coerce')
            col_df = pd.DataFrame({'x': clean_x, 'y': clean_y}).dropna()
            count = col_df.shape[0]

            if count < 3:
                _df[_new_col] = [np.nan] * _df.shape[0]
            else:
                x = col_df['x']
                y = col_df['y']
                slope, intercept, r_value, p_value, std_err = linregress(x,y)
                _df[_new_col] = clean_y - (clean_x*slope + intercept)
            
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_resid(df, col, self.y_col, col_name)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_resid(grp, col, self.y_col, col_name))

            cols[col_name] = df[col_name]

        return cols


class SimpleDifferenceTransformation(Transformation):
    def __init__(self, controller):
        super(SimpleDifferenceTransformation, self).__init__(controller)
        self.lag          = ""
        self.category_col = ""
        self.sort_col     = ""
        self.name         = "Simple Difference"
        self.description  = "X column sorted by a provided date column, and a longitudinal difference is created."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select X Columns:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        lag_label = QLabel("Lag (number of periods)", owner)
        self.lag_box = QLineEdit(owner)
        self.lag_box.setText("1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(lag_label)
        layout.addWidget(self.lag_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.lag          = self.lag_box.text()
        self.category_col = self.category_col_box.currentText()
        self.sort_col     = self.sort_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.lag = int(self.lag)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a lag.")
            return False
        return True

    def create_transformation(self, df):

        df = util.try_sort(df, self.sort_col)

        def create_diff(_df, _col, _new_col, _lag):
            n_col = pd.to_numeric(_df[_col], errors='coerce')
            _df[_new_col] = n_col.diff(periods=_lag)
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_diff(df, col, col_name, self.lag)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_diff(grp, col, col_name, self.lag))
            
            cols[col_name] = df[col_name].sort_index()

        return cols


class PercentChangeTransformation(Transformation):
    def __init__(self, controller):
        super(PercentChangeTransformation, self).__init__(controller)
        self.lag          = ""
        self.category_col = ""
        self.sort_col     = ""
        self.name         = "Percent Change - Standard"
        self.description  = "Transform column in percentage change, after being sorted by given sort column."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select X Columns:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        lag_label = QLabel("Lag (number of periods)", owner)
        self.lag_box = QLineEdit(owner)
        self.lag_box.setText("1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(lag_label)
        layout.addWidget(self.lag_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.lag          = self.lag_box.text()
        self.category_col = self.category_col_box.currentText()
        self.sort_col     = self.sort_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.lag = int(self.lag)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a lag.")
            return False
        return True

    def create_transformation(self, df):

        df = util.try_sort(df, self.sort_col)

        def create_pct_chg(_df, _col, _new_col, _lag):
            n_col = pd.to_numeric(_df[_col], errors='coerce')
            _df[_new_col] = n_col.pct_change(fill_method=None, periods=_lag)
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_pct_chg(df, col, col_name, self.lag)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_pct_chg(grp, col, col_name, self.lag))

            cols[col_name] = df[col_name].sort_index()

        return cols  


class RollingSumTransformation(Transformation):
    def __init__(self, controller):
        super(RollingSumTransformation, self).__init__(controller)
        self.lag          = ""
        self.category_col = ""
        self.sort_col     = ""
        self.name         = "Rolling Sum"
        self.description  = "Compute a rolling sum, for a given window time period and minimum data point threshold. If you want an ever-increasing expanding window, enter -1 as a window amount."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        window_label = QLabel("Window (number of periods in the rolling window. Input -1 to have an expanding window.)", owner)
        self.window_box = QLineEdit(owner)
        self.window_box.setText("-1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(window_label)
        layout.addWidget(self.window_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.window       = self.window_box.text()
        self.category_col = self.category_col_box.currentText()
        self.sort_col     = self.sort_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.window = int(self.window)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a window.")
            return False
        return True

    def create_transformation(self, df):

        df = util.try_sort(df, self.sort_col)

        def create_rolling_sum(_df, _col, _new_col, _window):
            s = pd.to_numeric(_df[_col], errors='coerce')
            if _window == -1:
                _df[_new_col] = s.expanding().apply(lambda s: s.sum())
            else:
                _df[_new_col] = s.rolling(window=_window).apply(lambda s: s.sum())
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_rolling_sum(df, col, col_name, self.window)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_rolling_sum(grp, col, col_name, self.window))

            cols[col_name] = df[col_name].sort_index()

        return cols  


class RollingMeanTransformation(Transformation):
    def __init__(self, controller):
        super(RollingMeanTransformation, self).__init__(controller)
        self.lag          = ""
        self.category_col = ""
        self.sort_col     = ""
        self.name         = "Rolling Average"
        self.description  = "Compute a rolling average, for a given window time period and minimum data point threshold. If you want an ever-increasing expanding window, enter -1 as a window amount."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        window_label = QLabel("Window (number of periods in the rolling window. Input -1 to have an expanding window.)", owner)
        self.window_box = QLineEdit(owner)
        self.window_box.setText("-1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(window_label)
        layout.addWidget(self.window_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.window       = self.window_box.text()
        self.category_col = self.category_col_box.currentText()
        self.sort_col     = self.sort_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.window = int(self.window)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a window.")
            return False
        return True

    def create_transformation(self, df):

        df = util.try_sort(df, self.sort_col)

        def create_rolling_mean(_df, _col, _new_col, _window):
            s = pd.to_numeric(_df[_col], errors='coerce')
            if _window == -1:
                _df[_new_col] = s.expanding().apply(lambda s: s.mean())
            else:
                _df[_new_col] = s.rolling(window=_window).apply(lambda s: s.mean())
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_rolling_mean(df, col, col_name, self.window)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_rolling_mean(grp, col, col_name, self.window))

            cols[col_name] = df[col_name].sort_index()

        return cols 


class RollingMedianTransformation(Transformation):
    def __init__(self, controller):
        super(RollingMedianTransformation, self).__init__(controller)
        self.lag          = ""
        self.category_col = ""
        self.sort_col     = ""
        self.name         = "Rolling Median"
        self.description  = "Compute a rolling median, for a given window time period and minimum data point threshold. If you want an ever-increasing expanding window, enter -1 as a window amount."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        window_label = QLabel("Window (number of periods in the rolling window. Input -1 to have an expanding window.)", owner)
        self.window_box = QLineEdit(owner)
        self.window_box.setText("-1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(window_label)
        layout.addWidget(self.window_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.window       = self.window_box.text()
        self.category_col = self.category_col_box.currentText()
        self.sort_col     = self.sort_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.window = int(self.window)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a window.")
            return False
        return True

    def create_transformation(self, df):

        df = util.try_sort(df, self.sort_col)

        def create_rolling_median(_df, _col, _new_col, _window):
            s = pd.to_numeric(_df[_col], errors='coerce')
            if _window == -1:
                _df[_new_col] = s.expanding().apply(lambda s: s.median())
            else:
                _df[_new_col] = s.rolling(window=_window).apply(lambda s: s.median())
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_rolling_median(df, col, col_name, self.window)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_rolling_median(grp, col, col_name, self.window))

            cols[col_name] = df[col_name].sort_index()
 
        return cols


class SubtractRollingMedianTransformation(Transformation):
    def __init__(self, controller):
        super(SubtractRollingMedianTransformation, self).__init__(controller)
        self.lag          = ""
        self.category_col = ""
        self.sort_col     = ""
        self.name         = "Subtract Rolling Median"
        self.description  = "Subtract a column's rolling median, for a given window time period and minimum data point threshold. If you want an ever-increasing expanding window, enter -1 as a window amount."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        window_label = QLabel("Window (number of periods in the rolling window. Input -1 to have an expanding window.)", owner)
        self.window_box = QLineEdit(owner)
        self.window_box.setText("-1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(window_label)
        layout.addWidget(self.window_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.window       = self.window_box.text()
        self.category_col = self.category_col_box.currentText()
        self.sort_col     = self.sort_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.window = int(self.window)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a window.")
            return False
        return True

    def create_transformation(self, df):

        df = util.try_sort(df, self.sort_col)

        def create_rolling_median(_df, _col, _new_col, _window):
            s = pd.to_numeric(_df[_col], errors='coerce')
            if _window == -1:
                m = s.expanding().apply(lambda s: s.median())
            else:
                m = s.rolling(window=_window).apply(lambda s: s.median())

            _df[_new_col] = s - m
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_rolling_median(df, col, col_name, self.window)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_rolling_median(grp, col, col_name, self.window))

            cols[col_name] = df[col_name].sort_index()

        return cols


class SubtractCrossSectionalMedian(Transformation):
    def __init__(self, controller):
        super(SubtractCrossSectionalMedian, self).__init__(controller)
        self.category_col = ""
        self.name         = "Subtract Cross Sectional Median"
        self.description  = "For each value, subtract the median of all values. If a category column is provided, this median will be calculated by first dividing the dataset by the values in that column."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        category_col_label = QLabel("Select Category Column", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.category_col = self.category_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        return True

    def create_transformation(self, df):

        def subtract_median(_df, _col, _new_col):
            s = pd.to_numeric(_df[_col], errors='coerce')
            _df[_new_col] = s - s.median()
            return _df

        cols = {}   
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)

            if not self.validate_category_col(self.category_col, df):
                df = subtract_median(df, col, col_name)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: subtract_median(grp, col, col_name))

            cols[col_name] = df[col_name]

        return cols


class PercentChangeMedianTransformation(Transformation):
    def __init__(self, controller):
        super(PercentChangeMedianTransformation, self).__init__(controller)
        self.lag          = ""
        self.category_col = ""
        self.sort_col     = ""
        self.name         = "Percent Change - Median"
        self.description  = "Like a percent change, except instead of the difference being divided by the earlier number, it's divided by the median for a given rolling period."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        window_label = QLabel("Median Window (number of periods in the rolling window. Input -1 to have an expanding window)", owner)
        self.window_box = QLineEdit(owner)
        self.window_box.setText("-1")
        lag_label = QLabel("Lag of Change (number of periods to go back for calculating change)", owner)
        self.lag_box = QLineEdit(owner)
        self.lag_box.setText("1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(lag_label)
        layout.addWidget(self.lag_box)
        layout.addWidget(window_label)
        layout.addWidget(self.window_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

        owner.setMinimumHeight(600)
        owner.setMinimumWidth(600)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.window       = self.window_box.text()
        self.lag          = self.lag_box.text()
        self.category_col = self.category_col_box.currentText()
        self.sort_col     = self.sort_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.window = int(self.window)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a window.")
            return False
        try:
            self.lag = int(self.lag)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a lag.")
            return False
        return True

    def create_transformation(self, df):

        df = util.try_sort(df, self.sort_col)

        def create_pct_chg_median(_df, _col, _new_col, _window, _lag):
            s = pd.to_numeric(_df[_col], errors='coerce')
            d = s.diff(periods=_lag)
            if _window == -1:
                m = s.expanding().apply(lambda s: s.median())
            else:
                m = s.rolling(window=_window).apply(lambda s: s.median())

            _df[_new_col] = d / m
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_pct_chg_median(df, col, col_name, self.window, self.lag)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_pct_chg_median(grp, col, col_name, self.window, self.lag))

            cols[col_name] = df[col_name].sort_index()

        return cols


class PercentChangeStdTransformation(Transformation):
    def __init__(self, controller):
        super(PercentChangeStdTransformation, self).__init__(controller)
        self.lag          = ""
        self.category_col = ""
        self.sort_col     = ""
        self.name         = "Percent Change - Std"
        self.description  = "Like a percent change, except instead of the difference being divided by the earlier number, it's divided by the standard deviation of a given rolling window."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        window_label = QLabel("Standard Deviation Window (number of periods in the rolling window. Input -1 to have an expanding window)", owner)
        self.window_box = QLineEdit(owner)
        self.window_box.setText("-1")
        lag_label = QLabel("Lag of Change (number of periods to go back for calculating change)", owner)
        self.lag_box = QLineEdit(owner)
        self.lag_box.setText("1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(lag_label)
        layout.addWidget(self.lag_box)
        layout.addWidget(window_label)
        layout.addWidget(self.window_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

        owner.setMinimumHeight(600)
        owner.setMinimumWidth(600)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.window       = self.window_box.text()
        self.lag          = self.lag_box.text()
        self.category_col = self.category_col_box.currentText()
        self.sort_col     = self.sort_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.window = int(self.window)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a window.")
            return False
        try:
            self.lag = int(self.lag)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a lag.")
            return False
        return True

    def create_transformation(self, df):

        df = util.try_sort(df, self.sort_col)

        def create_pct_chg_std(_df, _col, _new_col, _window, _lag):
            s = pd.to_numeric(_df[_col], errors='coerce')
            d = s.diff(periods=_lag)
            if _window == -1:
                m = s.expanding().apply(lambda s: s.std())
            else:
                m = s.rolling(window=_window).apply(lambda s: s.std())

            _df[_new_col] = d / m
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_pct_chg_std(df, col, col_name, self.window, self.lag)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_pct_chg_std(grp, col, col_name, self.window, self.lag))

            cols[col_name] = df[col_name].sort_index()

        return cols


class ZScoreLongitudinalTransformation(Transformation):
    def __init__(self, controller):
        super(ZScoreLongitudinalTransformation, self).__init__(controller)
        self.window       = ""
        self.category_col = ""
        self.sort_col     = ""
        self.name         = "Z-Score - Longitudinal"
        self.description  = "Create a z-score for a given rolling window size."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        window_label = QLabel("Window (number of periods in the rolling window. Input -1 to have an expanding window)", owner)
        self.window_box = QLineEdit(owner)
        self.window_box.setText("-1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(window_label)
        layout.addWidget(self.window_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(500)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.window       = self.window_box.text()
        self.category_col = self.category_col_box.currentText()
        self.sort_col     = self.sort_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.window = int(self.window)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer as a window.")
            return False
        return True

    def create_transformation(self, df):

        df = util.try_sort(df, self.sort_col)

        def create_zscore_long(_df, _col, _new_col, _window):
            s = pd.to_numeric(_df[_col], errors='coerce')
            if _window == -1:
                std = s.expanding().apply(lambda s: s.std())
                mn = s.expanding().apply(lambda s: s.mean())
            else:
                std = s.rolling(window=_window).apply(lambda s: s.std())
                mn = s.rolling(window=_window).apply(lambda s: s.mean())

            _df[_new_col] = (s - mn) / std
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_zscore_long(df, col, col_name, self.window)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_zscore_long(grp, col, col_name, self.window))

            cols[col_name] = df[col_name].sort_index()

        return cols


class ZScoreCrossSectionalTransformation(Transformation):
    def __init__(self, controller):
        super(ZScoreCrossSectionalTransformation, self).__init__(controller)
        self.category_col = ""
        self.name         = "Z-Score - Cross Sectional"
        self.description  = "Create a cross sectional z-score."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(500)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.category_col = self.category_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        return True

    def create_transformation(self, df):

        def create_zscore_cross(_df, _col, _new_col):
            s = pd.to_numeric(_df[_col], errors='coerce')
            mn = s.mean()
            std = s.std()
            _df[_new_col] = (s - mn) / std
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_zscore_cross(df, col, col_name)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_zscore_cross(grp, col, col_name))

            cols[col_name] = df[col_name]

        return cols


class OffsetTransformation(Transformation):
    def __init__(self, controller):
        super(OffsetTransformation, self).__init__(controller)
        self.category_col = ""
        self.lag          = ""
        self.name         = "Offset"
        self.description  = "Shift the values in a column up (backward) or down (forward)."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        sort_col_label = QLabel("Select Sort Column (probably a date)", owner)
        self.sort_col_box = widgets.ColumnComboBox(owner, controller, "Date")
        lag_label = QLabel("Periods to Shift Data (Negative values shift data 'up' (backward), positive values shift data 'down' (forward))", owner)
        self.lag_box = QLineEdit(owner)
        self.lag_box.setText("1")
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(sort_col_label)
        layout.addWidget(self.sort_col_box)
        layout.addWidget(lag_label)
        layout.addWidget(self.lag_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(575)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.category_col = self.category_col_box.currentText()
        self.lag          = self.lag_box.text()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        try:
            self.lag = int(self.lag)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide an integer (positive or negative) as a shift amount.")
            return False
        return True

    def create_transformation(self, df):

        def shift_data(_df, _col, _new_col, _periods):
            _df[_new_col] = _df[_col].shift(_periods)
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = shift_data(df, col, col_name, self.lag)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: shift_data(grp, col, col_name, self.lag))

            cols[col_name] = df[col_name]

        return cols


class ColumnArithmeticTransformation(Transformation):
    def __init__(self, controller):
        super(ColumnArithmeticTransformation, self).__init__(controller)
        self.name         = "Column Arithmetic"
        self.description  = "Perform arithmetic operations on a column."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        op_label = QLabel("Operator", owner)
        self.op_box = QComboBox(owner)
        self.op_box.addItems(["+", "-", "/", "*"])
        value_label = QLabel("Value", owner)
        self.value_box = QLineEdit(owner)
        self.value_box.setText("1")

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(op_label)
        layout.addWidget(self.op_box)
        layout.addWidget(value_label)
        layout.addWidget(self.value_box)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(500)

    def validate_inputs(self, owner):
        self.suffix = self.suffix_box.text()
        self.x_cols = self.x_cols_box.selectedItems()
        self.op     = self.op_box.currentText()
        self.value  = self.value_box.text()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        if self.op not in ['+', '-', '/', '*']:
            QMessageBox.about(owner, "Warning", "Please input an appropriate operator.")
            return False
        try:
            self.value = float(self.value)
        except ValueError:
            QMessageBox.about(owner, "Warning", "Please provide a numeric value to be used in the right side of the operation.")
            return False
        return True

    def create_transformation(self, df):

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            cols[col_name] = eval('pd.to_numeric(df[col], errors="coerce") ' + self.op + 'self.value')

        return cols


class QuintileTransformation(Transformation):
    def __init__(self, controller):
        super(QuintileTransformation, self).__init__(controller)
        self.category_col = ""
        self.name         = "Quintile"
        self.description  = "Transform the values in a column into quintiles."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(500)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.category_col = self.category_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        return True

    def create_transformation(self, df):

        def create_quintile(_df, _col, _new_col):
            q = util.create_quintile(pd.to_numeric(_df[_col], errors='coerce'))
            _df[_new_col] = q
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_quintile(df, col, col_name)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_quintile(grp, col, col_name))

            cols[col_name] = df[col_name]

        return cols


class BinaryMultipleTransformation(Transformation):
    def __init__(self, controller):
        super(BinaryMultipleTransformation, self).__init__(controller)
        self.cols_included = []
        self.name          = "Binary (Multiple Columns)"
        self.description   = "Return 1 if conditions are met. Checking the 'All' checkbox means the result will be 1 if all of the selected columns are 1, as opposed to any."
        self.description   = self.description + " If 'Treat NAs as 0s?' checkbox is left unchecked, any of the columns having an NA will mean the result is NA."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        column_name_label = QLabel("Resulting Column Name", owner)
        self.column_name_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to check for binaries:", track_columns=True)
        self.all_or_any_checkbox = QCheckBox("All 1s or Any 1s? (Checking thix box means the resulting column will be 1 if all columns are 1)")
        self.all_or_any_checkbox.setChecked(True)
        self.nas_checkbox = QCheckBox("Treat NAs as 0s? (Leaving this unchecked means any missing data will result in an NA)")

        layout.addWidget(column_name_label)
        layout.addWidget(self.column_name_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(self.all_or_any_checkbox)
        layout.addWidget(self.nas_checkbox)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(550)

    def validate_inputs(self, owner):
        self.column_name       = self.column_name_box.text()
        self.cols_included     = self.x_cols_box.selectedItems()
        self.all_or_any        = self.all_or_any_checkbox.isChecked()
        self.treat_nas_as_zero = self.nas_checkbox.isChecked()

        if self.column_name == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid column name for the resulting column(s).")
            return False
        if len(self.cols_included) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be bucketed.")
            return False
        return True

    def create_transformation(self, df):

        cols_df = df[self.cols_included]
        for col in cols_df.columns:
            cols_df[col] = pd.to_numeric(cols_df[col], errors='coerce')

        if self.all_or_any:
            new_col = np.where((cols_df == 1).all(axis=1), 1, 0)
        else:
            new_col = np.where((cols_df == 1).any(axis=1), 1, 0)

        if not self.treat_nas_as_zero:
            new_col = np.where(cols_df.isnull().any(axis=1), np.nan, new_col)

        col_name = self.generate_column_name(df, self.column_name)
        self.col_ids.append(col_name)
        cols = {}
        cols[col_name] = new_col

        return cols


class ColumnToColumnTransformation(Transformation):
    def __init__(self, controller):
        super(ColumnToColumnTransformation, self).__init__(controller)
        self.name         = "Column to Column"
        self.description  = "Perform arithmetic operations between 2 columns."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to transform:", track_columns=True)
        op_label = QLabel("Operator", owner)
        self.op_box = QComboBox(owner)
        self.op_box.addItems(["+", "-", "/", "*"])
        y_col_label = QLabel("Select Column to Apply Operation to", owner)
        self.y_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(op_label)
        layout.addWidget(self.op_box)
        layout.addWidget(y_col_label)
        layout.addWidget(self.y_col_box)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(500)

    def validate_inputs(self, owner):
        self.suffix = self.suffix_box.text()
        self.x_cols = self.x_cols_box.selectedItems()
        self.op     = self.op_box.currentText()
        self.y_col  = self.y_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        if self.op not in ['+', '-', '/', '*']:
            QMessageBox.about(owner, "Warning", "Please input an appropriate operator.")
            return False
        if self.y_col == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid column to apply operation to.")
            return False
        return True

    def create_transformation(self, df):

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            cols[col_name] = eval('pd.to_numeric(df[col], errors="coerce") ' + self.op + 'pd.to_numeric(df[self.y_col], errors="coerce")')

        return cols

class BucketColumnsTransformation(Transformation):
    def __init__(self, controller):
        super(BucketColumnsTransformation, self).__init__(controller)
        self.cols_included    = []
        self.resulting_column = ""
        self.name             = "Bucket Columns"
        self.description      = "Aggregate a list of given columns, using the given aggregation method."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        column_name_label = QLabel("Resulting Column Name", owner)
        self.column_name_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to bucket:", track_columns=True)
        op_label = QLabel("Select Aggregation Method", owner)
        self.op_box = QComboBox(owner)
        self.op_box.addItems(["Sum", "Mean", "Median", "Mode", "Max", "Min", "Concatenate"])

        treat_nas_as_label = QLabel("Treat NAs as (leave blank to keep NA): ", owner)
        self.treat_nas_as_box = QLineEdit(owner)

        layout.addWidget(column_name_label)
        layout.addWidget(self.column_name_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(op_label)
        layout.addWidget(self.op_box)
        layout.addWidget(treat_nas_as_label)
        layout.addWidget(self.treat_nas_as_box)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(500)

    def validate_inputs(self, owner):
        self.column_name   = self.column_name_box.text()
        self.cols_included = self.x_cols_box.selectedItems()
        self.op            = self.op_box.currentText()
        self.treat_nas_as  = self.treat_nas_as_box.text()

        if self.column_name == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid column name for the resulting column.")
            return False
        if len(self.cols_included) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        return True

    def create_transformation(self, df):

        cols_df = df[self.cols_included]

        if self.op == "Concatenate":
            for col in cols_df.columns:
                cols_df[col] = cols_df[col].astype(str)

            try:
                self.treat_nas_as = str(self.treat_nas_as)
            except:
                self.treat_nas_as = np.nan

            if self.treat_nas_as == "":
                self.treat_nas_as = np.nan

            if not pd.isnull(self.treat_nas_as):
                cols_df.replace("", np.nan, inplace=True)
                cols_df.fillna(self.treat_nas_as, inplace=True)

            if len(cols_df.columns) == 0:
                new_col = pd.Series([np.nan] * cols_df.shape[0])
            else:
                new_col = cols_df[cols_df.columns[0]]
                for col in cols_df.columns[1:]:
                    new_col = new_col.str.cat(cols_df[col], sep=", ", na_rep="Missing") # TODO: This isn't working properly. Blank values not being replaced
        else:
            for col in cols_df.columns:
                cols_df[col] = pd.to_numeric(cols_df[col], errors='coerce')

            try:
                self.treat_nas_as = float(self.treat_nas_as)
            except:
                self.treat_nas_as = np.nan

            if not pd.isnull(self.treat_nas_as):
                cols_df.fillna(self.treat_nas_as, inplace=True)

            if   self.op == "Sum":
                new_col = cols_df.sum(axis=1, min_count=1)
            elif self.op == "Mean":
                new_col = cols_df.mean(axis=1)
            elif self.op == "Median":
                new_col = cols_df.median(axis=1)
            elif self.op == "Mode":
                new_col = cols_df.mode(axis=1)
            elif self.op == "Max":
                new_col = cols_df.max(axis=1)
            elif self.op == "Min":
                new_col = cols_df.min(axis=1)

        col_name = self.generate_column_name(df, self.column_name)
        self.resulting_column = col_name
        self.col_ids.append(col_name)
        cols = {}
        cols[col_name] = new_col

        return cols


class DateFormatTransformation(Transformation):
    def __init__(self, controller):
        super(DateFormatTransformation, self).__init__(controller)
        self.name         = "Date Format Change"
        self.x_cols       = []
        self.format       = ""
        self.description  = "Take a given date column and change the level of detail using a given format."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select date column(s):", track_columns=True)
        format_label = QLabel("Format to transform date column(s) into", owner)
        format_guide_label = QLabel("https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior", owner)
        self.format_box = QLineEdit(owner)
        self.format_box.setText("%Y-%m")

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(format_label)
        layout.addWidget(format_guide_label)
        layout.addWidget(self.format_box)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(500)

    def validate_inputs(self, owner):
        self.suffix = self.suffix_box.text()
        self.x_cols = self.x_cols_box.selectedItems()
        self.format = self.format_box.text()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        if self.format == "":
            QMessageBox.about(owner, "Warning", "Please provide a date format. https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior")
            return False
        return True

    def create_transformation(self, df):

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            try:
                col_date = pd.to_datetime(df[col])
                cols[col_name] = col_date.dt.strftime(self.format)
            except:
                cols[col_name] = np.nan

        return cols


class CategoryAggregation(Transformation):
    def __init__(self, controller):
        super(CategoryAggregation, self).__init__(controller)
        self.cols_included    = []
        self.resulting_column = ""
        self.name             = "Category Aggregation"
        self.description      = """
        For each unique value in a given category column, aggregate the values in the given transformation column(s).
        These aggregated values will be duplicated across all rows for that unique category value.
        """

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column suffix:", owner)
        self.suffix_label_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select columns to aggregate:", track_columns=True)
        op_label = QLabel("Select Aggregation Method", owner)
        self.op_box = QComboBox(owner)
        self.op_box.addItems(["Sum", "Mean", "Median", "Mode", "Max", "Min"])
        category_col_label = QLabel("Select Category Column", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_label_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(op_label)
        layout.addWidget(self.op_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)

        owner.setMinimumHeight(500)
        owner.setMinimumWidth(500)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_label_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.op           = self.op_box.currentText()
        self.category_col = self.category_col_box.currentText()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if self.category_col == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid category column.")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        return True

    def create_transformation(self, df):

        def create_quintile(_df, _col, _new_col):
            num_col = pd.to_numeric(_df[_col], errors='coerce')

            if   self.op == "Sum":
                agg_val = num_col.sum()
            elif self.op == "Mean":
                agg_val = num_col.mean()
            elif self.op == "Median":
                agg_val = num_col.median()
            elif self.op == "Mode":
                agg_val = num_col.mode()
            elif self.op == "Max":
                agg_val = num_col.max()
            elif self.op == "Min":
                agg_val = num_col.min()

            _df[_new_col] = [agg_val] * _df.shape[0]
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_quintile(df, col, col_name)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_quintile(grp, col, col_name))

            cols[col_name] = df[col_name]

        return cols


class RankTransformation(Transformation):
    def __init__(self, controller):
        super(RankTransformation, self).__init__(controller)
        self.category_col = ""
        self.name         = "Rank"
        self.description  = "Transform a given column's data into rankings."

    def add_widgets(self, layout, owner, controller):
        self.controller = controller

        suffix_label = QLabel("Column Suffix", owner)
        self.suffix_box = QLineEdit(owner)
        self.x_cols_box = widgets.SearchableListView(owner, controller, title="Select X Columns:", track_columns=True)

        method_label = QLabel("How to handle ties?", owner)
        self.method_box = QComboBox(owner)
        self.method_box.addItems(["average", "min", "max"])
        na_option_label = QLabel("How to handle NAs? (keep = their rank will be NA)", owner)
        self.na_option_box = QComboBox(owner)
        self.na_option_box.addItems(["keep", "top", "bottom"])

        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)

        self.percentile_checkbox = QCheckBox("Percentile? (as opposed to rank)")
        self.ascending_checkbox = QCheckBox("Ascending? (Higher values = higher rank or percentile)")

        layout.addWidget(suffix_label)
        layout.addWidget(self.suffix_box)
        layout.addWidget(self.x_cols_box)
        layout.addWidget(method_label)
        layout.addWidget(self.method_box)
        layout.addWidget(na_option_label)
        layout.addWidget(self.na_option_box)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)
        layout.addWidget(self.percentile_checkbox)
        layout.addWidget(self.ascending_checkbox)

        owner.setMinimumHeight(700)
        owner.setMinimumWidth(700)

    def validate_inputs(self, owner):
        self.suffix       = self.suffix_box.text()
        self.x_cols       = self.x_cols_box.selectedItems()
        self.category_col = self.category_col_box.currentText()
        self.method       = self.method_box.currentText()
        self.treat_nas_as = self.na_option_box.currentText()
        self.percentile   = self.percentile_checkbox.isChecked()
        self.ascending    = self.ascending_checkbox.isChecked()

        if self.suffix == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid suffix for the resulting column(s).")
            return False
        if len(self.x_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please select at least one column to be transformed.")
            return False
        return True

    def create_transformation(self, df):

        def create_rank(_df, _col, _new_col, _pct, _ascending, _method, _na_option):
            n_col = pd.to_numeric(_df[_col], errors='coerce')
            _df[_new_col] = n_col.rank(method=_method,pct=_pct,ascending=_ascending,na_option=_na_option)
            return _df

        cols = {}
        for col in self.x_cols:
            col_name = self.generate_column_name(df, col+"_"+self.suffix)
            self.col_ids.append(col_name)
            
            if not self.validate_category_col(self.category_col, df):
                df = create_rank(df, col, col_name, self.percentile, self.ascending, self.method,self.treat_nas_as)
            else:
                df = df.groupby(self.category_col).apply(lambda grp: create_rank(grp, col, col_name, self.percentile, self.ascending, self.method,self.treat_nas_as))

            cols[col_name] = df[col_name].sort_index()

        return cols  



transformation_types = {
        "Binary (String)": BinaryStringTransformation,
        "Binary (Percentile)": BinaryPercentileTransformation,
        "Binary (Value)": BinaryValueTransformation,
        "Binary (Multiple Columns)": BinaryMultipleTransformation,
        "Single Linear Predicted Values": SingleLinearPredictedTransformation,
        "Single Linear Residual": SingleLinearResidualTransformation,
        "Simple Difference": SimpleDifferenceTransformation,
        "Percent Change - Standard": PercentChangeTransformation,
        "Percent Change - Median": PercentChangeMedianTransformation,
        "Percent Change - Std": PercentChangeStdTransformation,
        "Rolling Sum": RollingSumTransformation,
        "Rolling Average": RollingMeanTransformation,
        "Rolling Median": RollingMedianTransformation,
        "Subtract Rolling Median": SubtractRollingMedianTransformation,
        "Subtract Cross Sectional Median": SubtractCrossSectionalMedian,
        "Z Score - Longitudinal": ZScoreLongitudinalTransformation,
        "Z Score - Cross Sectional": ZScoreCrossSectionalTransformation,
        "Offset": OffsetTransformation,
        "Column to Column": ColumnToColumnTransformation,
        "Column Arithmetic": ColumnArithmeticTransformation,
        "Bucket Columns": BucketColumnsTransformation,
        "Quintile": QuintileTransformation,
        "Date Format": DateFormatTransformation,
        "Category Aggregation": CategoryAggregation,
        "Rank": RankTransformation
    }