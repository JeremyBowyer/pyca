import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.stats import pearsonr
import time
import math
import psutil
import dateutil
import csv
import datetime
import gc
import re
from abc import ABC, abstractmethod
import IPython

from data_objects import Coord

import widgets
from qrangeslider import QRangeSlider

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel, QLineEdit, QMessageBox, QCheckBox, QListWidget, QComboBox
from PyQt5.QtGui import *


class MetricDiveFilter(ABC):
    def __init__(self):
        self.name = "Metric Dive Filter"

    @abstractmethod
    def create_filter(self, df, cols):
        pass


class MetricDiveDateFilter(MetricDiveFilter):
    def __init__(self, dates, keep):
        self.keep = keep
        self.dates = dates
        self.name = "Date list filter"

    def create_filter(self, df, cols):
        bools = []
        for idx, row in df.iterrows():
            match = row[cols['date_col']] in self.dates
            if self.keep:
                bools.append(match)
            else:
                bools.append(not match)
        return bools


class MetricDivePlotlyPointFilter(MetricDiveFilter):
    def __init__(self, points, keep):
        self.keep = keep
        self.points = points
        self.indices = [point['text'] for point in points]
        self.name = "Plotly points filter"

    def create_filter(self, df, cols):
        matches = df.index.isin(self.indices)
        if self.keep:
            return list(matches)
        else:
            return list(~matches)


class MetricDiveCoordFilter(MetricDiveFilter):
    def __init__(self, coords, keep):
        self.keep = keep
        self.coords = coords
        self.name = "Scatter points filter"

    def create_filter(self, df, cols):
        bools = []
        for idx, row in df.iterrows():
            coord = Coord((row[cols['x_col']], row[cols['y_col']]))
            match = coord in self.coords
            if self.keep:
                bools.append(match)
            else:
                bools.append(not match)
        return bools


class Filter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def validate_inputs(self, owner):
        pass

    @abstractmethod
    def create_filter(self, df):
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
        self.type_name   = "Value Filter"
        self.description = "Filter your dataset where values in given columns fall between a given maximum and minimum value."
        self.label       = ""
        self.name        = ""

    def add_widgets(self, layout, owner, controller):
        filter_name_label = QLabel("Filter Name", owner)
        self.filter_name_box = QLineEdit(owner)
        filter_col_label = QLabel("Column to filter on", owner)
        self.filter_col_box = widgets.ColumnComboBox(owner, controller)
        self.filter_col_box.setFixedWidth(450)
        min_label = QLabel("Minimum Value (inclusive)", owner)
        self.min_value_le = QLineEdit(owner)
        max_label = QLabel("Maximum Value (inclusive)", owner)
        self.max_value_le = QLineEdit(owner)
        self.remove_matched_box = QCheckBox("REMOVE matched rows? (unchecking will KEEP matched rows")

        layout.addWidget(filter_name_label)
        layout.addWidget(self.filter_name_box)
        layout.addWidget(filter_col_label)
        layout.addWidget(self.filter_col_box)
        layout.addWidget(min_label)
        layout.addWidget(self.min_value_le)
        layout.addWidget(max_label)
        layout.addWidget(self.max_value_le)
        layout.addWidget(self.remove_matched_box)

    def add_summary_widgets(self, box, controller):
        box.filter = self
        box.filter_label.setText(self.type_name)
        box.filer_name_label = QLabel(self.name, box)
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

        box.filter_out_label = QLabel("REMOVE matched rows? (unchecking will KEEP matched rows", box)
        box.filter_out_label.setFont(controller.NORMAL_FONT)
        box.filter_out_value = QLabel(str(self.remove_matched), box)
        box.filter_out_value.setFont(controller.SMALL_FONT)

        box.content_layout.addWidget(box.filer_name_label)
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
        self.name       = self.filter_name_box.text() if self.filter_name_box.text() != "" else self.name
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

    def create_filter(self, df):
        num_col = pd.to_numeric(df[self.filter_col], errors='coerce')
        bools = num_col.between(self.min_value, self.max_value, inclusive=True) 
        if self.remove_matched:
            return list(~bools)
        else:
            return list(bools)


class CompleteCasesFilter(Filter):
    def __init__(self):
        super(CompleteCasesFilter).__init__()
        self.type_name   = "Complete Cases Filter"
        self.description = "The main data set will be filtered based on rows where the given columns all have data."
        self.label       = ""
        self.name        = ""

    def add_widgets(self, layout, owner, controller):
        filter_name_label = QLabel("Filter Name", owner)
        self.filter_name_box = QLineEdit(owner)
        self.filter_cols_box = widgets.SearchableListView(owner, controller, title="Columns to check:", track_columns=True)
        self.remove_matched_box = QCheckBox("REMOVE matched rows? (unchecking will KEEP matched rows)")

        layout.addWidget(filter_name_label)
        layout.addWidget(self.filter_name_box)
        layout.addWidget(self.filter_cols_box)
        layout.addWidget(self.remove_matched_box)

    def add_summary_widgets(self, box, controller):
        box.filter = self
        box.filter_label.setText(self.type_name)
        box.filer_name_label = QLabel(self.name, box)
        box.filter_cols_label = QLabel("Columns Filtered On:", box)
        box.filter_cols_label.setFont(controller.NORMAL_FONT)
        box.filter_cols = QListWidget(box)
        box.filter_cols.addItems(self.filter_cols)

        box.filter_out_label = QLabel("REMOVE matched rows? (or KEEP)", box)
        box.filter_out_label.setFont(controller.NORMAL_FONT)
        box.filter_out_value = QLabel(str(self.remove_matched), box)
        box.filter_out_value.setFont(controller.SMALL_FONT)

        box.content_layout.addWidget(box.filer_name_label)
        box.content_layout.addWidget(box.filter_cols_label)
        box.content_layout.addWidget(box.filter_cols)
        box.content_layout.addWidget(box.filter_out_label)
        box.content_layout.addWidget(box.filter_out_value)

        box.setFixedHeight(500)
        box.setFixedWidth(500)

    def validate_inputs(self, owner):
        self.name           = self.filter_name_box.text() if self.filter_name_box.text() != "" else self.name
        self.filter_cols    = self.filter_cols_box.selectedItems()
        self.remove_matched = self.remove_matched_box.isChecked()

        if len(self.filter_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please provide at least one column to filter on.")
            return False
        return True

    def create_filter(self, df):
        bools = ~(df[self.filter_cols].applymap(lambda x: x=="") | df[self.filter_cols].isnull()).any(axis=1)
        if self.remove_matched:
            return list(~bools)
        else:
            return list(bools)


class PercentileFilter(Filter):
    def __init__(self):
        super(PercentileFilter).__init__()
        self.type_name   = "Percentile Filter"
        self.description = "The main data set will be filtered based on the given percentiles of the given column."
        self.label       = ""
        self.name        = ""

    def add_widgets(self, layout, owner, controller):
        filter_name_label = QLabel("Filter Name", owner)
        self.filter_name_box = QLineEdit(owner)
        filter_col_label = QLabel("Column to filter on", owner)
        self.filter_col_box = widgets.ColumnComboBox(owner, controller)
        perc_label = QLabel("Select quantile range (inclusive)", owner)
        self.perc_slider = QRangeSlider(owner)
        category_col_label = QLabel("Select Category Column (optional)", owner)
        self.category_col_box = widgets.ColumnComboBox(owner, controller)
        self.remove_matched_box = QCheckBox("REMOVE matched rows? (unchecking will KEEP matched rows)")

        layout.addWidget(filter_name_label)
        layout.addWidget(self.filter_name_box)
        layout.addWidget(filter_col_label)
        layout.addWidget(self.filter_col_box)
        layout.addWidget(perc_label)
        layout.addWidget(self.perc_slider)
        layout.addWidget(category_col_label)
        layout.addWidget(self.category_col_box)
        layout.addWidget(self.remove_matched_box)

    def add_summary_widgets(self, box, controller):
        box.filter = self
        box.filter_label.setText(self.type_name)
        box.filer_name_label = QLabel(self.name, box)
        box.filter_col_label = QLabel("Column Filtered On:", box)
        box.filter_col_label.setFont(controller.NORMAL_FONT)
        box.filter_col = QLabel(self.filter_col, box)
        box.filter_col.setFont(controller.SMALL_FONT)

        box.upper_perc_label = QLabel("Upper Percentile:", box)
        box.upper_perc_label.setFont(controller.NORMAL_FONT)
        box.upper_perc = QLabel(str(self.upper_perc), box)
        box.upper_perc.setFont(controller.NORMAL_FONT)

        box.lower_perc_label = QLabel("Lower Percentile:", box)
        box.lower_perc_label.setFont(controller.NORMAL_FONT)
        box.lower_perc = QLabel(str(self.lower_perc), box)
        box.lower_perc.setFont(controller.NORMAL_FONT)

        box.filter_out_label = QLabel("REMOVE matched rows? (unchecking will KEEP matched rows", box)
        box.filter_out_label.setFont(controller.NORMAL_FONT)
        box.filter_out_value = QLabel(str(self.remove_matched), box)
        box.filter_out_value.setFont(controller.SMALL_FONT)

        box.content_layout.addWidget(box.box.filer_name_label)
        box.content_layout.addWidget(box.filter_col_label)
        box.content_layout.addWidget(box.filter_col)
        box.content_layout.addWidget(box.lower_perc_label)
        box.content_layout.addWidget(box.lower_perc)
        box.content_layout.addWidget(box.upper_perc_label)
        box.content_layout.addWidget(box.upper_perc)
        box.content_layout.addWidget(box.filter_out_label)
        box.content_layout.addWidget(box.filter_out_value)

        box.setFixedHeight(500)
        box.setFixedWidth(500)

    def validate_inputs(self, owner):
        self.name           = self.filter_name_box.text() if self.filter_name_box.text() != "" else self.name
        self.filter_col     = self.filter_col_box.currentText()
        self.lower_perc     = self.perc_slider.getRange()[0]
        self.upper_perc     = self.perc_slider.getRange()[1]
        self.category_col   = self.category_col_box.currentText()
        self.remove_matched = self.remove_matched_box.isChecked()

        if self.filter_col == "":
            QMessageBox.about(owner, "Warning", "Please provide a valid column to filter on.")
            return False
        return True

    def create_filter(self, df):

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
            try:
                _df[_new_col] = new_col
            except:
                pass
            return _df

        df_copy = df.copy()
        if self.category_col == "" or self.category_col not in df_copy.columns:
            df_copy = create_perc_binary(df_copy, self.upper_perc, self.lower_perc, self.filter_col, '__filter_col__')
        else:
            df_copy = df_copy.groupby(self.category_col).apply(lambda grp: create_perc_binary(grp, self.upper_perc, self.lower_perc, self.filter_col, '__filter_col__'))

        bools = np.where(df_copy['__filter_col__']==1, True, False)
        if self.remove_matched:
            return list(~bools)
        else:
            return list(bools)


class StringFilter(Filter):
    def __init__(self):
        super(StringFilter).__init__()
        self.type_name   = "String Filter"
        self.description = "Filter your dataset based on values in given columns matching a given string."
        self.label       = ""
        self.name        = ""

    def add_widgets(self, layout, owner, controller):
        filter_name_label = QLabel("Filter Name", owner)
        self.filter_name_box = QLineEdit(owner)
        self.filter_cols_box = widgets.SearchableListView(owner, controller, title="Columns to filter on:", track_columns=True)
        string_label = QLabel("String to match", owner)
        self.string_box = QLineEdit(owner)
        self.case_checkbox = QCheckBox("Case sensitive?")
        self.remove_matched_box = QCheckBox("REMOVE matched rows? (unchecking will KEEP matched rows)")

        layout.addWidget(filter_name_label)
        layout.addWidget(self.filter_name_box)
        layout.addWidget(self.filter_cols_box)
        layout.addWidget(string_label)
        layout.addWidget(self.string_box)
        layout.addWidget(self.case_checkbox)
        layout.addWidget(self.remove_matched_box)

    def add_summary_widgets(self, box, controller):
        box.filter = self
        box.filter_label.setText(self.type_name)

        box.filer_name_label = QLabel(self.name, box)

        box.filter_cols_label = QLabel("Columns Filtered On:", box)
        box.filter_cols_label.setFont(controller.NORMAL_FONT)
        box.filter_cols = QListWidget(box)
        box.filter_cols.addItems(self.filter_cols)

        box.string_label = QLabel("String Matched:", box)
        box.string_label.setFont(controller.NORMAL_FONT)
        box.string_value = QLabel(str(self.string), box)
        box.string_value.setFont(controller.SMALL_FONT)

        box.case_label = QLabel("Case Sensitive?", box)
        box.case_label.setFont(controller.NORMAL_FONT)
        box.case_value = QLabel(str(self.case_sensitive), box)
        box.case_value.setFont(controller.SMALL_FONT)

        box.filter_out_label = QLabel("REMOVED matched rows?", box)
        box.filter_out_label.setFont(controller.NORMAL_FONT)
        box.filter_out_value = QLabel(str(self.remove_matched), box)
        box.filter_out_value.setFont(controller.SMALL_FONT)

        box.content_layout.addWidget(box.filer_name_label)
        box.content_layout.addWidget(box.filter_cols_label)
        box.content_layout.addWidget(box.filter_cols)
        box.content_layout.addWidget(box.string_label)
        box.content_layout.addWidget(box.string_value)
        box.content_layout.addWidget(box.case_label)
        box.content_layout.addWidget(box.case_value)
        box.content_layout.addWidget(box.filter_out_label)
        box.content_layout.addWidget(box.filter_out_value)

        box.setFixedHeight(500)
        box.setFixedWidth(500)

    def validate_inputs(self, owner):
        self.name              = self.filter_name_box.text() if self.filter_name_box.text() != "" else self.name
        self.filter_cols       = self.filter_cols_box.selectedItems()
        self.string            = self.string_box.text()
        self.case_sensitive    = self.case_checkbox.isChecked()
        self.remove_matched    = self.remove_matched_box.isChecked()

        if len(self.filter_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please provide at least one column to filter on.")
            return False
        if self.string == "":
            QMessageBox.about(owner, "Warning", "Please provide a string to match.")
            return False
        return True

    def create_filter(self, df):
        df_copy = df[self.filter_cols].copy()
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.match(self.string, self.case_sensitive)

        bools = df_copy.any(axis=1)
        if self.remove_matched:
            return list(~bools)
        else:
            return list(bools)


class DateFilter(Filter):
    def __init__(self):
        super(DateFilter).__init__()
        self.type_name   = "Date Filter"
        self.description = "Filter your dataset based on dates in given columns within a given date range."
        self.label       = ""
        self.name        = ""

    def add_widgets(self, layout, owner, controller):
        filter_name_label = QLabel("Filter Name", owner)
        self.filter_name_box = QLineEdit(owner)
        self.filter_cols_box = widgets.SearchableListView(owner, controller, title="Columns to filter on:", track_columns=True)

        start_date_label = QLabel("Start Date (inclusive)", owner)
        self.start_date_box = QLineEdit(owner)

        end_date_label = QLabel("End Date (inclusive)", owner)
        self.end_date_box = QLineEdit(owner)

        self.remove_matched_box = QCheckBox("REMOVE matched rows? (unchecking will KEEP matched rows)")

        layout.addWidget(filter_name_label)
        layout.addWidget(self.filter_name_box)
        layout.addWidget(self.filter_cols_box)
        layout.addWidget(start_date_label)
        layout.addWidget(self.start_date_box)
        layout.addWidget(end_date_label)
        layout.addWidget(self.end_date_box)
        layout.addWidget(self.remove_matched_box)

    def add_summary_widgets(self, box, controller):
        box.filter = self
        box.filter_label.setText(self.type_name)

        box.filer_name_label = QLabel(self.name, box)

        box.filter_cols_label = QLabel("Columns Filtered On:", box)
        box.filter_cols_label.setFont(controller.NORMAL_FONT)
        box.filter_cols = QListWidget(box)
        box.filter_cols.addItems(self.filter_cols)

        box.start_date_label = QLabel("Start Date:", box)
        box.start_date_label.setFont(controller.NORMAL_FONT)
        box.start_date_value = QLabel(str(self.start_date_str), box)
        box.start_date_value.setFont(controller.SMALL_FONT)

        box.end_date_label = QLabel("End Date:", box)
        box.end_date_label.setFont(controller.NORMAL_FONT)
        box.end_date_value = QLabel(str(self.end_date_str), box)
        box.end_date_value.setFont(controller.SMALL_FONT)

        box.filter_out_label = QLabel("REMOVED matched rows?", box)
        box.filter_out_label.setFont(controller.NORMAL_FONT)
        box.filter_out_value = QLabel(str(self.remove_matched), box)
        box.filter_out_value.setFont(controller.SMALL_FONT)

        box.content_layout.addWidget(box.filer_name_label)
        box.content_layout.addWidget(box.filter_cols_label)
        box.content_layout.addWidget(box.filter_cols)
        box.content_layout.addWidget(box.start_date_label)
        box.content_layout.addWidget(box.start_date_value)
        box.content_layout.addWidget(box.end_date_label)
        box.content_layout.addWidget(box.end_date_value)
        box.content_layout.addWidget(box.filter_out_label)
        box.content_layout.addWidget(box.filter_out_value)

        box.setFixedHeight(500)
        box.setFixedWidth(500)

    def validate_inputs(self, owner):
        self.name              = self.filter_name_box.text() if self.filter_name_box.text() != "" else self.name
        self.filter_cols       = self.filter_cols_box.selectedItems()
        self.start_date_str    = self.start_date_box.text()
        self.end_date_str      = self.end_date_box.text()
        self.remove_matched    = self.remove_matched_box.isChecked()

        if len(self.filter_cols) == 0:
            QMessageBox.about(owner, "Warning", "Please provide at least one column to filter on.")
            return False
        try:
            self.start_date = pd.to_datetime(self.start_date_str)
        except (dateutil.parser._parser.ParserError, OverflowError):
            QMessageBox.about(owner, "Warning", "Please provide a start date in a proper format. https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior")
            return False

        try:
            self.end_date = pd.to_datetime(self.end_date_str)
        except (dateutil.parser._parser.ParserError, OverflowError):
            QMessageBox.about(owner, "Warning", "Please provide an end date in a proper format. https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior")
            return False

        if pd.isnull(self.start_date) and pd.isnull(self.end_date):
            QMessageBox.about(owner, "Warning", "Please provide at least one date (start or end or both) in a proper format. https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior")
            return False

        if pd.isnull(self.start_date):
            self.start_date = -math.inf

        if pd.isnull(self.end_date):
            self.end_date = math.inf

        return True

    def create_filter(self, df):
        df_copy = df[self.filter_cols].copy()

        for col in df_copy.columns:
            try:
                col_date = pd.to_datetime(df_copy[col])
            except ValueError:
                df_copy[col] = False

            if   self.start_date == -math.inf:
                df_copy[col] = col_date <= self.end_date
            elif self.end_date == math.inf:
                df_copy[col] = col_date >= self.start_date
            else:
                df_copy[col] = (col_date >= self.start_date) & (col_date <= self.end_date)

        bools = df_copy.any(axis=1)
        if self.remove_matched:
            return list(~bools)
        else:
            return list(bools)


filter_types = {
        "Value Filter": ValueFilter,
        "Complete Cases Filter": CompleteCasesFilter,
        "Percentile Filter": PercentileFilter,
        "String Filter": StringFilter,
        "Date Filter": DateFilter
    }