import numpy as np
import pandas as pd
import datetime

from functools import partial

from util import clean_string
from models import *
from widgets import *
from threads import DataWorker, ReportWorker, LoadCsvWorker
from decorators import pyca_profile
import cProfile

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import plotly.graph_objs as go


class LoadingPage(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        self.hbox = QHBoxLayout(self)
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.setAlignment(Qt.AlignCenter)
        
        self.vbox = QVBoxLayout()
        self.vbox.setAlignment(Qt.AlignCenter)  
        
        self.loadingLabel = QLabel("Loading...", self)
        self.loadingLabel.setFont(QFont("Raleway", 16))
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setFixedSize(400,20)
        
        self.loadingMessage = QLabel("", self)
        self.loadingMessage.setFont(QFont("Raleway", 12))
        
        self.vbox.addWidget(self.loadingLabel)
        self.vbox.addWidget(self.progress)
        self.vbox.addWidget(self.loadingMessage)
        
        self.hbox.addLayout(self.vbox)
        

class SplashPage(QWidget):

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        
        self.loadBtn = QPushButton("Load a CSV to Continue", self)
        self.loadBtn.clicked.connect(self.parent.load_csv)
        self.loadBtn.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.loadBtn.setFixedSize(400,200)
        self.loadBtn.setFont(parent.LARGE_BTN_FONT)
        
        self.hbox = QHBoxLayout()
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.addWidget(self.loadBtn)
        self.hbox.setAlignment(Qt.AlignCenter)
        
        self.vbox = QVBoxLayout(self)
        self.vbox.addLayout(self.hbox)
        self.vbox.setAlignment(Qt.AlignCenter)    


class SetupPage(QWidget):

    request_run_analysis_signal = pyqtSignal(dict)
    request_validate_columns_signal = pyqtSignal(dict)

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.build_ui()
        self.request_run_analysis_signal.connect(self.parent.dataWorker.run_analysis)
        self.request_validate_columns_signal.connect(self.parent.dataWorker.validate_columns)
        self.parent.dataWorker.output_models_completed_signal.connect(self.build_output_models)
        self.parent.dataWorker.columns_validated_signal.connect(self.on_columns_validated)
        self.parent.dataWorker.analysis_complete_signal.connect(self.analysis_complete)
        
    def build_ui(self):
        
        self.topLayout = QVBoxLayout(self)
        self.topLayout.setAlignment(Qt.AlignTop)
        
        # Top
        top = QVBoxLayout()
        top.setAlignment(Qt.AlignTop)
        top.addStretch()
        self.uploaded_dataset = QLabel("Uploaded File : ", self)
        self.uploaded_dataset.setFont(self.parent.NORMAL_FONT)
        self.time_uploaded = QLabel("Time Uploaded : ", self)
        self.time_uploaded.setFont(self.parent.NORMAL_FONT)
        self.number_of_rows = QLabel("Number of rows: ", self)
        self.number_of_rows.setFont(self.parent.NORMAL_FONT)
        top.addWidget(self.uploaded_dataset)
        top.addWidget(self.time_uploaded)
        top.addWidget(self.number_of_rows)

        # Middle
        middle = QVBoxLayout()
        middle.setAlignment(Qt.AlignTop)
        middle.addStretch()
        
        yLabel = QLabel("Select Y Colum", self)
        yLabel.setFont(self.parent.SETUP_WIDGET_FONT)
        self.yBox = ColumnComboBox(self, self.parent)
        self.yBox.setFont(self.parent.SETUP_WIDGET_FONT)
        
        dateLabel = QLabel("Select Date Column (optional)", self)
        dateLabel.setFont(self.parent.SETUP_WIDGET_FONT)
        self.dateBox = ColumnComboBox(self, self.parent)
        self.dateBox.setFont(self.parent.SETUP_WIDGET_FONT)
        
        categoryLabel = QLabel("Select Category Column (optional)", self)
        categoryLabel.setFont(self.parent.SETUP_WIDGET_FONT)
        self.categoryBox = ColumnComboBox(self, self.parent)
        self.categoryBox.setFont(self.parent.SETUP_WIDGET_FONT)
        
        middle.addWidget(yLabel)
        middle.addWidget(self.yBox)        
        
        middle.addWidget(dateLabel)
        middle.addWidget(self.dateBox)  
        
        middle.addWidget(categoryLabel)
        middle.addWidget(self.categoryBox)
        
        # Bottom Half
        bottom = QHBoxLayout()
        bottom.setAlignment(Qt.AlignCenter)

        self.ignore_box   = SearchableListView(self, self.parent, title="Select Columns to Ignore (optional)", track_columns=True)
        self.multi_box    = SearchableListView(self, self.parent, title="Select Columns for Multilinear (optional)", track_columns=True)
        
        bottom.addWidget(self.ignore_box)
        bottom.addWidget(self.multi_box)
        
        # Button
        buttonLayout = QHBoxLayout()
        buttonLayout.setAlignment(Qt.AlignCenter)
        
        self.run_analysis_btn = QPushButton("Run Analysis", self)
        self.run_analysis_btn.setFont(self.parent.NORMAL_BTN_FONT)
        self.run_analysis_btn.clicked.connect(self.validate_columns)
        
        buttonLayout.addWidget(self.run_analysis_btn)
        
        self.topLayout.addLayout(top, 1)
        self.topLayout.addLayout(middle, 4)
        self.topLayout.addLayout(bottom, 4)
        self.topLayout.addLayout(buttonLayout, 1)

    def validate_columns(self):
        if self.parent.is_df_loaded:
            cols = {
                "y_col": str(self.yBox.currentText()),
                "date_col": str(self.dateBox.currentText()),
                "category_col": str(self.categoryBox.currentText()),
                "ignore_cols": self.ignore_box.selectedItems(),
                "multi_cols": self.multi_box.selectedItems()
                }
            self.request_validate_columns_signal.emit(cols)
            self.parent.start_status_task("Validating columns...")
            self.run_analysis_btn.setEnabled(False)


    @pyqtSlot(bool, list)
    def on_columns_validated(self, valid, confirm_msgs):
        if not valid:
            self.run_analysis_btn.setEnabled(True)
            self.parent.finished_status_task()
            return False

        if isinstance(confirm_msgs,list) and len(confirm_msgs) > 0:
            confirm_msg = "Warning! \n \n"
            for msg in confirm_msgs:
                confirm_msg += msg + '\n \n'
            confirm_msg += '\n Continue?'
            choice = QMessageBox.question(self, "Confirm", confirm_msg, QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                self.parent.finished_status_task()
                self.run_analysis()
                return True
            else:
                self.run_analysis_btn.setEnabled(True)
                self.parent.finished_status_task()
                return False
        else:
            self.run_analysis()
            self.parent.finished_status_task()
            return True

    def run_analysis(self):
        if self.parent.is_df_loaded:
            cols = {
                "y_col": str(self.yBox.currentText()),
                "date_col": str(self.dateBox.currentText()),
                "category_col": str(self.categoryBox.currentText()),
                "ignore_cols": self.ignore_box.selectedItems(),
                "multi_cols": self.multi_box.selectedItems()
                }
            self.request_run_analysis_signal.emit(cols)

    def update_df_name(self, df, file_nm):
        self.uploaded_dataset.setText("Uploaded File      : " + file_nm)
        self.time_uploaded.setText("Time Uploaded   : " + datetime.datetime.now().strftime("%H:%M on %m-%d-%Y"))
        self.number_of_rows.setText("Number of Rows : " + str(df.shape[0]))

    @pyqtSlot(object, object, object, object, object, object)
    def build_output_models(self, splash_model, all_dates_model, by_date_cor_model, by_category_cor_model, by_date_dp_model, by_category_dp_model):
        self.parent.pages[MetricComparisonPage].build_splash_model(splash_model)
        self.parent.pages[MetricComparisonPage].build_all_dates_model(all_dates_model)
        self.parent.pages[MetricComparisonPage].build_by_date_cor_model(by_date_cor_model)
        self.parent.pages[MetricComparisonPage].build_by_category_cor_model(by_category_cor_model)
        self.parent.pages[MetricComparisonPage].build_by_date_dp_model(by_date_dp_model)
        self.parent.pages[MetricComparisonPage].build_by_category_dp_model(by_category_dp_model)

    @pyqtSlot()
    def analysis_complete(self):
        self.run_analysis_btn.setEnabled(True)
        self.parent.finished_task(MetricComparisonPage)
    

class TransformationPage(QWidget):

    create_transformation_signal = pyqtSignal(transformations.Transformation)
    remove_transformation_signal = pyqtSignal(transformations.Transformation)

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.transformation_popups = []
        self.transformation_boxes = []
        self.create_transformation_signal.connect(self.parent.dataWorker.create_transformation)
        self.remove_transformation_signal.connect(self.parent.dataWorker.remove_transformation)
        self.parent.dataWorker.transformation_completed_signal.connect(self.create_completed_transformation_box)
        self.build_main_layout()

    def build_main_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        
        # Button
        buttonLayout = QHBoxLayout()
        buttonLayout.setAlignment(Qt.AlignCenter)
        
        add_tran_btn = QPushButton("Add Transformation", self)
        add_tran_btn.setFont(self.parent.NORMAL_BTN_FONT)
        add_tran_btn.clicked.connect(self.add_transformation)
        add_tran_btn.setFixedSize(200, 75)
        
        buttonLayout.addWidget(add_tran_btn)

        # Bottom Half
        self.tran_layout = QtWidgets.QVBoxLayout()
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QtWidgets.QWidget(self)
        self.transformations = FlowLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.tran_layout.addWidget(self.scrollArea)
        
        self.main_layout.addLayout(buttonLayout, 1)
        self.main_layout.addLayout(self.tran_layout, 10)

    def add_transformation(self):
        popup = TransformationPopupBox(self, self.parent)
        self.transformation_popups.append(popup)

    def create_completed_transformation_box(self, transformation):
        box = TransformationSummaryBox(self, self.parent, self.transformations)
        transformation.add_summary_widgets(box, self.parent)
        box.content_layout.addStretch()
        self.transformations.addWidget(box)
        self.transformation_boxes.append(box)

    def remove_all_transformations(self):
        cnt = self.transformations.count()-1
        while(cnt >= 0):
            box = self.transformations.itemAt(cnt).widget()
            if isinstance(box, TransformationSummaryBox):
                box.remove_widget()
            cnt -= 1
        self.transformation_boxes = []


class FilterPage(QWidget):

    create_filter_signal = pyqtSignal(filters.Filter)
    remove_filter_signal = pyqtSignal(filters.Filter)

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.filter_popups = []
        self.filter_boxes = []
        self.create_filter_signal.connect(self.parent.dataWorker.create_filter)
        self.remove_filter_signal.connect(self.parent.dataWorker.remove_filter)
        self.parent.dataWorker.filter_applied_signal.connect(self.create_applied_filter_box)
        self.build_main_layout()

    def build_main_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        
        # Button
        buttonLayout = QHBoxLayout()
        buttonLayout.setAlignment(Qt.AlignCenter)
        
        add_filter_btn = QPushButton("Add Filter", self)
        add_filter_btn.setFont(self.parent.NORMAL_BTN_FONT)
        add_filter_btn.clicked.connect(self.add_filter)
        add_filter_btn.setFixedSize(200, 75)
        
        buttonLayout.addWidget(add_filter_btn)

        # Bottom Half
        self.filter_layout = QtWidgets.QVBoxLayout()
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QtWidgets.QWidget(self)
        self.filters = FlowLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.filter_layout.addWidget(self.scrollArea)
        
        self.main_layout.addLayout(buttonLayout, 1)
        self.main_layout.addLayout(self.filter_layout, 10)

    def add_filter(self):
        popup = FilterPopupBox(self, self.parent)
        self.filter_popups.append(popup)

    def create_applied_filter_box(self, _filter):
        box = FilterSummaryBox(self, self.parent, self.filters)
        _filter.add_summary_widgets(box, self.parent)
        box.content_layout.addStretch()
        self.filters.addWidget(box)
        self.filter_boxes.append(box)

    def remove_all_filters(self):
        cnt = self.filters.count()-1
        while(cnt >= 0):
            box = self.filters.itemAt(cnt).widget()
            if isinstance(box, FilterSummaryBox):
                box.remove_widget()
            cnt -= 1
        self.filter_boxes = []


class DateAggregationPage(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent

    request_date_aggregation_signal = pyqtSignal(dict)
    request_remove_date_aggregation_signal = pyqtSignal()

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.build_ui()
        self.request_date_aggregation_signal.connect(self.parent.dataWorker.aggregate_by_date)
        self.request_remove_date_aggregation_signal.connect(self.parent.dataWorker.remove_date_aggregation)
        self.parent.dataWorker.date_aggregation_complete_signal.connect(self.on_aggregate_complete)
        self.parent.dataWorker.remove_date_aggregation_complete_signal.connect(self.on_aggregate_removed)
        
    def build_ui(self):
        
        self.topLayout = QVBoxLayout(self)
        self.topLayout.setAlignment(Qt.AlignTop)
        
        # Top Half
        top = QVBoxLayout()
        top.setAlignment(Qt.AlignTop)
        top.addStretch()
        
        date_label = QLabel("Select Date Colum", self)
        date_label.setFont(self.parent.SETUP_WIDGET_FONT)
        self.date_box = ColumnComboBox(self, self.parent)
        self.date_box.setFont(self.parent.SETUP_WIDGET_FONT)
        
        categoryLabel = QLabel("Select Category Column (optional)", self)
        categoryLabel.setFont(self.parent.SETUP_WIDGET_FONT)
        self.categoryBox = ColumnComboBox(self, self.parent)
        self.categoryBox.setFont(self.parent.SETUP_WIDGET_FONT)
        
        agg_level_label = QLabel("Select Aggregation Level (your data must be more granular than this)", self)
        agg_guide_label = QLabel("https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior", self)
        self.agg_level_box = QLineEdit(self)
        self.agg_level_box.setText("%Y")

        agg_func_label = QLabel("Select Aggregation Level (your data must be more granular than this)", self)
        self.agg_func_box = QComboBox(self)
        self.agg_func_box.addItems(["Sum", "Mean", "Median", "Mode", "Max", "Min"])

        top.addWidget(date_label)
        top.addWidget(self.date_box)        
        
        top.addWidget(categoryLabel)
        top.addWidget(self.categoryBox)

        top.addWidget(agg_level_label)
        top.addWidget(agg_guide_label)
        top.addWidget(self.agg_level_box)

        top.addWidget(agg_func_label)
        top.addWidget(self.agg_func_box)
        
        # Button
        buttonLayout = QHBoxLayout()
        buttonLayout.setAlignment(Qt.AlignCenter)
        
        self.aggregate_dates_btn = QPushButton("Aggregate Dates", self)
        self.aggregate_dates_btn.setFont(self.parent.NORMAL_BTN_FONT)
        self.aggregate_dates_btn.setFixedSize(250,75)
        self.aggregate_dates_btn.clicked.connect(self.aggregate_dates)
        
        buttonLayout.addWidget(self.aggregate_dates_btn)
        
        self.topLayout.addLayout(top, 1)
        self.topLayout.addLayout(buttonLayout, 1)

    @pyqtSlot()
    def on_aggregate_complete(self):
        try:
            self.aggregate_dates_btn.disconnect()
        except:
            pass
        self.aggregate_dates_btn.setText("Remove Date Aggregation")
        self.aggregate_dates_btn.setStyleSheet('background-color : red')
        self.aggregate_dates_btn.setEnabled(True)
        self.aggregate_dates_btn.clicked.connect(self.remove_date_aggregation)

    @pyqtSlot()
    def on_aggregate_removed(self):
        try:
            self.aggregate_dates_btn.disconnect()
        except:
            pass
        self.aggregate_dates_btn.setText("Aggregate Dates")
        self.aggregate_dates_btn.setStyleSheet('background-color : #474747')
        self.aggregate_dates_btn.setEnabled(True)
        self.aggregate_dates_btn.clicked.connect(self.aggregate_dates)

    def aggregate_dates(self):

        choice = QMessageBox.question(self, "Aggregate by date", "Warning! Aggregating by date will remove any filters you have created. And any columns that can't be transformed into a numeric will have their data lost. This process will likely take a very long time, especially if you provide a category column. Continue?", QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            pass
        else:
            return

        if self.parent.is_df_loaded:
            self.parent.pages[FilterPage].remove_all_filters()
            info = {
                "date_col": str(self.date_box.currentText()),
                "category_col": str(self.categoryBox.currentText()),
                "agg_level": str(self.agg_level_box.text()),
                "agg_func": str(self.agg_func_box.currentText()),
                }
            self.request_date_aggregation_signal.emit(info)
            self.aggregate_dates_btn.setEnabled(False)
        else:
            QMessageBox.about(self.parent, "Warning", "Please load a CSV data set first.")

    def remove_date_aggregation(self):
        
        choice = QMessageBox.question(self, "Aggregate by date", "Warning! Aggregating by date will remove any filters you have created. And any columns that can't be transformed into a numeric will have their data lost. Continue?", QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            pass
        else:
            return

        self.parent.pages[FilterPage].remove_all_filters()
        self.aggregate_dates_btn.setEnabled(False)
        self.request_remove_date_aggregation_signal.emit()

class DataPreviewPage(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent

        self.table = ColumnFilterTable(
            self, self.parent,
            cell_font=self.parent.TABLE_FONT_MEDIUM,
            header_font=self.parent.TABLE_FONT_LARGE
        )
    
    def buildModel(self, model):
        self.table.buildModel(model)


class MetricComparisonPage(QWidget):

    request_run_cor_mat_signal = pyqtSignal(list)
    
    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.parent.dataWorker.correlation_scatter_complete_signal.connect(self.display_cor_scatter)
        self.parent.dataWorker.correlation_histogram_complete_signal.connect(self.display_cor_hist)
        self.parent.dataWorker.correlation_matrix_completed_signal.connect(self.build_cor_mat_model)
        self.request_run_cor_mat_signal.connect(self.parent.dataWorker.run_correlation_matrix)

        # Create Tabs
        self.tabs = QTabWidget(self)
        # Create layout
        self.gridLayout = QGridLayout(self)
        self.gridLayout.addWidget(self.tabs, 0, 0)

        self.build_splash_tab()
        self.build_all_dates_tab()
        self.build_by_date_cor_tab()
        self.build_by_category_cor_tab()
        self.build_by_date_dp_tab()
        self.build_by_category_dp_tab()
        self.build_all_dates_cor_mat_tab()

    def build_splash_tab(self):
        splash_tab = QTabWidget(self.tabs)
        self.tabs.addTab(splash_tab, "Splash Page")
        self.splash_table = QuickFilterTable(
            splash_tab,
            self.parent,
            filter_col=0,
            resize_to_contents_cols=list(range(1,8)),
            cell_font=self.parent.TABLE_FONT_MEDIUM,
            header_font=self.parent.TABLE_FONT_LARGE,
            sort_enabled=True
            )

    def build_by_date_cor_tab(self):
        by_date_cor_tab = QTabWidget(self.tabs)
        self.tabs.addTab(by_date_cor_tab, "Correlations - By Date")
        self.by_date_cor_table = QuickFilterTable(
            by_date_cor_tab,
            self.parent,
            filter_col=0,
            resize_to_contents_cols=list(range(1,8)),
            cell_font=self.parent.TABLE_FONT_MEDIUM,
            header_font=self.parent.TABLE_FONT_LARGE,
            sort_enabled=True
        )

    def build_by_category_cor_tab(self):
        by_category_cor_tab = QTabWidget(self.tabs)
        self.tabs.addTab(by_category_cor_tab, "Correlations - By Category")
        self.by_category_cor_table = QuickFilterTable(
            by_category_cor_tab,
            self.parent,
            filter_col=0,
            resize_to_contents_cols=list(range(1,8)),
            cell_font=self.parent.TABLE_FONT_MEDIUM,
            header_font=self.parent.TABLE_FONT_LARGE,
            sort_enabled=True
        )

    def build_by_date_dp_tab(self):
        by_date_dp_tab = QTabWidget(self.tabs)
        self.tabs.addTab(by_date_dp_tab, "Data Points - By Date")
        self.by_date_dp_table = QuickFilterTable(
            by_date_dp_tab,
            self.parent,
            filter_col=0,
            cell_font=self.parent.TABLE_FONT_MEDIUM,
            header_font=self.parent.TABLE_FONT_LARGE,
            sort_enabled=True
        )

    def build_by_category_dp_tab(self):
        by_category_dp_tab = QTabWidget(self.tabs)
        self.tabs.addTab(by_category_dp_tab, "Data Points - By Category")
        self.by_category_dp_table = QuickFilterTable(
            by_category_dp_tab,
            self.parent,
            filter_col=0,
            cell_font=self.parent.TABLE_FONT_MEDIUM,
            header_font=self.parent.TABLE_FONT_LARGE,
            sort_enabled=True
        )

    def build_all_dates_tab(self):
        self.allDatesTab = QTabWidget(self.tabs)
        self.tabs.addTab(self.allDatesTab, "Summary - All Dates")
        self.build_all_dates_table_tab()
        self.build_all_dates_hist_tab()
        self.build_all_dates_scatter_tab()

    def build_all_dates_table_tab(self):
        tableTab = QTabWidget(self.allDatesTab)
        self.allDatesTab.addTab(tableTab, "Table")
        self.all_dates_table = QuickFilterTable(
            tableTab,
            self.parent,
            filter_col=0,
            resize_to_contents_cols=list(range(1,8)),
            cell_font=self.parent.TABLE_FONT_MEDIUM,
            header_font=self.parent.TABLE_FONT_LARGE,
            sort_enabled=True
        )

    def build_all_dates_cor_mat_tab(self):
        cor_mat_tab = QWidget()
        self.grid = QGridLayout(cor_mat_tab)

        self.cor_mat_columns = SearchableListView(self, self.parent, title="Select Columns for correlation matrix", track_columns=True)
        self.run_cor_mat_button = QPushButton("Run Correlation Matrix", cor_mat_tab)
        self.run_cor_mat_button.clicked.connect(self.request_run_cor_mat)

        self.cor_matrix = QuickFilterTable(
            cor_mat_tab,
            self.parent,
            filter_col=0,
            cell_font=self.parent.TABLE_FONT_MEDIUM,
            header_font=self.parent.TABLE_FONT_LARGE
        )
        
        self.grid.addWidget(self.cor_mat_columns   , 0, 0, 3, 1)
        self.grid.addWidget(self.run_cor_mat_button, 3, 0, 1, 1)
        self.grid.addLayout(self.cor_matrix        , 0, 1, 4, 1)

        #cor_mat_tab.setLayout(self.grid)
        self.tabs.addTab(cor_mat_tab, "Correlation Matrix")

    def build_all_dates_hist_tab(self):
        self.histogram = PlotlyView(self.parent)
        self.allDatesTab.addTab(self.histogram, "Histogram")

    def build_all_dates_scatter_tab(self):
        self.scatter = PlotlyView(self.parent)
        self.allDatesTab.addTab(self.scatter, "Scatter")

    def build_splash_model(self, model):
        self.splash_table.buildModel(model)

    def build_by_date_cor_model(self, model):
        self.by_date_cor_table.buildModel(model)

    def build_by_category_cor_model(self, model):
        self.by_category_cor_table.buildModel(model)

    def build_by_date_dp_model(self, model):
        self.by_date_dp_table.buildModel(model)

    def build_by_category_dp_model(self, model):
        self.by_category_dp_table.buildModel(model)

    def build_all_dates_model(self, model):
        self.all_dates_table.buildModel(model)

    def build_cor_mat_model(self, model):
        self.run_cor_mat_button.setEnabled(True)
        self.cor_matrix.buildModel(model)

    @pyqtSlot(pd.DataFrame)
    def display_cor_scatter(self, df):
        self.scatter.scatter(df=df, x_col="Data Points", y_col="Correlation")

    @pyqtSlot(list)
    def display_cor_hist(self, vals):
        self.histogram.histogram(vals)

    def request_run_cor_mat(self):
        self.run_cor_mat_button.setEnabled(False)
        cols = self.cor_mat_columns.selectedItems()
        self.request_run_cor_mat_signal.emit(cols)

class MetricDivePage(QWidget):

    request_run_metric_dive_signal = pyqtSignal(dict, bool)
    request_run_metric_3d_signal = pyqtSignal(str)
    request_save_metric_dive_report_signal = pyqtSignal(str)

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent  = parent
        self.x_col   = ""
        self.cat_col = ""
        self.y_col   = ""
        self.z_col   = ""

        # connect thread signals
        self.parent.dataWorker.metric_dive_df_signal.connect(self.display_metric_dive_plots)
        self.parent.dataWorker.metric_dive_anova_stats.connect(self.display_metric_dive_anova_stats)
        self.parent.dataWorker.metric_dive_summary_stats.connect(self.display_metric_dive_summary_stats)
        self.parent.dataWorker.metric_dive_sens_spec.connect(self.display_sens_spec_plot)
        self.parent.dataWorker.metric_dive_box_and_whisker.connect(self.display_box_and_whisker)
        self.parent.dataWorker.metric_dive_dp_by_date.connect(self.display_dp_by_date)
        self.parent.dataWorker.metric_dive_qq_norm.connect(self.display_qq_norm)
        self.parent.dataWorker.metric_dive_qq_y.connect(self.display_qq_y)
        self.parent.dataWorker.metric_dive_dp_by_category.connect(self.display_dp_by_category)
        self.parent.dataWorker.metric_dive_metric_turnover.connect(self.display_metric_dive_metric_turnover)
        self.parent.dataWorker.metric_dive_3d_signal.connect(self.display_3d_plot)
        self.parent.dataWorker.metric_dive_started_signal.connect(lambda: self.enable_actions(False))
        self.parent.dataWorker.metric_dive_complete_signal.connect(lambda: self.enable_actions(True))
        self.parent.dataWorker.metric_dive_3d_standalone_complete_signal.connect(lambda: self.enable_actions(True))
        self.parent.dataWorker.metric_dive_data_preview_model_signal.connect(self.update_metric_dive_model)

        self.request_run_metric_dive_signal.connect(self.parent.dataWorker.run_metric_dive)
        self.request_run_metric_3d_signal.connect(self.parent.dataWorker.run_metric_3d_standalone)
        self.request_save_metric_dive_report_signal.connect(self.parent.dataWorker.prepare_metric_dive_for_report)
        #self.parent.dataWorker.y_col_change_signal.connect(self.update_y_col)
        #self.parent.dataWorker.analysis_complete_signal.connect(self.clear_all) <- this was used when metric dive was tied to run analysis
        
        # Create layout
        self.gridLayout = QGridLayout(self)

        # Create X column box
        x_col_label = QLabel('X Column: ')
        x_col_label.setFont(self.parent.NORMAL_FONT)
        self.x_col_box = ColumnComboBox(self, self.parent)
        self.x_col_box.setFont(QFont("Raleway", 16))
        
        # Create Y column box
        y_col_label = QLabel('Y Column: ')
        y_col_label.setFont(self.parent.NORMAL_FONT)
        self.y_col_box = ColumnComboBox(self, self.parent)
        self.y_col_box.setFont(QFont("Raleway", 16))

        # Create date column box
        date_col_label = QLabel('Date Column (optional): ')
        date_col_label.setFont(self.parent.NORMAL_FONT)
        self.date_col_box = ColumnComboBox(self, self.parent)
        self.date_col_box.setFont(QFont("Raleway", 16))

        # Create category column box
        category_col_label = QLabel('Category Column (optional): ')
        category_col_label.setFont(self.parent.NORMAL_FONT)
        self.category_col_box = ColumnComboBox(self, self.parent)
        self.category_col_box.setFont(QFont("Raleway", 16))

        # Create button for running metric dive
        self.run_metric_dive_btn = QPushButton("Run Metric Dive", self)
        self.run_metric_dive_btn.setFont(QFont("Raleway", 16))
        self.run_metric_dive_btn.clicked.connect(self.run_metric_dive)
        
        # Create report button
        self.save_btn = QPushButton("Save Report of Metric", self)
        self.save_btn.clicked.connect(self.save_report)
        # self.save_btn.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        # self.save_btn.setFixedSize(400,200)
        self.save_btn.setFont(parent.NORMAL_BTN_FONT)
        self.save_btn.setEnabled(False)

        # Create Tabs
        self.tabs = QTabWidget(self)
        self.plotTabs = QTabWidget(self)

        self.tabs.addTab(self.plotTabs, "Plots")
        self.build_scatter_tab()
        self.build_xy_line_chart_tab()
        self.build_histogram_tab()
        self.build_dp_tab()
        self.build_qq_tab()
        self.build_summary_tab()
        self.build_metric_turnover_tab()
        self.build_sens_spec_tab()
        self.build_box_whisker_tab()
        self.build_3d_tab()

        self.build_data_preview_tab()
        # Add widgets to layout
        #self.gridLayout.addWidget(self.y_col_label, 0, 0, 1, 1)
        self.gridLayout.addWidget(x_col_label, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.x_col_box, 0, 1, 1, 4)
        self.gridLayout.addWidget(y_col_label, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.y_col_box, 1, 1, 1, 4)
        self.gridLayout.addWidget(date_col_label, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.date_col_box, 2, 1, 1, 4)
        self.gridLayout.addWidget(category_col_label, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.category_col_box, 3, 1, 1, 4)
        self.gridLayout.addWidget(self.run_metric_dive_btn, 0, 5, 4, 1)
        self.gridLayout.addWidget(self.tabs, 5, 0, 1, 6)
        self.gridLayout.addWidget(self.save_btn, 6, 0, 1, 6)
    
    def update_metric_dive_model(self, model):
        self.metric_dive_table.buildModel(model)

    def enable_report_btn(self, enable):
        self.save_btn.setEnabled(enable)

    def enable_actions(self, enable):
        self.save_btn.setEnabled(enable)
        self.x_col_box.setEnabled(enable)
        self.y_col_box.setEnabled(enable)
        self.date_col_box.setEnabled(enable)
        self.z_col_box.setEnabled(enable)
        self.run_3d_btn.setEnabled(enable)
        self.category_col_box.setEnabled(enable)
        self.run_metric_dive_btn.setEnabled(enable)
        if hasattr(self, 'filter_scatter_widget'):
            try:
                self.filter_scatter_widget.enable_actions(enable)
            except:
                pass

        if enable:
            self.parent.finished_status_task()

    def build_data_preview_tab(self):
        self.dataTab = QTabWidget(self)
        self.tabs.addTab(self.dataTab, "Data")
        self.metric_dive_table = ColumnFilterTable(self.dataTab, self.parent, numeric_sort=False)

    def build_summary_tab(self):
        self.detailsTab = DetailsTab(self, self.parent)
        self.tabs.addTab(self.detailsTab, "Details")

    def build_scatter_tab(self):
        self.filter_scatter_widget = FilterScatterWidget(self, self.parent)
        self.plotTabs.addTab(self.filter_scatter_widget, "Scatter")

        self.filter_scatter_widget.metric_dive_filter_signal.connect(lambda x: self.enable_actions(False))
        self.filter_scatter_widget.remove_metric_dive_filter_signal.connect(lambda: self.enable_actions(True))

    def build_xy_line_chart_tab(self):
        self.xy_line_chart = PlotlyView(self.parent)
        #self.xy_line_chart_loader = Loader(self.xy_line_chart)
        self.plotTabs.addTab(self.xy_line_chart, "Line Chart")

    def build_histogram_tab(self):
        self.histogram = PlotlyView(self.parent)
        self.plotTabs.addTab(self.histogram, "Histogram")
    
    def build_dp_tab(self):
        self.dp_tabs = QTabWidget(self)
        self.plotTabs.addTab(self.dp_tabs, "Data Points")

        self.dp_date_plot     = PlotlyView(self.parent)
        self.dp_category_plot = PlotlyView(self.parent)

        self.dp_tabs.addTab(self.dp_date_plot, "By Date")
        self.dp_tabs.addTab(self.dp_category_plot, "By Category")

    def build_qq_tab(self):
        self.qq_tabs = QTabWidget(self)
        self.plotTabs.addTab(self.qq_tabs, "QQ Plots")

        self.qq_norm_plot = PlotlyView(self.parent)
        self.qq_norm_y    = PlotlyView(self.parent)

        self.qq_tabs.addTab(self.qq_norm_plot, "vs Theoretical Normal")
        self.qq_tabs.addTab(self.qq_norm_y, "vs Y's Distribution")

    def build_sens_spec_tab(self):
        self.sens_spec_tab = QWidget()
        self.sens_spec_layout = QHBoxLayout()
        self.sens_spec_tab.setLayout(self.sens_spec_layout)

        table_layout = QVBoxLayout()
        self.sens_spec_table_1 = PrettyTable(n_rows=4, n_columns=2, parent=self, controller=self.parent, title="Categories", column_headers=["Category", "Count"])
        self.sens_spec_table_1.set_column_values(0, ['True Negative', 'False Negative', 'True Positive', 'False Positive'])
        self.sens_spec_table_2 = PrettyTable(n_rows=5, n_columns=2, parent=self, controller=self.parent, title="Summary Stats", column_headers=["Metric", "Value"])
        self.sens_spec_table_2.set_column_values(0, [
            '% of flagged events were catastrophic losses?',
            '% of unflagged events were catastrophic losses?',
            '% of catastrophic losses were flagged by the model?',
            '% of non-catastrophic losses were flagged by the model?',
            '% of the universe is flagged?'])
        table_layout.addWidget(self.sens_spec_table_1)
        table_layout.addWidget(self.sens_spec_table_2)

        self.sens_spec_plot = ScatterPlot(self.plotTabs, self.parent, tooltip=True, legend=False, selectable=False)
        self.sens_spec_layout.addLayout(table_layout)
        self.sens_spec_layout.addWidget(self.sens_spec_plot.view)
        self.plotTabs.addTab(self.sens_spec_tab, "Sensitivity/Specificity")

    def build_metric_turnover_tab(self):
        self.metric_turnover_plot = PlotlyView(self.parent)
        self.plotTabs.addTab(self.metric_turnover_plot, "Rank Volatility")

    def build_3d_tab(self):
        self.main_widget_3d = QWidget()

        self.main_layout_3d = QVBoxLayout(self.main_widget_3d)
        self.plot_3d = PlotlyView(self.parent)

        # Filter Side
        self.inputs_3d = QHBoxLayout()

        zLabel = QLabel("Select Z Column", self)
        zLabel.setFont(self.parent.SETUP_WIDGET_FONT)
        self.z_col_box = ColumnComboBox(self, self.parent)
        self.z_col_box.setFont(self.parent.SETUP_WIDGET_FONT)

        self.run_3d_btn = QPushButton("Run 3D", self)
        self.run_3d_btn.clicked.connect(self.run_metric_3d)

        self.inputs_3d.addWidget(zLabel)
        self.inputs_3d.addWidget(self.z_col_box)
        self.inputs_3d.addWidget(self.run_3d_btn)

        # Scatter Side
        self.scatter_side_3d = QVBoxLayout()
        self.scatter_side_3d.addWidget(self.plot_3d)

        self.main_layout_3d.addLayout(self.inputs_3d)
        self.main_layout_3d.addLayout(self.scatter_side_3d)

        self.plotTabs.addTab(self.main_widget_3d, "3D Plot")

    def build_box_whisker_tab(self):
        pass
        # self.box_and_whisker = MplCanvas(self.plotTabs, self.parent)
        # self.plotTabs.addTab(self.box_and_whisker, "Box and Whisker")
        # self.display_box_and_whisker(None)

    def run_metric_dive(self):
        x_col = self.x_col_box.currentText()
        if x_col == "":
            self.parent.show_error("Please select an X column.")
            return

        if x_col not in [self.x_col_box.itemText(i) for i in range(self.x_col_box.count())]:
            self.parent.show_error("Column: '"+x_col+"' not found in dataset.")
            return

        y_col = self.y_col_box.currentText()
        if y_col == "":
            self.parent.show_error("Please select a Y column.")
            return
            
        if y_col not in [self.y_col_box.itemText(i) for i in range(self.y_col_box.count())]:
            self.parent.show_error("Column: '"+y_col+"' not found in dataset.")
            return

        date_col = self.date_col_box.currentText()
        if date_col not in [self.date_col_box.itemText(i) for i in range(self.date_col_box.count())]:
            self.parent.show_error("Column: '"+date_col+"' not found in dataset.")
            return

        cat_col = self.category_col_box.currentText()
        if cat_col not in [self.category_col_box.itemText(i) for i in range(self.category_col_box.count())]:
            self.parent.show_error("Column: '"+cat_col+"' not found in dataset.")
            return

        if date_col in [x_col, y_col]:
            self.parent.show_error("Date column can't be the same as X or Y column.")
            return

        self.enable_actions(False)
        self.clear_all(False)
        self.parent.start_status_task("Running metric dive for column: " + x_col)
        self.x_col = x_col
        self.y_col = y_col
        self.date_col = date_col
        self.cat_col = cat_col
        cols = {
            "x_col": x_col,
            "y_col": y_col,
            "date_col": date_col,
            "category_col": cat_col
        }
        self.request_run_metric_dive_signal.emit(cols, True)

    def run_metric_3d(self):
        z_col = self.z_col_box.currentText()

        if z_col == "":
            return

        if z_col not in [self.z_col_box.itemText(i) for i in range(self.z_col_box.count())]:
            self.parent.show_error("Column: '"+z_col+"' not found in dataset.")
            return

        self.z_col = z_col
        self.enable_actions(False)
        self.parent.start_status_task("Running 3D plot for columns: " + self.x_col + ", " + self.y_col + ", " + self.z_col)
        self.request_run_metric_3d_signal.emit(z_col)
        return

    def create_proposed_nm(self, x, y):
        clean_x = clean_string(x)
        clean_y = clean_string(y)

        proposed_nm = clean_x + ' against ' + clean_y + '.html'
        return proposed_nm

    def clean_file_nm(self, nm):
        if not nm.endswith(".html"):
            return nm+".html"
        return nm

    def save_report(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        proposed_name = self.create_proposed_nm(self.x_col, self.y_col)
        fileName, _ = QFileDialog.getSaveFileName(self,"Save metric report", proposed_name,"HTML Files (*.html)", options=options)
        if fileName:
            clean_nm = self.clean_file_nm(fileName)
            self.request_save_metric_dive_report_signal.emit(clean_nm)

    @pyqtSlot(dict)
    def on_analysis_complete(self, output_dict):
        # This is not being used anymore??
        self.display_qq_norm(output_dict['qq_norm'])
        self.display_qq_y(output_dict['qq_y'])
        self.display_dp_by_date(output_dict['dp_date'])
        self.display_dp_by_category(output_dict['dp_category'])
        self.display_metric_dive_metric_turnover(output_dict['metric_turnover'])
        self.display_metric_dive_anova_stats(output_dict['anova'])
        self.display_metric_dive_summary_stats(output_dict['summary'])
        self.display_metric_dive_plots(output_dict['metric_dive_data'][0], output_dict['metric_dive_data'][1])
        self.detailsTab.build_metric_dive_performance_date_proxy(output_dict['perf_date_table'])
        self.detailsTab.build_metric_dive_performance_category_proxy(output_dict['perf_category_table'])

    def clear_all(self, col_box=True):
        if col_box:
            self.x_col_box.setCurrentIndex(0)
            self.x_col = ""
            self.category_col_box.setCurrentIndex(0)
            self.category_col = ""

        #self.xy_line_chart_loader.start_load()

        self.display_qq_norm(None)
        self.display_qq_y(None)
        self.display_dp_by_date(None)
        self.display_dp_by_category(None)
        self.display_metric_dive_metric_turnover(None)
        self.display_metric_dive_anova_stats(None)
        self.display_metric_dive_summary_stats(None)
        self.display_metric_dive_plots(None, None)
        self.detailsTab.build_metric_dive_performance_date_proxy(None)
        self.detailsTab.build_metric_dive_performance_category_proxy(None)
        self.display_box_and_whisker(None)
        #self.filter_scatter_widget.enable_filter(False)

    @pyqtSlot(object)
    def display_qq_norm(self, df):
        if df is None or df.empty:
            self.qq_norm_plot.clear_plot()
        else:
            self.qq_norm_plot.scatter(df=df, x_col=df.columns[0], y_col='Normal')

    @pyqtSlot(object)
    def display_qq_y(self, df):
        if df is None or df.empty:
            self.qq_norm_y.clear_plot()
        else:
            self.qq_norm_y.scatter(df=df, x_col=df.columns[0], y_col=df.columns[1])

    @pyqtSlot(object)
    def display_dp_by_date(self, df):
        if df is None or df.empty:
            self.dp_date_plot.clear_plot()
        else:
            self.dp_date_plot.line_graph(df['Date'], df['Data Points'])

    @pyqtSlot(object)
    def display_dp_by_category(self, df):
        if df is None or df.empty:
            self.dp_category_plot.clear_plot()
        else:
            self.dp_category_plot.bar_graph(df['Category'], df['Data Points'])

    @pyqtSlot(object)
    def display_metric_dive_metric_turnover(self, df):
        if df is None or df.empty:
            self.metric_turnover_plot.clear_plot()
        else:
            self.metric_turnover_plot.line_graph(df['Date'], df['Metric Turnover'])

    @pyqtSlot(dict)
    def display_metric_dive_anova_stats(self, stats_dict):
        self.detailsTab.set_anova_values(stats_dict)

    @pyqtSlot(dict)
    def display_metric_dive_summary_stats(self, stats_dict):
        self.detailsTab.set_summary_values(stats_dict)

    @pyqtSlot(object, object)
    def display_sens_spec_plot(self, df, category_size_dict):
        if df is None or df.empty:
            self.sens_spec_plot.clear_plot()
            return

        category_color_dict = {
            'True Negative': (34, 139, 34),
            'False Negative': (255, 0, 0),
            'True Positive': (34, 139, 34),
            'False Positive': (255, 165, 0)
        }

        count_dict = {
            'True Negative': list(df[df['Category'] == 'True Negative']['Count'])[0],
            'False Negative': list(df[df['Category'] == 'False Negative']['Count'])[0],
            'True Positive': list(df[df['Category'] == 'True Positive']['Count'])[0],
            'False Positive': list(df[df['Category'] == 'False Positive']['Count'])[0]
        }

        self.sens_spec_plot.quadrant(df, 'x', 'y', category='Category', category_color_dict=category_color_dict, category_size_dict=category_size_dict, alpha=1)

        counts = [str(count) for count in count_dict.values()]
        self.sens_spec_table_1.set_column_values(1, counts)

        first  = '%.4f' % (float(count_dict['True Positive'] / (count_dict['True Positive'] + count_dict['False Positive']))*100) + "%"
        second = '%.4f' % (float(count_dict['False Negative'] / (count_dict['True Negative'] + count_dict['False Negative']))*100) + "%"
        third  = '%.4f' % (float(count_dict['True Positive'] / (count_dict['True Positive'] + count_dict['False Negative']))*100) + "%"
        fourth = '%.4f' % (float(count_dict['False Positive'] / (count_dict['False Positive'] + count_dict['True Negative']))*100) + "%"
        fifth  = '%.4f' % (float((count_dict['True Positive'] + count_dict['False Positive']) / sum([float(count) for count in count_dict.values()]))*100) + "%"

        self.sens_spec_table_2.set_column_values(1, [first, second, third, fourth, fifth])

    @pyqtSlot(list)
    def display_box_and_whisker(self, vals):
        pass
        # if vals is None:
        #     self.box_and_whisker.clear_plot()
        # else:
        #     self.box_and_whisker.boxplot(vals)

    @pyqtSlot(str)
    def update_y_col(self, y_col):
        pass
        # self.y_col = y_col
        # self.y_col_label.setText("Y Column: "+y_col)

    @pyqtSlot(pd.DataFrame, dict)
    def display_3d_plot(self, df, cols):
        if df is None  or df.empty or not isinstance(cols,dict):
            self.plot_3d.clear_plot()
            return
        if cols["z_col"] is None or cols["z_col"] == "" or cols["z_col"] not in df.columns:
            self.plot_3d.clear_plot()
            return
        
        self.plot_3d.scatter3d(df=df, x_col=cols["x_col"], y_col=cols["y_col"], z_col=cols["z_col"], date_col=cols["date_col"], category_col=cols["category_col"])

    @pyqtSlot(pd.DataFrame, dict)
    def display_metric_dive_plots(self, df, cols):
        if (
            df is None
            or df.empty
            or not isinstance(cols, dict)
            or any([col_valid not in df.columns for col_valid in [col_name for col_name in cols.values() if col_name != "" and col_name is not None]])
        ):
            self.filter_scatter_widget.scatter.clear_plot()
            self.filter_scatter_widget.set_dates(None, None)
            self.histogram.clear_plot()
            self.xy_line_chart.clear_plot()
        else:
            self.filter_scatter_widget.scatter.scatter(df=df, x_col=cols["x_col"], y_col=cols["y_col"], category_col=cols["category_col"], date_col=cols["date_col"], add_trendline=True)
            self.filter_scatter_widget.set_dates(df, date_col=cols["date_col"])
            self.histogram.histogram(x=df[cols["x_col"]])
            if(cols["date_col"] in df.columns and cols["x_col"] in df.columns and cols["y_col"] in df.columns):
                self.xy_line_chart.line_graph_dual_axis(df[cols["date_col"]], df[cols["x_col"]], df[cols["y_col"]])
            else:
                self.xy_line_chart.clear_plot()
            #self.xy_line_chart_loader.finish_load()

        self.display_3d_plot(df, cols)
   