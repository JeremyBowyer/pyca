import numpy as np
import pandas as pd

from functools import partial
from IPython import embed

from models import *
from ui_objects import *
from threads import *

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.build_ui()
        self.request_run_analysis_signal.connect(self.parent.dataWorker.run_analysis)
        self.parent.dataWorker.analysis_complete_signal.connect(self.analysis_complete)
        
    def build_ui(self):
        
        self.topLayout = QVBoxLayout(self)
        self.topLayout.setAlignment(Qt.AlignCenter)
        
        # Top Half
        top = QVBoxLayout()
        top.setAlignment(Qt.AlignCenter)
        
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
        
        top.addWidget(yLabel)
        top.addWidget(self.yBox)        
        
        top.addWidget(dateLabel)
        top.addWidget(self.dateBox)  
        
        top.addWidget(categoryLabel)
        top.addWidget(self.categoryBox)
        
        # Bottom Half
        bottom = QHBoxLayout()
        bottom.setAlignment(Qt.AlignCenter)
        
        ignoreLayout = QVBoxLayout()
        ignoreLabel = QLabel("Select Columns to Ignore (optional)", self)
        ignoreLabel.setFont(self.parent.SETUP_WIDGET_FONT)
        self.ignoreBox = ColumnList(self, self.parent)
        self.ignoreBox.setFont(self.parent.SETUP_WIDGET_FONT)
        ignoreLayout.addWidget(ignoreLabel)
        ignoreLayout.addWidget(self.ignoreBox)
     
        multiLayout = QVBoxLayout()
        multiLabel = QLabel("Select Columns for Multilinear (optional)", self)
        multiLabel.setFont(self.parent.SETUP_WIDGET_FONT)
        self.multiBox = ColumnList(self, self.parent)
        self.multiBox.setFont(self.parent.SETUP_WIDGET_FONT)
        multiLayout.addWidget(multiLabel)
        multiLayout.addWidget(self.multiBox)
        
        bottom.addLayout(ignoreLayout)
        bottom.addLayout(multiLayout)
        
        # Button
        buttonLayout = QHBoxLayout()
        buttonLayout.setAlignment(Qt.AlignCenter)
        
        runAnalysisButton = QPushButton("Run Analysis", self)
        runAnalysisButton.setFont(self.parent.NORMAL_BTN_FONT)
        runAnalysisButton.clicked.connect(self.run_analysis)
        
        buttonLayout.addWidget(runAnalysisButton)
        
        self.topLayout.addLayout(top, 4)
        self.topLayout.addLayout(bottom, 4)
        self.topLayout.addLayout(buttonLayout, 1)

    def run_analysis(self):
        if self.parent.is_df_loaded:
            cols = {
                "yCol": str(self.yBox.currentText()),
                "dateCol": str(self.dateBox.currentText()),
                "categoryCol": str(self.categoryBox.currentText()),
                "ignoreCols": [x.text() for x in self.ignoreBox.selectedItems()],
                "multiCols": [x.text() for x in self.multiBox.selectedItems()]
                }
            self.request_run_analysis_signal.emit(cols)
            
    @pyqtSlot(object, object, object, object)
    def analysis_complete(self, all_dates_model, by_date_cor_model, by_date_dp_model, cor_mat_model):
        self.parent.pages[MetricComparisonPage].build_all_dates_model(all_dates_model)
        self.parent.pages[MetricComparisonPage].build_by_date_cor_model(by_date_cor_model)
        self.parent.pages[MetricComparisonPage].build_by_date_dp_model(by_date_dp_model)
        self.parent.pages[MetricComparisonPage].build_all_dates_cor_mat_model(cor_mat_model)
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
        for box in self.transformation_boxes:
            box.remove_widget()

class FilterPage(QWidget):

    apply_filter_signal = pyqtSignal(filters.Filter)
    remove_filter_signal = pyqtSignal(filters.Filter)

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.filter_popups = []
        self.filter_boxes = []
        self.apply_filter_signal.connect(self.parent.dataWorker.apply_filter)
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
        for box in self.filter_boxes:
            box.remove_widget()
            
class DataPreviewPage(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        
        self.lineEdit  = QLineEdit(self)
        self.tableView = QTableView(self)
        self.comboBox  = ColumnComboBox(self, self.parent)
        self.label     = QLabel(self)
        self.btn       = QPushButton("Apply Filter", self)

        self.gridLayout = QGridLayout(self)
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.comboBox, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.btn, 0, 3, 1, 1)
        self.gridLayout.addWidget(self.tableView, 1, 0, 1, 4)

        self.label.setText("Filter")
        self.btn.clicked.connect(self.search_model)
        
        self.tableView.setSortingEnabled(True)
    
    def buildModel(self, model):
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(model)
        
        self.tableView.setModel(self.proxy)
    
 
    def search_model(self):
        text = self.lineEdit.text()
        index = self.comboBox.currentIndex()
        search = QRegExp(text,
                         Qt.CaseInsensitive,
                         QRegExp.RegExp
                         )
                         
        self.proxy.setFilterKeyColumn(index)
        self.proxy.setFilterRegExp(search)


class MetricComparisonPage(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.parent.dataWorker.correlation_scatter_complete_signal.connect(self.display_cor_scatter)
        self.parent.dataWorker.correlation_histogram_complete_signal.connect(self.display_cor_hist)

        # Create Tabs
        self.tabs = QTabWidget(self)
        # Create layout
        self.gridLayout = QGridLayout(self)
        self.gridLayout.addWidget(self.tabs, 0, 0)

        self.build_all_dates_tab()
        self.build_by_date_cor_tab()
        self.build_by_date_dp_tab()

    def build_by_date_cor_tab(self):
        byDatesTab = QTabWidget(self.tabs)
        self.tabs.addTab(byDatesTab, "Correlations - By Date")

        self.by_date_cor_le   = QLineEdit(byDatesTab)
        self.by_date_cor_view = QTableView(byDatesTab)
        label                 = QLabel(byDatesTab)
        # Set up grid
        gridLayout = QGridLayout(byDatesTab)
        gridLayout.addWidget(label, 0, 0, 1, 1)
        gridLayout.addWidget(self.by_date_cor_le, 0, 1, 1, 1)
        gridLayout.addWidget(self.by_date_cor_view, 1, 0, 1, 4)
        byDatesTab.setLayout(gridLayout)
        # Set object options
        label.setText("Filter")
        self.by_date_cor_le.textChanged.connect(self.filter_by_date_cor)

    def build_by_date_dp_tab(self):
        by_date_dp_tab = QTabWidget(self.tabs)
        self.tabs.addTab(by_date_dp_tab, "Data Points - By Date")

        self.by_date_dp_le   = QLineEdit(by_date_dp_tab)
        self.by_date_dp_view = QTableView(by_date_dp_tab)
        label                = QLabel(by_date_dp_tab)
        # Set up grid
        gridLayout = QGridLayout(by_date_dp_tab)
        gridLayout.addWidget(label, 0, 0, 1, 1)
        gridLayout.addWidget(self.by_date_dp_le, 0, 1, 1, 1)
        gridLayout.addWidget(self.by_date_dp_view, 1, 0, 1, 4)
        by_date_dp_tab.setLayout(gridLayout)
        # Set object options
        label.setText("Filter")
        self.by_date_dp_le.textChanged.connect(self.filter_by_date_dp)

    def build_all_dates_tab(self):
        self.allDatesTab = QTabWidget(self.tabs)
        self.tabs.addTab(self.allDatesTab, "Summary - All Dates")
        self.build_all_dates_table_tab()
        self.build_all_dates_hist_tab()
        self.build_all_dates_scatter_tab()
        self.build_all_dates_cor_mat_tab()

    def build_all_dates_table_tab(self):
        tableTab = QTabWidget(self.allDatesTab)
        self.allDatesTab.addTab(tableTab, "Table")

        self.all_dates_le     = QLineEdit(tableTab)
        self.allDatesView = QTableView(tableTab)
        label             = QLabel(tableTab)
        # Set up grid
        gridLayout = QGridLayout(tableTab)
        gridLayout.addWidget(label, 0, 0, 1, 1)
        gridLayout.addWidget(self.all_dates_le, 0, 1, 1, 1)
        gridLayout.addWidget(self.allDatesView, 1, 0, 1, 4)
        tableTab.setLayout(gridLayout)
        # Set object options
        label.setText("Filter")
        self.all_dates_le.textChanged.connect(self.filter_all_dates)

    def build_all_dates_cor_mat_tab(self):
        cor_mat_tab = QTabWidget(self.allDatesTab)
        self.allDatesTab.addTab(cor_mat_tab, "Correlation Matrix")

        self.cor_mat_le   = QLineEdit(cor_mat_tab)
        self.cor_mat_view = QTableView(cor_mat_tab)
        label             = QLabel(cor_mat_tab)
        # Set up grid
        gridLayout = QGridLayout(cor_mat_tab)
        gridLayout.addWidget(label, 0, 0, 1, 1)
        gridLayout.addWidget(self.cor_mat_le, 0, 1, 1, 1)
        gridLayout.addWidget(self.cor_mat_view, 1, 0, 1, 4)
        cor_mat_tab.setLayout(gridLayout)
        # Set object options
        label.setText("Filter")
        self.cor_mat_le.textChanged.connect(self.filter_cor_mat)

    def build_all_dates_hist_tab(self):
        self.histogram = Histogram(self.allDatesTab, self.parent)
        self.allDatesTab.addTab(self.histogram.view, "Histogram")

    def build_all_dates_scatter_tab(self):
        self.scatter = ScatterPlot(self.allDatesTab, self.parent)
        self.allDatesTab.addTab(self.scatter.view, "Scatter")

    def build_by_date_cor_model(self, model):
        self.by_dates_cor_proxy = QSortFilterProxyModel(self)
        self.by_dates_cor_proxy.setSourceModel(model)
        self.by_dates_cor_proxy.setFilterKeyColumn(0)
        
        self.by_date_cor_view.setModel(self.by_dates_cor_proxy)
        self.by_date_cor_view.setFont(self.parent.TABLE_FONT_MEDIUM)
        self.by_date_cor_view.horizontalHeader().setFont(self.parent.TABLE_FONT_LARGE)
        self.by_date_cor_view.setSortingEnabled(True)

    def build_by_date_dp_model(self, model):
        self.by_dates_dp_proxy = QSortFilterProxyModel(self)
        self.by_dates_dp_proxy.setSourceModel(model)
        self.by_dates_dp_proxy.setFilterKeyColumn(0)
        
        self.by_date_dp_view.setModel(self.by_dates_dp_proxy)
        self.by_date_dp_view.setFont(self.parent.TABLE_FONT_MEDIUM)
        self.by_date_dp_view.horizontalHeader().setFont(self.parent.TABLE_FONT_LARGE)
        self.by_date_dp_view.setSortingEnabled(True)

    def build_all_dates_model(self, model):
        self.all_dates_proxy = QSortFilterProxyModel(self)
        self.all_dates_proxy.setSourceModel(model)
        self.all_dates_proxy.setFilterKeyColumn(0)
        
        self.allDatesView.setModel(self.all_dates_proxy)
        self.allDatesView.setFont(self.parent.TABLE_FONT_MEDIUM)
        # self.allDatesView.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        # self.allDatesView.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        # self.allDatesView.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        # self.allDatesView.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        # self.allDatesView.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        # self.allDatesView.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.allDatesView.horizontalHeader().setFont(self.parent.TABLE_FONT_LARGE)
        self.allDatesView.setSortingEnabled(True)

    def build_all_dates_cor_mat_model(self, model):
        self.cor_mat_proxy = QSortFilterProxyModel(self)
        self.cor_mat_proxy.setSourceModel(model)
        self.cor_mat_proxy.setFilterKeyColumn(0)
        
        self.cor_mat_view.setModel(self.cor_mat_proxy)
        self.cor_mat_view.setFont(self.parent.TABLE_FONT_MEDIUM)
        self.cor_mat_view.horizontalHeader().setFont(self.parent.TABLE_FONT_LARGE)
 
    @pyqtSlot(pd.DataFrame)
    def display_cor_scatter(self, df):
        self.scatter.scatter(x=df["Data Points"], y=df["Correlation"])

    @pyqtSlot(list)
    def display_cor_hist(self, vals):
        self.histogram.histogram(vals)

    @QtCore.pyqtSlot(str)
    def filter_all_dates(self, text):
        text = self.all_dates_le.text()
        search = QRegExp(text,
                         Qt.CaseInsensitive,
                         QRegExp.RegExp
                         )
        self.all_dates_proxy.setFilterKeyColumn(0)
        self.all_dates_proxy.setFilterRegExp(search)

    @QtCore.pyqtSlot(str)
    def filter_by_date_cor(self, text):
        text = self.by_date_cor_le.text()
        search = QRegExp(text,
                         Qt.CaseInsensitive,
                         QRegExp.RegExp
                         )
        self.by_dates_cor_proxy.setFilterKeyColumn(0)
        self.by_dates_cor_proxy.setFilterRegExp(search)

    @QtCore.pyqtSlot(str)
    def filter_by_date_dp(self, text):
        text = self.by_date_dp_le.text()
        search = QRegExp(text,
                         Qt.CaseInsensitive,
                         QRegExp.RegExp
                         )
        self.by_dates_dp_proxy.setFilterKeyColumn(0)
        self.by_dates_dp_proxy.setFilterRegExp(search)

    @QtCore.pyqtSlot(str)
    def filter_cor_mat(self, text):
        text = self.cor_mat_le.text()
        search = QRegExp(text,
                         Qt.CaseInsensitive,
                         QRegExp.RegExp
                         )
        self.cor_mat_proxy.setFilterKeyColumn(0)
        self.cor_mat_proxy.setFilterRegExp(search)

class MetricDivePage(QWidget):

    request_run_metric_dive_signal = pyqtSignal(str)

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        
        # connect thread signals
        self.request_run_metric_dive_signal.connect(self.parent.dataWorker.run_metric_dive)
        self.parent.dataWorker.run_metric_dive_signal.connect(self.display_metric_dive)
        
        # Create layout
        self.gridLayout = QGridLayout(self)
        
        # Create Combo Box with column names
        self.xColBox = XColumnComboBox(self, self.parent)
        self.xColBox.setFont(QFont("Raleway", 16))
        self.xColBox.currentTextChanged.connect(self.request_run_metric_dive)
        
        # Create Tabs
        self.tabs = QTabWidget(self)
        self.plotTabs = QTabWidget(self)
        self.detailsTabs = QTabWidget(self)

        self.tabs.addTab(self.plotTabs, "Plots")
        self.tabs.addTab(self.detailsTabs, "Details")
        self.build_scatter_tab()
        
        # Add widgets to layout
        self.gridLayout.addWidget(self.xColBox, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.tabs, 1, 0, 1, 1)
    
    def build_scatter_tab(self):
        self.scatter = ScatterPlot(self.plotTabs, self.parent)
        self.plotTabs.addTab(self.scatter.view, "Scatter")
    
    def request_run_metric_dive(self, col):
        if col != '' and self.parent.is_df_loaded:
            self.request_run_metric_dive_signal.emit(col)
        
    @pyqtSlot(pd.DataFrame, dict)
    def display_metric_dive(self, df, cols):
        self.scatter.scatter(x=df[cols["x"]], y=df[cols["y"]])
   