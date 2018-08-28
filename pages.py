import numpy as np
import pandas as pd

from functools import partial

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
        self.parent.dataWorker.run_analysis_output_signal.connect(self.load_analysis)
        
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
            self.parent.start_task("Calculating summary statistics for metrics...")
            self.request_run_analysis_signal.emit(cols)
    
    def load_analysis(self, dfmodel):
        self.parent.pages[MetricComparisonPage].buildModel(dfmodel)
        self.parent.finished_task(MetricComparisonPage)
    
class TransformationPage(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
            
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
        
        self.lineEdit  = QLineEdit(self)
        self.tableView = QTableView(self)
        self.label     = QLabel(self)

        self.gridLayout = QGridLayout(self)
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.tableView, 1, 0, 1, 4)

        self.label.setText("Filter")
        self.lineEdit.textChanged.connect(self.on_le_change)
        
        self.tableView.setSortingEnabled(True)
    
    def buildModel(self, model):
        
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(model)
        self.proxy.setFilterKeyColumn(0)
        
        self.tableView.setModel(self.proxy)
        self.tableView.setFont(self.parent.TABLE_FONT_MEDIUM)
        self.tableView.resizeColumnsToContents()
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableView.horizontalHeader().setFont(self.parent.TABLE_FONT_LARGE)
 
    @QtCore.pyqtSlot(str)
    def on_le_change(self, text):
        text = self.lineEdit.text()
        search = QRegExp(text,
                         Qt.CaseInsensitive,
                         QRegExp.RegExp
                         )
        self.proxy.setFilterRegExp(search)

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
        self.xColBox = ColumnComboBox(self, self.parent)
        self.xColBox.setFont(QFont("Raleway", 16))
        self.xColBox.currentTextChanged.connect(self.request_run_metric_dive)
        
        # Create Tabs
        self.tabs = QTabWidget(self)
        self.scatterTab = QWidget()
        self.histTab = QWidget()
            
        self.tabs.addTab(self.scatterTab,"Scatter")
        self.tabs.addTab(self.histTab,"Histogram")
        self.build_scatter_tab()
        
        # Add widgets to layout
        self.gridLayout.addWidget(self.xColBox, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.tabs, 1, 0, 1, 1)
    
    def build_scatter_tab(self):
        self.canvas = ScatterCanvas(self.scatterTab, width=8, height=4) 
        self.toolbar = NavigationToolbar(self.canvas, self.scatterTab)
    
    def request_run_metric_dive(self, col):
        if col != '' and self.parent.is_df_loaded:
            print("requesting: "+col)
            self.request_run_metric_dive_signal.emit(col)
        
    @pyqtSlot(pd.DataFrame, dict)
    def display_metric_dive(self, df, cols):
        self.canvas.scatter(df, x=cols["x"], y=cols["y"], size=cols["size"])
        print("done")
   