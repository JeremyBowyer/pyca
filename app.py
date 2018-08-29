import sys
import logging
import datetime
import psutil

from models import *
from pages import *
from ui_objects import *
from threads import *

from styles import dark

import pandas as pd
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasAgg
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QSize 

from functools import partial

     
class Pyca(QMainWindow):
    PAGE_MASTER = [LoadingPage, SplashPage, SetupPage, TransformationPage, DataPreviewPage, MetricComparisonPage, MetricDivePage]
    
    # FONTS #
    TOOLBAR_FONT = QFont("Raleway", 14)
    NORMAL_BTN_FONT = QFont("Raleway", 14)
    LARGE_BTN_FONT = QFont("Raleway", 24)
    SETUP_WIDGET_FONT = QFont("Raleway", 12)
    TABLE_FONT_MEDIUM = QFont("Raleway", 12)
    TABLE_FONT_LARGE = QFont("Raleway", 16)
    
    # COLORS #
    BACKGROUND_PRIMARY = "#363636"
    BACKGROUND_SECONDARY = "#434343"
    FONT_COLOR = "#b4b4b4"
        
    columnLists = []
    
    def __init__(self):
        super(self.__class__, self).__init__()
        
        self.setMinimumSize(QSize(720, 575))    
        self.setWindowTitle("Python Correlation App")
        
        self.is_df_loaded = False
        self.is_loading = False
        self.start_data_worker()
        
        # Build UI
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)
        
        self.pages = {}
        for P in self.PAGE_MASTER:
            self.pages[P] = P(self)
            self.stack.addWidget(self.pages[P])
            
        self.build_status_bar()
        self.build_menu()
        self.build_toolbar()
        
        self.home() 
    
    def home(self):
        self.showMaximized()
        self.raise_page(SplashPage, force=True)
    
    def build_status_bar(self):
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusProgress = QtWidgets.QProgressBar()
        self.statusBar.addPermanentWidget(self.statusProgress)
        self.statusProgress.setGeometry(30, 40, 200, 25)
        
    def build_menu(self):
        self.mainMenu = self.menuBar()
        # File Menu
        # Load Data
        loadAction = QAction("&Load CSV", self)
        loadAction.setShortcut("Ctrl+L")
        loadAction.setStatusTip("Load a CSV dataset")
        loadAction.triggered.connect(self.load_csv)
                
        # Quit
        self.quitAction = QAction("&Quit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.setStatusTip("Leave the application")
        self.quitAction.triggered.connect(self.close_application)
        
        self.fileMenu = self.mainMenu.addMenu("&File")
        self.fileMenu.addAction(loadAction)
        self.fileMenu.addAction(self.quitAction)
    
    def build_toolbar(self):
        self.toolBar = QToolBar(self)
        self.addToolBar(Qt.LeftToolBarArea, self.toolBar)
        
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.toolBar.setIconSize(QSize(60, 60))
        
        setupAction = QAction(self)
        setupAction.setIcon(QIcon("icons/data-centre-white.png"))
        setupAction.setIconText("Setup")
        setupAction.triggered.connect(partial(self.raise_page, SetupPage))
        setupAction.setFont(self.TOOLBAR_FONT)

        transformationAction = QAction(self)
        transformationAction.setIcon(QIcon("icons/timeline-chart-white.png"))
        transformationAction.setIconText("Transformations")
        transformationAction.triggered.connect(partial(self.raise_page, TransformationPage))
        transformationAction.setFont(self.TOOLBAR_FONT)
        
        dataPreviewAction = QAction(self)
        dataPreviewAction.setIcon(QIcon("icons/data-table-white.png"))
        dataPreviewAction.setIconText("Data Preview")
        dataPreviewAction.triggered.connect(partial(self.raise_page, DataPreviewPage))
        dataPreviewAction.setFont(self.TOOLBAR_FONT)
        
        metricComparisonAction = QAction(self)
        metricComparisonAction.setIcon(QIcon("icons/data-network-white.png"))
        metricComparisonAction.setIconText("Metric Comparison")
        metricComparisonAction.triggered.connect(partial(self.raise_page, MetricComparisonPage))
        metricComparisonAction.setFont(self.TOOLBAR_FONT)
        
        metricDiveAction = QAction(self)
        metricDiveAction.setIcon(QIcon("icons/regression-analysis-white.png"))
        metricDiveAction.setIconText("Metric Dive")
        metricDiveAction.triggered.connect(partial(self.raise_page, MetricDivePage))
        metricDiveAction.setFont(self.TOOLBAR_FONT)
        
        self.toolBar.addAction(setupAction)
        self.toolBar.addAction(transformationAction)
        self.toolBar.addAction(dataPreviewAction)
        self.toolBar.addAction(metricComparisonAction)
        self.toolBar.addAction(metricDiveAction)
        
        for action in self.toolBar.actions():
            widget = self.toolBar.widgetForAction(action)
            widget.setFixedSize(190, 100)
    
    def save_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save data as CSV", "","CSV files (*.csv)", options=options)
        if fileName:
            self.df.to_csv(fileName, index=False)
    
    def load_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Upload a CSV file", "","CSV files (*.csv)", options=options)
        if fileName:
            self.start_task("Loading (this may take a few minutes)...")
            
            # Create Worker; Connect Signals
            self.loadWorker = LoadCsvWorker(self, fileName)
            self.loadWorker.df_signal.connect(self.dataWorker.load_df)
            self.loadWorker.df_model_signal.connect(self.load_df_model)
            self.loadWorker.loading_msg_signal.connect(self.update_loading_msg)
            
            # Create Thread
            thread = QThread(self)
            thread.started.connect(self.loadWorker.work)
            
            # Start Thread
            self.loadWorker.moveToThread(thread)
            thread.start()
    
    def start_task(self, msg):
        self.pages[LoadingPage].loadingLabel.setText(msg)
        self.pages[LoadingPage].loadingMessage.setText("")
        self.raise_page(LoadingPage, force=True)
        self.pages[LoadingPage].progress.setRange(0,0)
        self.is_loading = True
    
    def finished_task(self, page=None):
        self.is_loading = False
        self.pages[LoadingPage].progress.setRange(0,1)
        self.pages[LoadingPage].loadingMessage.setText("")
        if page:
            self.raise_page(page)
    
    def start_status_task(self, msg):
        self.statusBar.showMessage(msg)
        self.statusProgress.setRange(0,0)
        
    def finished_status_task(self, msg=""):
        self.statusBar.showMessage(msg)
        self.statusProgress.setRange(0,1)
    
    def raise_page(self, page, force=False):
        if (self.is_df_loaded and not self.is_loading) or force:
            idx = self.PAGE_MASTER.index(page)
            for i, action in enumerate(self.toolBar.actions()):
                font = action.font()
                font.setBold(idx - 2 == i)
                action.setFont(font)
            self.stack.setCurrentIndex(idx)
    
    def close_application(self):
        if not self.is_df_loaded:
            self.start_task("Closing application, please wait...")
            sys.exit()
            
        choice = QMessageBox.question(self, "Quit", "Are you sure?", QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            self.start_task("Closing application, please wait...")
            sys.exit()
        else:
            pass        
    
    def closeEvent(self, event):
        event.ignore()
        self.close_application()
    
    ########################
    # Multi-Thread Methods #
    ########################
    def start_data_worker(self):
        # Create Worker; Connect Signals
        self.dataWorker = DataWorker()
        self.dataWorker.update_cols_signal.connect(self.update_cols)
        self.dataWorker.loading_msg_signal.connect(self.update_loading_msg)

        # Create Thread
        self.dataThread = QThread(self)

        # Start Thread
        self.dataWorker.moveToThread(self.dataThread)
        self.dataThread.start()
    
    @pyqtSlot(object)
    def load_df_model(self, dfmodel):
        self.pages[DataPreviewPage].buildModel(dfmodel)
        
        self.is_df_loaded = True
        # Save Data Action
        saveAction = QAction("&Save CSV", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.setStatusTip("Save your data as CSV")
        saveAction.triggered.connect(self.save_csv)
        self.fileMenu.insertAction(self.quitAction, saveAction)
        self.finished_task(SetupPage)
    
    @pyqtSlot(list)
    def update_cols(self, cols):
        for colList in self.columnLists:
            colList.clear()
            colList.addItems(cols)
        
    @pyqtSlot(str)
    def update_loading_msg(self, msg):
        self.pages[LoadingPage].loadingMessage.setText(msg)

    
if __name__ == "__main__":
    try:
        app = QtWidgets.QApplication(sys.argv)
        dark(app)
        pyca = Pyca()
        sys.exit( app.exec_() )
    except Exception as e:
        now = datetime.datetime.now()
        logging.basicConfig(filename=f'logs/{now.strftime("%Y-%m-%d %H %M %S")}.log',level=logging.ERROR)
        logging.error("Error in app:", exc_info=True)
