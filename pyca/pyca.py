import sys
import cProfile
import datetime
import psutil
import os
import shutil
import pandas as pd
import numpy as np
from functools import partial
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QStackedWidget, QAction, QToolBar, QFileDialog, QMessageBox
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QSize, pyqtSignal, pyqtSlot, QThread, Qt

import warnings
warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficent is not defined.")
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

# Pyca modules
from pages import LoadingPage, SplashPage, SetupPage, DataPreviewPage, TransformationPage, FilterPage, DateAggregationPage, MetricComparisonPage, MetricDivePage
from threads import DataWorker, LoadCsvWorker, ReportWorker
from decorators import pyca_profile
from styles import dark


class PycaApp(QtWidgets.QApplication):
    pass
    # t = QElapsedTimer()

    # def notify(self, receiver, event):
    #     self.t.start()
    #     ret = QApplication.notify(self, receiver, event)
    #     if(self.t.elapsed() > 10):
    #         print(f"processing event type {event.type()} for object {receiver.objectName()} " 
    #               f"took {self.t.elapsed()}ms")
    #     return ret


class PycaPalette:
    def __init__(self):
        self.current = -1

    def __getitem__(self, idx):
        return self.PALETTE[idx % len(self.PALETTE)]

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < len(self.PALETTE):
            return self.PALETTE[self.current]
        raise StopIteration


class Rgb:
    def __init__(self, r, g, b, a=None):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def get_tuple(self):
        if self.a is None:
            return (self.r, self.g, self.b)
        else:
            return (self.r, self.g, self.b, self.a)

    def get_string(self):
        if self.a is None:
            return f"RGB({self.r}, {self.g}, {self.b})"
        else:
            return f"RGBA({self.r}, {self.g}, {self.b}, {self.a})"

    def set_r(self, r):
        self.r = r
        return self

    def set_g(self, g):
        self.g = g
        return self

    def set_b(self, b):
        self.b = b
        return self

    def set_a(self, a):
        self.a = a
        return self

class RgbPalette(PycaPalette):
    def __init__(self):
        super(RgbPalette, self).__init__()
        self.PALETTE = [
            Rgb(38, 70, 83),
            Rgb(42, 157, 143),
            Rgb(233, 196, 106),
            Rgb(244, 162, 97),
            Rgb(231, 111, 81)
        ]


class HexPalette(PycaPalette):
    def __init__(self):
        super(HexPalette, self).__init__()
        self.PALETTE = [
            '#5032FF',
            '#FFCCFF',
            '#2a9d8f',
            '#e9c46a',
            '#f4a261',
            '#e76f51'
        ]


class Pyca(QMainWindow):
    # None's represent separator on the toolbar
    PAGE_MASTER = [LoadingPage, SplashPage, DataPreviewPage, TransformationPage, FilterPage, DateAggregationPage, None, SetupPage, MetricComparisonPage, None, MetricDivePage]
    
    # FONTS #
    PLOT_FONT_FAMILY  = "Courier New"
    FONT_BASE         = "Raleway"
    TOOLBAR_FONT      = QFont("Raleway", 14)
    SMALL_FONT        = QFont("Raleway", 12)
    NORMAL_FONT       = QFont("Raleway", 16)
    LARGE_FONT        = QFont("Raleway", 24)
    NORMAL_BTN_FONT   = QFont("Raleway", 14)
    LARGE_BTN_FONT    = QFont("Raleway", 24)
    SETUP_WIDGET_FONT = QFont("Raleway", 12)
    TABLE_FONT_MEDIUM = QFont("Raleway", 12)
    TABLE_FONT_LARGE  = QFont("Raleway", 16)
    
    # COLORS #
    BACKGROUND_PRIMARY        = "#363636"
    BACKGROUND_SECONDARY      = "#434343"
    BACKGROUND_SECONDARY_RGBA = "rgba(67, 67, 67,1)"
    FONT_COLOR                = "#b4b4b4"
    FONT_COLOR_SECONDARY      = "#FFFFFF"

    PLOT_PALETTE_RGB = RgbPalette()
    PLOT_PALETTE_HEX = HexPalette()

    request_save_df_signal = pyqtSignal(str)

    column_lists = []
    
    def __init__(self):
        super(self.__class__, self).__init__()
        
        self.setMinimumSize(QSize(720, 575))    
        self.setWindowTitle("Python Correlation App")

        self.is_df_loaded = False
        self.is_loading = False
        self.start_data_worker()
        self.start_report_worker()
        self.connect_signals()
        
        self.all_cols = []
        self.x_cols   = []

        # Build UI            
        self.build_main_layout()
        self.build_stack()
        self.build_status_bar()
        self.build_menu()
        self.build_toolbar()
        
        self.home() 
    
    def home(self):
        self.showMaximized()
        self.raise_page(SplashPage, force=True)
    
    def build_main_layout(self):
        self.main_view = QWidget(self)
        self.main_view_layout = QVBoxLayout()
        self.main_view.setLayout(self.main_view_layout)
        self.setCentralWidget(self.main_view)

        self.y_col_label = QLabel("")
        self.y_col_label.setFont(self.NORMAL_FONT)
        self.y_col_label.setWordWrap(True)
        self.main_view_layout.addWidget(self.y_col_label)

    def build_stack(self):
        self.stack = QStackedWidget(self)
        self.main_view_layout.addWidget(self.stack)

        self.pages = {}
        for P in self.PAGE_MASTER:
            if P is None:
                self.stack.addWidget(QWidget())
                continue
            self.pages[P] = P(self)
            self.stack.addWidget(self.pages[P])

    def build_status_bar(self):
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusProgress = QtWidgets.QProgressBar()
        self.statusBar.addPermanentWidget(self.statusProgress)
        
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
        setupAction.setCheckable(True)
        setupAction.setIcon(QIcon("icons/data-centre-white.png"))
        setupAction.setIconText("Run Analysis")
        setupAction.triggered.connect(partial(self.raise_page, SetupPage))
        setupAction.setFont(self.TOOLBAR_FONT)

        dataPreviewAction = QAction(self)
        dataPreviewAction.setCheckable(True)
        dataPreviewAction.setIcon(QIcon("icons/data-table-white.png"))
        dataPreviewAction.setIconText("Data Preview")
        dataPreviewAction.triggered.connect(partial(self.raise_page, DataPreviewPage))
        dataPreviewAction.setFont(self.TOOLBAR_FONT)

        transformationAction = QAction(self)
        transformationAction.setCheckable(True)
        transformationAction.setIcon(QIcon("icons/timeline-chart-white.png"))
        transformationAction.setIconText("Transformations")
        transformationAction.triggered.connect(partial(self.raise_page, TransformationPage))
        transformationAction.setFont(self.TOOLBAR_FONT)

        filterAction = QAction(self)
        filterAction.setCheckable(True)
        filterAction.setIcon(QIcon("icons/filter.png"))
        filterAction.setIconText("Filters")
        filterAction.triggered.connect(partial(self.raise_page, FilterPage))
        filterAction.setFont(self.TOOLBAR_FONT)

        dateAggregationAction = QAction(self)
        dateAggregationAction.setCheckable(True)
        dateAggregationAction.setIcon(QIcon("icons/pyramid-white.png"))
        dateAggregationAction.setIconText("Date Aggregation")
        dateAggregationAction.triggered.connect(partial(self.raise_page, DateAggregationPage))
        dateAggregationAction.setFont(self.TOOLBAR_FONT)
        
        metricComparisonAction = QAction(self)
        metricComparisonAction.setCheckable(True)
        metricComparisonAction.setIcon(QIcon("icons/data-network-white.png"))
        metricComparisonAction.setIconText("Metric Comparison")
        metricComparisonAction.triggered.connect(partial(self.raise_page, MetricComparisonPage))
        metricComparisonAction.setFont(self.TOOLBAR_FONT)
        
        metricDiveAction = QAction(self)
        metricDiveAction.setCheckable(True)
        metricDiveAction.setIcon(QIcon("icons/regression-analysis-white.png"))
        metricDiveAction.setIconText("Metric Dive")
        metricDiveAction.triggered.connect(partial(self.raise_page, MetricDivePage))
        metricDiveAction.setFont(self.TOOLBAR_FONT)
        
        self.toolBar.addAction(dataPreviewAction)
        self.toolBar.addAction(transformationAction)
        self.toolBar.addAction(filterAction)
        self.toolBar.addAction(dateAggregationAction)
        self.toolBar.addSeparator()
        self.toolBar.addAction(setupAction)
        self.toolBar.addAction(metricComparisonAction)
        self.toolBar.addSeparator()
        self.toolBar.addAction(metricDiveAction)
        
        for action in self.toolBar.actions():
            widget = self.toolBar.widgetForAction(action)
            widget.setFixedSize(190, 100)
    
    def save_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save data as CSV", "","CSV files (*.csv)", options=options)
        if fileName:
            self.request_save_df_signal.emit(fileName)
    
    def load_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Upload a CSV file", "","CSV files (*.csv)", options=options)
        if fileName:
            self.start_task("Loading (this may take a few minutes)...")
            
            # Create Worker; Connect Signals
            self.loadWorker = LoadCsvWorker(self, fileName)
            self.loadWorker.loaded_df_signal.connect(self.dataWorker.load_df)
            self.loadWorker.loaded_df_signal.connect(self.pages[SetupPage].update_df_name)
            self.loadWorker.loaded_df_model_signal.connect(self.load_df_model)
            self.loadWorker.loading_msg_signal.connect(self.update_loading_msg)
            self.loadWorker.finish_task_signal.connect(self.finished_task)
            self.loadWorker.show_error_signal.connect(self.show_error)
            
            # Create Thread
            thread = QThread(self)
            thread.started.connect(self.loadWorker.work)
            
            # Start Thread
            self.loadWorker.moveToThread(thread)
            thread.start()
    
    def update_y_col(self, col):
        self.y_col_label.setText("Y Column: " + col)

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
        if (self.is_df_loaded and not self.is_loading) or True:
            idx = self.PAGE_MASTER.index(page)
            for i, action in enumerate(self.toolBar.actions()):
                selected = idx - 2 == i
                
                font = action.font()
                font.setBold(selected)
                font.setOverline(selected)
                font.setUnderline(selected)
                action.setFont(font)

                action.setChecked(selected)
                
            self.stack.setCurrentIndex(idx)
    
    def close_application(self):
        if not self.is_df_loaded:
            self.start_task("Closing application, please wait...")
            self.clear_tmp_files()
            sys.exit()
            
        choice = QMessageBox.question(self, "Quit", "Are you sure?", QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            self.start_task("Closing application, please wait...")
            self.clear_tmp_files()
            sys.exit()
        else:
            pass        
    
    def clear_tmp_files(self):
        folder = 'pyca/tmp'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def closeEvent(self, event):
        event.ignore()
        self.close_application()
    
    ########################
    # Multi-Thread Methods #
    ########################
    def start_data_worker(self):
        # Create Worker
        self.dataWorker = DataWorker()

        # Create Thread
        self.dataThread = QThread(self)

        # Start Thread
        self.dataWorker.moveToThread(self.dataThread)
        self.dataThread.start()
    
    def start_report_worker(self):
        self.reportWorker = ReportWorker(self.PLOT_PALETTE_HEX)

    def connect_signals(self):
        # Data Worker Signals
        self.dataWorker.update_cols_signal.connect(self.update_cols)
        self.dataWorker.loading_msg_signal.connect(self.update_loading_msg)
        self.dataWorker.start_task_signal.connect(self.start_task)
        self.dataWorker.start_status_task_signal.connect(self.start_status_task)
        self.dataWorker.finish_status_task_signal.connect(self.finished_status_task)
        self.dataWorker.show_error_signal.connect(self.show_error)
        self.dataWorker.data_preview_model_signal.connect(self.update_df_model)
        self.dataWorker.data_loaded_signal.connect(self.data_loaded)
        self.dataWorker.metric_dive_report_signal.connect(self.reportWorker.generate_report)
        self.dataWorker.y_col_change_signal.connect(self.update_y_col)

        # Report Worker Signals
        self.reportWorker.start_status_task_signal.connect(self.start_status_task)
        self.reportWorker.finish_status_task_signal.connect(self.finished_status_task)

        # Self Signals
        self.request_save_df_signal.connect(self.dataWorker.save_df)

    def data_loaded(self):
        self.pages[TransformationPage].remove_all_transformations()
        self.pages[FilterPage].remove_all_filters()

    @pyqtSlot(str)
    def show_error(self, msg):
        QMessageBox.about(self, "Warning", msg)

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

    @pyqtSlot(object)
    def update_df_model(self, dfmodel):
        self.pages[DataPreviewPage].buildModel(dfmodel)
    
    @pyqtSlot(list, list)
    def update_cols(self, all_cols, x_cols):
        self.all_cols = all_cols
        self.x_cols   = x_cols
        for colList in self.column_lists:
            colList.update_cols(all_cols, x_cols)
        
    @pyqtSlot(str)
    def update_loading_msg(self, msg):
        self.pages[LoadingPage].loadingMessage.setText(msg)

    
if __name__ == "__main__":
    app = PycaApp(sys.argv)
    dark(app)
    pyca = Pyca()
    sys.exit( app.exec_() )
    # import logging
    # now = datetime.datetime.now()
    # logging.basicConfig(filename=f'logs/{now.strftime("%Y-%m-%d %H %M %S")}.log',level=logging.ERROR)
    # logging.error("Error in app:", exc_info=True)
