import numpy as np
import pandas as pd
from scipy.stats import linregress
import time
import psutil
import csv

from models import NumpyModel, PandasModel
from ui_objects import ScatterCanvas

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtGui import *


class Worker(QObject):
    
    def __init__(self):
        super().__init__()
        self.__abort = False
    
    def abort(self):
        self.__abort = True


class DataWorker(Worker):
    
    df_loaded_signal = pyqtSignal()
    update_cols_signal = pyqtSignal(list)
    run_analysis_output_signal = pyqtSignal(object)
    loading_msg_signal = pyqtSignal(str)
    
    run_metric_dive_signal = pyqtSignal(pd.DataFrame, dict)
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.yCol = ""
        
    @pyqtSlot(pd.DataFrame)
    def load_df(self, df):
        self.df = df
        self.df_loaded_signal.emit()
        self.update_cols_signal.emit(list(self.df.columns))
    
    @pyqtSlot(dict)
    def run_analysis(self, cols):
        if self.df_is_loaded():
            # Store y column for metric dive
            self.yCol = cols["yCol"]
            
            # Create column lists
            ignoreCols = [cols["yCol"]] + [cols["dateCol"]] + [cols["categoryCol"]] + cols["ignoreCols"]
            xCols = [col for col in self.df.columns if col not in ignoreCols]
            
            # Coerce Y column to numeric
            self.df[cols["yCol"]] = pd.to_numeric(self.df[cols["yCol"]], errors="coerce")
            
            metrics = []
            correls = []
            rsqrds = []
            slopes = []
            pvalues = []
            counts = []
            
            cnt = 1
            n_cols = len(xCols)
            for metric in xCols:
                # Loading message
                # msg = str(cnt) + " out of " + str(n_cols) + ": " + metric
                # msg = (msg[:50] + '...') if len(msg) > 50 else msg
                # self.loading_msg_signal.emit(msg)
                # cnt += 1
                
                metric_df = pd.DataFrame({
                    "x":pd.to_numeric(self.df[metric], errors="coerce"),
                    "y":self.df[cols["yCol"]]
                    })
                metric_df.dropna(inplace=True)
                
                count = metric_df.shape[0]
                
                if count > 1:
                    x = metric_df["x"]
                    y = metric_df["y"]

                    slope, intercept, r_value, p_value, std_err = linregress(x,y)
                    if r_value != 0:
                        metrics.append(metric)
                        correls.append(r_value)
                        rsqrds.append(r_value*r_value)
                        slopes.append(slope)
                        pvalues.append(p_value)
                        counts.append(count)
            
            comparison_df = pd.DataFrame({
                "Metric": metrics,
                "Correlation": correls,
                "R-Squared": rsqrds,
                "Slope": slopes,
                "P-Value": pvalues,
                "Data Points": counts
                })
            comparison_df = comparison_df[["Metric",
                                           "Correlation",
                                           "R-Squared",
                                           "Slope",
                                           "P-Value",
                                           "Data Points"]]
            comparison_df = comparison_df.round({
                "Correlation": 4,
                "R-Squared": 4,
                "Slope": 4,
                "P-Value": 4
            })
            
            dfmodel = NumpyModel(comparison_df, format_cols={1:(True, True), 2:(False, True), 3:(True, True), 4:(False, False), 5:(False, True)})
            self.run_analysis_output_signal.emit(dfmodel)

    def df_is_loaded(self):
        return(self.df is not None)
    
    @pyqtSlot(str)
    def run_metric_dive(self, col):
        if self.df_is_loaded() and self.yCol != '':
            cols = {
                "x": col,
                "y": self.yCol,
                "size": "price"
                }
                
            self.metric_df = self.df[[col, self.yCol, "price"]]
            
            self.metric_df[col] = pd.to_numeric(self.metric_df[col], errors="coerce")
            self.metric_df[self.yCol] = pd.to_numeric(self.metric_df[self.yCol], errors="coerce")
            self.metric_df["price"] = pd.to_numeric(self.metric_df["price"], errors="coerce")
            self.metric_df.dropna(inplace=True)
            self.run_metric_dive_signal.emit(self.metric_df, cols)
    
    
class LoadCsvWorker(Worker):
    
    df_signal = pyqtSignal(pd.DataFrame)
    df_model_signal = pyqtSignal(object)
    loading_msg_signal = pyqtSignal(str)
    
    def __init__(self, parent, fileName):
        super().__init__()
        self.parent = parent
        self.fileName = fileName

    @pyqtSlot()
    def work(self):
        data = []
        row_limit = 50000
        starting_mem = psutil.virtual_memory().percent
        available_mem = 100-starting_mem
        with open(self.fileName) as f:
            reader = csv.reader(f, quotechar='"')
            cnt = 0
            for i,x in enumerate(reader):
                cnt += 1
                data.append(x)
                # Every 1,000 rows, check memory
                if i%1000 == 0:
                    mem_usage = psutil.virtual_memory().percent
                    self.loading_msg_signal.emit("Rows Loaded: " + str(i) + "   Memory Usage: " + str(mem_usage) + "%")
                    delta_mem = mem_usage - starting_mem
                    # if memory usage is over 95%, stop process
                    if mem_usage > 95:
                        break
                    # if the current process has used over 60%
                    # of available memory, add current data to
                    # dataframe (or create dataframe) and continue
                    if delta_mem >= available_mem*0.6:
                        if cnt == len(data):
                            cols = data.pop(0)
                            df = pd.DataFrame(data, columns=cols)
                        else:
                            temp = pd.DataFrame(data, columns=cols)
                            df = pd.concat([df,temp])
                        # Reset variables for next iteration
                        data = []
                        temp = None
                        starting_mem = psutil.virtual_memory().percent
                        available_mem = 100-starting_mem
         
        # If previous loop never used enough memory
        # to warrant offloading the data to the 
        # dataframe, do so now.
        if len(data) > 0:
            if cnt == len(data):
                cols = data.pop(0)
                df = pd.DataFrame(data, columns=cols)
            else:
                temp = pd.DataFrame(data, columns=cols)
                df = pd.concat([df,temp])
            data = []
            temp = None
         
        self.df_signal.emit(df)
        dfmodel = NumpyModel(df)
        self.df_model_signal.emit(dfmodel)